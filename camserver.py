#!/usr/bin/env python3
"""
camserver.py — Dead-simple webcam broadcaster.

Pick one protocol at launch:
  python camserver.py http   →  MJPEG stream at http://0.0.0.0:1080/stream
  python camserver.py rtsp   →  RTSP stream at rtsp://localhost:1081/cam  (needs ffmpeg + mediamtx)

Camera backends:
  --backend cv2   →  OpenCV V4L2 (USB webcams)
  --backend zed   →  ZED SDK (ZED X / ZED X Mini on Jetson)
  auto (default)  →  tries ZED first, falls back to OpenCV
"""

import argparse
import logging
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
from flask import Flask, Response, render_template_string

try:
    import pyzed.sl as sl

    HAS_ZED = True
except ImportError:
    HAS_ZED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("camserver")

# ── Shared state ────────────────────────────────────────────────────────────

latest_raw_frame = None
latest_jpeg: bytes | None = None
frame_lock = threading.Lock()
shutdown_event = threading.Event()
stream_fps: int = 30

# ── Camera capture ─────────────────────────────────────────────────────────

def capture_loop(camera_index: int, width: int | None, height: int | None, fps: int = 30):
    global latest_raw_frame, latest_jpeg

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open camera %d", camera_index)
        if platform.system() == "Darwin":
            log.error(
                "On macOS, ensure Terminal has camera permission: "
                "System Settings → Privacy & Security → Camera"
            )
        shutdown_event.set()
        return

    cap.set(cv2.CAP_PROP_FPS, fps)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Camera %d opened at %dx%d @ %.0f fps", camera_index, actual_w, actual_h, actual_fps)

    while not shutdown_event.is_set():
        ok, frame = cap.read()
        if not ok:
            log.warning("Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_raw_frame = frame
            if ok:
                latest_jpeg = jpeg.tobytes()

    cap.release()
    log.info("Camera released.")

# ── ZED SDK capture ────────────────────────────────────────────────────────

def zed_capture_loop(camera_index: int, width: int | None, height: int | None, fps: int = 30):
    global latest_raw_frame, latest_jpeg

    cam = sl.Camera()
    params = sl.InitParameters()
    params.camera_fps = fps
    params.depth_mode = sl.DEPTH_MODE.NONE
    params.camera_resolution = sl.RESOLUTION.AUTO

    target_size = (width, height) if (width and height) else None

    # Select camera by index (serial number)
    devs = sl.Camera.get_device_list()
    available = [d for d in devs if str(d.camera_state) == "AVAILABLE"]
    if not available:
        log.error("No ZED cameras available. Try: sudo systemctl restart zed_x_daemon")
        shutdown_event.set()
        return
    if camera_index >= len(available):
        log.error("ZED camera index %d out of range (%d available)", camera_index, len(available))
        shutdown_event.set()
        return

    dev = available[camera_index]
    params.set_from_serial_number(dev.serial_number)
    log.info("Opening ZED %s (serial %d)...", dev.camera_model, dev.serial_number)

    status = cam.open(params)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error("ZED open failed: %s", status)
        shutdown_event.set()
        return

    info = cam.get_camera_information()
    res = info.camera_configuration.resolution
    log.info("ZED opened at %dx%d @ %d fps", res.width, res.height, fps)
    if target_size:
        log.info("Resizing output to %dx%d", target_size[0], target_size[1])

    image = sl.Mat()
    runtime = sl.RuntimeParameters()

    while not shutdown_event.is_set():
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3].copy()  # BGRA → BGR

            if target_size:
                frame = cv2.resize(frame, target_size)

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with frame_lock:
                latest_raw_frame = frame
                if ok:
                    latest_jpeg = jpeg.tobytes()
        else:
            time.sleep(0.01)

    cam.close()
    log.info("ZED camera closed.")

# ── HTTP mode (Flask MJPEG) ────────────────────────────────────────────────

app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>camserver</title>
  <style>
    body { background: #111; color: #eee; font-family: monospace;
           display: flex; flex-direction: column; align-items: center;
           margin-top: 2em; }
    img  { max-width: 95vw; border: 2px solid #444; }
  </style>
</head>
<body>
  <h2>📷 camserver</h2>
  <img src="/stream" alt="webcam stream" />
  <p style="margin-top:1em;color:#888;">MJPEG stream: <code>/stream</code> · Snapshot: <code>/snapshot</code></p>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


def mjpeg_generator():
    interval = 1.0 / stream_fps
    while not shutdown_event.is_set():
        with frame_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(interval)


@app.route("/stream")
def stream():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/snapshot")
def snapshot():
    with frame_lock:
        frame = latest_jpeg
    if frame is None:
        return "No frame available", 503
    return Response(frame, mimetype="image/jpeg")


def run_http(port: int):
    log.info("HTTP stream:  http://0.0.0.0:%d/stream", port)
    log.info("Web UI:       http://0.0.0.0:%d/", port)
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)

# ── RTSP mode (FFmpeg → mediamtx) ─────────────────────────────────────────

def run_rtsp(port: int):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        log.error("ffmpeg not found. Install with: brew install ffmpeg")
        shutdown_event.set()
        return

    # Wait for first frame so we know the resolution
    while latest_raw_frame is None and not shutdown_event.is_set():
        time.sleep(0.1)
    if shutdown_event.is_set():
        return

    with frame_lock:
        h, w = latest_raw_frame.shape[:2]

    rtsp_url = f"rtsp://localhost:{port}/cam"
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(stream_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]
    log.info("RTSP publishing to %s", rtsp_url)
    log.info("Connect with VLC: rtsp://localhost:%d/cam", port)

    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        log.error("ffmpeg launch failed.")
        shutdown_event.set()
        return

    while not shutdown_event.is_set():
        with frame_lock:
            frame = latest_raw_frame
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            log.warning("FFmpeg pipe broken — RTSP stream stopped.")
            break

    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=5)
    log.info("FFmpeg process terminated.")

# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Webcam broadcaster (HTTP or RTSP)")
    p.add_argument("protocol", choices=["http", "rtsp"], help="Stream protocol")
    p.add_argument("--port", type=int, default=None, help="Port (default: 1080 for http, 1081 for rtsp)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--resolution", type=str, default=None, help="WxH e.g. 1280x720")
    p.add_argument("--fps", type=int, default=30, help="Capture frame rate (default: 30)")
    p.add_argument(
        "--backend", choices=["auto", "cv2", "zed"], default="auto",
        help="Camera backend: auto (default), cv2 (OpenCV/V4L2), zed (ZED SDK)",
    )
    return p.parse_args()


def main():
    global stream_fps
    args = parse_args()

    port = args.port
    if port is None:
        port = 1080 if args.protocol == "http" else 1081

    width, height = None, None
    if args.resolution:
        try:
            width, height = (int(x) for x in args.resolution.split("x"))
        except ValueError:
            log.error("Invalid resolution format. Use WxH, e.g. 1280x720")
            sys.exit(1)

    def _signal_handler(sig, _frame):
        log.info("Shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Select capture backend
    backend = args.backend
    if backend == "auto":
        if HAS_ZED:
            devs = sl.Camera.get_device_list()
            available = [d for d in devs if str(d.camera_state) == "AVAILABLE"]
            if available:
                backend = "zed"
                log.info("Auto-detected %d ZED camera(s), using ZED backend", len(available))
            else:
                backend = "cv2"
                log.info("No ZED cameras available, falling back to OpenCV")
        else:
            backend = "cv2"

    if backend == "zed" and not HAS_ZED:
        log.error("ZED SDK not installed. Install pyzed or use --backend cv2")
        sys.exit(1)

    # Start camera capture
    if backend == "zed":
        capture_fn = zed_capture_loop
    else:
        capture_fn = capture_loop

    stream_fps = args.fps

    cam_thread = threading.Thread(
        target=capture_fn, args=(args.camera, width, height, args.fps), daemon=True,
    )
    cam_thread.start()
    time.sleep(1.0)
    if shutdown_event.is_set():
        sys.exit(1)

    # Run selected protocol
    if args.protocol == "http":
        flask_thread = threading.Thread(target=run_http, args=(port,), daemon=True)
        flask_thread.start()
    else:
        rtsp_thread = threading.Thread(target=run_rtsp, args=(port,), daemon=True)
        rtsp_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()

    log.info("Goodbye.")


if __name__ == "__main__":
    main()
