#!/usr/bin/env python3
"""
devcam server — Dead-simple webcam broadcaster.

Pick one protocol at launch:
  python -m devcam http   →  MJPEG stream at http://0.0.0.0:1080/stream
  python -m devcam rtsp   →  RTSP stream at rtsp://localhost:1081/cam  (needs ffmpeg + mediamtx)

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
from flask import Flask, Response, jsonify, render_template_string, request

from devcam.detection import HumanDetector

try:
    import pyzed.sl as sl

    HAS_ZED = True
except ImportError:
    HAS_ZED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("devcam")

# ── Shared state ────────────────────────────────────────────────────────────

latest_raw_frame = None
latest_jpeg: bytes | None = None
frame_lock = threading.Lock()
shutdown_event = threading.Event()
stream_fps: int = 30

camera_control_lock = threading.Lock()
capture_stop_event: threading.Event | None = None
capture_thread: threading.Thread | None = None
capture_fn = None
active_backend = "cv2"
camera_config = {
    "camera": 0,
    "width": None,
    "height": None,
    "fps": 30,
    "enabled": True,
    "mirror": False,
}
camera_status = {
    "running": False,
    "error": None,
    "actual_width": None,
    "actual_height": None,
    "actual_fps": None,
}

COMMON_RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]
COMMON_FPS = [5, 10, 15, 24, 30, 60]

detection_lock = threading.Lock()
human_detector = HumanDetector()
detection_config = {
    "enabled": False,
    "confidence": 0.5,
    "interval_ms": 1000,
    "model": "yolov8n.pt",
}


def clear_latest_frame():
    global latest_raw_frame, latest_jpeg
    with frame_lock:
        latest_raw_frame = None
        latest_jpeg = None


def update_camera_status(**updates):
    with camera_control_lock:
        camera_status.update(updates)


def current_state():
    with camera_control_lock:
        state = {
            "backend": active_backend,
            "camera": camera_config["camera"],
            "width": camera_config["width"],
            "height": camera_config["height"],
            "fps": camera_config["fps"],
            "enabled": camera_config["enabled"],
            "mirror": camera_config["mirror"],
            **camera_status,
        }
    with frame_lock:
        state["has_frame"] = latest_jpeg is not None
    return state


def detection_state():
    with detection_lock:
        config = dict(detection_config)
    return {
        **config,
        "confidence_threshold": config["confidence"],
        "available": human_detector.available,
        "loaded": human_detector.loaded,
        "error": human_detector.error if human_detector.available else "ultralytics is not installed",
    }


def update_detection_config(data: dict):
    with detection_lock:
        if "enabled" in data:
            detection_config["enabled"] = bool(data["enabled"])
        if "confidence" in data:
            confidence = float(data["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence must be between 0 and 1")
            detection_config["confidence"] = confidence
        if "interval_ms" in data:
            interval_ms = int(data["interval_ms"])
            if not 50 <= interval_ms <= 5000:
                raise ValueError("Detection interval must be between 50 and 5000 ms")
            detection_config["interval_ms"] = interval_ms
        if "model" in data and data["model"]:
            detection_config["model"] = str(data["model"])
        human_detector.configure(
            confidence_threshold=detection_config["confidence"],
            model_name=detection_config["model"],
        )
        return dict(detection_config)

# ── Camera capture ─────────────────────────────────────────────────────────

def capture_loop(
    camera_index: int,
    width: int | None,
    height: int | None,
    fps: int = 30,
    stop_event: threading.Event | None = None,
):
    global latest_raw_frame, latest_jpeg
    stop_event = stop_event or shutdown_event

    clear_latest_frame()
    update_camera_status(
        running=False,
        error=None,
        actual_width=None,
        actual_height=None,
        actual_fps=None,
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        error = f"Cannot open camera {camera_index}"
        log.error(error)
        if platform.system() == "Darwin":
            log.error(
                "On macOS, ensure Terminal has camera permission: "
                "System Settings → Privacy & Security → Camera"
            )
        update_camera_status(running=False, error=error)
        return

    cap.set(cv2.CAP_PROP_FPS, fps)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Camera %d opened at %dx%d @ %.0f fps", camera_index, actual_w, actual_h, actual_fps)
    update_camera_status(
        running=True,
        error=None,
        actual_width=actual_w,
        actual_height=actual_h,
        actual_fps=round(actual_fps or fps, 1),
    )

    while not shutdown_event.is_set() and not stop_event.is_set():
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
    update_camera_status(running=False)
    log.info("Camera released.")

# ── ZED SDK capture ────────────────────────────────────────────────────────

def zed_capture_loop(
    camera_index: int,
    width: int | None,
    height: int | None,
    fps: int = 30,
    stop_event: threading.Event | None = None,
):
    global latest_raw_frame, latest_jpeg
    stop_event = stop_event or shutdown_event

    clear_latest_frame()
    update_camera_status(
        running=False,
        error=None,
        actual_width=None,
        actual_height=None,
        actual_fps=None,
    )
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
        error = "No ZED cameras available. Try: sudo systemctl restart zed_x_daemon"
        log.error(error)
        update_camera_status(running=False, error=error)
        return
    if camera_index >= len(available):
        error = f"ZED camera index {camera_index} out of range ({len(available)} available)"
        log.error(error)
        update_camera_status(running=False, error=error)
        return

    dev = available[camera_index]
    params.set_from_serial_number(dev.serial_number)
    log.info("Opening ZED %s (serial %d)...", dev.camera_model, dev.serial_number)

    status = cam.open(params)
    if status != sl.ERROR_CODE.SUCCESS:
        error = f"ZED open failed: {status}"
        log.error(error)
        update_camera_status(running=False, error=error)
        return

    info = cam.get_camera_information()
    res = info.camera_configuration.resolution
    log.info("ZED opened at %dx%d @ %d fps", res.width, res.height, fps)
    if target_size:
        log.info("Resizing output to %dx%d", target_size[0], target_size[1])
    update_camera_status(
        running=True,
        error=None,
        actual_width=target_size[0] if target_size else res.width,
        actual_height=target_size[1] if target_size else res.height,
        actual_fps=fps,
    )

    image = sl.Mat()
    runtime = sl.RuntimeParameters()

    while not shutdown_event.is_set() and not stop_event.is_set():
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
    update_camera_status(running=False)
    log.info("ZED camera closed.")


def start_capture(camera_index: int, width: int | None, height: int | None, fps: int):
    global capture_stop_event, capture_thread, stream_fps
    if capture_fn is None:
        raise RuntimeError("Camera backend has not been initialized")

    stop_capture()
    clear_latest_frame()
    stream_fps = fps
    with camera_control_lock:
        camera_config.update(
            camera=camera_index,
            width=width,
            height=height,
            fps=fps,
            enabled=True,
        )
        camera_status.update(
            running=False,
            error=None,
            actual_width=None,
            actual_height=None,
            actual_fps=None,
        )

    capture_stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=capture_fn,
        args=(camera_index, width, height, fps, capture_stop_event),
        daemon=True,
    )
    capture_thread.start()


def stop_capture():
    global capture_stop_event, capture_thread
    thread = capture_thread
    if capture_stop_event is not None:
        capture_stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    capture_stop_event = None
    capture_thread = None
    clear_latest_frame()
    with camera_control_lock:
        camera_config["enabled"] = False
        camera_status.update(running=False)


def enumerate_cameras(max_index: int = 10):
    state = current_state()
    cameras = []

    if active_backend == "zed":
        if not HAS_ZED:
            return cameras
        devs = sl.Camera.get_device_list()
        for idx, dev in enumerate(devs):
            cameras.append(
                {
                    "index": idx,
                    "label": f"ZED {dev.camera_model} ({dev.serial_number})",
                    "available": str(dev.camera_state) == "AVAILABLE",
                    "active": idx == state["camera"],
                }
            )
        return cameras

    seen = set()
    if state["running"]:
        cameras.append(
            {
                "index": state["camera"],
                "label": f"Camera {state['camera']}",
                "available": True,
                "active": True,
            }
        )
        seen.add(state["camera"])

    for index in range(max_index):
        if index in seen:
            continue
        cap = cv2.VideoCapture(index)
        available = cap.isOpened()
        if available:
            cameras.append(
                {
                    "index": index,
                    "label": f"Camera {index}",
                    "available": True,
                    "active": index == state["camera"],
                }
            )
        cap.release()
    return sorted(cameras, key=lambda item: item["index"])


def available_modes():
    state = current_state()
    resolutions = [{"width": None, "height": None, "label": "Auto"}]
    resolutions.extend(
        {"width": width, "height": height, "label": f"{width}x{height}"}
        for width, height in COMMON_RESOLUTIONS
    )
    if state["actual_width"] and state["actual_height"]:
        actual = (state["actual_width"], state["actual_height"])
        if actual not in COMMON_RESOLUTIONS:
            resolutions.insert(
                0,
                {
                    "width": actual[0],
                    "height": actual[1],
                    "label": f"{actual[0]}x{actual[1]} (current)",
                },
            )
    return {"resolutions": resolutions, "fps": COMMON_FPS}

# ── HTTP mode (Flask MJPEG) ────────────────────────────────────────────────

app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>devcam</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
        :root { color-scheme: dark; }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: #111315;
            color: #eef2f3;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        main {
            min-height: 100vh;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 340px;
            gap: 20px;
            padding: 20px;
        }
        .viewer {
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .topbar {
            min-height: 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: 0; }
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #b9c3c7;
            font-size: 13px;
            white-space: nowrap;
        }
        .dot {
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: #9aa3a7;
            box-shadow: 0 0 0 3px rgba(154, 163, 167, 0.15);
        }
        .dot.on { background: #30d158; box-shadow: 0 0 0 3px rgba(48, 209, 88, 0.16); }
        .dot.err { background: #ff9f0a; box-shadow: 0 0 0 3px rgba(255, 159, 10, 0.16); }
        .frame-wrap {
            position: relative;
            flex: 1;
            min-height: 420px;
            display: grid;
            place-items: center;
            overflow: hidden;
            background: #050607;
            border: 1px solid #30363a;
            border-radius: 8px;
        }
        img {
            display: block;
            width: 100%;
            height: 100%;
            max-height: calc(100vh - 96px);
            object-fit: contain;
        }
        img.mirrored { transform: scaleX(-1); }
        .overlay-canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        .placeholder {
            position: absolute;
            inset: 0;
            display: none;
            place-items: center;
            color: #8e989d;
            font-size: 14px;
            text-align: center;
            padding: 24px;
            background: #08090a;
        }
        .placeholder.show { display: grid; }
        aside {
            align-self: start;
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 16px;
            border: 1px solid #30363a;
            border-radius: 8px;
            background: #181b1e;
        }
        .group { display: grid; gap: 8px; }
        .panel-title { margin: 4px 0 0; color: #eef2f3; font-size: 13px; font-weight: 700; }
        label { color: #b9c3c7; font-size: 12px; font-weight: 650; }
        select, button {
            width: 100%;
            min-height: 38px;
            border: 1px solid #3a4247;
            border-radius: 6px;
            background: #22272b;
            color: #f4f7f8;
            font: inherit;
        }
        select { padding: 0 10px; }
        button {
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-weight: 650;
        }
        button.primary { background: #1f7a4d; border-color: #24985e; }
        button.warn { background: #6b3d12; border-color: #925a1b; }
        button:disabled { opacity: 0.5; cursor: wait; }
        input[type="range"] { width: 100%; accent-color: #30d158; }
        input[type="checkbox"] { width: 16px; height: 16px; accent-color: #30d158; }
        .check-row { display: flex; align-items: center; gap: 8px; color: #d6dcdf; font-size: 13px; }
        .range-row { display: grid; grid-template-columns: 1fr 48px; align-items: center; gap: 8px; }
        .range-value { color: #d6dcdf; font-size: 12px; text-align: right; }
        .meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            color: #b9c3c7;
            font-size: 12px;
        }
        .meta span {
            min-width: 0;
            padding: 8px;
            background: #121517;
            border-radius: 6px;
            overflow-wrap: anywhere;
        }
        .detection-line { color: #b9c3c7; font-size: 12px; min-height: 18px; overflow-wrap: anywhere; }
        .error { color: #ffbe5c; font-size: 13px; min-height: 18px; overflow-wrap: anywhere; }
        @media (max-width: 880px) {
            main { grid-template-columns: 1fr; padding: 12px; }
            .frame-wrap { min-height: 280px; }
            img { max-height: 58vh; }
            aside { order: -1; }
        }
  </style>
</head>
<body>
    <main>
        <section class="viewer" aria-label="Camera stream">
            <div class="topbar">
                <h1>devcam</h1>
                <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Loading</span></div>
            </div>
            <div class="frame-wrap">
                <img id="stream" src="/stream" alt="webcam stream" />
                <canvas id="overlay" class="overlay-canvas"></canvas>
                <div id="placeholder" class="placeholder">Camera is off</div>
            </div>
        </section>
        <aside aria-label="Camera controls">
            <button id="toggleButton" class="warn" type="button">Turn off camera</button>
            <div class="group">
                <label for="cameraSelect">Camera</label>
                <select id="cameraSelect"></select>
            </div>
            <div class="group">
                <label for="resolutionSelect">Resolution</label>
                <select id="resolutionSelect"></select>
            </div>
            <div class="group">
                <label for="fpsSelect">FPS</label>
                <select id="fpsSelect"></select>
            </div>
            <label class="check-row" for="mirrorToggle">
                <input id="mirrorToggle" type="checkbox" />
                Mirror camera view
            </label>
            <button id="applyButton" class="primary" type="button">Apply</button>
            <div class="meta">
                <span id="backendMeta">Backend</span>
                <span id="actualMeta">Actual</span>
            </div>
            <div class="panel-title">Human detection</div>
            <button id="detectionToggle" type="button">Enable detection</button>
            <div class="group">
                <label for="confidenceRange">Confidence</label>
                <div class="range-row">
                    <input id="confidenceRange" type="range" min="0.1" max="0.95" step="0.05" value="0.5" />
                    <span id="confidenceValue" class="range-value">0.50</span>
                </div>
            </div>
            <div class="group">
                <label for="detectionHzRange">Detection rate</label>
                <div class="range-row">
                    <input id="detectionHzRange" type="range" min="1" max="20" step="1" value="1" />
                    <span id="detectionHzValue" class="range-value">1 Hz</span>
                </div>
            </div>
            <div id="detectionText" class="detection-line">Detection idle</div>
            <div id="errorText" class="error"></div>
        </aside>
    </main>
    <script>
        const cameraSelect = document.querySelector("#cameraSelect");
        const resolutionSelect = document.querySelector("#resolutionSelect");
        const fpsSelect = document.querySelector("#fpsSelect");
        const mirrorToggle = document.querySelector("#mirrorToggle");
        const applyButton = document.querySelector("#applyButton");
        const toggleButton = document.querySelector("#toggleButton");
        const statusDot = document.querySelector("#statusDot");
        const statusText = document.querySelector("#statusText");
        const streamImage = document.querySelector("#stream");
        const overlayCanvas = document.querySelector("#overlay");
        const placeholder = document.querySelector("#placeholder");
        const backendMeta = document.querySelector("#backendMeta");
        const actualMeta = document.querySelector("#actualMeta");
        const errorText = document.querySelector("#errorText");
        const detectionToggle = document.querySelector("#detectionToggle");
        const confidenceRange = document.querySelector("#confidenceRange");
        const confidenceValue = document.querySelector("#confidenceValue");
        const detectionHzRange = document.querySelector("#detectionHzRange");
        const detectionHzValue = document.querySelector("#detectionHzValue");
        const detectionText = document.querySelector("#detectionText");

        let state = null;
        let detectionState = null;
        let detectionTimer = null;

        async function fetchJson(url, options) {
            const response = await fetch(url, options);
            const body = await response.json();
            if (!response.ok) throw new Error(body.error || response.statusText);
            return body;
        }

        function option(value, label, selected = false) {
            const element = document.createElement("option");
            element.value = value;
            element.textContent = label;
            element.selected = selected;
            return element;
        }

        async function loadControls() {
            const [cameras, modes] = await Promise.all([
                fetchJson("/api/cameras"),
                fetchJson("/api/modes"),
            ]);

            cameraSelect.replaceChildren(...cameras.cameras.map((camera) => {
                const label = camera.available ? camera.label : `${camera.label} unavailable`;
                const element = option(camera.index, label, state && camera.index === state.camera);
                element.disabled = !camera.available;
                return element;
            }));

            resolutionSelect.replaceChildren(
                ...modes.resolutions.map((resolution) => {
                    const value = resolution.width && resolution.height ? `${resolution.width}x${resolution.height}` : "auto";
                    const selected = state && state.width === resolution.width && state.height === resolution.height;
                    return option(value, resolution.label, selected);
                })
            );

            fpsSelect.replaceChildren(
                ...modes.fps.map((fps) => option(fps, `${fps}`, state && fps === state.fps))
            );
        }

        function renderStatus() {
            if (!state) return;
            statusDot.className = "dot";
            if (state.error) statusDot.classList.add("err");
            if (state.running) statusDot.classList.add("on");

            statusText.textContent = state.running ? "Live" : state.enabled ? "Starting" : "Off";
            if (state.error) statusText.textContent = "Attention";
            toggleButton.textContent = state.enabled ? "Turn off camera" : "Turn on camera";
            toggleButton.className = state.enabled ? "warn" : "primary";
            mirrorToggle.checked = Boolean(state.mirror);
            streamImage.classList.toggle("mirrored", Boolean(state.mirror));
            placeholder.classList.toggle("show", !state.enabled || !state.has_frame);
            placeholder.textContent = state.enabled ? "Waiting for camera" : "Camera is off";
            backendMeta.textContent = `Backend: ${state.backend}`;
            const actual = state.actual_width && state.actual_height
                ? `${state.actual_width}x${state.actual_height} @ ${state.actual_fps || state.fps} fps`
                : "No active mode";
            actualMeta.textContent = actual;
            errorText.textContent = state.error || "";
        }

        function renderDetectionStatus(result = null) {
            if (!detectionState) return;
            const threshold = detectionState.confidence_threshold ?? detectionState.confidence ?? 0.5;
            const detectionHz = Math.max(1, Math.min(20, Math.round(1000 / Number(detectionState.interval_ms || 1000))));
            confidenceRange.value = threshold;
            confidenceValue.textContent = Number(threshold).toFixed(2);
            detectionHzRange.value = detectionHz;
            detectionHzValue.textContent = `${detectionHz} Hz`;
            detectionToggle.textContent = detectionState.enabled ? "Disable detection" : "Enable detection";
            detectionToggle.className = detectionState.enabled ? "warn" : "primary";

            if (!detectionState.available) {
                detectionText.textContent = detectionState.error || "Detection unavailable";
                clearOverlay();
                return;
            }
            if (!detectionState.enabled) {
                detectionText.textContent = "Detection idle";
                clearOverlay();
                return;
            }
            if (!result) {
                detectionText.textContent = detectionState.loaded ? "Detection active" : "Model will load on first scan";
                return;
            }
            if (result.error) {
                detectionText.textContent = result.error;
                clearOverlay();
                return;
            }
            detectionText.textContent = result.human_detected
                ? `${result.count} person${result.count === 1 ? "" : "s"} · top ${Number(result.confidence).toFixed(2)} · ${result.latency_ms} ms`
                : `No person · ${result.latency_ms} ms`;
        }

        function overlayRect(frameWidth, frameHeight) {
            const bounds = overlayCanvas.getBoundingClientRect();
            const canvasRatio = bounds.width / bounds.height;
            const frameRatio = frameWidth / frameHeight;
            let width = bounds.width;
            let height = bounds.height;
            let x = 0;
            let y = 0;
            if (frameRatio > canvasRatio) {
                height = width / frameRatio;
                y = (bounds.height - height) / 2;
            } else {
                width = height * frameRatio;
                x = (bounds.width - width) / 2;
            }
            return { x, y, width, height };
        }

        function resizeOverlay() {
            const bounds = overlayCanvas.getBoundingClientRect();
            const scale = window.devicePixelRatio || 1;
            const width = Math.max(1, Math.round(bounds.width * scale));
            const height = Math.max(1, Math.round(bounds.height * scale));
            if (overlayCanvas.width !== width || overlayCanvas.height !== height) {
                overlayCanvas.width = width;
                overlayCanvas.height = height;
            }
        }

        function clearOverlay() {
            resizeOverlay();
            const context = overlayCanvas.getContext("2d");
            context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }

        function drawDetections(result) {
            clearOverlay();
            if (!result || !result.frame_width || !result.frame_height || !result.detections.length) return;

            const context = overlayCanvas.getContext("2d");
            const scale = window.devicePixelRatio || 1;
            const rect = overlayRect(result.frame_width, result.frame_height);
            context.save();
            context.scale(scale, scale);
            context.lineWidth = 3;
            context.strokeStyle = "#30d158";
            context.fillStyle = "rgba(48, 209, 88, 0.18)";
            context.font = "12px system-ui, sans-serif";

            for (const detection of result.detections) {
                const [x1, y1, x2, y2] = detection.bbox;
                const x = state && state.mirror
                    ? rect.x + ((result.frame_width - x2) / result.frame_width) * rect.width
                    : rect.x + (x1 / result.frame_width) * rect.width;
                const y = rect.y + (y1 / result.frame_height) * rect.height;
                const width = ((x2 - x1) / result.frame_width) * rect.width;
                const height = ((y2 - y1) / result.frame_height) * rect.height;
                context.fillRect(x, y, width, height);
                context.strokeRect(x, y, width, height);
                const label = `person ${Number(detection.confidence).toFixed(2)}`;
                const labelWidth = context.measureText(label).width + 10;
                const labelY = Math.max(rect.y + 2, y - 22);
                context.fillStyle = "#30d158";
                context.fillRect(x, labelY, labelWidth, 20);
                context.fillStyle = "#071009";
                context.fillText(label, x + 5, labelY + 14);
                context.fillStyle = "rgba(48, 209, 88, 0.18)";
            }
            context.restore();
        }

        function stopDetectionLoop() {
            if (detectionTimer) clearTimeout(detectionTimer);
            detectionTimer = null;
        }

        async function runDetectionOnce() {
            if (!detectionState || !detectionState.enabled) return;
            try {
                const result = await fetchJson("/api/detection/run", { method: "POST" });
                detectionState = { ...detectionState, ...result, confidence_threshold: result.confidence_threshold ?? detectionState.confidence_threshold };
                drawDetections(result);
                renderDetectionStatus(result);
            } catch (error) {
                detectionText.textContent = error.message;
                clearOverlay();
            } finally {
                if (detectionState && detectionState.enabled) {
                    detectionTimer = setTimeout(runDetectionOnce, detectionState.interval_ms);
                }
            }
        }

        function startDetectionLoop() {
            stopDetectionLoop();
            if (detectionState && detectionState.enabled) runDetectionOnce();
        }

        async function refreshDetectionStatus() {
            detectionState = await fetchJson("/api/detection/status");
            renderDetectionStatus();
            startDetectionLoop();
        }

        async function applyDetectionConfig(enabled = null) {
            detectionToggle.disabled = true;
            try {
                detectionState = await fetchJson("/api/detection/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        enabled: enabled === null ? detectionState.enabled : enabled,
                        confidence: Number(confidenceRange.value),
                        interval_ms: Math.round(1000 / Number(detectionHzRange.value || 1)),
                    }),
                });
                renderDetectionStatus();
                startDetectionLoop();
            } catch (error) {
                detectionText.textContent = error.message;
            } finally {
                detectionToggle.disabled = false;
            }
        }

        async function refreshStatus() {
            state = await fetchJson("/api/status");
            renderStatus();
        }

        async function applyConfig(enabled = true) {
            applyButton.disabled = true;
            toggleButton.disabled = true;
            const [width, height] = resolutionSelect.value === "auto"
                ? [null, null]
                : resolutionSelect.value.split("x").map(Number);
            try {
                state = await fetchJson("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        enabled,
                        camera: Number(cameraSelect.value || 0),
                        width,
                        height,
                        fps: Number(fpsSelect.value || 30),
                        mirror: mirrorToggle.checked,
                    }),
                });
                streamImage.src = enabled ? `/stream?ts=${Date.now()}` : "";
                clearOverlay();
                renderStatus();
                await loadControls();
            } catch (error) {
                errorText.textContent = error.message;
            } finally {
                applyButton.disabled = false;
                toggleButton.disabled = false;
            }
        }

        applyButton.addEventListener("click", () => applyConfig(true));
        toggleButton.addEventListener("click", () => applyConfig(!state || !state.enabled));
        mirrorToggle.addEventListener("change", async () => {
            state = await fetchJson("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mirror: mirrorToggle.checked }),
            });
            renderStatus();
            clearOverlay();
        });
        detectionToggle.addEventListener("click", () => applyDetectionConfig(!detectionState || !detectionState.enabled));
        confidenceRange.addEventListener("input", () => { confidenceValue.textContent = Number(confidenceRange.value).toFixed(2); });
        confidenceRange.addEventListener("change", () => applyDetectionConfig());
        detectionHzRange.addEventListener("input", () => { detectionHzValue.textContent = `${detectionHzRange.value} Hz`; });
        detectionHzRange.addEventListener("change", () => applyDetectionConfig());
        window.addEventListener("resize", () => { if (detectionState && detectionState.enabled) clearOverlay(); });

        async function init() {
            await refreshStatus();
            await refreshDetectionStatus();
            await loadControls();
            renderStatus();
            setInterval(refreshStatus, 1500);
        }

        init().catch((error) => { errorText.textContent = error.message; });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/status")
def api_status():
    return jsonify(current_state())


@app.route("/api/cameras")
def api_cameras():
    return jsonify({"cameras": enumerate_cameras()})


@app.route("/api/modes")
def api_modes():
    return jsonify(available_modes())


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    if "mirror" in data:
        with camera_control_lock:
            camera_config["mirror"] = bool(data["mirror"])
    if set(data.keys()) == {"mirror"}:
        return jsonify(current_state())
    if not enabled:
        stop_capture()
        if "mirror" in data:
            with camera_control_lock:
                camera_config["mirror"] = bool(data["mirror"])
        return jsonify(current_state())

    try:
        camera_index = int(data.get("camera", camera_config["camera"]))
        fps = int(data.get("fps", camera_config["fps"]))
        width = data.get("width", camera_config["width"])
        height = data.get("height", camera_config["height"])
        width = int(width) if width else None
        height = int(height) if height else None
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid camera configuration"}), 400

    if fps <= 0:
        return jsonify({"error": "FPS must be greater than zero"}), 400
    if (width is None) != (height is None):
        return jsonify({"error": "Resolution must include width and height"}), 400

    start_capture(camera_index, width, height, fps)
    if "mirror" in data:
        with camera_control_lock:
            camera_config["mirror"] = bool(data["mirror"])
    return jsonify(current_state())


@app.route("/api/detection/status")
def api_detection_status():
    return jsonify(detection_state())


@app.route("/api/detection/config", methods=["POST"])
def api_detection_config():
    data = request.get_json(silent=True) or {}
    try:
        update_detection_config(data)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(detection_state())


@app.route("/api/detection/run", methods=["GET", "POST"])
def api_detection_run():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if data:
            try:
                update_detection_config(data)
            except (TypeError, ValueError) as exc:
                return jsonify({"error": str(exc)}), 400

    state = detection_state()
    if not state["enabled"]:
        return jsonify({**state, **human_detector.disabled_result(enabled=False)})

    with frame_lock:
        frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
    result = human_detector.detect(frame, enabled=True)
    response = {**detection_state(), **result, "confidence_threshold": detection_config["confidence"]}
    if frame is None:
        return jsonify(response), 503
    if not response["available"]:
        return jsonify(response), 503
    return jsonify(response)


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

MEDIAMTX_VERSION = "1.12.2"


def _find_mediamtx() -> str | None:
    """Locate mediamtx binary on PATH or in common locations."""
    found = shutil.which("mediamtx")
    if found:
        return found
    # Check next to this file, the cwd, and the user data dir
    import pathlib
    candidates = [
        pathlib.Path.cwd() / "mediamtx",
        pathlib.Path.home() / ".local" / "bin" / "mediamtx",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _download_mediamtx() -> str:
    """Download mediamtx to ~/.local/bin and return the path."""
    import pathlib
    import tarfile
    import tempfile
    import urllib.request

    system = platform.system().lower()
    if system == "darwin":
        os_name = "darwin"
    elif system == "linux":
        os_name = "linux"
    else:
        raise RuntimeError(f"Unsupported OS for auto-download: {system}")

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "amd64"
    else:
        raise RuntimeError(f"Unsupported architecture for auto-download: {machine}")

    dest_dir = pathlib.Path.home() / ".local" / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "mediamtx"

    tarball_name = f"mediamtx_v{MEDIAMTX_VERSION}_{os_name}_{arch}.tar.gz"
    url = f"https://github.com/bluenviron/mediamtx/releases/download/v{MEDIAMTX_VERSION}/{tarball_name}"

    log.info("Downloading mediamtx v%s for %s/%s...", MEDIAMTX_VERSION, os_name, arch)
    with tempfile.TemporaryDirectory() as tmp:
        tarball_path = pathlib.Path(tmp) / tarball_name
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path, "r:gz") as tf:
            member = tf.getmember("mediamtx")
            tf.extract(member, tmp)
            extracted = pathlib.Path(tmp) / "mediamtx"
            import shutil as _shutil
            _shutil.move(str(extracted), str(dest))

    dest.chmod(0o755)
    log.info("Installed mediamtx to %s", dest)
    return str(dest)


def _ensure_mediamtx() -> str:
    """Find or download mediamtx, return path to binary."""
    path = _find_mediamtx()
    if path:
        return path
    return _download_mediamtx()


def _start_mediamtx(port: int) -> subprocess.Popen:
    """Start mediamtx as a subprocess configured for the given RTSP port."""
    binary = _ensure_mediamtx()
    env = {
        **dict(platform.system() == "Windows" and {} or {}),
        "MTX_PROTOCOLS": "tcp",
        "MTX_RTSPADDRESS": f":{port}",
    }
    full_env = {**dict(subprocess.os.environ), **env}
    log.info("Starting mediamtx on :%d...", port)
    proc = subprocess.Popen(
        [binary],
        env=full_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Give it a moment to bind the port
    time.sleep(1.0)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"mediamtx exited immediately: {stderr}")
    log.info("mediamtx started (pid %d)", proc.pid)
    return proc


def run_rtsp(port: int):
    mediamtx_proc = None
    try:
        mediamtx_proc = _start_mediamtx(port)
    except Exception as exc:
        log.error("Failed to start mediamtx: %s", exc)
        shutdown_event.set()
        return

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        if platform.system() == "Darwin":
            install_hint = "brew install ffmpeg"
        elif platform.system() == "Linux":
            install_hint = "sudo apt install ffmpeg"
        elif platform.system() == "Windows":
            install_hint = "winget install Gyan.FFmpeg"
        else:
            install_hint = "install ffmpeg and ensure it is on PATH"
        log.error("ffmpeg not found. Install with: %s", install_hint)
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

    if mediamtx_proc and mediamtx_proc.poll() is None:
        log.info("Stopping mediamtx (pid %d)...", mediamtx_proc.pid)
        mediamtx_proc.terminate()
        try:
            mediamtx_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mediamtx_proc.kill()
        log.info("mediamtx stopped.")

# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Webcam broadcaster (HTTP or RTSP)")
    p.add_argument("protocol", choices=["http", "rtsp"], help="Stream protocol")
    p.add_argument("--port", type=int, default=None, help="Port (default: 1080 for http, 8554 for rtsp)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--resolution", type=str, default=None, help="WxH e.g. 1280x720")
    p.add_argument("--fps", type=int, default=30, help="Capture frame rate (default: 30)")
    p.add_argument(
        "--backend", choices=["auto", "cv2", "zed"], default="auto",
        help="Camera backend: auto (default), cv2 (OpenCV/V4L2), zed (ZED SDK)",
    )
    return p.parse_args()


def main():
    global active_backend, capture_fn, stream_fps
    args = parse_args()

    port = args.port
    if port is None:
        port = 1080 if args.protocol == "http" else 8554

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

    active_backend = backend

    # Start camera capture
    if backend == "zed":
        capture_fn = zed_capture_loop
    else:
        capture_fn = capture_loop

    stream_fps = args.fps
    start_capture(args.camera, width, height, args.fps)
    time.sleep(1.0)
    if shutdown_event.is_set():
        sys.exit(1)
    if args.protocol == "rtsp" and current_state()["error"]:
        stop_capture()
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

    stop_capture()
    log.info("Goodbye.")


if __name__ == "__main__":
    main()
