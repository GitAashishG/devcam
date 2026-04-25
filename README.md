# devcam

Easy local camera feeds for development and testing. `devcam` can expose a camera as HTTP MJPEG, publish RTSP, and optionally run YOLO human detection from the same web UI.

## Components

| Service | Port | Description |
|---|---|---|
| `python -m devcam http` | 1080 | Web UI, HTTP MJPEG stream, snapshots, camera controls, optional human detection |
| `python -m devcam rtsp` | 1081 | RTSP publisher (requires ffmpeg + mediamtx) |
| `human_detector/detector.py` | 1082 | Legacy standalone YOLOv8n person detection API |

## Quick Start

```bash
pip install -r requirements.txt
python -m devcam http
```

Open `http://localhost:1080/` to view the stream, switch cameras, change resolution/FPS, mirror webcam view, turn capture on or off, and enable human detection. Detection is lazy-loaded: the app starts without loading YOLO, and the model loads only when detection is enabled or `/api/detection/run` is called.

The legacy entrypoint still works:

```bash
python camserver.py http
```

## Camera Server

Captures webcam and broadcasts as HTTP MJPEG or RTSP. Pick one at launch.

```
python -m devcam {http,rtsp} [--port PORT] [--camera INDEX] [--resolution WxH] [--fps FPS] [--backend {auto,cv2,zed}]
```

- **HTTP mode** → `http://localhost:1080/` (web viewer), `/stream` (MJPEG), `/snapshot` (single JPEG)
- **RTSP mode** → `rtsp://localhost:1081/cam` (requires ffmpeg + mediamtx)

The HTTP web viewer includes controls for camera selection, resolution, FPS, mirrored webcam display, turning capture on or off, enabling human detection, changing confidence threshold, changing detection rate up to 20 Hz, and drawing detection boxes over the live stream.

| Flag | Default | Description |
|---|---|---|
| `protocol` | *(required)* | `http` or `rtsp` |
| `--port` | 1080 / 1081 | Stream port |
| `--camera` | 0 | Camera index |
| `--resolution` | auto | `WxH` e.g. `1280x720` |
| `--fps` | 30 | Capture and stream frame rate |
| `--backend` | auto | `auto` (ZED first, then OpenCV), `cv2` (V4L2/USB), `zed` (ZED SDK) |

### HTTP Control API

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Current backend, selected camera, requested mode, actual mode, and capture state |
| `/api/cameras` | GET | Available cameras for the active backend |
| `/api/modes` | GET | Common resolution and FPS choices for the UI |
| `/api/config` | POST | Start, stop, mirror, or switch camera settings |

Example:

```bash
curl -X POST http://localhost:1080/api/config \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"camera":0,"width":1280,"height":720,"fps":15,"mirror":true}'
```

### Human Detection API

Human detection runs in the same Flask process against the latest in-memory camera frame. It does not require running the old detector service in a second terminal.

| Endpoint | Method | Description |
|---|---|---|
| `/api/detection/status` | GET | Detection availability, enabled state, confidence threshold, interval, model loaded state, and errors |
| `/api/detection/config` | POST | Enable/disable detection and update settings |
| `/api/detection/run` | GET/POST | Run YOLO person detection on the latest frame |

Configuration example:

```bash
curl -X POST http://localhost:1080/api/detection/config \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"confidence":0.55,"interval_ms":50}'
```

`interval_ms` controls detection cadence. The UI slider supports 1-20 Hz, where 20 Hz is `50 ms`.

Detection response shape:

```json
{
  "enabled": true,
  "available": true,
  "human_detected": true,
  "confidence": 0.87,
  "confidence_threshold": 0.55,
  "count": 1,
  "frame_width": 1920,
  "frame_height": 1080,
  "latency_ms": 42.1,
  "detections": [
    {"confidence": 0.87, "bbox": [420.0, 160.0, 780.0, 940.0]}
  ]
}
```

If `ultralytics` is not installed, camera streaming still works. Detection endpoints return `available: false` with a clear error.

### Camera Backends

- **cv2** — OpenCV V4L2 for USB webcams
- **zed** — ZED SDK for ZED X / ZED X Mini on Jetson (requires `pyzed`)
- **auto** *(default)* — tries ZED first, falls back to OpenCV

### Jetson / ZED X Notes

- If ZED cameras show as "NOT AVAILABLE", restart the daemon: `sudo systemctl restart zed_x_daemon`
- Resolution is always opened at the camera's native mode; `--resolution` downscales in software
- `--fps 5` or `--fps 15` recommended for low-bandwidth use cases

### RTSP Setup

One-time setup on macOS/Linux:

```bash
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
bash setup_mediamtx.sh   # downloads mediamtx for this OS/CPU
```

One-time setup on Windows PowerShell:

> Windows setup has not been tested or validated yet.

```powershell
winget install Gyan.FFmpeg
.\setup_mediamtx.ps1    # downloads mediamtx for Windows/CPU
```

Each time you use RTSP on macOS/Linux:

```bash
./mediamtx               # run in a separate terminal
python camserver.py rtsp
```

Each time you use RTSP on Windows PowerShell:

```powershell
.\mediamtx.exe           # run in a separate terminal
python camserver.py rtsp
```

## human_detector

REST API for person detection using YOLOv8n (~6MB model, CPU-only, auto-downloaded).

```
python human_detector/detector.py [--port PORT] [--preload]
```

| Endpoint | Method | Input | Response |
|---|---|---|---|
| `/detect?url=...` | GET | Image URL | `{human_detected, confidence, count, detections}` |
| `/detect` | POST | File upload / raw bytes / JSON `{"url":"..."}` | Same |
| `/health` | GET | — | `{"status": "ok"}` |

### Benchmark

```bash
python human_detector/benchmark.py -n 50
```

Measures detection throughput (Hz) using a pre-fetched snapshot.

## macOS Notes

- **Camera permission**: System Settings → Privacy & Security → Camera → Terminal
- All ports are unprivileged (≥1024) — no `sudo` needed
