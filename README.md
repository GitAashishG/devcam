# devcam

[![PyPI](https://img.shields.io/pypi/v/devcam)](https://pypi.org/project/devcam/)
[![Python](https://img.shields.io/pypi/pyversions/devcam)](https://pypi.org/project/devcam/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Easy local camera feeds for development and testing. Expose a webcam as HTTP MJPEG or RTSP, with an optional built-in YOLO human detection overlay — all from one command.

## Install

```bash
pip install devcam
```

To enable human detection (YOLOv8), install with the `vision` extra:

```bash
# quote the brackets in zsh
pip install 'devcam[vision]'
```

## Quick Start

```bash
devcam http            # web UI + MJPEG stream on http://localhost:1080
devcam rtsp            # RTSP stream on rtsp://localhost:1081/cam
```

Open `http://localhost:1080/` to view the live stream, switch cameras, change resolution/FPS, mirror the view, and toggle human detection. Detection is lazy-loaded: the YOLO model only loads when you first enable it.

## Usage

```
devcam {http,rtsp} [--port PORT] [--camera INDEX] [--resolution WxH] [--fps FPS] [--backend {auto,cv2,zed}]
```

| Flag | Default | Description |
|---|---|---|
| `protocol` | *(required)* | `http` or `rtsp` |
| `--port` | 1080 / 1081 | Stream port |
| `--camera` | 0 | Camera index |
| `--resolution` | auto | `WxH` e.g. `1280x720` |
| `--fps` | 30 | Capture and stream frame rate |
| `--backend` | auto | `auto` (ZED first, then OpenCV), `cv2` (V4L2/USB), `zed` (ZED SDK) |

### HTTP Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI with live viewer and controls |
| `/stream` | GET | Raw MJPEG stream |
| `/snapshot` | GET | Single JPEG frame |
| `/api/status` | GET | Camera state and backend info |
| `/api/cameras` | GET | Available cameras |
| `/api/modes` | GET | Supported resolutions and FPS |
| `/api/config` | POST | Start, stop, mirror, or reconfigure capture |

### Human Detection Endpoints

Requires `devcam[vision]`. Without it, streaming works normally and detection endpoints return `available: false`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/detection/status` | GET | Detection state, confidence, interval, model loaded |
| `/api/detection/config` | POST | Enable/disable detection and update settings |
| `/api/detection/run` | GET/POST | Run detection on the latest frame |

Example:

```bash
curl -X POST http://localhost:1080/api/detection/config \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"confidence":0.55,"interval_ms":50}'
```

Detection response:

```json
{
  "human_detected": true,
  "confidence": 0.87,
  "count": 1,
  "latency_ms": 42.1,
  "detections": [
    {"confidence": 0.87, "bbox": [420.0, 160.0, 780.0, 940.0]}
  ]
}
```

## Camera Backends

| Backend | Description |
|---|---|
| `cv2` | OpenCV V4L2 for USB webcams |
| `zed` | ZED SDK for ZED X / ZED X Mini on Jetson (requires `pyzed`) |
| `auto` *(default)* | Tries ZED first, falls back to OpenCV |

### Jetson / ZED X Notes

- If ZED cameras show as "NOT AVAILABLE": `sudo systemctl restart zed_x_daemon`
- Resolution opens at the camera's native mode; `--resolution` downscales in software
- `--fps 5` or `--fps 15` recommended for low-bandwidth use cases

## RTSP Setup

RTSP requires ffmpeg and [mediamtx](https://github.com/bluenviron/mediamtx). One-time setup:

```bash
# macOS
brew install ffmpeg && bash setup_mediamtx.sh

# Linux
sudo apt install ffmpeg && bash setup_mediamtx.sh
```

Then each time:

```bash
./mediamtx &             # media server in the background
devcam rtsp
```

## Development

```bash
git clone https://github.com/GitAashishG/devcam.git
cd devcam
pip install -e '.[dev,vision]'
python -m pytest tests/ -v
```

Releases happen automatically when the `version` in `pyproject.toml` is bumped and merged to `main`.

## macOS Notes

- **Camera permission**: System Settings → Privacy & Security → Camera → Terminal
- All ports are unprivileged (≥ 1024) — no `sudo` needed

## License

[MIT](LICENSE)
