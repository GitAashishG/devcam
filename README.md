# camserver

Dead-simple webcam broadcaster + human detection — two lightweight Python services.

## Components

| Service | Port | Description |
|---|---|---|
| `camserver.py` | 1080 | Webcam → HTTP MJPEG or RTSP stream |
| `human_detector/detector.py` | 1082 | YOLOv8n person detection API |

## Quick Start

```bash
# Terminal 1: start webcam stream
pip install -r requirements.txt
python camserver.py http

# Terminal 2: start human detector
pip install -r human_detector/requirements.txt
python human_detector/detector.py

# Terminal 3: detect humans from live camera
curl "http://localhost:1082/detect?url=http://localhost:1080/snapshot"
```

## camserver

Captures webcam and broadcasts as HTTP MJPEG or RTSP. Pick one at launch.

```
python camserver.py {http,rtsp} [--port PORT] [--camera INDEX] [--resolution WxH] [--fps FPS] [--backend {auto,cv2,zed}]
```

- **HTTP mode** → `http://localhost:1080/` (web viewer), `/stream` (MJPEG), `/snapshot` (single JPEG)
- **RTSP mode** → `rtsp://localhost:1081/cam` (requires ffmpeg + mediamtx)

| Flag | Default | Description |
|---|---|---|
| `protocol` | *(required)* | `http` or `rtsp` |
| `--port` | 1080 / 1081 | Stream port |
| `--camera` | 0 | Camera index |
| `--resolution` | auto | `WxH` e.g. `1280x720` |
| `--fps` | 30 | Capture and stream frame rate |
| `--backend` | auto | `auto` (ZED first, then OpenCV), `cv2` (V4L2/USB), `zed` (ZED SDK) |

### Camera Backends

- **cv2** — OpenCV V4L2 for USB webcams
- **zed** — ZED SDK for ZED X / ZED X Mini on Jetson (requires `pyzed`)
- **auto** *(default)* — tries ZED first, falls back to OpenCV

### Jetson / ZED X Notes

- If ZED cameras show as "NOT AVAILABLE", restart the daemon: `sudo systemctl restart zed_x_daemon`
- Resolution is always opened at the camera's native mode; `--resolution` downscales in software
- `--fps 5` or `--fps 15` recommended for low-bandwidth use cases

### RTSP Setup

```bash
brew install ffmpeg
bash setup_mediamtx.sh   # downloads mediamtx binary
./mediamtx               # run in separate terminal
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
