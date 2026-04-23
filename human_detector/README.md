# human_detector

Human detection API powered by YOLOv8n. Send it an image, get back whether humans are present.

## Quick Start

```bash
pip install -r requirements.txt
python detector.py
```

API runs on **http://localhost:1082**.

## Usage

### With camserver snapshot

```bash
# camserver running on :1080
curl "http://localhost:1082/detect?url=http://localhost:1080/snapshot"
```

### Upload an image file

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:1082/detect
```

### Send raw image bytes

```bash
curl -X POST -H "Content-Type: image/jpeg" --data-binary @photo.jpg http://localhost:1082/detect
```

## Response

```json
{
  "human_detected": true,
  "confidence": 0.93,
  "count": 2,
  "detections": [
    {"confidence": 0.93, "bbox": [120.5, 45.2, 380.1, 510.8]},
    {"confidence": 0.71, "bbox": [450.0, 60.3, 620.4, 490.2]}
  ]
}
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--port` | 1082 | API port |
| `--preload` | off | Load model at startup (otherwise loads on first request) |

## Model

Uses **YOLOv8n** (~6MB, auto-downloaded on first run). Detects COCO "person" class. Runs on CPU — no GPU required.
