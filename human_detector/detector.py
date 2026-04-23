#!/usr/bin/env python3
"""
detector.py — Human detection API using YOLOv8n.

POST /detect  — upload an image or pass a URL, get back human detection results.
GET  /detect?url=...  — fetch image from URL (e.g. camserver's /snapshot).

Response:
  {
    "human_detected": true,
    "confidence": 0.93,
    "count": 2,
    "detections": [
      {"confidence": 0.93, "bbox": [x1, y1, x2, y2]},
      ...
    ]
  }
"""

import argparse
import io
import logging

import requests as http_requests
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("detector")

app = Flask(__name__)

# YOLOv8n auto-downloads on first run (~6MB)
model: YOLO | None = None
PERSON_CLASS_ID = 0  # COCO class 0 = "person"


def load_model():
    global model
    if model is None:
        log.info("Loading YOLOv8n model (first run downloads ~6MB)...")
        model = YOLO("yolov8n.pt")
        log.info("Model loaded.")
    return model


def detect_humans(image_bytes: bytes) -> dict:
    """Run YOLOv8n on image bytes, return person detections."""
    m = load_model()
    results = m(io.BytesIO(image_bytes), verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != PERSON_CLASS_ID:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    top_confidence = detections[0]["confidence"] if detections else 0.0

    return {
        "human_detected": len(detections) > 0,
        "confidence": top_confidence,
        "count": len(detections),
        "detections": detections,
    }


@app.route("/detect", methods=["GET", "POST"])
def detect():
    image_bytes = None

    if request.method == "POST":
        # Accept file upload
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        # Accept raw body
        elif request.content_type and "image" in request.content_type:
            image_bytes = request.get_data()
        # Accept JSON with url
        elif request.is_json:
            url = request.json.get("url")
            if url:
                try:
                    resp = http_requests.get(url, timeout=5)
                    resp.raise_for_status()
                    image_bytes = resp.content
                except http_requests.RequestException as e:
                    return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

    if request.method == "GET":
        url = request.args.get("url")
        if url:
            try:
                resp = http_requests.get(url, timeout=5)
                resp.raise_for_status()
                image_bytes = resp.content
            except http_requests.RequestException as e:
                return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

    if image_bytes is None:
        return jsonify({
            "error": "No image provided. POST a file, send image bytes, "
                     "or pass ?url=http://..."
        }), 400

    result = detect_humans(image_bytes)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def parse_args():
    p = argparse.ArgumentParser(description="Human detection API (YOLOv8n)")
    p.add_argument("--port", type=int, default=1082, help="Port (default: 1082)")
    p.add_argument("--preload", action="store_true", help="Load model at startup")
    return p.parse_args()


def main():
    args = parse_args()

    if args.preload:
        load_model()

    log.info("Human detector API: http://0.0.0.0:%d/detect", args.port)
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
