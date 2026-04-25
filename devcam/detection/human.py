"""YOLO-backed human detection with optional runtime dependency."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    HAS_ULTRALYTICS = False

log = logging.getLogger("devcam.detection")

PERSON_CLASS_ID = 0


class HumanDetector:
    """Lazy YOLO person detector for BGR OpenCV frames."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model: Any | None = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return HAS_ULTRALYTICS

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def configure(self, confidence_threshold: float | None = None, model_name: str | None = None):
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None

    def load_model(self) -> bool:
        if not HAS_ULTRALYTICS:
            self.error = "ultralytics is not installed"
            return False
        if self.model is not None:
            return True
        try:
            log.info("Loading YOLO model %s", self.model_name)
            self.model = YOLO(self.model_name)
            self.error = None
            return True
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.error = f"failed to load {self.model_name}: {exc}"
            log.exception("Failed to load YOLO model")
            return False

    def disabled_result(self, enabled: bool = False) -> dict:
        return {
            "enabled": enabled,
            "available": HAS_ULTRALYTICS,
            "loaded": self.loaded,
            "error": None if HAS_ULTRALYTICS else "ultralytics is not installed",
            "human_detected": False,
            "confidence": 0.0,
            "count": 0,
            "detections": [],
            "frame_width": None,
            "frame_height": None,
            "processed_at": None,
            "latency_ms": 0.0,
        }

    def detect(self, frame: np.ndarray | None, enabled: bool = True) -> dict:
        if not enabled:
            return self.disabled_result(enabled=False)
        if frame is None:
            result = self.disabled_result(enabled=True)
            result.update({"error": "no frame available"})
            return result
        frame_height, frame_width = frame.shape[:2]
        if not self.load_model():
            result = self.disabled_result(enabled=True)
            result.update({"frame_width": frame_width, "frame_height": frame_height, "error": self.error})
            return result

        started = time.perf_counter()
        processed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            results = self.model(frame, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id != PERSON_CLASS_ID:
                        continue
                    confidence = float(box.conf[0])
                    if confidence < self.confidence_threshold:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        {
                            "confidence": round(confidence, 4),
                            "bbox": [round(value, 1) for value in (x1, y1, x2, y2)],
                        }
                    )
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            top_confidence = detections[0]["confidence"] if detections else 0.0
            self.error = None
            return {
                "enabled": True,
                "available": True,
                "loaded": True,
                "error": None,
                "human_detected": bool(detections),
                "confidence": top_confidence,
                "count": len(detections),
                "detections": detections,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "processed_at": processed_at,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            self.error = str(exc)
            log.exception("Human detection failed")
            result = self.disabled_result(enabled=True)
            result.update(
                {
                    "available": True,
                    "loaded": True,
                    "error": self.error,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "processed_at": processed_at,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
            return result
