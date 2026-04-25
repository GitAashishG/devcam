"""Tests for update_detection_config validation logic."""

import pytest

from devcam import server


@pytest.fixture(autouse=True)
def _reset_detection_config():
    server.detection_config.update(
        enabled=False, confidence=0.5, interval_ms=1000, model="yolov8n.pt",
    )
    yield


def test_enable_detection():
    server.update_detection_config({"enabled": True})
    assert server.detection_config["enabled"] is True


def test_set_valid_confidence():
    server.update_detection_config({"confidence": 0.75})
    assert server.detection_config["confidence"] == 0.75


def test_confidence_below_zero_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        server.update_detection_config({"confidence": -0.1})


def test_confidence_above_one_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        server.update_detection_config({"confidence": 1.1})


def test_set_valid_interval():
    server.update_detection_config({"interval_ms": 200})
    assert server.detection_config["interval_ms"] == 200


def test_interval_too_low():
    with pytest.raises(ValueError, match="between 50 and 5000"):
        server.update_detection_config({"interval_ms": 10})


def test_interval_too_high():
    with pytest.raises(ValueError, match="between 50 and 5000"):
        server.update_detection_config({"interval_ms": 10000})


def test_set_model():
    server.update_detection_config({"model": "yolov8s.pt"})
    assert server.detection_config["model"] == "yolov8s.pt"


def test_combined_update():
    server.update_detection_config({
        "enabled": True,
        "confidence": 0.8,
        "interval_ms": 100,
        "model": "yolov8s.pt",
    })
    assert server.detection_config["enabled"] is True
    assert server.detection_config["confidence"] == 0.8
    assert server.detection_config["interval_ms"] == 100
    assert server.detection_config["model"] == "yolov8s.pt"
