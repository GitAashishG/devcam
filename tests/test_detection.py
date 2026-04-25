"""Tests for devcam.detection.human — HumanDetector."""

import numpy as np
import pytest

from devcam.detection.human import HumanDetector


# ── Construction & properties ──────────────────────────────────────────────


def test_defaults():
    d = HumanDetector()
    assert d.model_name == "yolov8n.pt"
    assert d.confidence_threshold == 0.5
    assert d.loaded is False
    assert d.error is None


def test_custom_init():
    d = HumanDetector(model_name="custom.pt", confidence_threshold=0.8)
    assert d.model_name == "custom.pt"
    assert d.confidence_threshold == 0.8


# ── configure() ────────────────────────────────────────────────────────────


def test_configure_confidence():
    d = HumanDetector()
    d.configure(confidence_threshold=0.75)
    assert d.confidence_threshold == 0.75


def test_configure_clamps_confidence():
    d = HumanDetector()
    d.configure(confidence_threshold=-0.5)
    assert d.confidence_threshold == 0.0
    d.configure(confidence_threshold=1.5)
    assert d.confidence_threshold == 1.0


def test_configure_model_change_resets_model():
    d = HumanDetector()
    d.model = "fake"  # pretend a model was loaded
    d.configure(model_name="other.pt")
    assert d.model_name == "other.pt"
    assert d.model is None  # should have been cleared


def test_configure_same_model_keeps_loaded():
    d = HumanDetector()
    d.model = "fake"
    d.configure(model_name="yolov8n.pt")
    assert d.model == "fake"  # not cleared


# ── disabled_result() ──────────────────────────────────────────────────────


def test_disabled_result_shape():
    d = HumanDetector()
    result = d.disabled_result(enabled=False)
    assert result["enabled"] is False
    assert result["human_detected"] is False
    assert result["count"] == 0
    assert result["detections"] == []
    assert result["confidence"] == 0.0
    assert result["frame_width"] is None
    assert result["frame_height"] is None


# ── detect() without ultralytics ───────────────────────────────────────────


def test_detect_disabled():
    d = HumanDetector()
    result = d.detect(np.zeros((480, 640, 3), dtype=np.uint8), enabled=False)
    assert result["enabled"] is False
    assert result["human_detected"] is False


def test_detect_no_frame():
    d = HumanDetector()
    result = d.detect(None, enabled=True)
    assert result["error"] == "no frame available"
    assert result["human_detected"] is False


def test_detect_frame_dimensions_reported_even_without_model():
    """If model fails to load, frame dims are still reported."""
    d = HumanDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = d.detect(frame, enabled=True)
    # Without ultralytics installed the model won't load, but dims should appear
    assert result["frame_width"] == 1280
    assert result["frame_height"] == 720
