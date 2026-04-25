"""Tests for the Flask HTTP API exposed by devcam.server."""

import json
import threading

import numpy as np
import pytest

from devcam import server


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset module-level globals between tests."""
    server.shutdown_event.clear()
    server.latest_raw_frame = None
    server.latest_jpeg = None
    server.camera_config.update(
        camera=0, width=None, height=None, fps=30, enabled=True, mirror=False,
    )
    server.camera_status.update(
        running=False, error=None,
        actual_width=None, actual_height=None, actual_fps=None,
    )
    server.detection_config.update(
        enabled=False, confidence=0.5, interval_ms=1000, model="yolov8n.pt",
    )
    yield


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


# ── GET /api/status ────────────────────────────────────────────────────────


def test_status(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["backend"] == "cv2"
    assert data["camera"] == 0
    assert "has_frame" in data
    assert "running" in data


# ── GET /api/modes ─────────────────────────────────────────────────────────


def test_modes(client):
    resp = client.get("/api/modes")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "resolutions" in data
    assert "fps" in data
    assert isinstance(data["fps"], list)
    # Auto is always the first resolution option
    assert data["resolutions"][0]["label"] == "Auto"


# ── POST /api/config — mirror-only ────────────────────────────────────────


def test_config_mirror_only(client):
    resp = client.post(
        "/api/config",
        data=json.dumps({"mirror": True}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["mirror"] is True


def test_config_disable(client):
    """Disabling capture doesn't require a running camera."""
    resp = client.post(
        "/api/config",
        data=json.dumps({"enabled": False}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["enabled"] is False


def test_config_invalid_fps(client):
    resp = client.post(
        "/api/config",
        data=json.dumps({"enabled": True, "camera": 0, "fps": -1}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "FPS" in resp.get_json()["error"]


def test_config_partial_resolution_rejected(client):
    resp = client.post(
        "/api/config",
        data=json.dumps({"enabled": True, "camera": 0, "fps": 30, "width": 640}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "Resolution" in resp.get_json()["error"]


# ── Detection API ──────────────────────────────────────────────────────────


def test_detection_status(client):
    resp = client.get("/api/detection/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["enabled"] is False
    assert "available" in data
    assert "loaded" in data


def test_detection_config_enable(client):
    resp = client.post(
        "/api/detection/config",
        data=json.dumps({"enabled": True, "confidence": 0.7}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["enabled"] is True
    assert data["confidence"] == 0.7


def test_detection_config_bad_confidence(client):
    resp = client.post(
        "/api/detection/config",
        data=json.dumps({"confidence": 1.5}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_detection_config_bad_interval(client):
    resp = client.post(
        "/api/detection/config",
        data=json.dumps({"interval_ms": 10}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_detection_run_disabled(client):
    """Detection run when disabled returns cleanly."""
    resp = client.post("/api/detection/run")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["human_detected"] is False


def test_detection_run_no_frame(client):
    """Detection enabled but no frame available gives 503."""
    server.detection_config["enabled"] = True
    resp = client.post("/api/detection/run")
    assert resp.status_code == 503


# ── Snapshot ───────────────────────────────────────────────────────────────


def test_snapshot_no_frame(client):
    resp = client.get("/snapshot")
    assert resp.status_code == 503


def test_snapshot_with_frame(client):
    import cv2
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", frame)
    server.latest_jpeg = jpeg.tobytes()
    resp = client.get("/snapshot")
    assert resp.status_code == 200
    assert resp.content_type == "image/jpeg"


# ── Index page ─────────────────────────────────────────────────────────────


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"devcam" in resp.data
