"""Tests for package metadata and CLI entry point."""

import subprocess
import sys

import devcam


def test_version_exists():
    assert hasattr(devcam, "__version__")
    assert isinstance(devcam.__version__, str)
    # Follows semver-ish pattern
    parts = devcam.__version__.split(".")
    assert len(parts) >= 2


def test_main_module_help():
    """python -m devcam --help should exit 0 and print usage."""
    result = subprocess.run(
        [sys.executable, "-m", "devcam", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "http" in result.stdout
    assert "rtsp" in result.stdout


def test_console_script_help():
    """The `devcam` console script should be callable with --help."""
    result = subprocess.run(
        [sys.executable, "-m", "devcam", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0


def test_parse_args_http():
    from devcam.server import parse_args
    sys.argv = ["devcam", "http", "--port", "9090", "--camera", "1", "--fps", "15"]
    args = parse_args()
    assert args.protocol == "http"
    assert args.port == 9090
    assert args.camera == 1
    assert args.fps == 15


def test_parse_args_rtsp_defaults():
    from devcam.server import parse_args
    sys.argv = ["devcam", "rtsp"]
    args = parse_args()
    assert args.protocol == "rtsp"
    assert args.port is None
    assert args.backend == "auto"


def test_parse_args_resolution():
    from devcam.server import parse_args
    sys.argv = ["devcam", "http", "--resolution", "1280x720"]
    args = parse_args()
    assert args.resolution == "1280x720"
