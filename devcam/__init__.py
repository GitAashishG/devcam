"""devcam - unified camera streaming and optional vision tools."""

__version__ = "0.2.1"

from devcam.detection import HAS_ULTRALYTICS, HumanDetector

__all__ = ["__version__", "HAS_ULTRALYTICS", "HumanDetector"]
