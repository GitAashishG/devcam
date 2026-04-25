#!/usr/bin/env python3
"""Benchmark human detection throughput using devcam.detection.

Usage:
  python -m devcam.detection.benchmark              # use built-in test frame
  python -m devcam.detection.benchmark --image photo.jpg
  python -m devcam.detection.benchmark --snapshot http://localhost:1080/snapshot
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np


def _make_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a synthetic BGR frame for benchmarking."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark devcam human detection")
    parser.add_argument(
        "-n", type=int, default=50, help="Number of detection passes (default: 50)",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to an image file to use as input",
    )
    parser.add_argument(
        "--snapshot", type=str, default=None,
        help="URL to fetch a JPEG snapshot (e.g. http://localhost:1080/snapshot)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    from devcam.detection import HAS_ULTRALYTICS, HumanDetector

    if not HAS_ULTRALYTICS:
        print("Error: ultralytics is not installed. Install with: pip install 'devcam[vision]'")
        sys.exit(1)

    # Prepare the frame
    if args.snapshot:
        import urllib.request
        print(f"Fetching snapshot from {args.snapshot}...")
        with urllib.request.urlopen(args.snapshot, timeout=5) as resp:
            data = np.frombuffer(resp.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            print("Error: could not decode image from snapshot URL")
            sys.exit(1)
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: could not read image file: {args.image}")
            sys.exit(1)
    else:
        frame = _make_test_frame()

    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    detector = HumanDetector(confidence_threshold=args.confidence)

    # Warm up (first call loads the model)
    print("Loading model...")
    result = detector.detect(frame)
    print(f"  Model loaded — {result['count']} person(s) detected in warm-up")

    # Benchmark
    print(f"Running {args.n} detections...")
    times = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        detector.detect(frame)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    fastest = min(times)
    slowest = max(times)

    print(f"\nResults ({args.n} passes):")
    print(f"  Avg: {avg * 1000:.1f} ms  ({1 / avg:.1f} Hz)")
    print(f"  Min: {fastest * 1000:.1f} ms  ({1 / fastest:.1f} Hz)")
    print(f"  Max: {slowest * 1000:.1f} ms  ({1 / slowest:.1f} Hz)")


if __name__ == "__main__":
    main()
