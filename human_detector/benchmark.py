#!/usr/bin/env python3
"""Quick benchmark: how many detections/sec can the detector handle?"""

import argparse
import time
import requests

def main():
    p = argparse.ArgumentParser(description="Benchmark human detector")
    p.add_argument("--detector", default="http://localhost:1082/detect")
    p.add_argument("--snapshot", default="http://localhost:1080/snapshot")
    p.add_argument("-n", type=int, default=50, help="Number of requests (default: 50)")
    args = p.parse_args()

    # Pre-fetch one snapshot to reuse (removes network jitter from camserver)
    print(f"Fetching snapshot from {args.snapshot}...")
    img = requests.get(args.snapshot, timeout=5).content
    print(f"  Image size: {len(img)} bytes")

    # Warm up (first request loads model)
    print("Warming up model...")
    requests.post(args.detector, data=img, headers={"Content-Type": "image/jpeg"}, timeout=30)

    # Benchmark
    print(f"Running {args.n} detections...")
    times = []
    for i in range(args.n):
        t0 = time.perf_counter()
        r = requests.post(args.detector, data=img, headers={"Content-Type": "image/jpeg"}, timeout=30)
        dt = time.perf_counter() - t0
        times.append(dt)
        r.raise_for_status()

    avg = sum(times) / len(times)
    fastest = min(times)
    slowest = max(times)
    hz = 1.0 / avg

    print(f"\nResults ({args.n} requests):")
    print(f"  Avg: {avg*1000:.1f} ms  ({hz:.1f} Hz)")
    print(f"  Min: {fastest*1000:.1f} ms  ({1/fastest:.1f} Hz)")
    print(f"  Max: {slowest*1000:.1f} ms  ({1/slowest:.1f} Hz)")

if __name__ == "__main__":
    main()
