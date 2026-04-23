#!/usr/bin/env bash
# setup_mediamtx.sh — Download mediamtx (RTSP server) for macOS.
# Usage: bash setup_mediamtx.sh

set -euo pipefail

VERSION="1.12.2"
OS="darwin"

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    arm64|aarch64) ARCH="arm64" ;;
    x86_64)        ARCH="amd64" ;;
    *)             echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

TARBALL="mediamtx_v${VERSION}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/bluenviron/mediamtx/releases/download/v${VERSION}/${TARBALL}"

echo "Downloading mediamtx v${VERSION} for ${OS}/${ARCH}..."
curl -L -o "$TARBALL" "$URL"

echo "Extracting..."
tar xzf "$TARBALL" mediamtx
rm -f "$TARBALL"
chmod +x mediamtx

echo ""
echo "Done! Run mediamtx with:"
echo "  ./mediamtx"
echo ""
echo "Then start camserver with RTSP enabled:"
echo "  python camserver.py"
