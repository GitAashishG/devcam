#!/usr/bin/env bash
# setup_mediamtx.sh — Download mediamtx (RTSP server) for this OS.
# Usage: bash setup_mediamtx.sh

set -euo pipefail

VERSION="1.12.2"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1"
        exit 1
    fi
}

require_command curl
require_command tar

# Detect operating system
case "$(uname -s)" in
    Darwin) OS="darwin" ;;
    Linux)  OS="linux" ;;
    *)      echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

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
echo "  python camserver.py rtsp"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo ""
    echo "Note: ffmpeg is also required for RTSP mode."
    case "$OS" in
        darwin) echo "Install it with: brew install ffmpeg" ;;
        linux)  echo "Install it with your package manager, for example: sudo apt install ffmpeg" ;;
    esac
fi
