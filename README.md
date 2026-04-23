# camserver

Dead-simple webcam broadcaster — pick HTTP or RTSP at launch.

## Quick Start (HTTP)

```bash
pip install -r requirements.txt
python camserver.py http
```

Open **http://localhost:1080/** in a browser.

## RTSP Mode

Requires **ffmpeg** and **mediamtx**.

```bash
# Install ffmpeg
brew install ffmpeg

# Download mediamtx (one-time)
bash setup_mediamtx.sh

# Start mediamtx in one terminal (configure it to listen on port 1081)
./mediamtx

# Start camserver in another terminal
python camserver.py rtsp
```

Connect with VLC: **rtsp://localhost:1081/cam**

## Usage

```
python camserver.py {http,rtsp} [--port PORT] [--camera INDEX] [--resolution WxH]
```

| Flag | Default | Description |
|---|---|---|
| `protocol` | *(required)* | `http` or `rtsp` |
| `--port` | 1080 (http) / 1081 (rtsp) | Stream port |
| `--camera` | 0 | Camera index |
| `--resolution` | auto | `WxH` e.g. `1280x720` |

## macOS Camera Permission

If the camera won't open, grant Terminal access:
**System Settings → Privacy & Security → Camera → Terminal** (toggle on).
