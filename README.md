# Local Companion

A talking face on your own computer. You speak or type, it answers out loud with a lip-synced face.
Everything runs locally: no accounts, no cloud, no telemetry.

![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![Platform: Windows (WSL2) / Linux](https://img.shields.io/badge/platform-Windows%20(WSL2)%20%7C%20Linux-lightgrey)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

Drop in one front-facing portrait, press the green phone button, and talk. The face is animated locally
with [Ditto](https://github.com/antgroup/ditto-talkinghead), the voice is [Kokoro](https://github.com/thewh1teagle/kokoro-onnx),
and the chat runs on [Ollama](https://ollama.com). An 8 GB laptop GPU is enough.

## What it does

- **One photo becomes a talking face.** Open-mouth, blinking, lip-synced video rendered on your GPU.
- **Hands-free conversation.** Speak, it answers aloud, then listens again. You can interrupt it by talking over it.
- **Typing works too.** The speech-bubble button opens chat and settings.
- **Optional camera.** Mood and hand gestures become hints for the reply, it can wave back, and
  "what do you see?" sends a camera frame to a local vision model.
- **Voice, chat and vision models are selectable** in the settings panel.
- **Private by design.** Models download once into `checkpoints/`; after that the app works offline.
  It listens only on `127.0.0.1` and rejects requests from other websites.

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (Docker Desktop, WSL2 backend) or Linux. **macOS is not supported** (the face engine needs an NVIDIA GPU). |
| GPU | NVIDIA with at least 8 GB VRAM, recent driver. RTX 40xx laptops work. |
| Docker | Docker Desktop or Docker Engine with the NVIDIA Container Toolkit. Docker Compose v2. |
| Python | 3.8 or newer (only used to run `start.py`). |
| Disk | About 25 GB free for the first run (image, models, chat weights). |
| Browser | Current Chrome, Edge or Firefox. Chrome/Edge use built-in speech recognition; Firefox is transcribed locally by the app. |

## Start it

```bash
git clone https://github.com/SamuelH98/local-avatar.git
cd local-avatar
python3 start.py            # Windows: double-click start.bat
```

1. `start.py` checks Docker and your GPU, then builds the image (10–20 minutes the first time).
2. The app downloads the face, voice, chat and speech-recognition models (a few GB) and shows progress
   in the terminal and in the web page.
3. When it prints `Ready`, your browser opens. Later starts take a few seconds.

Open the address it prints, usually **http://127.0.0.1:8080**. If that port is busy, `start.py` picks a free one
and remembers it in `.env`.

## Using it

1. Drop a front-facing portrait onto the page (or click to choose one). Mouth closed, face large and well lit
   works best.
2. Tap the green phone. It greets you out loud and starts listening.
3. Talk, or type in the chat panel. The amber mic button mutes/unmutes; the red button hangs up.

Behind the speech-bubble button: conversation log, chat input, and settings for the chat model, vision model,
voice, zoom, and interruption handling. The camera button enables mood/gesture hints and the
"what do you see?" question.

### Commands

| Command | What it does |
|---|---|
| `python3 start.py` | Start the app, or check on it if it is already running. |
| `python3 start.py --stop` | Stop it. Models and settings are kept. |
| `python3 start.py --logs` | Show recent log lines, for troubleshooting. |
| `python3 start.py --rebuild` | Rebuild the image from scratch (rarely needed). |
| `python3 start.py --no-browser` | Start without opening a browser. |

## How it works

```
Browser (index.html + app.js)
   |  HTTP on 127.0.0.1
   v
companion.py  ──  /api/speak   ──> Kokoro (voice, CPU) ──> Ditto (lip sync, GPU) ──> mp4
   |          ──  /api/listen  ──> faster-whisper (CPU, only for browsers without built-in recognition)
   |          ──  /ollama/*    ──> Ollama (chat engine, in a second container)
   v
checkpoints/  mounted model cache, survives rebuilds and upgrades
```

The Docker image contains the app, the Ditto source, Kokoro and MediaPipe browser assets. Model weights are **not**
baked into the image; they download on first run into `checkpoints/` (mounted, so they survive rebuilds).

### HTTP endpoints

The page talks to one server, so there is nothing to configure.

| Endpoint | Purpose |
|---|---|
| `GET /` | The web page (`index.html`, `app.js`, `vendor/`). |
| `GET /api/status` | Setup progress, readiness, loaded models, voices. |
| `POST /api/face` | Set the face photo (image bytes). |
| `GET /api/idle` | Looping video of the face at rest. |
| `POST /api/speak` | `{"text": "...", "voice": "af_heart"}` → lip-synced `video/mp4`. |
| `POST /api/listen` | Raw 16 kHz mono int16 PCM → `{"text": "..."}`. |
| `/ollama/api/tags`, `/ollama/api/chat` | Whitelisted pass-through to the chat engine. |

## Settings (`.env`)

`.env` is created on first run and is intentionally not committed. Edit it, then run `python3 start.py` again.

| Setting | Meaning |
|---|---|
| `APP_PORT` | Port of the web page. `start.py` picks a free one if it is taken and writes it back. |
| `CHAT_MODEL` | Ollama chat model (default `qwen2.5:1.5b`). |
| `VISION_MODEL` | Model for "what do you see" (default `moondream`). Leave empty to switch vision off. |
| `OLLAMA_URL` | Remove the `#` to use an Ollama already installed on this computer (it must run with `OLLAMA_HOST=0.0.0.0`). Otherwise a chat engine runs in Docker. |

When `OLLAMA_URL` points at your own Ollama, `start.py` does not start the bundled chat container.

## What is in the box

```
start.py                the only thing you run
start.bat               Windows launcher
docker-compose.yml      the two containers (app + optional chat engine)
Dockerfile              builds the app image
.env                    your settings (created on first run; not in git)
app/
  companion.py          the one server: web page, face, voice, chat pass-through, first-run setup
  index.html            the page
  app.js                page logic: photo, call, chat, camera
docker/
  fetch_assets.py       build-time helper that bakes MediaPipe into the image
checkpoints/            downloaded models (created on first run; not in git)
tests/                  automated tests (no GPU, Docker or internet needed)
```

There is no separate web server or setup script. `start.py` starts Docker; the app inside Docker does the rest,
including downloading its own models.

## Good to know

- **Safe by default:** the app only listens on your own computer (`127.0.0.1`), refuses requests from other
  websites (Host/Origin check), serves only its own page files, and exposes just the two chat calls it needs.
  Uploads are capped at 30 MB.
- **Your GPU stays free:** the face model loads when needed and unloads after 5 minutes of silence
  (`IDLE_UNLOAD_SECONDS`). One face render runs at a time so VRAM never spikes. Plug the laptop in.
- **Problems fix themselves:** if a download fails or Ollama is slow to start, the app retries every 30 seconds
  and shows the reason. If the prebuilt GPU engines are rejected by your card, it converts the models for your GPU
  once and remembers.
- **First reply is slower:** each sentence is rendered, then played, so there is a short pause before the face speaks.
- **Conversation is not saved:** chat history lives in the page and disappears when you close it. Settings and
  the photo are remembered in the browser; models live in `checkpoints/`.

## Troubleshooting

| You see | Do this |
|---|---|
| "Docker is not running" | Start Docker Desktop, wait until it says running, then run `python3 start.py` again. |
| "cannot see your NVIDIA GPU" | Update the NVIDIA driver. Windows: Docker Desktop with the WSL2 backend. Linux: install the NVIDIA Container Toolkit and restart Docker. |
| Page says "Setting up…" for a long time | Normal on the first run (big downloads). If the number stops growing, check `python3 start.py --logs`. |
| "No face found" / "could not be animated" | Use a larger, well-lit, front-facing photo with the mouth closed. |
| Hands-free listening does nothing | Allow the microphone for the page. Chrome/Edge need internet for their built-in recognition; Firefox uses the local engine. Typing always works. |
| "Speech recognition is not ready yet" (Firefox) | The first run downloads a small Whisper model. Wait a minute; the page retries by itself. |
| Port 8080 is already in use | `start.py` automatically uses the next free port and saves it in `.env`. |
| Setup stops with a message | It keeps retrying by itself. Fix the cause, then run `python3 start.py` again; the full log is in `python3 start.py --logs`. |
| No disk space during build/download | Free about 25 GB, then run `python3 start.py --rebuild`. |

## Development and tests

The test suite uses fake Docker, GPU, Ollama and model downloads, so it needs no GPU, Docker or internet:

```bash
python3 -m unittest discover -s tests -v
```

That covers the server (`tests/test_companion.py`) and the launcher (`tests/test_start.py`).
To include the real-browser test:

```bash
pip install playwright pillow && playwright install chromium
python3 -m unittest tests.test_browser -v
```

`tests/fakeworld.py` is the pretend machine shared by the tests. The app itself is plain Python with no
third-party imports at module level; `app/companion.py` and `app/app.js` explain the design in comments.

## Coming from an older version

Just run `python3 start.py`. Old containers are replaced, your `checkpoints/` folder and downloaded chat models are
reused, and Docker's cached build layers mean no 20-minute rebuild. `server.py`, `ditto_server.py`, `setup.py` and
`vendor/` are gone; you can delete an old `vendor/` folder.

## License

[MIT](LICENSE) © 2026 Samuel Hale.

Third-party components keep their own licenses:

- [Ditto](https://github.com/antgroup/ditto-talkinghead) (face animation), Apache-2.0 — cloned during the image build.
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) and the Kokoro voice model, MIT / Apache-2.0 — downloaded at runtime.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and Whisper models, MIT — installed in the image, downloaded at runtime.
- [Ollama](https://github.com/ollama/ollama) and the selected chat/vision models — separate containers and their own model licenses.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) tasks and runtime, Apache-2.0 — fetched during the image build.

Model weights are not distributed with this repository.
