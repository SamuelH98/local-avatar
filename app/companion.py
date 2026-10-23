#!/usr/bin/env python3
"""
Companion server. This is the ONE server: it runs inside the Docker container and you never start it by hand
(start.py does). One address serves everything, so the browser needs no settings:

    /                          the web page (index.html, app.js, vendor/)
    /api/status                first-run setup progress and what is ready
    /api/face   (POST image)   set the face photo
    /api/idle   (GET)          looping mp4 of the face at rest
    /api/speak  (POST json)    {"text", "voice"} -> lip-synced mp4 (neural voice + face in one step)
    /api/listen (POST audio)   raw 16 kHz mono int16 -> {"text"}: speech recognition for browsers without their own (Firefox)
    /ollama/api/tags|chat      the chat engine, passed through

First run: a background thread downloads the face model, the voice and the chat models, checks the GPU and
prepares the face engine. Progress shows in the page. Every step is skipped when already done, so a restart is quick.
Light on an 8 GB laptop GPU: the face model loads when first needed, unloads after IDLE_UNLOAD_SECONDS of silence,
and only one face render runs at a time.
"""
import gc, importlib, io, json, mimetypes, os, re, shutil, subprocess, sys, tempfile, threading, time, traceback
import urllib.request, wave
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

# --------------------------------------------------------------------------------------------- settings (env vars)
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.environ.get("WEB_DIR", HERE)                                   # index.html, app.js, vendor/
HOST = os.environ.get("HOST", "0.0.0.0")                                # 0.0.0.0 inside Docker; compose publishes it on 127.0.0.1 only
PORT = int(os.environ.get("PORT", "8790"))
REPO = os.path.abspath(os.environ.get("DITTO_REPO", "/ditto/ditto-talkinghead"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434").rstrip("/")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:1.5b")
VISION_MODEL = os.environ.get("VISION_MODEL", "moondream")              # empty = no vision
IDLE_UNLOAD_SECONDS = float(os.environ.get("IDLE_UNLOAD_SECONDS", "300"))   # 0 = keep the face model loaded
RETRY_SECONDS = float(os.environ.get("RETRY_SECONDS", "30"))
OLLAMA_WAIT_SECONDS = float(os.environ.get("OLLAMA_WAIT_SECONDS", "120"))
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") != "0"
HF_REPO = os.environ.get("HF_REPO", "digital-avatar/ditto-talkinghead")
KOKORO_URL = os.environ.get("KOKORO_URL", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0")

CKPT = os.path.join(REPO, "checkpoints")                                # a mounted folder: weights survive rebuilds
CFG = os.path.join(CKPT, "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")
ENGINES_PREBUILT = os.path.join(CKPT, "ditto_trt_Ampere_Plus")          # RTX 40xx is compatible
ENGINES_CUSTOM = os.path.join(CKPT, "ditto_trt_custom")                 # made from ONNX if the prebuilt ones are rejected
ONNX = os.path.join(CKPT, "ditto_onnx")
KOKORO_DIR = os.path.join(CKPT, "kokoro")
KOKORO_FILES = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")             # speech-to-text for browsers with no built-in one (Firefox); runs on the CPU
WHISPER_DIR = os.path.join(CKPT, "whisper")
VERIFIED = os.path.join(CKPT, ".face_engine_ok")
WORK = tempfile.mkdtemp(prefix="companion_")
MAX_UPLOAD = 30 * 2**20
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
OLLAMA_PATHS = {"/api/tags", "/api/chat"}                               # the only Ollama calls the page can make

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".task")


# --------------------------------------------------------------------------------------------- shared state
class SetupError(Exception):
    """A problem the person can act on. The message is shown in the page and by start.py."""


STATUS = {"stage": "Starting…", "detail": "", "error": None, "ready": False}
FACE = {"sdk": None, "infer": None, "src": None, "idle": None, "last": 0.0, "n": 0, "error": None}
VOICE = {"engine": None, "names": []}
EAR = {"model": None, "error": None}
EAR_LOCK = threading.Lock()
SETUP_STOP = threading.Event()    # set to end the setup loop (used when shutting down and in tests)
FACE_LOCK = threading.Lock()      # one face render at a time, so VRAM never spikes
VOICE_LOCK = threading.Lock()


def step(stage, detail=""):
    STATUS.update(stage=stage, detail=detail, error=None)
    print(f"[setup] {stage}", flush=True)


def _files_in(folder):
    return os.listdir(folder) if os.path.isdir(folder) else []


# --------------------------------------------------------------------------------------------- first-run setup
def download(url, dest, label):
    """Download with a visible MB counter. Written to .part first so a half file is never mistaken for a whole one."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest + ".part", "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            STATUS["detail"] = f"{label}: {done >> 20}" + (f" / {total >> 20}" if total else "") + " MB"
    os.replace(dest + ".part", dest)


def check_gpu():
    if not REQUIRE_GPU:
        return
    step("Checking the GPU")
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = ""
    if "GPU" not in out:
        raise SetupError("The container cannot see your NVIDIA GPU. Linux: install the NVIDIA Container Toolkit. "
                         "Windows: Docker Desktop with the WSL2 backend and an up-to-date NVIDIA driver. Then run start.py again.")
    print(f"[setup] {out.splitlines()[0]}", flush=True)


def face_weights_ok():
    return os.path.exists(CFG) and bool(_files_in(ENGINES_PREBUILT))


def folder_mb(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs if os.path.exists(os.path.join(d, f))) >> 20


def get_face_weights():
    if face_weights_ok():
        return
    step("Downloading the face model (a few GB, first run only)")
    stop = threading.Event()

    def show_size():
        while not stop.wait(2):
            STATUS["detail"] = f"{folder_mb(CKPT)} MB downloaded"
    threading.Thread(target=show_size, daemon=True).start()
    try:
        os.environ.setdefault("HF_HOME", "/tmp/hf")
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, local_dir=CKPT, allow_patterns=["ditto_cfg/*", "ditto_trt_Ampere_Plus/*", "ditto_onnx/*"])
    finally:
        stop.set()
    if not face_weights_ok():
        raise SetupError("The face model download finished but files are missing. Retrying.")


def get_voice_files():
    for name in KOKORO_FILES:
        dest = os.path.join(KOKORO_DIR, name)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 1e6):
            step("Downloading the voice (first run only)")
            download(f"{KOKORO_URL}/{name}", dest, name)


def ollama_open(path, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(OLLAMA_URL + path, data, {"Content-Type": "application/json"}), timeout=timeout)


def ollama_models():
    with ollama_open("/api/tags") as r:
        return [m["name"] for m in json.load(r)["models"]]


def get_chat_models():
    step("Waiting for the chat engine (Ollama)")
    deadline = time.time() + OLLAMA_WAIT_SECONDS
    while True:
        try:
            have = ollama_models()
            break
        except Exception as e:
            if time.time() > deadline:
                raise SetupError(f"Cannot reach Ollama at {OLLAMA_URL} ({e}). If you use your own Ollama, check OLLAMA_URL in .env.")
            time.sleep(3)
    for model in filter(None, (CHAT_MODEL, VISION_MODEL)):
        if model in have or model + ":latest" in have:
            continue
        step(f"Downloading chat model {model} (first run only)")
        with ollama_open("/api/pull", {"model": model, "name": model, "stream": True}, timeout=3600) as r:
            for line in r:
                j = json.loads(line)
                if j.get("error"):
                    raise SetupError(f"Could not download {model}: {j['error']}")
                pct = f" {j.get('completed', 0) * 100 // j['total']}%" if j.get("total") else ""
                STATUS["detail"] = f"{model}: {j.get('status', '')}{pct}"


def start_face_engine():
    """First run only: load the face model once so problems show up now, not in the middle of a conversation."""
    if os.path.exists(VERIFIED):
        return
    step("Starting the face engine for the first time (can take a few minutes)")
    with FACE_LOCK:
        load_face()
    open(VERIFIED, "w").write("ok")


def start_voice():
    step("Starting the voice")
    voice_engine()


def get_ear():
    """Speech recognition for browsers without it (Firefox). Optional: if it fails, Chrome and Edge still work, and it is retried on first use."""
    if EAR["model"] is not None:
        return
    step("Getting the ears: speech recognition (first run only, about 150 MB)")
    try:
        with EAR_LOCK:
            load_ear()
    except Exception as e:
        EAR["error"] = f"{type(e).__name__}: {e}"
        print(f"[ear] could not start: {EAR['error']}", flush=True)


SETUP_STEPS = (check_gpu, get_face_weights, get_voice_files, get_chat_models, start_face_engine, start_voice, get_ear)


def run_setup():
    """Runs the steps in order. On any problem: report it, wait, try again (every step skips what is already done)."""
    while not SETUP_STOP.is_set():
        try:
            for fn in SETUP_STEPS:
                fn()
            step("Ready")
            STATUS["ready"] = True
            return
        except Exception as e:
            traceback.print_exc()
            STATUS["error"] = str(e) if isinstance(e, SetupError) else f"{type(e).__name__}: {e}"
            SETUP_STOP.wait(RETRY_SECONDS)


# --------------------------------------------------------------------------------------------- face (Ditto)
def engines_dir():
    return ENGINES_CUSTOM if _files_in(ENGINES_CUSTOM) else ENGINES_PREBUILT


def build_sdk(data):
    inference = importlib.import_module("inference")
    pipeline = "stream_pipeline_online" if "online" in os.path.basename(CFG) else "stream_pipeline_offline"
    t = time.time()
    sdk = importlib.import_module(pipeline).StreamSDK(CFG, data)
    print(f"[face] model loaded in {time.time() - t:.1f}s", flush=True)
    return inference, sdk


def convert_engines():
    """The prebuilt TensorRT engines were rejected by this GPU: build engines from the ONNX models instead."""
    step("Optimizing the face model for your GPU (one time, several minutes)")
    tmp = ENGINES_CUSTOM + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    rc = subprocess.run([sys.executable, "scripts/cvt_onnx_to_trt.py", "--onnx_dir", ONNX, "--trt_dir", tmp], cwd=REPO).returncode
    if rc != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SetupError("Converting the face model for your GPU failed (see the container log: python3 start.py --logs).")
    os.replace(tmp, ENGINES_CUSTOM)


def load_face():
    """Build the Ditto SDK once. Call with FACE_LOCK held."""
    if FACE["sdk"] is not None:
        return
    for p in (REPO, CFG, engines_dir()):
        if not os.path.exists(p):
            raise SetupError(f"Face model files are missing ({p}). Run start.py again.")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(REPO)                                       # the Ditto repo uses relative paths
    try:
        FACE["infer"], FACE["sdk"] = build_sdk(engines_dir())
    except (ImportError, SetupError):
        raise
    except Exception:
        if engines_dir() == ENGINES_CUSTOM or not os.path.isdir(ONNX):
            raise
        traceback.print_exc()
        convert_engines()
        FACE["infer"], FACE["sdk"] = build_sdk(engines_dir())


def unload_face():
    FACE["sdk"] = FACE["infer"] = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("[face] unloaded (idle)", flush=True)


def render_locked(wav):
    """wav bytes -> mp4 bytes. Call with FACE_LOCK held."""
    if not FACE["src"]:
        raise RuntimeError("no face photo yet")
    load_face()
    audio, out = os.path.join(WORK, "in.wav"), os.path.join(WORK, "out.mp4")
    with open(audio, "wb") as f:
        f.write(wav)
    if os.path.exists(out):
        os.remove(out)
    t = time.time()
    try:
        FACE["infer"].run(FACE["sdk"], audio, FACE["src"], out, {})
    except Exception:
        FACE["sdk"] = None                               # state is unknown after a failure: rebuild next time
        raise
    finally:
        FACE["last"] = time.time()
    FACE["error"] = None
    print(f"[face] rendered in {time.time() - t:.1f}s", flush=True)
    with open(out, "rb") as f:
        return f.read()


def render(wav):
    with FACE_LOCK:
        return render_locked(wav)


def set_face(image):
    with FACE_LOCK:
        FACE["n"] += 1
        path = os.path.join(WORK, f"face{FACE['n']}" + (".png" if image[:4] == b"\x89PNG" else ".jpg"))
        with open(path, "wb") as f:
            f.write(image)
        if FACE["src"] and os.path.exists(FACE["src"]):
            os.remove(FACE["src"])
        FACE["src"], FACE["idle"] = path, None           # new face: its idle loop must be rendered again


def idle_clip():
    with FACE_LOCK:
        if FACE["idle"] is None:
            FACE["idle"] = render_locked(silent_wav())
        return FACE["idle"]


def reaper():
    """Gives the GPU back after IDLE_UNLOAD_SECONDS without speech."""
    while IDLE_UNLOAD_SECONDS > 0:
        time.sleep(min(5, IDLE_UNLOAD_SECONDS))
        if FACE["sdk"] is not None and time.time() - FACE["last"] > IDLE_UNLOAD_SECONDS and FACE_LOCK.acquire(blocking=False):
            try:
                if time.time() - FACE["last"] > IDLE_UNLOAD_SECONDS:
                    unload_face()
            finally:
                FACE_LOCK.release()


# --------------------------------------------------------------------------------------------- voice (Kokoro, CPU)
def voice_engine():
    if VOICE["engine"] is not None:
        return VOICE["engine"]
    model, voices = (os.path.join(KOKORO_DIR, f) for f in KOKORO_FILES)
    if not (os.path.exists(model) and os.path.exists(voices)):
        raise SetupError("The voice files are missing. Run start.py again.")
    from kokoro_onnx import Kokoro
    engine = Kokoro(model, voices)
    names = sorted(v for v in engine.get_voices() if v[:2] in ("af", "am", "bf", "bm"))
    VOICE["names"] = sorted(names, key=lambda v: (v != "af_heart", v != "af_sarah", v))
    VOICE["engine"] = engine
    return engine


def wav_bytes(samples, rate):
    import numpy as np
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def synthesize(text, voice):
    with VOICE_LOCK:
        engine = voice_engine()
        samples, rate = engine.create(text, voice=voice if voice in VOICE["names"] else VOICE["names"][0], speed=1.0, lang="en-us")
    return wav_bytes(samples, rate)


def load_ear():
    """Whisper on the CPU (int8), so it takes no VRAM from the face. Call with EAR_LOCK held."""
    if EAR["model"] is None:
        from faster_whisper import WhisperModel
        os.makedirs(WHISPER_DIR, exist_ok=True)
        EAR["model"] = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root=WHISPER_DIR)
        EAR["error"] = None
        print(f"[ear] speech recognition ready ({WHISPER_MODEL})", flush=True)
    return EAR["model"]


NOT_SPEECH = {"you", "thanks for watching", "thank you for watching"}      # what Whisper tends to invent from silence


def transcribe(pcm):
    """16 kHz mono 16-bit little-endian audio -> text ('' if nothing was said)."""
    import numpy as np
    audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype("float32") / 32768.0
    if len(audio) < 16000 * 0.3:
        return ""
    with EAR_LOCK:
        model = load_ear()
        segments, _ = model.transcribe(audio, language="en" if WHISPER_MODEL.endswith(".en") else None, beam_size=1,
                                       vad_filter=False, condition_on_previous_text=False, temperature=0.0)
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0).strip()
    return "" if text.lower().strip(" .!?,") in NOT_SPEECH else text


def silent_wav(seconds=4):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * seconds)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- web server
def status_json():
    return {**STATUS, "face_loaded": FACE["sdk"] is not None, "face_photo": bool(FACE["src"]), "face_error": FACE["error"],
            "voices": VOICE["names"], "chat_model": CHAT_MODEL, "vision_model": VISION_MODEL,
            "ear": EAR["model"] is not None, "ear_error": EAR["error"]}


class Handler(BaseHTTPRequestHandler):
    server_version = "companion"

    def log_message(self, *args):
        pass

    # ---- small helpers
    def reply(self, code, body=b"", ctype="application/json", cache="no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def reply_json(self, code, obj):
        self.reply(code, json.dumps(obj).encode())

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("upload too large")
        return self.rfile.read(n) if n else b""

    def is_local_request(self):
        """Refuses requests that did not come from this computer's own address (stops other websites from using the app)."""
        host = urlsplit("//" + (self.headers.get("Host") or "")).hostname
        origin = self.headers.get("Origin")
        return host in LOCAL_HOSTS and (not origin or urlsplit(origin).netloc == self.headers.get("Host"))

    # ---- routing
    def dispatch(self):
        try:
            if not self.is_local_request():
                return self.reply_json(403, {"error": "forbidden"})
            path = urlsplit(self.path).path
            if path.startswith("/ollama/"):
                return self.proxy_ollama(path[len("/ollama"):])
            if self.command == "POST":
                return self.post(path)
            if path == "/api/status":
                return self.reply_json(200, status_json())
            if path == "/api/idle":
                return self.reply(200, idle_clip(), "video/mp4")
            return self.static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            FACE["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            try:
                self.reply_json(400 if isinstance(e, ValueError) else 500, {"error": FACE["error"]})
            except Exception:
                pass

    do_GET = do_HEAD = do_POST = dispatch

    def post(self, path):
        body = self.read_body()
        if path == "/api/face":
            set_face(body)
            return self.reply_json(200, {"ok": True})
        if path == "/api/speak":
            j = json.loads(body or b"{}")
            text = str(j.get("text", "")).strip()[:1000]
            if not text:
                raise ValueError("no text to speak")
            return self.reply(200, render(synthesize(text, str(j.get("voice", "")))), "video/mp4")
        if path == "/api/listen":                                # raw 16 kHz mono int16 audio from the page -> what was said
            try:
                return self.reply_json(200, {"text": transcribe(body)})
            except Exception as e:                               # kept apart from FACE["error"]: the face still works without ears
                EAR["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                return self.reply_json(503, {"error": EAR["error"]})
        self.reply_json(404, {"error": "not found"})

    def proxy_ollama(self, path):
        if path not in OLLAMA_PATHS:
            return self.reply_json(404, {"error": "not found"})
        body = self.read_body() or None
        target = urlsplit(OLLAMA_URL)
        try:
            conn = HTTPConnection(target.hostname, target.port or 11434, timeout=3600)
            conn.request(self.command, path, body, {"Content-Type": self.headers.get("Content-Type") or "application/json"})
            r = conn.getresponse()
        except Exception as e:
            return self.reply_json(502, {"error": f"Ollama is not reachable at {OLLAMA_URL} ({e})"})
        self.send_response(r.status)
        self.send_header("Content-Type", r.getheader("Content-Type") or "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536)                   # streams the reply token by token as it arrives
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()

    def static(self, path):
        """Only index.html, app.js and vendor/ are ever served; nothing else in the container is reachable."""
        page = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js"}.get(path)
        root = os.path.realpath(WEB)
        if page:
            target = os.path.join(root, page)
        elif path.startswith("/vendor/"):
            target = os.path.realpath(os.path.join(root, path[1:]))
            if not target.startswith(os.path.join(root, "vendor") + os.sep):
                return self.reply_json(404, {"error": "not found"})
        else:
            return self.reply_json(404, {"error": "not found"})
        if not os.path.isfile(target):
            return self.reply_json(404, {"error": f"{path} not found"})
        with open(target, "rb") as f:
            data = f.read()
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        self.reply(200, data, ctype, "max-age=86400" if path.startswith("/vendor/") else "no-cache")


def main():
    threading.Thread(target=run_setup, daemon=True).start()
    threading.Thread(target=reaper, daemon=True).start()
    ThreadingHTTPServer.daemon_threads = True
    print(f"[companion] http://{HOST}:{PORT}  face model loads on first reply, unloads after {IDLE_UNLOAD_SECONDS:.0f}s idle", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()