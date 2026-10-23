"""A pretend machine for testing: fake Ditto repo, fake Kokoro, fake Hugging Face, fake Ollama, fake download host.
No GPU, Docker or internet needed. Used by test_companion.py and test_browser.py."""
import json, os, shutil, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

FAKE_INFERENCE = '''
import os, shutil
RENDERS = []
def run(sdk, audio, src, out, cfg):
    RENDERS.append((os.path.getsize(audio), os.path.basename(src)))
    shutil.copy(os.environ["FAKE_CLIP"], out)
'''
FAKE_PIPELINE = '''
import os
class StreamSDK:
    BUILDS = []
    def __init__(self, cfg, data):
        StreamSDK.BUILDS.append(data)
        if os.environ.get("FAKE_REJECT_PREBUILT") and "Ampere_Plus" in data:
            raise RuntimeError("Error Code 6: could not deserialize engine")
'''
FAKE_CONVERT = '''
import argparse, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--onnx_dir"); ap.add_argument("--trt_dir"); a = ap.parse_args()
if os.environ.get("FAKE_CONVERT_FAILS"): sys.exit(2)
os.makedirs(a.trt_dir); open(os.path.join(a.trt_dir, "model.engine"), "w").write("engine")
'''
FAKE_KOKORO = '''
import numpy as np
class Kokoro:
    def __init__(self, model, voices): pass
    def get_voices(self): return ["bm_george", "af_sarah", "am_adam", "af_heart", "zz_other"]
    def create(self, text, voice, speed, lang):
        assert voice in self.get_voices() and lang == "en-us"
        return np.sin(np.linspace(0, 440, 4800)).astype("float32"), 24000
'''
FAKE_HF = '''
import fnmatch, os
FILES = {"ditto_cfg/v0.4_hubert_cfg_trt_online.pkl": b"cfg", "ditto_trt_Ampere_Plus/a.engine": b"e", "ditto_onnx/a.onnx": b"o", "other/ignored.bin": b"x"}
def snapshot_download(repo, local_dir, allow_patterns):
    for name, data in FILES.items():
        if any(fnmatch.fnmatch(name, p) for p in allow_patterns):
            os.makedirs(os.path.dirname(os.path.join(local_dir, name)), exist_ok=True)
            open(os.path.join(local_dir, name), "wb").write(data)
'''


def make_clip(path):
    """A real 1-second clip with sound if ffmpeg can make one, else placeholder bytes."""
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x445566:s=96x96:d=1:r=10", "-f", "lavfi",
                        "-i", "sine=d=1", "-shortest", "-c:v", "libvpx", "-c:a", "libvorbis", "-f", "webm", path], check=True, timeout=60)
    except Exception:
        write(path, b"\x1a\x45\xdf\xa3fake-clip")


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)


class Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a): pass


class FakeOllama(ThreadingHTTPServer):
    """Answers /api/tags, /api/pull (adds the model) and /api/chat (streams a reply word by word)."""
    daemon_threads = True

    def __init__(self, models=()):
        self.models, self.chats, self.pulled = list(models), [], []
        outer = self

        class H(Quiet):
            def do_GET(self):
                body = json.dumps({"models": [{"name": m} for m in outer.models]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()

                def chunk(obj):
                    b = (json.dumps(obj) + "\n").encode()
                    self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
                if self.path == "/api/pull":
                    outer.pulled.append(req["model"]); chunk({"status": "pulling", "completed": 50, "total": 100})
                    outer.models.append(req["model"]); chunk({"status": "success"})
                elif self.path == "/api/chat":
                    outer.chats.append(req)
                    reply = "Hello there. " if req.get("stream") else "I see a cup. "
                    parts = (reply + "This is a test reply!").split(" ")
                    if req.get("stream"):
                        for w in parts: chunk({"message": {"content": w + " "}, "done": False})
                        chunk({"message": {"content": ""}, "done": True})
                    else:
                        chunk({"message": {"content": " ".join(parts)}, "done": True})
                self.wfile.write(b"0\r\n\r\n")
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def url(self): return f"http://127.0.0.1:{self.server_address[1]}"


class FakeDownloads(ThreadingHTTPServer):
    """Serves the Kokoro files (1.5 MB each)."""
    daemon_threads = True

    def __init__(self):
        self.fail = False
        outer = self

        class H(Quiet):
            def do_GET(self):
                if outer.fail:
                    self.send_response(404); self.end_headers(); return
                data = b"k" * 1_500_000
                self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        super().__init__(("127.0.0.1", 0), H)
        threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def url(self): return f"http://127.0.0.1:{self.server_address[1]}"


class World:
    """One pretend machine. world.apply(companion) points the server module at it and resets its state."""

    def __init__(self, weights=True):
        self.tmp = tempfile.mkdtemp(prefix="world_")
        self.repo = os.path.join(self.tmp, "ditto")
        self.web = os.path.join(self.tmp, "web")
        self.clip = os.path.join(self.tmp, "clip.webm")
        make_clip(self.clip)
        os.environ["FAKE_CLIP"] = self.clip
        os.makedirs(os.path.join(self.repo, "scripts"))
        for name, text in (("inference.py", FAKE_INFERENCE), ("stream_pipeline_online.py", FAKE_PIPELINE), ("kokoro_onnx.py", FAKE_KOKORO),
                           ("huggingface_hub.py", FAKE_HF), ("scripts/cvt_onnx_to_trt.py", FAKE_CONVERT)):
            write(os.path.join(self.repo, name), text)
        os.makedirs(os.path.join(self.web, "vendor", "mediapipe"))
        shutil.copy(os.path.join(ROOT, "app", "index.html"), self.web)
        shutil.copy(os.path.join(ROOT, "app", "app.js"), self.web)
        write(os.path.join(self.web, "vendor", "hello.txt"), "hi")
        write(os.path.join(self.web, "secret.txt"), "do not serve")
        self.ollama, self.downloads = FakeOllama(), FakeDownloads()
        if weights:
            self.put_weights()

    def put_weights(self):
        for name in ("ditto_cfg/v0.4_hubert_cfg_trt_online.pkl", "ditto_trt_Ampere_Plus/a.engine", "ditto_onnx/a.onnx"):
            write(os.path.join(self.repo, "checkpoints", name), b"x")
        for f in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"):
            write(os.path.join(self.repo, "checkpoints", "kokoro", f), b"k" * 1_100_000)

    def apply(self, c):
        ck = os.path.join(self.repo, "checkpoints")
        c.REPO, c.CKPT, c.WEB = self.repo, ck, self.web
        c.CFG = os.path.join(ck, "ditto_cfg", "v0.4_hubert_cfg_trt_online.pkl")
        c.ENGINES_PREBUILT, c.ENGINES_CUSTOM, c.ONNX = (os.path.join(ck, d) for d in ("ditto_trt_Ampere_Plus", "ditto_trt_custom", "ditto_onnx"))
        c.KOKORO_DIR, c.VERIFIED = os.path.join(ck, "kokoro"), os.path.join(ck, ".face_engine_ok")
        c.OLLAMA_URL, c.KOKORO_URL = self.ollama.url, self.downloads.url
        c.REQUIRE_GPU, c.RETRY_SECONDS, c.OLLAMA_WAIT_SECONDS, c.IDLE_UNLOAD_SECONDS = False, 0.2, 1, 0
        c.SETUP_STOP = threading.Event()
        c.STATUS.update(stage="Starting…", detail="", error=None, ready=False)
        c.FACE.update(sdk=None, infer=None, src=None, idle=None, last=0.0, n=0, error=None)
        c.VOICE.update(engine=None, names=[])
        for m in ("inference", "stream_pipeline_online", "kokoro_onnx", "huggingface_hub"):
            sys.modules.pop(m, None)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        return self

    def serve(self, c):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), c.Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        return srv

    def close(self):
        for s in (self.ollama, self.downloads):
            s.shutdown(); s.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)
