"""Server tests: python3 -m unittest discover -s tests -v   (no GPU, Docker or internet needed)"""
import http.client, json, os, sys, threading, time, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakeworld import World
import companion as c

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def call(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
    conn.request(method, path, body, headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, r.getheader("Content-Type"), data


def wait_until(cond, secs=15):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class Base(unittest.TestCase):
    weights = True

    def setUp(self):
        self.w = World(weights=self.weights).apply(c)
        self.srv = self.w.serve(c)
        self.threads = []
        os.environ.pop("FAKE_REJECT_PREBUILT", None)
        os.environ.pop("FAKE_CONVERT_FAILS", None)

    def start_setup(self):
        t = threading.Thread(target=c.run_setup, daemon=True)
        t.start()
        self.threads.append(t)

    def tearDown(self):
        c.SETUP_STOP.set()                      # end any setup loop still retrying, so tests cannot leak into each other
        c.IDLE_UNLOAD_SECONDS = 0
        for t in self.threads:
            t.join(10)
        self.srv.shutdown(); self.srv.server_close()
        self.w.close()
        os.environ.pop("FAKE_REJECT_PREBUILT", None)
        os.environ.pop("FAKE_CONVERT_FAILS", None)

    def run_setup_to_ready(self):
        self.start_setup()
        self.assertTrue(wait_until(lambda: c.STATUS["ready"]), f"never became ready: {c.STATUS}")


class StaticAndSecurity(Base):
    def test_serves_page_script_and_vendor(self):
        code, ctype, body = call(self.srv, "GET", "/")
        self.assertEqual((code, ctype.split(";")[0]), (200, "text/html")); self.assertIn(b"Local Companion", body)
        self.assertEqual(call(self.srv, "GET", "/app.js")[1].split(";")[0], "text/javascript")
        self.assertEqual(call(self.srv, "GET", "/vendor/hello.txt")[2], b"hi")

    def test_nothing_else_is_reachable(self):
        for p in ("/secret.txt", "/companion.py", "/vendor/../secret.txt", "/vendor/%2e%2e/secret.txt", "/vendor/..%2fsecret.txt", "/etc/passwd", "/vendor/nope"):
            self.assertEqual(call(self.srv, "GET", p)[0], 404, p)

    def test_other_websites_and_dns_rebinding_are_refused(self):
        self.assertEqual(call(self.srv, "GET", "/api/status", headers={"Host": "evil.example.com"})[0], 403)
        port = self.srv.server_address[1]
        self.assertEqual(call(self.srv, "POST", "/api/face", PNG, {"Origin": "http://evil.example.com"})[0], 403)
        self.assertEqual(call(self.srv, "POST", "/api/face", PNG, {"Origin": f"http://127.0.0.1:{port}"})[0], 200)
        self.assertEqual(call(self.srv, "GET", "/api/status", headers={"Host": f"localhost:{port}"})[0], 200)
        self.assertEqual(call(self.srv, "GET", "/api/status", headers={"Host": f"[::1]:{port}"})[0], 200)

    def test_unknown_post_is_404(self):
        self.assertEqual(call(self.srv, "POST", "/api/nothing", b"{}")[0], 404)


class OllamaProxy(Base):
    def test_only_tags_and_chat_pass_through(self):
        self.w.ollama.models.append("qwen2.5:1.5b")
        code, _, body = call(self.srv, "GET", "/ollama/api/tags")
        self.assertEqual((code, json.loads(body)["models"][0]["name"]), (200, "qwen2.5:1.5b"))
        for p in ("/ollama/api/pull", "/ollama/api/delete", "/ollama/api/push", "/ollama/api/create", "/ollama/"):
            self.assertEqual(call(self.srv, "POST", p, b"{}")[0], 404, p)
        self.assertEqual(self.w.ollama.pulled, [])

    def test_chat_streams_token_by_token(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.srv.server_address[1], timeout=10)
        conn.request("POST", "/ollama/api/chat", json.dumps({"model": "m", "stream": True, "messages": []}), {"Content-Type": "application/json"})
        r = conn.getresponse()
        lines = [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]
        text = "".join(l["message"]["content"] for l in lines)
        self.assertEqual(text.strip(), "Hello there. This is a test reply!"); self.assertTrue(lines[-1]["done"]); self.assertGreater(len(lines), 5)

    def test_unreachable_ollama_gives_clear_502(self):
        c.OLLAMA_URL = "http://127.0.0.1:9"
        code, _, body = call(self.srv, "GET", "/ollama/api/tags")
        self.assertEqual(code, 502); self.assertIn("not reachable", json.loads(body)["error"])


class FaceAndVoice(Base):
    def setUp(self):
        super().setUp()
        self.run_setup_to_ready()

    def renders(self):
        return len(sys.modules["inference"].RENDERS)

    def test_idle_clip_is_cached_until_the_face_changes(self):
        self.assertEqual(call(self.srv, "POST", "/api/face", PNG)[0], 200)
        a, b = call(self.srv, "GET", "/api/idle"), call(self.srv, "GET", "/api/idle")
        self.assertEqual((a[0], a[1]), (200, "video/mp4")); self.assertEqual(a[2], b[2]); self.assertEqual(self.renders(), 1)
        call(self.srv, "POST", "/api/face", b"\xff\xd8jpeg")
        call(self.srv, "GET", "/api/idle")
        self.assertEqual(self.renders(), 2)
        self.assertTrue(sys.modules["inference"].RENDERS[-1][1].endswith(".jpg"))
        self.assertEqual(len([f for f in os.listdir(c.WORK) if f.startswith("face")]), 1, "old face files are cleaned up")

    def test_speak_returns_a_lip_synced_clip(self):
        call(self.srv, "POST", "/api/face", PNG)
        code, ctype, body = call(self.srv, "POST", "/api/speak", json.dumps({"text": "Hello", "voice": "af_heart"}))
        self.assertEqual((code, ctype), (200, "video/mp4")); self.assertGreater(len(body), 10)
        audio_bytes = sys.modules["inference"].RENDERS[-1][0]
        self.assertGreater(audio_bytes, 9000, "the rendered audio is the synthesized speech (4800 samples, 16-bit)")

    def test_speak_problems_are_explained(self):
        code, _, body = call(self.srv, "POST", "/api/speak", json.dumps({"text": "Hi"}))
        self.assertEqual(code, 500); self.assertIn("no face photo", json.loads(body)["error"])
        call(self.srv, "POST", "/api/face", PNG)
        self.assertEqual(call(self.srv, "POST", "/api/speak", json.dumps({"text": "   "}))[0], 400)
        self.assertEqual(call(self.srv, "POST", "/api/speak", b"not json")[0], 400)
        self.assertEqual(call(self.srv, "POST", "/api/speak", json.dumps({"text": "ok", "voice": "nonsense"}))[0], 200, "unknown voice falls back to the first")

    def test_status_lists_voices_best_first_and_models(self):
        s = json.loads(call(self.srv, "GET", "/api/status")[2])
        self.assertEqual(s["voices"][:2], ["af_heart", "af_sarah"]); self.assertNotIn("zz_other", s["voices"])
        self.assertTrue(s["ready"] and s["error"] is None); self.assertEqual(s["chat_model"], c.CHAT_MODEL)

    def test_oversized_upload_is_rejected(self):
        c.MAX_UPLOAD = 100
        self.assertEqual(call(self.srv, "POST", "/api/face", b"x" * 500)[0], 400)

    def test_idle_unload_frees_the_model_then_reloads(self):
        call(self.srv, "POST", "/api/face", PNG); call(self.srv, "POST", "/api/speak", json.dumps({"text": "Hi"}))
        self.assertIsNotNone(c.FACE["sdk"])
        c.IDLE_UNLOAD_SECONDS = 0.3
        t = threading.Thread(target=c.reaper, daemon=True); t.start(); self.threads.append(t)
        self.assertTrue(wait_until(lambda: c.FACE["sdk"] is None, 5), "model should unload when idle")
        c.IDLE_UNLOAD_SECONDS = 0
        t.join(10)
        self.assertEqual(call(self.srv, "POST", "/api/speak", json.dumps({"text": "Back"}))[0], 200)
        self.assertIsNotNone(c.FACE["sdk"])


class FirstRunSetup(Base):
    weights = False

    def test_downloads_everything_then_reports_ready(self):
        stages = []
        real_step = c.step
        c.step = lambda s, d="": (stages.append(s), real_step(s, d))[1]
        try:
            self.run_setup_to_ready()
        finally:
            c.step = real_step
        text = " | ".join(stages)
        for part in ("face model", "voice", "chat model qwen2.5:1.5b", "chat model moondream", "face engine", "Ready"):
            self.assertIn(part, text)
        self.assertTrue(c.face_weights_ok()); self.assertFalse(os.path.exists(os.path.join(c.CKPT, "other")), "only the allowed patterns are fetched")
        self.assertEqual(sorted(os.listdir(c.KOKORO_DIR)), sorted(c.KOKORO_FILES), "no .part leftovers")
        self.assertEqual(self.w.ollama.pulled, ["qwen2.5:1.5b", "moondream"]); self.assertTrue(os.path.exists(c.VERIFIED))

    def test_second_run_downloads_nothing(self):
        self.run_setup_to_ready()
        self.w.ollama.pulled.clear(); self.w.downloads.fail = True
        c.STATUS.update(ready=False)
        mtimes = {f: os.path.getmtime(os.path.join(c.KOKORO_DIR, f)) for f in c.KOKORO_FILES}
        self.run_setup_to_ready()
        self.assertEqual(self.w.ollama.pulled, [])
        self.assertEqual(mtimes, {f: os.path.getmtime(os.path.join(c.KOKORO_DIR, f)) for f in c.KOKORO_FILES})

    def test_vision_can_be_switched_off(self):
        c.VISION_MODEL = ""
        self.run_setup_to_ready()
        self.assertEqual(self.w.ollama.pulled, ["qwen2.5:1.5b"])

    def test_failed_download_reports_error_then_recovers(self):
        self.w.downloads.fail = True
        self.start_setup()
        self.assertTrue(wait_until(lambda: c.STATUS["error"]), "the problem is shown")
        self.assertFalse(c.STATUS["ready"]); self.assertEqual([f for f in os.listdir(c.KOKORO_DIR)] if os.path.isdir(c.KOKORO_DIR) else [], [])
        self.w.downloads.fail = False
        self.assertTrue(wait_until(lambda: c.STATUS["ready"]), "it retries by itself once the problem is gone")
        self.assertIsNone(c.STATUS["error"])

    def test_unreachable_ollama_is_reported(self):
        c.OLLAMA_URL = "http://127.0.0.1:9"
        self.start_setup()
        self.assertTrue(wait_until(lambda: c.STATUS["error"] and "Cannot reach Ollama" in c.STATUS["error"]))

    def test_missing_gpu_is_explained(self):
        c.REQUIRE_GPU = True
        old = os.environ["PATH"]; os.environ["PATH"] = "/nonexistent"
        try:
            with self.assertRaises(c.SetupError) as cm:
                c.check_gpu()
        finally:
            os.environ["PATH"] = old
        self.assertIn("NVIDIA", str(cm.exception)); self.assertIn("start.py", str(cm.exception))


class EngineFallback(Base):
    def test_rejected_prebuilt_engines_are_converted_once_and_remembered(self):
        os.environ["FAKE_REJECT_PREBUILT"] = "1"
        self.run_setup_to_ready()
        builds = sys.modules["stream_pipeline_online"].StreamSDK.BUILDS
        self.assertEqual([os.path.basename(b) for b in builds], ["ditto_trt_Ampere_Plus", "ditto_trt_custom"])
        self.assertTrue(os.path.exists(os.path.join(c.ENGINES_CUSTOM, "model.engine"))); self.assertFalse(os.path.exists(c.ENGINES_CUSTOM + ".tmp"))
        c.unload_face()
        with c.FACE_LOCK:
            c.load_face()
        self.assertEqual(os.path.basename(builds[-1]), "ditto_trt_custom", "later loads go straight to the converted engines")

    def test_failed_conversion_leaves_no_half_engines_and_says_so(self):
        os.environ["FAKE_REJECT_PREBUILT"] = "1"; os.environ["FAKE_CONVERT_FAILS"] = "1"
        self.start_setup()
        self.assertTrue(wait_until(lambda: c.STATUS["error"] and "Converting" in c.STATUS["error"]))
        self.assertFalse(os.path.exists(c.ENGINES_CUSTOM)); self.assertFalse(os.path.exists(c.ENGINES_CUSTOM + ".tmp"))

    def test_missing_python_package_is_not_mistaken_for_a_gpu_problem(self):
        os.remove(os.path.join(self.w.repo, "stream_pipeline_online.py"))
        with self.assertRaises(ImportError):
            with c.FACE_LOCK:
                c.load_face()
        self.assertFalse(os.path.exists(c.ENGINES_CUSTOM))


if __name__ == "__main__":
    unittest.main()
