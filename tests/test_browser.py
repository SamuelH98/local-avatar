"""Real-browser test of the whole page against the real server (fake Ditto/Ollama behind it).
Needs:  pip install playwright && playwright install chromium        Run:  python3 -m unittest tests.test_browser -v"""
import io, json, os, sys, threading, time, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakeworld import World
import companion as c

try:
    from playwright.sync_api import sync_playwright
    from PIL import Image, ImageDraw
except ImportError:
    sync_playwright = None


def face_png():
    img = Image.new("RGB", (400, 480), (210, 170, 140)); d = ImageDraw.Draw(img)
    d.ellipse((120, 140, 160, 170), fill="white"); d.ellipse((240, 140, 280, 170), fill="white"); d.arc((150, 260, 250, 340), 20, 160, fill="black", width=5)
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()


@unittest.skipIf(sync_playwright is None, "playwright not installed")
class PageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World(weights=True).apply(c)
        cls.srv = cls.w.serve(c)
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/"
        cls.photo = os.path.join(cls.w.tmp, "face.png")
        with open(cls.photo, "wb") as f:
            f.write(face_png())
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"])

    @classmethod
    def tearDownClass(cls):
        c.SETUP_STOP.set(); c.IDLE_UNLOAD_SECONDS = 0
        cls.browser.close(); cls.pw.stop(); cls.srv.shutdown(); cls.srv.server_close(); cls.w.close()

    def new_page(self):
        ctx = self.browser.new_context(permissions=["camera", "microphone"], viewport={"width": 1100, "height": 760})
        page = ctx.new_page()
        self.errors, self.speak, self.requests = [], [], []
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and "vendor" not in m.text and "404" not in m.text and self.errors.append(m.text))
        def on_request(r):
            self.requests.append(r.url.replace(self.url, "/"))
            if r.url.endswith("/api/speak"):
                self.speak.append(json.loads(r.post_data))
        page.on("request", on_request)
        return ctx, page

    def state(self, page):
        return page.inner_text("#state")

    def logs(self, page):
        return page.inner_text("#log")

    def test_whole_journey(self):
        ctx, page = self.new_page()
        # ---- first run: setup is still going, the page says so
        page.goto(self.url)
        page.wait_for_function("document.querySelector('#state').textContent.startsWith('Setting up')", timeout=5000)
        self.assertTrue(page.is_disabled("#call"))
        # ---- setup finishes: the page notices by itself
        threading.Thread(target=c.run_setup, daemon=True).start()
        page.wait_for_function("document.querySelector('#log').textContent.includes('Ready. Chat model: qwen2.5:1.5b')", timeout=15000)
        self.assertIn("vision: moondream", self.logs(page))
        self.assertEqual(page.eval_on_selector("#vo", "e => [...e.options].map(o => o.value).slice(0, 2)"), ["k:af_heart", "k:af_sarah"])
        self.assertEqual(page.eval_on_selector("#m", "e => e.value"), "qwen2.5:1.5b")
        # ---- add a face
        page.set_input_files("#file", self.photo)
        page.wait_for_function("!document.querySelector('#call').disabled", timeout=15000)
        self.assertIn("Face ready.", self.logs(page)); self.assertEqual(self.state(page), "Ready, tap call")
        self.assertTrue(page.eval_on_selector("#dv", "v => v.classList.contains('on') && v.loop && v.muted"), "idle loop is playing")
        # ---- call: rings, then greets with a lip-synced clip
        page.click("#call")
        self.assertEqual(self.state(page), "Calling…")
        page.wait_for_function("document.body.classList.contains('incall')", timeout=8000)
        deadline = time.time() + 10
        while not self.speak and time.time() < deadline: time.sleep(0.1)
        self.assertEqual(self.speak[0]["text"], "Hi! It's so good to see you. What's on your mind?"); self.assertEqual(self.speak[0]["voice"], "af_heart")
        page.wait_for_function("/connected|speaking|listening/.test(document.querySelector('#state').textContent)", timeout=5000)
        self.assertRegex(self.state(page), r"· \d+:\d\d$")
        # ---- type a question: reply streams in and is spoken sentence by sentence
        page.click("#chat"); page.fill("#in", "How are you?"); page.press("#in", "Enter")
        page.wait_for_function("document.querySelector('#log').textContent.includes('This is a test reply!')", timeout=8000)
        deadline = time.time() + 10
        while len(self.speak) < 3 and time.time() < deadline: time.sleep(0.1)
        self.assertEqual([s["text"] for s in self.speak[1:3]], ["Hello there.", "This is a test reply!"])
        sent = self.w.ollama.chats[-1]
        self.assertTrue(sent["stream"]); self.assertEqual(sent["messages"][-1], {"role": "user", "content": "How are you?"})
        self.assertIn("one or two short spoken sentences", sent["messages"][0]["content"])
        # ---- conversation memory
        page.fill("#in", "And now?"); page.press("#in", "Enter")
        deadline = time.time() + 10
        while len(self.w.ollama.chats) < 2 and time.time() < deadline: time.sleep(0.1)
        roles = [m["role"] for m in self.w.ollama.chats[-1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(self.w.ollama.chats[-1]["messages"][2]["content"], "Hello there. This is a test reply!")
        # ---- camera + "what do you see?" uses the vision model with a frame
        page.wait_for_function("document.querySelector('#state').textContent.startsWith('connected')", timeout=15000)
        page.click("#camb")
        page.wait_for_function("!document.querySelector('#look').hidden", timeout=8000)
        before = len(self.w.ollama.chats)
        page.click("#look")
        deadline = time.time() + 10
        while len(self.w.ollama.chats) == before and time.time() < deadline: time.sleep(0.1)
        vis = self.w.ollama.chats[-1]
        self.assertEqual((vis["model"], vis["stream"]), ("moondream:latest" if "moondream:latest" in self.w.ollama.models else "moondream", False))
        self.assertGreater(len(vis["messages"][0]["images"][0]), 200, "a real camera frame was sent")
        page.wait_for_function("document.querySelector('#log').textContent.includes('I see a cup')", timeout=8000)
        # ---- hang up
        page.click("#hang")
        self.assertEqual(self.state(page), "Ready, tap call")
        self.assertFalse(page.eval_on_selector("body", "b => b.classList.contains('incall') || b.classList.contains('calling')"))
        # ---- reload: the remembered photo is used again, nothing to pick
        page.reload()
        page.wait_for_function("!document.querySelector('#call').disabled", timeout=15000)
        self.assertTrue(page.is_hidden("#drop"))
        self.assertEqual(self.errors, [], "no JavaScript errors")
        ctx.close()

    def test_works_via_the_localhost_name_too(self):
        ctx, page = self.new_page()
        page.goto(self.url.replace("127.0.0.1", "localhost"))
        page.wait_for_function("document.querySelector('#state').textContent.length > 0")
        status = page.evaluate("fetch('/api/status').then(r => r.status)")
        self.assertEqual(status, 200)
        ctx.close()


if __name__ == "__main__":
    unittest.main()
