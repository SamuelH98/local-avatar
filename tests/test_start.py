"""start.py tests with a fake `docker` on PATH (no Docker needed). Run: python3 -m unittest tests.test_start -v"""
import os, shutil, socket, stat, subprocess, sys, tempfile, threading, time, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakeworld import World, ROOT
import companion as c

FAKE_DOCKER = r'''#!%(py)s
import os, sys
S = os.environ["FAKE_STATE"]; a = sys.argv[1:]
open(S + "/calls", "a").write(" ".join(a) + "\n")
def has(f): return os.path.exists(S + "/" + f)
if os.environ.get("FAKE_DOCKER_DOWN") and a[:1] == ["info"]: sys.exit(1)
if a[:1] == ["info"] or a[:2] == ["compose", "version"]: sys.exit(0)
if a[:1] == ["ps"]:
    name = [x for x in a if x.startswith("name=")][0][6:-1]
    if name == "companion-ditto" and has("old"): print("running")
    sys.exit(0)
if a[:1] == ["port"]:
    if has("up"): print("127.0.0.1:" + open(S + "/port").read())
    sys.exit(0 if has("up") else 1)
if a[:1] == ["images"]:
    if has("built"): print("abc123")
    sys.exit(0)
if a[:1] == ["compose"]:
    if "build" in a:
        assert os.environ.get("BUILDKIT_PROGRESS") == "plain"
        for i in (1, 2, 3): print("#%%d [ %%d/3] RUN something" %% (i + 4, i), flush=True)
        if os.environ.get("FAKE_BUILD_FAILS"): print("ERROR: boom"); sys.exit(1)
        open(S + "/built", "w").write("1")
    elif "up" in a:
        for l in open(os.environ["FAKE_ENV_FILE"]):
            if l.startswith("APP_PORT="): open(S + "/port", "w").write(l.split("=")[1].strip())
        open(S + "/up", "w").write("1")
    elif "down" in a:
        if has("up"): os.remove(S + "/up")
    elif "logs" in a: print("app  | hello from the log")
    sys.exit(0)
'''


class StartTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="start_")
        self.proj, self.bin, self.state = (os.path.join(self.tmp, d) for d in ("proj", "bin", "state"))
        for d in (self.proj, self.bin, self.state):
            os.makedirs(d)
        for f in ("start.py", "docker-compose.yml"):
            shutil.copy(os.path.join(ROOT, f), self.proj)
        path = os.path.join(self.bin, "docker")
        with open(path, "w") as f:
            f.write(FAKE_DOCKER % {"py": sys.executable})
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        self.env = dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"], FAKE_STATE=self.state,
                        FAKE_ENV_FILE=os.path.join(self.proj, ".env"))
        self.world = World().apply(c)
        self.stop = threading.Event(); self.srv = None; self.thread = None

    def tearDown(self):
        self.stop.set()
        c.SETUP_STOP.set()
        if self.thread: self.thread.join(5)
        if self.srv: self.srv.shutdown(); self.srv.server_close()
        self.world.close(); shutil.rmtree(self.tmp, ignore_errors=True)

    def app_comes_up(self, ready=True, error=None):
        """Plays the container: once fake `up` ran, serve the real server on the port start.py chose."""
        def go():
            while not self.stop.is_set() and not os.path.exists(self.state + "/up"):
                time.sleep(0.05)
            if self.stop.is_set(): return
            port = int(open(self.state + "/port").read())
            from http.server import ThreadingHTTPServer
            self.srv = ThreadingHTTPServer(("127.0.0.1", port), c.Handler); self.srv.daemon_threads = True
            threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
            c.STATUS.update(stage="Downloading the voice (first run only)", detail="5 / 100 MB")
            time.sleep(3.5)
            c.STATUS.update(stage="Ready", detail="", ready=ready, error=error) if ready or error else None
        self.thread = threading.Thread(target=go, daemon=True); self.thread.start()

    def start(self, *args, **env):
        return subprocess.run([sys.executable, os.path.join(self.proj, "start.py"), "--no-browser", *args], capture_output=True, text=True,
                              env={**self.env, **env}, timeout=60, cwd=self.proj)

    def calls(self):
        return open(self.state + "/calls").read().splitlines() if os.path.exists(self.state + "/calls") else []

    def envfile(self):
        return dict(l.split("=", 1) for l in open(os.path.join(self.proj, ".env")).read().splitlines() if "=" in l and not l.startswith("#"))

    def test_first_run_builds_starts_and_waits_for_setup(self):
        self.app_comes_up()
        r = self.start()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for part in ("First run", "building: step 3 of 3", "Downloading the voice", "Ready:  http://127.0.0.1:"):
            self.assertIn(part, r.stdout)
        up = [x for x in self.calls() if " up " in x + " "][0]
        self.assertIn("--profile bundled-ollama", up); self.assertIn("--remove-orphans", up); self.assertIn("-d", up)
        self.assertEqual(self.envfile()["VISION_MODEL"], "moondream"); self.assertTrue(self.envfile()["APP_PORT"].isdigit())

    def test_second_start_skips_first_run_message(self):
        open(self.state + "/built", "w").write("1")
        self.app_comes_up()
        r = self.start()
        self.assertEqual(r.returncode, 0); self.assertNotIn("First run", r.stdout)

    def test_busy_port_is_avoided_and_remembered(self):
        with socket.socket() as hog:
            hog.bind(("127.0.0.1", 0)); hog.listen(); busy = hog.getsockname()[1]
            with open(os.path.join(self.proj, ".env"), "w") as f: f.write(f"APP_PORT={busy}\n")
            self.app_comes_up()
            r = self.start()
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("is used by another program", r.stdout); self.assertNotEqual(int(self.envfile()["APP_PORT"]), busy)

    def test_own_host_ollama_means_no_bundled_container(self):
        with open(os.path.join(self.proj, ".env"), "w") as f: f.write("OLLAMA_URL=http://host.docker.internal:11434\n")
        self.app_comes_up()
        r = self.start()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse([x for x in self.calls() if "bundled-ollama" in x])

    def test_old_v13_container_is_announced(self):
        open(self.state + "/old", "w").write("1"); self.app_comes_up()
        self.assertIn("previous version", self.start().stdout)

    def test_docker_not_running_is_explained(self):
        r = self.start(FAKE_DOCKER_DOWN="1")
        self.assertEqual(r.returncode, 1); self.assertIn("not running", r.stdout); self.assertEqual(self.calls(), ["info"])

    def test_failed_build_shows_the_reason(self):
        r = self.start(FAKE_BUILD_FAILS="1")
        self.assertEqual(r.returncode, 1); self.assertIn("build failed", r.stdout); self.assertIn("boom", r.stdout)
        self.assertFalse([x for x in self.calls() if " up " in x + " "])

    def test_setup_problem_is_reported_and_exit_code_is_nonzero(self):
        self.app_comes_up(ready=False, error="The container cannot see your NVIDIA GPU.")
        r = self.start()
        self.assertEqual(r.returncode, 1); self.assertIn("cannot see your NVIDIA GPU", r.stdout)

    def test_stop_and_logs(self):
        self.assertEqual(self.start("--stop").returncode, 0); self.assertTrue([x for x in self.calls() if " down" in x])
        self.assertIn("hello from the log", self.start("--logs").stdout)


if __name__ == "__main__":
    unittest.main()
