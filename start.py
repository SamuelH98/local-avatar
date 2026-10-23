#!/usr/bin/env python3
"""
Local Companion launcher. The only thing you need to run:

    python3 start.py              start the app (or check on it) and open it in your browser
    python3 start.py --stop       stop it
    python3 start.py --logs       show what it has been doing (for troubleshooting)
    python3 start.py --rebuild    rebuild the Docker image from scratch (rarely needed)
    python3 start.py --no-browser don't open the browser

Needs only Python 3 and Docker (with GPU support). First run: builds the image (10-20 min), then the app downloads its
models by itself (a few GB) and shows progress here and in the web page. Every later start takes a few seconds.
Settings live in .env, which this script creates. It never touches containers, ports or files that are not its own.
"""
import argparse, json, os, re, shutil, socket, subprocess, sys, time, urllib.request, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")
CONTAINER, IMAGE, PROFILE = "companion", "companion:latest", "bundled-ollama"
BUNDLED_OLLAMA = "http://ollama:11434"
ENV_TEMPLATE = """# Settings for Local Companion. Edit, then run start.py again.
# Port of the web page. start.py picks a free one if this one is taken, and writes it back here.
APP_PORT=8080
# Chat model, and vision model (the "what do you see" feature). Leave VISION_MODEL empty to switch vision off.
CHAT_MODEL=qwen2.5:1.5b
VISION_MODEL=moondream
# Chat engine: by default one that comes with this package runs in Docker.
# To use an Ollama already installed on this computer instead, remove the # below
# (that Ollama must be started with OLLAMA_HOST=0.0.0.0 so Docker can reach it):
# OLLAMA_URL=http://host.docker.internal:11434
"""


def say(msg=""):
    print(msg, flush=True)


def run(cmd, **kw):
    """Runs a command and captures its output. A missing program is reported as exit code 127, not a crash."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", "")


# ----------------------------------------------------------------------------------------- Docker
def find_compose():
    """Returns (compose command, None) or (None, what is wrong in plain words)."""
    if shutil.which("docker") is None:
        return None, ("Docker is not installed.\n  Windows: install Docker Desktop (WSL2 backend, GPU support on).\n"
                      "  Linux: install Docker Engine and the NVIDIA Container Toolkit.\n  Then run this again.")
    if run(["docker", "info"]).returncode != 0:
        return None, "Docker is installed but not running. Start Docker Desktop (or the docker service), wait until it says it is running, then run this again."
    if run(["docker", "compose", "version"]).returncode == 0:
        return ["docker", "compose"], None
    if shutil.which("docker-compose"):
        return ["docker-compose"], None
    return None, "Docker Compose is missing. Install Docker Compose v2 (it comes with current Docker Desktop)."


def container_state(name):
    return run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.State}}"]).stdout.strip()


def running_port():
    """The port our container is already published on, or None."""
    out = run(["docker", "port", CONTAINER, "8790/tcp"]).stdout.strip().splitlines()
    m = re.search(r":(\d+)\s*$", out[0]) if out else None
    return int(m.group(1)) if m else None


def image_exists():
    return bool(run(["docker", "images", "-q", IMAGE]).stdout.strip())


# ----------------------------------------------------------------------------------------- settings (.env)
def read_env():
    values = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip()
    return values


def set_env(key, value):
    lines = (open(ENV_FILE, encoding="utf-8-sig").read() if os.path.exists(ENV_FILE) else ENV_TEMPLATE).splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"^\s*{key}\s*=", line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(.4)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket() as s:                     # also catches ports reserved by the system
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def choose_port(env):
    wanted = int(env["APP_PORT"]) if env.get("APP_PORT", "").isdigit() else 8080
    if running_port() == wanted:
        return wanted                               # our own container already has it
    port = next((p for p in range(wanted, wanted + 20) if not port_busy(p)), None)
    if port is not None and port != wanted:
        say(f"  Port {wanted} is used by another program, using {port} instead.")
    return port


# ----------------------------------------------------------------------------------------- build / start / wait
def compose_args(compose, bundled):
    return compose + ["--project-directory", HERE] + (["--profile", PROFILE] if bundled else [])


def build_image(compose, bundled, rebuild):
    """Runs `docker compose build` and shows one calm progress line instead of hundreds of lines."""
    cmd = compose_args(compose, bundled) + ["build"] + (["--no-cache"] if rebuild else [])
    env = dict(os.environ, BUILDKIT_PROGRESS="plain", COMPOSE_PROGRESS="plain", DOCKER_BUILDKIT="1")
    log_path = os.path.join(HERE, "build.log")
    started, step, tail = time.time(), "", []
    tty = sys.stdout.isatty()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env)
        for line in proc.stdout:
            log.write(line)
            tail = (tail + [line.rstrip()])[-25:]
            m = re.match(r"^#\d+ \[(?:[\w.-]+ )?\s*(\d+)/(\d+)\]", line)
            if m and m.group(0) != step:
                step = m.group(0)
                if not tty:
                    say(f"  building: step {m.group(1)} of {m.group(2)}")
            if tty and step:
                n, total = re.findall(r"(\d+)/(\d+)", step)[0]
                secs = int(time.time() - started)
                print(f"\r  building the image: step {n} of {total}  ({secs // 60}:{secs % 60:02d})   ", end="", flush=True)
        ok = proc.wait() == 0
    if tty and step:
        print()
    if not ok:
        say("\n  The build failed. Last lines:")
        for line in tail:
            say("    " + line[:180])
        say(f"  (full log: {log_path})")
    return ok


def start_containers(compose, bundled):
    r = run(compose_args(compose, bundled) + ["up", "-d", "--remove-orphans"])
    if r.returncode == 0:
        return True
    text = (r.stdout + r.stderr).strip()
    say("\n  Could not start the containers:\n    " + "\n    ".join(text.splitlines()[-8:]))
    if re.search(r"nvidia|gpu|device driver", text, re.I):
        say("\n  This usually means Docker cannot use your NVIDIA GPU.\n  Windows: Docker Desktop, WSL2 backend, up-to-date NVIDIA driver.\n"
            "  Linux: install the NVIDIA Container Toolkit, then 'sudo systemctl restart docker'.")
    return False


def wait_until_ready(url):
    """Shows what the app is doing (first run: downloads). Returns True when ready."""
    stage, inline, offline_since = None, False, time.time()
    while True:
        try:
            with urllib.request.urlopen(url + "/api/status", timeout=3) as r:
                s = json.load(r)
            offline_since = None
        except Exception:
            offline_since = offline_since or time.time()
            if time.time() - offline_since > 120:
                say("\n  The app is not answering. See what went wrong with:  python3 start.py --logs")
                return False
            time.sleep(1.5)
            continue
        if s.get("ready"):
            return True
        if s.get("error"):
            if inline:
                print()
            say(f"\n  Setup hit a problem:\n    {s['error']}\n  It keeps retrying by itself. Fix the cause, then run start.py again.  Details: python3 start.py --logs")
            return False
        if s["stage"] != stage:
            if inline:
                print()
            stage, inline = s["stage"], False
            say(f"  - {stage}")
        if s.get("detail") and sys.stdout.isatty():
            print(f"\r      {s['detail']}".ljust(70), end="", flush=True)
            inline = True
        time.sleep(1)


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Start the Local Companion.")
    ap.add_argument("--stop", action="store_true", help="stop the app")
    ap.add_argument("--logs", action="store_true", help="show recent log lines")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the image from scratch")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")    # old Windows consoles cannot print every character
    except Exception:
        pass
    if sys.version_info < (3, 8):
        say("Python 3.8 or newer is needed.")
        return 1

    compose, problem = find_compose()
    if not compose:
        say(f"\n  Problem: {problem}")
        return 1
    if not os.path.exists(ENV_FILE):
        set_env("APP_PORT", read_env().get("APP_PORT", "8080"))     # creates .env from the template
    env = read_env()
    bundled = env.get("OLLAMA_URL", BUNDLED_OLLAMA).rstrip("/") == BUNDLED_OLLAMA

    if args.stop:
        say("Stopping...")
        run(compose_args(compose, True) + ["down"])
        say("Stopped. Your models and settings are kept. Start again with: python3 start.py")
        return 0
    if args.logs:
        out = run(compose_args(compose, bundled) + ["logs", "--tail", "100", "--no-color"])
        say((out.stdout + out.stderr).strip() or "No logs yet. Is it running? Try: python3 start.py")
        return 0

    say("Local Companion")
    if container_state("companion-ditto"):
        say("  Found the container from the previous version; it is replaced automatically (your downloaded models are kept).")
    first_build = args.rebuild or not image_exists()
    if first_build:
        free = shutil.disk_usage(HERE).free / 2**30
        say("  First run: building the image takes 10-20 minutes, then the models download (a few GB). Later starts take seconds.")
        if free < 25:
            say(f"  Warning: only {free:.0f} GB free on this drive; about 25 GB are needed.")
    port = choose_port(env)
    if port is None:
        say("\n  Problem: no free port found near 8080. Set APP_PORT in .env to a free port and run again.")
        return 1
    if env.get("APP_PORT") != str(port):
        set_env("APP_PORT", port)

    if not build_image(compose, bundled, args.rebuild) or not start_containers(compose, bundled):
        return 1
    url = f"http://127.0.0.1:{port}"
    say(f"  Starting... {url}")
    try:
        if not wait_until_ready(url):
            return 1
    except KeyboardInterrupt:
        say(f"\n  Stopped waiting. The app keeps running in the background; run start.py again to check on it.\n  Address: {url}")
        return 0
    say(f"\n  Ready:  {url}\n  (You can close this window. Stop the app with: python3 start.py --stop)")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
