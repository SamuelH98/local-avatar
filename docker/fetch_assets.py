#!/usr/bin/env python3
"""Runs during `docker build`: puts the browser-side camera/face-check files (MediaPipe) inside the image,
so nothing needs downloading when the app runs and the page works fully offline.
Usage: fetch_assets.py [destination]   (default /app/vendor)"""
import io, os, sys, tarfile, time, urllib.request

DEST = sys.argv[1] if len(sys.argv) > 1 else "/app/vendor"
MP_VERSION = "0.10.14"
GS = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "face_landmarker.task": f"{GS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task": f"{GS}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}
RUNTIME = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{MP_VERSION}.tgz"


def fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {url} ({e})", flush=True)
            time.sleep(2 * (i + 1))


def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main(models=MODELS):
    for name, url in models.items():
        data = fetch(url)
        if len(data) < 1e6:
            sys.exit(f"{name} looks incomplete ({len(data)} bytes)")
        save(os.path.join(DEST, name), data)
        print(f"  {name}: {len(data) >> 10} KB", flush=True)
    with tarfile.open(fileobj=io.BytesIO(fetch(RUNTIME)), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name == "package/vision_bundle.mjs":
                save(os.path.join(DEST, "mediapipe", "vision_bundle.mjs"), tar.extractfile(m).read())
            elif m.isfile() and m.name.startswith("package/wasm/"):
                save(os.path.join(DEST, "mediapipe", "wasm", os.path.basename(m.name)), tar.extractfile(m).read())
    need = [os.path.join(DEST, "mediapipe", "vision_bundle.mjs")]
    wasm = os.path.join(DEST, "mediapipe", "wasm")
    if not all(os.path.exists(p) for p in need) or not os.path.isdir(wasm) or not os.listdir(wasm):
        sys.exit("MediaPipe runtime files missing after download")
    print(f"  mediapipe runtime v{MP_VERSION}: {len(os.listdir(wasm))} wasm files", flush=True)


if __name__ == "__main__":
    main()
