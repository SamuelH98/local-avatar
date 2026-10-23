# Local Companion: ONE image holding the web page, the face (Ditto), the voice (Kokoro) and the server that ties them together.
# GPU build for NVIDIA (RTX 40xx laptop OK). Model weights are NOT baked in: they download on first run into ./checkpoints (mounted).
# The heavy layers below are unchanged from earlier versions on purpose, so Docker's build cache keeps working. Only the last layers are new.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ffmpeg libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip setuptools wheel
RUN pip3 install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install librosa tqdm filetype imageio imageio-ffmpeg opencv-python-headless scikit-image cython cuda-python colored polygraphy \
      huggingface_hub numpy==2.0.1
# TensorRT 8.6.1 from NVIDIA's apt repo (already configured in this base image). The pip "tensorrt" 8.6.1 package cannot be built in Docker
# and its wheels alone provide no "tensorrt" module. The exact build string is looked up, not hard-coded.
RUN set -e; apt-get update; \
    V=$(apt-cache madison libnvinfer8 | awk '{print $3}' | grep '^8\.6\.1' | head -1); \
    if [ -z "$V" ]; then echo "No TensorRT 8.6.1 in apt. Available:"; apt-cache madison libnvinfer8; exit 1; fi; \
    echo "Installing TensorRT $V"; \
    apt-get install -y --no-install-recommends libnvinfer8=$V libnvinfer-plugin8=$V libnvonnxparsers8=$V python3-libnvinfer=$V; \
    rm -rf /var/lib/apt/lists/*
# Ditto does "from cuda import cuda": that API was removed in cuda-python 13, so pin the 12.x line (own layer: no heavy rebuild).
RUN pip3 install "cuda-python==12.6.0"
# (no import test here: libcuda only exists at runtime, not during build)
RUN pip3 show cuda-python | grep -i '^version'
# Neural voice (Kokoro, CPU onnx so it does not take VRAM from Ditto). numpy is re-pinned after, as Ditto expects.
RUN pip3 install kokoro-onnx soundfile && pip3 install numpy==2.0.1
RUN python3 -c "import kokoro_onnx; print('kokoro-onnx OK')"
RUN python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
RUN git clone --depth 1 https://github.com/antgroup/ditto-talkinghead /ditto/ditto-talkinghead
WORKDIR /ditto/ditto-talkinghead
# Speech recognition for browsers that have none (Firefox): faster-whisper on the CPU (int8), so it takes no VRAM from the face.
# numpy is re-pinned after, as Ditto expects. The model itself (about 150 MB) downloads on first run into ./checkpoints/whisper.
RUN pip3 install faster-whisper && pip3 install numpy==2.0.1
RUN python3 -c "import faster_whisper, numpy; print('faster-whisper OK, numpy', numpy.__version__)"
# --- new in v14 (small layers, cheap to rebuild) ---
# Browser-side camera/face-check files (MediaPipe) baked in, so the page works with nothing to download at run time.
COPY docker/fetch_assets.py /tmp/fetch_assets.py
RUN python3 /tmp/fetch_assets.py /app/vendor && rm /tmp/fetch_assets.py
# The app itself (last layer: editing it never rebuilds anything above).
COPY app/ /app/
EXPOSE 8790
CMD ["python3", "/app/companion.py"]
