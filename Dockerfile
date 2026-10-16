# glassfix container: runs in the client's pod (Kubernetes, RunPod, any Docker host).
#
# CPU image (default):
#   docker build -t glassfix:cpu .
# NVIDIA GPU image (host needs the NVIDIA driver + NVIDIA Container Toolkit):
#   docker build -t glassfix:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu132 .
#
# Run on a folder of photos (<input>/<car_id>/*.jpg) -> cut-out PNGs in <output>/<run-name>/<car_id>/:
#   docker run --rm -v /path/in:/data/input -v /path/out:/data/output glassfix:cpu \
#       run --input /data/input --output /data/output --run-name job123
#   (GPU: add --gpus all)

FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG PREFETCH_WEIGHTS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # CPU threads for torch; raise on a big pod (more threads = more scratch memory).
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url ${TORCH_INDEX}
COPY requirements-pod.txt .
RUN pip install -r requirements-pod.txt

COPY glassfix/ glassfix/
COPY config.yaml LICENSES.md README.md ./

# Bake the model weights (~1.6 GB) into the image so pods start without downloading.
# Build with --build-arg PREFETCH_WEIGHTS=0 to download on first run instead (mount /app/models as a volume).
RUN if [ "$PREFETCH_WEIGHTS" = "1" ]; then python -m glassfix doctor --download; fi

# Run as a non-root user.
RUN useradd --create-home --uid 10001 glassfix && chown -R glassfix /app
USER glassfix

ENTRYPOINT ["python", "-m", "glassfix"]
CMD ["run", "--input", "/data/input", "--output", "/data/output", "--run-name", "job"]
