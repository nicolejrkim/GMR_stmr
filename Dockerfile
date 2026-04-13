# GPU-friendly runtime image for GMR (MuJoCo + Python 3.10)
# Build: docker build -t gmr:latest .
# Run (GPU): docker run --rm -it --gpus all -v $(pwd):/workspace/GMR gmr:latest
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

# System dependencies for Python, MuJoCo rendering, OpenCV, and video output.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libegl1 \
    libglfw3 \
    libosmesa6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/GMR

# Copy project and install in editable mode.
COPY . /workspace/GMR
RUN pip install -e .

CMD ["bash"]
