FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    TORCH_HOME=/cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/emb2face

COPY requirements.txt pyproject.toml ./

# Install a CUDA-enabled PyTorch stack first, then the rest of the pinned deps.
RUN python3 -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install -r requirements.txt

COPY . .

RUN python3 -m pip install -e .

CMD ["python3", "-m", "emb2face", "infer", "--help"]
