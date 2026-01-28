# ───────────────────────── Base image ─────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
LABEL maintainer="Alessandro Amato"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y \
        git python3-pip python3-opencv python3-opengl \
        libsm6 libxext6 xvfb vim unzip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /workspace
VOLUME ["/workspace/output"]

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir \
        torch-cluster -f https://pytorch-geometric.com/whl/torch-2.8.0+cu128.html

COPY . .
