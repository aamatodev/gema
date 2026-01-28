# ───────────────────────── Base image ─────────────────────────
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
LABEL maintainer="Alessandro Amato"

RUN apt-get update && \
    apt-get install -y \
        git python3-pip python3-opencv python3-opengl \
        libsm6 libxext6 xvfb vim unzip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /workspace
VOLUME ["/workspace/output"]

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        torch-cluster -f https://pytorch-geometric.com/whl/torch-2.5.1+cu124.html \
        torch_geometric \
        cmake

COPY . .

RUN pip install -r ./external/epymarl/requirements.txt
