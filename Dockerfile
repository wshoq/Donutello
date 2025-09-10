# Dockerfile dla treningu Donut - naver-clova-ix/donut-base-finetuned-cord-v2

FROM nvidia/cuda:12.8.0-cudnn8-runtime-ubuntu22.04

# Zmienne środowiskowe
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# Systemowe zależności
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget curl unzip build-essential python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie pip
RUN python3.11 -m ensurepip
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Instalacja PyTorch CUDA 12.8
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.7.0+cu128 torchvision==0.18.0+cu128 to
