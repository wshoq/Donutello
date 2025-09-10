# Bazowy obraz: NVIDIA CUDA 12.8 z cuDNN na Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Instalacja podstawowych narzędzi i Python 3.11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential wget curl git ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalacja pip dla Pythona 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Instalacja PyTorch z CUDA 12.8 i Transformers
ENV TORCH_VERSION=2.7.0
ENV CUDA_VERSION=12.8
RUN python3.11 -m pip install \
        torch==${TORCH_VERSION}+cu128 torchvision==0.18.0+cu128 torchaudio==2.7.0+cu128 \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && python3.11 -m pip install \
        transformers==4.34.0 datasets accelerate peft pillow tqdm evaluate \
        timm==0.5.4 pytorch-lightning==1.6.4

# Klonowanie i instalacja biblioteki Donut
RUN git clone https://github.com/clovaai/donut.git /opt/donut \
    && python3.11 -m pip install -r /opt/donut/requirements.txt \
    && python3.11 -m pip install /opt/donut

# Ustawienie Pythona 3.11 jako domyślnego, przygotowanie workspace
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN mkdir /workspace
WORKDIR /workspace

# Domyślna komenda (opcjonalnie można zmienić na np. uruchomienie skryptu treningowego)
CMD ["sleep", "infinity"]
