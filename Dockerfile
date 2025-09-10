FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common build-essential wget curl git ca-certificates unzip && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# instalacja pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py && \
    python3.11 -m pip install --upgrade pip

# instalacja PyTorch z CUDA 12.8 i Transformers
RUN python3.11 -m pip install \
      torch==2.7.0+cu128 torchvision==0.18.0+cu128 torchaudio==2.7.0+cu128 \
      -f https://download.pytorch.org/whl/torch_stable.html && \
    python3.11 -m pip install transformers==4.34.0 datasets accelerate peft pillow tqdm evaluate

# ustawiamy python3 jako 3.11 i przygotowujemy katalog roboczy
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN mkdir /workspace
WORKDIR /workspace

# polecenie domyślne (podtrzymanie kontenera w RunPod)
CMD ["sleep", "infinity"]
