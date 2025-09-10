# --- Bazowy obraz: PyTorch z CUDA 12.8 i cuDNN9 ---
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# --- ENV ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
WORKDIR /workspace

# --- Systemowe zależności ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates build-essential \
    libgl1 libglib2.0-0 ffmpeg libsndfile1 \
    openssh-server procps python3-dev python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# --- pip upgrade ---
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Pythonowe paczki (bez torch – jest w base image!) ---
RUN pip install --no-cache-dir \
    "transformers>=4.34.0" \
    datasets \
    accelerate \
    peft \
    sentencepiece \
    Pillow \
    tqdm \
    evaluate \
    jsonlines \
    opencv-python-headless

# --- Git LFS (modele na HF) ---
RUN apt-get update && apt-get install -y git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

# --- User (opcjonalnie) ---
ARG USERNAME=runner
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} || true && \
    mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

# --- Expose SSH (jeśli chcesz wchodzić do poda) ---
EXPOSE 22

# --- Default ---
CMD ["/bin/bash"]
