# ----------------------------
# Dockerfile dla Donut Training
# ----------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- Podstawowe narzędzia i Python ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget unzip curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY train.py /workspace/

# --- PyTorch + CUDA 12.8 ---
RUN pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchvision==0.23.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# --- Hugging Face: transformers + tokenizers ---
RUN pip install --no-cache-dir \
    transformers>=4.57.0 \
    tokenizers<=0.23.0

# --- Pozostałe zależności ---
RUN pip install --no-cache-dir \
    datasets>=2.14.0,<3.0.0 \
    protobuf==4.24.3 \
    accelerate==0.34.2 \
    peft \
    sentencepiece \
    Pillow \
    tqdm \
    evaluate \
    jsonlines \
    scikit-learn \
    nltk \
    opencv-python-headless

# --- Domyślny CMD, kontener nie odpala treningu od razu ---
CMD ["sleep", "infinity"]
