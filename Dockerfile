FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- Podstawowe narzędzia ---
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

# --- Podstawowe paczki Hugging Face (kompatybilne wersje) ---
RUN pip install --no-cache-dir \
    transformers==4.25.1 \
    tokenizers==0.22.0 \
    datasets==2.21.0 \
    protobuf==4.24.3 \
    accelerate==0.34.2 \
    sentencepiece \
    Pillow

# --- Pozostałe paczki (mniejsze ryzyko konfliktów) ---
RUN pip install --no-cache-dir \
    peft \
    tqdm \
    evaluate \
    jsonlines \
    scikit-learn \
    nltk \
    opencv-python-headless

# --- Domyślny CMD ---
CMD ["sleep", "infinity"]
