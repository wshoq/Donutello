# --- Base: PyTorch z CUDA 12.8 + cuDNN9 (torch już jest) ---
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# --- Systemowe pakiety ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates build-essential \
    libgl1 libglib2.0-0 ffmpeg libsndfile1 \
    python3-dev python3-distutils \
 && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip ---
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Hugging Face i inne ---
RUN pip install --no-cache-dir \
    transformers==4.55.4 \
    datasets==3.0.1 \
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

# --- Git LFS ---
RUN apt-get update && apt-get install -y git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY train.py /workspace/

CMD ["/bin/bash"]
