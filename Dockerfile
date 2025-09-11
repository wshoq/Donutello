FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# --- Podstawowe narzędzia ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git wget unzip curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
COPY train.py /workspace/
COPY requirements.txt /workspace/

# --- Instalacja wszystkiego z requirements ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Domyślny CMD ---
CMD ["sleep", "infinity"]
