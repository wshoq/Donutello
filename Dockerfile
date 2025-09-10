# --- Base: official PyTorch image (contains torch preinstalled with CUDA) ---
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# --- Metadata / env ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
WORKDIR /workspace

# --- System deps ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git wget curl unzip ca-certificates build-essential openssh-server procps python3-dev python3-distutils \
      && rm -rf /var/lib/apt/lists/*

# Ensure pip is up-to-date and use python3 from base image
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Python packages (transformers/datasets/accelerate/peft etc.) ---
# Keep torch out of this list because it's already in base image.
RUN python3 -m pip install --no-cache-dir \
      "transformers>=4.34.0" \
      "datasets" \
      "accelerate" \
      "peft" \
      "sentencepiece" \
      "Pillow" \
      "tqdm" \
      "evaluate" \
      "jsonlines" \
      "opencv-python-headless" \
      "pyyaml"

# (Optional) install git-lfs if you plan to pull LFS models
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs && git-lfs install || true && rm -rf /var/lib/apt/lists/*

# Create user for safety (optional)
ARG USERNAME=runner
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} || true && \
    mkdir -p /workspace && chown -R ${USERNAME}:${USERNAME} /workspace

# Copy training scripts (optionally - you can mount them as volumes)
# If you want to embed train.py into image, uncomment the COPY lines
# COPY train.py /workspace/train.py
# COPY src/ /workspace/src/

# Expose SSH (optional) and default to bash
EXPOSE 22
CMD ["/bin/bash"]
