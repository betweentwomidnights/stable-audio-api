# API Dockerfile - Clean version that avoids blinker conflicts
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm
ENV HF_HOME=/app/.cache/huggingface
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Remove problematic system packages that conflict with pip
RUN apt-get remove -y python3-blinker || true

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache

# Upgrade pip and use virtual environment to avoid system conflicts
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5+ with CUDA support
RUN pip install torch>=2.5.0 torchaudio>=2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install ML dependencies
RUN pip install \
    einops \
    transformers \
    accelerate \
    diffusers \
    scipy \
    librosa \
    soundfile \
    huggingface-hub

# Install web dependencies
RUN pip install flask flask-cors

# Clone and install stable-audio-tools
RUN git clone https://github.com/Stability-AI/stable-audio-tools.git /tmp/stable-audio-tools
RUN cd /tmp/stable-audio-tools && pip install .

# Copy API script
COPY api.py /app/
COPY riff_manager.py /app/

COPY riffs/ /app/riffs/

# Create output directory (for debugging/testing)
RUN mkdir -p /app/outputs

# Expose Flask port
EXPOSE 8005

# Run the API
CMD ["python", "api.py"]