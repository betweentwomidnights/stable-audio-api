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
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
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
RUN git clone https://github.com/sskalnik/stable-audio-tools.git /tmp/stable-audio-tools
RUN cd /tmp/stable-audio-tools && pip install .

# NEW (PyTorch 2.6 - required for sskalnik's stable-audio-tools fork)
RUN pip install torch>=2.6.0 torchaudio>=2.6.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torchcodec

# Copy API script
COPY api.py /app/
COPY riff_manager.py /app/
COPY model_loader_enhanced.py /app/
COPY base_model_config.json /app/ 

COPY riffs/ /app/riffs/

COPY patch_generation.py /app/
RUN python /app/patch_generation.py

# Create output directory (for debugging/testing)
RUN mkdir -p /app/outputs

# Expose Flask port
EXPOSE 8005

# Run the API
CMD ["python", "api.py"]