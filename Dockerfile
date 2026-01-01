FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install system dependencies
# git for installing python packages from git
# espeak-ng for MeloTTS/Piper
# ffmpeg for audio processing
# cmake/build-essential for compiling some python extensions
# libsndfile1 for soundfile
RUN apt-get update && apt-get install -y \
    git \
    espeak-ng \
    ffmpeg \
    cmake \
    build-essential \
    curl \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set HF_HOME early so it persists for both build and runtime
ENV HF_HOME=/app/models/huggingface

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache "numpy<2" setuptools wheel "Cython<3"
# Install spacy-pkuseg (Python 3.11 compatible fork) instead of broken pkuseg
RUN uv pip install --system --no-cache spacy-pkuseg
RUN uv pip install --system --no-cache --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
# Install Chatterbox without dependencies (since we used spacy-pkuseg instead of pkuseg)


# Bake in Piper Model (en_US-lessac-medium)
RUN mkdir -p /app/models/piper && \
    wget -q -O /app/models/piper/en_US-lessac-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx" && \
    wget -q -O /app/models/piper/en_US-lessac-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

# Bake in Chatterbox Models
# Triggers download to HF_HOME location defined above
RUN python3 -c "from chatterbox.tts import ChatterboxTTS; from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTTS.from_pretrained(device='cpu'); ChatterboxTurboTTS.from_pretrained(device='cpu')"

# Copy application code
COPY . .

# Set entrypoint to run the module
ENTRYPOINT ["python", "-m", "bitvoice"]
CMD ["--help"]
