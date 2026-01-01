FROM python:3.10-slim-bullseye

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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in Kokoro Model
# Create models directory and download files
RUN mkdir -p /app/models && \
    wget -q -O /app/models/kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" && \
    wget -q -O /app/models/voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Copy application code
COPY . .

# Set entrypoint to just bash for debugging, or directly to bitvoice if preferred. 
# User wanted it to be treated as container-only app.
ENTRYPOINT ["python", "bitvoice.py"]
CMD ["--help"]
