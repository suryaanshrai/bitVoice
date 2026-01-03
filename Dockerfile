# Use PyTorch runtime image which has torch pre-installed (saving ~2GB download)
FROM python:3.11

# Set environment variables
ENV TZ=Asia/Kolkata
# DEBIAN_FRONTEND=noninteractive \
# # Prevent Python from writing pyc files to disc (equivalent to python -B)
# PYTHONDONTWRITEBYTECODE=1 \
# # Helper to ensure output is not buffered
# PYTHONUNBUFFERED=1 \
# # Hugging Face cache location
ENV HF_HOME=/app/models/huggingface
# Ensure bitvoice module is found even when WORKDIR is changed
ENV PYTHONPATH=/app

# Install system dependencies
# ffmpeg, espeak-ng, libsndfile1: Required for audio processing and TTS
# python3: Required as base image is bare

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 \
#     tzdata \
#     espeak-ng \
#     ffmpeg \
#     libsndfile1 \
#     build-essential \
#     wget \
#     python3-dev \
#     && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Torch/Audio are already in the base image, so we just install the rest.
# Pre-install numpy (required for pkuseg build) and Cython (to regenerate C++ files for Py3.10)
# RUN uv pip install --system --no-cache "numpy<2" "Cython<3"

# Manually download pkuseg, delete incompatible C++ files to force regeneration, and install
# RUN wget https://files.pythonhosted.org/packages/source/p/pkuseg/pkuseg-0.0.25.tar.gz \
#     && tar -xvf pkuseg-0.0.25.tar.gz \
#     && rm pkuseg-0.0.25/pkuseg/inference.cpp \
#     && rm pkuseg-0.0.25/pkuseg/feature_extractor.cpp \
#     && rm pkuseg-0.0.25/pkuseg/postag/feature_extractor.cpp \
#     && cd pkuseg-0.0.25 \
#     && python3 setup.py install \
#     && cd .. \
#     && rm -rf pkuseg-0.0.25*

RUN uv pip install --system --no-cache -r requirements.txt \
    && rm -rf /root/.cache

# Bake in Chatterbox Models
# This ensures models are present in the image and don't need runtime download.
# We use device='cpu' here because the build environment might not have a GPU.
# At runtime, the application will load these cached models onto the GPU if available.
RUN python3 -c "from chatterbox.tts import ChatterboxTTS; from chatterbox.tts_turbo import ChatterboxTurboTTS; ChatterboxTTS.from_pretrained(device='cpu'); ChatterboxTurboTTS.from_pretrained(device='cpu')"

# Copy application code (ONLY necessary source files)
COPY bitvoice/ bitvoice/
COPY README.md LICENSE ./

# Set entrypoint
ENTRYPOINT ["python", "-m", "bitvoice"]
CMD ["--help"]
