FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/output /app/caches

# Copy application code
COPY bitvoice.py /app/bitvoice.py

ENTRYPOINT ["python", "bitvoice.py"]
CMD ["--help"]
