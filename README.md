# BitVoice

Convert documents (Markdown, PDF, EPUB, Word, Text) into natural-sounding speech using advanced local AI models.

## Features
- **Models**: Kokoro, F5-TTS (Voice Cloning), Piper, XTTS v2, pyttsx3, gTTS.
- **Modules**: Can be installed and used as a Python library.
- **Formats**: `.md`, `.txt`, `.pdf`, `.docx`, `.epub`.
- **Input**: Process single files or entire directories recursively.
- **Performance**: Parallel processing support (`--parallel`).
- **Smart**: SHA256 caching to avoid re-generating existing audio.

## Dependencies

Core dependencies are listed in `requirements.txt`.
*Optional*: For F5-TTS support, you'll need `f5-tts`, `torch`, `torchaudio`. The Docker image includes core dependencies.

## Installation

### Local (Python)
```bash
# Clone
git clone https://github.com/yourusername/bitvoice.git
cd bitvoice

# Install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install as command line tool (legacy wrapper)
python bitvoice.py --install

# Install as a proper Python library (Recommended)
# *Interactive*: Prompts to optionally install F5-TTS extras.
python bitvoice.py --install-library

# Manual Install (Advanced)
pip install -e .[f5]

# Install F5-TTS Dependencies (Optional, ~3GB)
python bitvoice.py --install-f5-tts
```

### Usage as a Library
```python
from bitvoice import BitVoice

# Initialize
bv = BitVoice(model="kokoro", voice="af_heart")

# Convert text
bv.convert_text("Hello from Python!", "output.wav")

# Convert file (supports md, pdf, epub, docx, txt)
bv.convert_file("my_book.epub", "book_audio.wav")
```

### Docker (Pre-built)
Pull the ready-to-use image from DockerHub:
```bash
# Pull the image
docker pull suryaanshrai515/bitvoice:latest

# Run it on your current directory
# This mounts the current folder ($PWD) to /data inside the container
docker run --rm -v "$(pwd):/data" suryaanshrai515/bitvoice:latest --input /data --output /data
```

### Docker (Build Locally)
```bash
docker build -t bitvoice .
docker run --rm -v "$(pwd):/data" bitvoice --input /data --output /data
```

## Usage

```bash
# Basic usage (Kokoro model)
bitvoice --input ./docs --output ./audio

# Single file
bitvoice --input README.md --output intro.wav

# Parallel processing with Piper (Fast)
bitvoice --model piper --input ./books --parallel


# Voice Cloning with F5-TTS
bitvoice --model f5-tts --voice ./my_voice_sample.wav --input ./script.md
```

## Development & Testing
The project includes a comprehensive test suite using `pytest`.

```bash
# Run Unit and Integration tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_unit.py
```

### CI/CD Pipeline
Automated via GitHub Actions (`.github/workflows/ci-cd.yml`):
1.  **Test**: Runs `pytest` on every push to `main` and pull requests.
2.  **Deploy**: If tests pass on `main`, builds and pushes the Docker image to Docker Hub (`suryaanshrai515/bitvoice:latest`).
