# BitVoice

Convert Markdown documents into natural-sounding speech using advanced local AI models.

**Now 100% Dockerized.** Run anywhere with zero dependency conflicts.

## Features
- **Models**: 
    - **Chatterbox**: High-quality, fast, and supports voice cloning (Baked-in).
    - **Chatterbox Turbo**: Optimized for speed (Baked-in).
- **Formats**: `.md`.
- **Input**: Process single files or entire directories recursively.
- **Performance**: Parallel processing support (`--parallel`).
- **Smart**: SHA256 caching keeps things fast.

## 🚀 Quick Start

## 🚀 Quick Start

### 1. Install Alias
We provide scripts to set up a convenient `bitvoice` command. This will automatically pull the latest ready-to-use image from Docker Hub (`suryaanshrai515/bitvoice:latest`).

**Windows (Powershell)**:
```powershell
iex (irm https://raw.githubusercontent.com/suryaanshrai/bitVoice/main/scripts/install_windows.ps1)
```

**Linux / macOS**:
```bash
curl -fsSL https://raw.githubusercontent.com/suryaanshrai/bitVoice/main/scripts/install_linux.sh | bash
# or for macOS
# curl -fsSL https://raw.githubusercontent.com/suryaanshrai/bitVoice/main/scripts/install_mac.sh | bash
```

### 2. Run
Now you can use `bitvoice` just like a local tool. It works on files in your current directory.

```bash
# Convert a file
bitvoice --input README.md --output intro.wav
```

### (Optional) Build Locally
If you prefer to build the image yourself:
```bash
docker build -t suryaanshrai515/bitvoice:latest .
```

# Convert a folder using Chatterbox
bitvoice --model chatterbox --input ./books --parallel

# List available models
bitvoice --model-list
```

## Usage

### Supported Models
| Model | Description | Status |
|-------|-------------|--------|
| `chatterbox` | High-quality, supports voice cloning | **Baked-in** |
| `chatterbox-turbo` | Optimized for speed | **Baked-in** |

### Command Line Arguments
```text
  --input, -i       Input directory or single file.
  --output, -o      Output directory or filename.
  --model, -m       TTS Model (chatterbox, chatterbox-turbo). Default: chatterbox
  --voice, -v       Voice name (or reference audio for cloning).
  --parallel, -p    Enable parallel processing.
  --model-list      List supported TTS models.
  --voice-list      List voices for a specific model.
  --verbose         Enable verbose logging.
```

### Examples

**Using Chatterbox with Params**:
```bash
bitvoice --model chatterbox --input book.txt --cb-speed 1.2
```

## Development

The project is now fully containerized.
- **Dockerfile**: Builds the environment and downloads core models.
- **bitvoice.py**: Core logic.

To run tests within the container:
```bash
docker run --rm bitvoice:latest -m pytest tests/
```
