# BitVoice

Convert documents (Markdown, PDF, EPUB, Word, Text) into natural-sounding speech using advanced local AI models.

**Now 100% Dockerized.** Run anywhere with zero dependency conflicts.

## Features
- **Models**: 
    - **Kokoro**: High-quality, lightweight (Baked-in).
    - **Piper**: Fast, local, low-latency (Baked-in).
    - **F5-TTS**: Zero-shot Voice Cloning.
    - **MeloTTS**: Multilingual, high-quality.
    - **Chatterbox**: Voice cloning specialized.
    - **XTTS v2**: Heavy, high-fidelity.
- **Formats**: `.md`, `.txt`, `.pdf`, `.docx`, `.epub`.
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

# Convert a folder using Piper (Fast)
bitvoice --model piper --input ./books --parallel

# List available models
bitvoice --model-list
```

## Usage

### Supported Models
| Model | Description | Status |
|-------|-------------|--------|
| `kokoro` | High-quality, lightweight | **Baked-in** |
| `piper` | Fast, works on CPU | **Baked-in** |
| `f5-tts` | Voice Cloning (Requires GPU recomm.) | Download on demand |
| `melo` | MeloTTS Multilingual | Experimental |
| `chatterbox` | Chatterbox | Experimental |
| `xtts` | XTTS v2 | Download on demand |

### Command Line Arguments
```text
  --input, -i       Input directory or single file.
  --output, -o      Output directory or filename.
  --model, -m       TTS Model (kokoro, piper, f5-tts, etc). Default: kokoro
  --voice, -v       Voice name (or reference audio for cloning).
  --parallel, -p    Enable parallel processing.
  --model-list      List supported TTS models.
  --voice-list      List voices for a specific model.
  --verbose         Enable verbose logging.
```

### Examples

**Voice Cloning with F5-TTS**:
```bash
bitvoice --model f5-tts --voice ./my_reference_audio.wav --input ./script.txt
```

**Using a Specific Kokoro Voice**:
```bash
bitvoice --model kokoro --voice af_bella --input book.txt
```ī

## Development

The project is now fully containerized.
- **Dockerfile**: Builds the environment and downloads core models.
- **bitvoice.py**: Core logic.

To run tests within the container:
```bash
docker run --rm bitvoice:latest -m pytest tests/
```
