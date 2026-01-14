# bitVoice 🎙️

A Python-based text-to-speech (TTS) system that converts markdown files to high-quality audio using the ChatterboxTTS model. The system features intelligent text chunking, silence removal, and content hashing for efficient batch processing.

>[!note]
> This is an AI generated README since I made this project solely for my own use. [This branch](https://github.com/suryaanshrai/bitVoice/tree/legacy/v1/vibed) has the original *vibe coded* code which I initially started to deploy as a CLI utility through docker, but since it did not *vibed* well and started taking too much time and resources, I ended up writing a basic python script that only suffices my use (which can still be leveraged with slight modifications on your local system as well).

## Features

✨ **Smart Processing**
- Recursive markdown file scanning with frontmatter filtering
- Intelligent text chunking for optimal audio generation
- Automatic silence removal (removes silences longer than 1 second)
- Content hashing to skip already-generated files

🎵 **High-Quality Audio**
- Powered by ChatterboxTTS (CUDA-accelerated)
- Voice cloning from audio samples
- Customizable generation parameters (exaggeration, CFG weight, temperature)
- Maintains natural speech rhythm with preserved pauses

📁 **Organized Output**
- Preserves directory structure from input to output
- WAV format output with configurable sample rates
- Batch processing with cooldown periods

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- PyTorch with CUDA support
- ChatterboxTTS

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/suryaanshrai/bitVoice.git
cd bitVoice
```

2. **Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install chatterbox-tts markdown beautifulsoup4
```

4. **Set up ChatterboxTTS model**
The model will be automatically downloaded on first run.

## Configuration

### Directory Structure

```
bitVoice/
├── content/          # Input markdown files
├── audios/           # Generated audio outputs
├── mysample.wav      # Voice sample for cloning
├── main2.py          # Main processing script
└── hashes.json       # Content hash cache
```

### Audio Generation Settings

Edit the `SETTINGS` dictionary in `main2.py`:

```python
SETTINGS = {
    "audio_prompt_path": "mysample.wav",  # Your voice sample
    "exaggeration": 0.675,     # Range: 0.25 - 2.0
    "cfg_weight": 0.3,         # Range: 0.02 - 1.0
    "temperature": 0.4,        # Range: 0.05 - 5.0
}
```

### Silence Removal Configuration

Adjust silence removal parameters in the `remove_silence()` call (line 133):

```python
combined_wav = remove_silence(
    combined_wav, 
    MODEL.sr, 
    min_silence_duration=1.0,      # Trim silences longer than 1 second
    # silence_threshold=0.01,        # Amplitude threshold (optional)
    # keep_silence_duration=0.5      # Duration to keep as pause (optional)
)
```

### Text Chunking

Modify the `CHUNK_LENGTH` constant to control text splitting:

```python
CHUNK_LENGTH = 150  # Characters per chunk
```

## Usage

### Prepare Your Content

1. Create markdown files in the `content/` directory
2. Add frontmatter to files you want to process:

```markdown
---
generate-audio: true
---

# Your Title

Your content here...
```

### Generate Audio

Run the script:

```bash
python main2.py
```

The script will:
1. Scan `content/` for markdown files with `generate-audio: true`
2. Clean and chunk the text
3. Generate audio using your voice sample
4. Remove long silences
5. Save to `audios/` with the same directory structure

### Cooldown Period

The script includes a 180-second cooldown between files to prevent GPU overheating. Adjust in `main2.py` line 196:

```python
time.sleep(180)  # Cooldown in seconds
```

## How It Works

1. **File Discovery**: Recursively scans `INPUT_DIR` for `.md` files
2. **Frontmatter Check**: Only processes files with `generate-audio` in frontmatter
3. **Content Cleaning**: Removes frontmatter and converts markdown to plain text
4. **Text Chunking**: Splits text at sentence boundaries into optimal lengths
5. **Audio Generation**: Generates audio for each chunk with your voice
6. **Silence Removal**: Trims long silences while preserving natural pauses
7. **Hash Caching**: Saves content hash to skip regeneration
8. **Output**: Saves WAV file to corresponding path in `audios/`

## Advanced Features

### Content Hashing

Set `HASHING = False` to disable content hashing and regenerate all audio:

```python
HASHING = False  # Regenerate even if already processed
```

### Custom Voice Sample

Replace `mysample.wav` with your own voice recording (5-10 seconds recommended):

```python
SETTINGS = {
    "audio_prompt_path": "your_voice.wav",
    # ... other settings
}
```

## Parameters Reference

### Exaggeration (0.25 - 2.0)
Controls expressiveness and prosody variation
- **Lower**: More monotone, consistent
- **Higher**: More dramatic, varied intonation

### CFG Weight (0.02 - 1.0)
Classifier-free guidance strength
- **Lower**: More creative, potentially inconsistent
- **Higher**: Closer to training data, more predictable

### Temperature (0.05 - 5.0)
Sampling randomness
- **Lower**: More deterministic, consistent
- **Higher**: More varied, creative

## Troubleshooting

**GPU Out of Memory**
- Reduce `CHUNK_LENGTH`
- Increase cooldown time
- Close other GPU applications

**Audio Quality Issues**
- Use a clearer voice sample
- Adjust `exaggeration` and `temperature`
- Ensure input text is well-formatted

**Files Not Processing**
- Check frontmatter includes `generate-audio`
- Verify file encoding is UTF-8
- Check console for error messages

## License

MIT License - feel free to use and modify as needed.

## Credits

Powered by [ChatterboxTTS](https://github.com/lifeiteng/Chatterbox)

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.
