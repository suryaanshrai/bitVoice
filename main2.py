from datetime import datetime

start_time = datetime.now()

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import re
from markdown import markdown
from bs4 import BeautifulSoup
import torch
import os, json, time
from pathlib import Path
import hashlib


MODEL = ChatterboxTTS.from_pretrained(device="cuda")
OUTPUT_DIR = "audios"
INPUT_DIR = "content"
CHUNK_LENGTH = 150
# SETTINGS = {
#     "audio_prompt_path": "mysample.wav",
#     "exaggeration": 0.72, # [0.25 - 2]
#     "cfg_weight": 0.25, # [0.02 - 1]
#     "temperature": 0.15, # [0.05 - 5]
# }
SETTINGS = {
    "audio_prompt_path": "mysample.wav",
    "exaggeration": 0.675, # [0.25 - 2]
    "cfg_weight": 0.3, # [0.02 - 1]
    "temperature": 0.4, # [0.05 - 5]
}

HASH_JSON = "hashes.json"
HASHING = True


def remove_silence(audio, sample_rate, silence_threshold=0.01, min_silence_duration=0.5, keep_silence_duration=0.5):
    """
    Remove longer silences from audio while keeping natural pauses.
    
    Args:
        audio: torch tensor of audio (channels, samples)
        sample_rate: audio sample rate
        silence_threshold: amplitude threshold for silence detection (0.01 = -40dB)
        min_silence_duration: minimum silence duration to remove (in seconds)
        keep_silence_duration: duration of silence to keep as natural pause (in seconds)
    """
    # Convert to mono if stereo
    if audio.dim() > 1 and audio.shape[0] > 1:
        audio_mono = audio.mean(dim=0)
    else:
        audio_mono = audio.squeeze()
    
    # Calculate energy/amplitude
    energy = torch.abs(audio_mono)
    
    # Detect silence
    is_silent = energy < silence_threshold
    
    # Find silence segments
    min_silence_samples = int(min_silence_duration * sample_rate)
    keep_silence_samples = int(keep_silence_duration * sample_rate)
    
    segments_to_keep = []
    i = 0
    
    while i < len(is_silent):
        if not is_silent[i]:
            # Non-silent region - find where it ends
            start = i
            while i < len(is_silent) and not is_silent[i]:
                i += 1
            segments_to_keep.append((start, i))
        else:
            # Silent region - check duration
            silence_start = i
            while i < len(is_silent) and is_silent[i]:
                i += 1
            silence_duration = i - silence_start
            
            # If silence is long, keep only a short pause
            if silence_duration > min_silence_samples:
                # Add a short pause
                pause_end = min(silence_start + keep_silence_samples, i)
                if segments_to_keep:  # Only add pause if not at the start
                    segments_to_keep.append((silence_start, pause_end))
            else:
                # Keep short silences as-is
                segments_to_keep.append((silence_start, i))
    
    # Reconstruct audio
    if not segments_to_keep:
        return audio
    
    output_segments = [audio[..., start:end] for start, end in segments_to_keep]
    return torch.cat(output_segments, dim=-1)


def split_text(text, title=None):
    chunks = re.split(r'(?<=[.?!])\s*', text)
    result = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        while i < len(chunks) - 1 and len(chunk) + len(chunks[i+1]) < CHUNK_LENGTH:
            chunk += " " + chunks[i+1]
            i += 1
        result.append(chunk)
        i += 1
    if title:
        result = [title] + result
    return result


def generate_audio(text, settings):
    if HASHING:
        with open(HASH_JSON, "r") as f:
            hash_dict = json.load(f)
        content_hash = hashlib.sha256(" ".join(text).encode('utf-8')).hexdigest()
        if content_hash in hash_dict:
            print(f"\n\nSkipping audio file generation for {text[0]}\n\n")
            return None
    audio_segments = []

    for para in text:
        if para.strip():  
            print(f"\n\nGENERATING CHUNK of size {len(para)}. CHUNK content: {para}")            
            wav = MODEL.generate(para, **settings)
            audio_segments.append(wav)
    
    combined_wav = torch.cat(audio_segments, dim=-1)
    
    # Remove longer silences while keeping natural pauses
    combined_wav = remove_silence(combined_wav, MODEL.sr, min_silence_duration=1.0)
    
    # ta.save(dest, combined_wav, MODEL.sr)
    if HASHING:
        hash_dict[content_hash] = True
        with open(HASH_JSON, "w") as f:
            json.dump(hash_dict, f)
    return combined_wav

def clean_md(md_content):
    md_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', md_content, flags=re.DOTALL)
    html_content = markdown(md_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return text



def parse_dir():    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Find all markdown files recursively
    md_files = list(input_path.rglob("*.md"))
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for generate-audio:true in frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, flags=re.DOTALL)
        if not frontmatter_match:
            continue
        
        frontmatter = frontmatter_match.group(1)
        if not re.search(r'generate-audio', frontmatter, re.IGNORECASE):
            continue
        
        print(f"Processing: {md_file}")
        
        # Clean the markdown content
        cleaned_text = clean_md(content)
        
        # Get the title (filename without .md extension)
        title = md_file.stem
        
        # Split the text into chunks
        chunks = split_text(cleaned_text, title=title)
        
        # Generate audio
        combined_audio = generate_audio(text=chunks, settings=SETTINGS)
        
        if combined_audio is None:
            continue
        
        # Create output path maintaining directory structure
        relative_path = md_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix(".wav")
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the audio
        ta.save(str(output_file), combined_audio, MODEL.sr)
        print(f"Saved: {output_file}")
        print("Sleeping for 180 seconds for system cooldown")
        time.sleep(180)








parse_dir()

duration = datetime.now() - start_time
print(f"\n\nTIME TAKEN TO EXECUTE: {duration}")