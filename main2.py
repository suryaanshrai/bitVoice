from datetime import datetime

start_time = datetime.now()

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import re
from markdown import markdown
from bs4 import BeautifulSoup
import torch
import os
from pathlib import Path


MODEL = ChatterboxTTS.from_pretrained(device="cuda")
OUTPUT_DIR = "audios"
INPUT_DIR = "content"
CHUNK_LENGTH = 150
SETTINGS = {
    "audio_prompt_path": "mysample.wav",
    "exaggeration": 0.6, # [0.25 - 2]
    "cfg_weight": 0.3, # [0.02 - 1]
    # "temperature": 0.9, # [0.05 - 5]
}


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
    audio_segments = []

    for para in text:
        if para.strip():  
            print(f"\n\nGENERATING CHUNK of size {len(para)}. CHUNK content: {para}")            
            wav = MODEL.generate(para, **settings)
            audio_segments.append(wav)
    
    combined_wav = torch.cat(audio_segments, dim=-1)
    # ta.save(dest, combined_wav, MODEL.sr)
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
        combined_audio = generate_audio(text=chunks[:2], settings=SETTINGS)
        
        # Create output path maintaining directory structure
        relative_path = md_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix(".wav")
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the audio
        ta.save(str(output_file), combined_audio, MODEL.sr)
        print(f"Saved: {output_file}")







parse_dir()

duration = datetime.now() - start_time
print(f"\n\nTIME TAKEN TO EXECUTE: {duration}")