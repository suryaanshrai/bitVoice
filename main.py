from datetime import datetime

start_time = datetime.now()

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import re
from markdown import markdown
from bs4 import BeautifulSoup
import torch


CHUNK_LENGTH = 200
# SETTINGS = {
#     "audio_prompt_path": "mysample.wav",
#     "exaggeration": 0.65, # [0.25 - 2]
#     "cfg_weight": 0.05, # [0.02 - 1]
#     "temperature": 0.9, # [0.05 - 5]
# }

SETTINGS = {
    "audio_prompt_path": "mysample.wav",
    "exaggeration": 0.72, # [0.25 - 2]
    "cfg_weight": 0.02, # [0.02 - 1]
    # "temperature": 0.4, # [0.05 - 5]
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

def read_dir():
    pass


def generate_audio(text, dest, settings):
    model = ChatterboxTTS.from_pretrained(device="cuda")
    audio_segments = []

    for para in text:
        if para.strip():  
            print(f"\n\nGENERATING CHUNK of size {len(para)}. CHUNK content: {para}")            
            wav = model.generate(para, **settings)
            audio_segments.append(wav)
    
    combined_wav = torch.cat(audio_segments, dim=-1)
    ta.save(dest, combined_wav, model.sr)

def clean_md(md_content):
    md_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', md_content, flags=re.DOTALL)
    html_content = markdown(md_content)
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return text


with open("Reading a bit.md", "r") as f:
    content = f.read()
    poem = clean_md(content)

chunks = split_text(poem)
generate_audio(text=chunks[:2], dest="reading_surya_3.wav", settings=SETTINGS)

























duration = datetime.now() - start_time
print(f"\n\nTIME TAKEN TO EXECUTE: {duration}")