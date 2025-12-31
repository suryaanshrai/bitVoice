#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import pickle
import re
from pathlib import Path
import soundfile as sf
import warnings

# Suppress some common warnings from libraries
warnings.filterwarnings("ignore")

# --- Configuration ---
CACHE_DIR = "caches"
CACHE_FILE = os.path.join(CACHE_DIR, "cache.pkl")
MODELS_DIR = "models"
KOKORO_MODEL = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.join(MODELS_DIR, "voices-v1.0.bin")
PIPER_MODEL = os.path.join(MODELS_DIR, "en_US-lessac-medium.onnx")
PIPER_CONFIG = os.path.join(MODELS_DIR, "en_US-lessac-medium.onnx.json")

# --- Utilities ---
def get_file_hash(content, model_name, voice_name):
    """Generate SHA256 hash for content and parameters."""
    full_string = f"{content}|{model_name}|{voice_name}"
    return hashlib.sha256(full_string.encode('utf-8')).hexdigest()

def clean_markdown(text):
    """Basic markdown cleaning for better TTS."""
    # Remove images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove links: [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove bold/italic: **text** or *text* -> text
    text = re.sub(r'(\*\*|__|(\*|_))(.*?)\1', r'\3', text)
    # Remove headers: # Header -> Header
    text = re.sub(r'^#+\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
    # Remove code blocks: ```code```
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code: `code`
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def read_file_content(file_path):
    """Read text from supported file formats."""
    if isinstance(file_path, str): file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.md' or suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
            
        elif suffix == '.docx':
            from docx import Document
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
            
        elif suffix == '.epub':
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(str(file_path))
            chapters = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    chapters.append(soup.get_text())
            return "\n".join(chapters)

        else:
            print(f"Unsupported format: {suffix}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# --- Engines ---
class TTSEngine:
    def get_voices(self): raise NotImplementedError
    def generate(self, text, voice, output_path): raise NotImplementedError

class KokoroEngine(TTSEngine):
    def __init__(self):
        if not os.path.exists(KOKORO_MODEL) or not os.path.exists(KOKORO_VOICES):
            # Try to load from package resources if used as library? For now, simplistic check.
             pass 
        try:
            from kokoro_onnx import Kokoro
            # Fallback pathing for library usage?
            # If we are in library mode, we might not have models in cwd.
            # Ideally models should be downloaded or paths configures.
            # We will assume user manages models for now or they are in valid path.
            if os.path.exists(KOKORO_MODEL):
                self.kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
            else:
                 raise FileNotFoundError(f"Kokoro model not found at {KOKORO_MODEL}")
        except ImportError:
            raise ImportError("kokoro-onnx not installed.")
    
    def get_voices(self): return self.kokoro.get_voices()
    def generate(self, text, voice, output_path):
        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        sf.write(output_path, samples, sample_rate)

class Pyttsx3Engine(TTSEngine):
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
    def get_voices(self): return [v.name for v in self.engine.getProperty('voices')]
    def generate(self, text, voice, output_path):
        voices = self.engine.getProperty('voices')
        selected_voice = None
        for v in voices:
            if voice.lower() in v.name.lower():
                selected_voice = v.id
                break
        if selected_voice: self.engine.setProperty('voice', selected_voice)
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()

class PiperEngine(TTSEngine):
    def __init__(self):
        if not os.path.exists(PIPER_MODEL): raise FileNotFoundError(f"Piper model not found at {PIPER_MODEL}")
        try:
            from piper import PiperVoice
            self.voice = PiperVoice.load(PIPER_MODEL, config_path=PIPER_CONFIG)
        except ImportError: raise ImportError("piper-tts not installed.")
    def get_voices(self): return ["default"]
    def generate(self, text, voice, output_path):
        with open(output_path, "wb") as wav_file:
            self.voice.synthesize(text, wav_file)

class GTTSEngine(TTSEngine):
    def __init__(self):
        from gtts import gTTS
        self.gTTS = gTTS
    def get_voices(self): return ["en", "fr", "es"]
    def generate(self, text, voice, output_path):
        tts = self.gTTS(text=text, lang=voice if voice in self.get_voices() else "en")
        tts.save(output_path)

class F5TTSEngine(TTSEngine):
    def __init__(self):
        from f5_tts.api import F5TTS
        from importlib.resources import files
        self.files = files
        self.f5tts = F5TTS()
    def get_voices(self): return ["default"]
    def generate(self, text, voice, output_path):
        ref_file = voice
        if voice == "default" or not os.path.exists(voice):
             ref_file = str(self.files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        self.f5tts.infer(ref_file=ref_file, ref_text="Ref text.", gen_text=text, file_wave=output_path)

class XTTSEngine(TTSEngine):
    def __init__(self):
        from TTS.api import TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    def get_voices(self): return ["default"]
    def generate(self, text, voice, output_path):
        speaker_wav = voice if os.path.exists(voice) else None
        if not speaker_wav: self.tts.tts_to_file(text=text, file_path=output_path, speaker=self.tts.speakers[0], language="en")
        else: self.tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language="en")

class FishSpeechEngine(TTSEngine):
    def get_voices(self): return []
    def generate(self, text, voice, output_path): raise NotImplementedError("Fish Speech not manually configured.")

def get_engine(model_name):
    if model_name == "kokoro": return KokoroEngine()
    elif model_name == "pyttsx3": return Pyttsx3Engine()
    elif model_name == "piper": return PiperEngine()
    elif model_name == "gtts": return GTTSEngine()
    elif model_name == "f5-tts": return F5TTSEngine()
    elif model_name == "xtts": return XTTSEngine()
    else: raise ValueError(f"Unknown model: {model_name}")

# --- Worker for Parallel ---
def process_single_item(item_data):
    text, model_name, voice_name, output_path = item_data
    try:
        engine = get_engine(model_name)
        engine.generate(text, voice_name, output_path)
        return True, None
    except Exception as e:
        return False, str(e)

# --- BitVoice Library Class ---
class BitVoice:
    """
    Main entry point for using BitVoice as a library.
    Example:
      bv = BitVoice(model='kokoro', voice='af_heart')
      bv.convert_file('book.txt', 'audio.wav')
    """
    def __init__(self, model="kokoro", voice=None):
        self.model = model
        self.voice = voice
        self.engine = None
        # Lazy init engine
    
    def _init_engine(self):
        if not self.engine:
            self.engine = get_engine(self.model)
            # Resolve voice default if needed
            if not self.voice:
                try:
                    voices = self.engine.get_voices()
                    self.voice = "af_heart" if self.model == "kokoro" else (voices[0] if voices else "default")
                except:
                    self.voice = "default"
    
    def convert_text(self, text, output_path):
        self._init_engine()
        self.engine.generate(text, self.voice, output_path)

    def convert_file(self, input_file, output_path):
        text = read_file_content(input_file)
        if not text:
            raise ValueError(f"Could not read text from {input_file}")
        cleaned = clean_markdown(text)
        self.convert_text(cleaned, output_path)

# --- Installation Logic ---
def install_tool():
    """Install CLI wrapper."""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    # ... (Keep existing script wrapper logic if desired, or prefer library install now)
    # The user asked for "Library installation". 
    pass # Replaced by install_library_package

def install_library_package():
    """Install the current directory as a pip package."""
    # Check venv
    in_venv = (sys.prefix != sys.base_prefix)
    if not in_venv:
        print("Warning: You are NOT running in a virtual environment.")
        confirm = input("Do you want to install 'bitvoice' into your global python environment? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    print("Installing bitvoice as a library (editable mode)...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("\nSuccess! You can now use 'bitvoice' as a command or 'import bitvoice' in Python.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="BitVoice: Convert text/doc files to Speech.")
    parser.add_argument("--input", "-i", type=str, help="Input directory or single file.")
    parser.add_argument("--output", "-o", type=str, help="Output directory or filename.")
    parser.add_argument("--model", "-m", type=str, default="kokoro", help="TTS Model.")
    parser.add_argument("--voice", "-v", type=str, help="Voice name.")
    parser.add_argument("--parallel", "-p", action="store_true", help="Enable parallel processing.")
    parser.add_argument("--install", action="store_true", help="Install CLI wrapper script (Legacy).")
    parser.add_argument("--install-library", action="store_true", help="Install as a Python library.")
    
    args = parser.parse_args()

    if args.install_library:
        install_library_package()
        return

    if args.install:
        install_tool()
        return
        
    if not args.input:
        parser.print_help()
        return

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return
        
    is_single_file = input_path.is_file()
    supported_exts = {'.md', '.txt', '.pdf', '.docx', '.epub'}
    
    files_to_process = []
    if is_single_file:
         if input_path.suffix.lower() in supported_exts: files_to_process = [input_path]
         else: 
             print(f"Warning: Unknown extension {input_path.suffix}, trying anyway.")
             files_to_process = [input_path]
    else:
        for ext in supported_exts:
             files_to_process.extend(input_path.rglob(f"*{ext}"))
             
    if not files_to_process:
        print("No supported files found.")
        return

    # Logic for processing
    # We need to make sure 'engines_map' validation logic is preserved or adapted.
    try:
        dummy = get_engine(args.model) # Validate model
        # Get voices
        try: 
            available_voices = dummy.get_voices()
        except: 
            available_voices = ["default"]
            
        voice = args.voice
        if not voice:
             if args.model == "kokoro": voice = "af_heart"
             elif args.model in ["piper", "f5-tts", "xtts"]: voice = "default"
             else: voice = "default"
    except Exception as e:
        print(f"Error initializing model {args.model}: {e}")
        return

    # Cache load
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except:
            pass

    work_items = []
    for file_path in files_to_process:
        if CACHE_DIR in file_path.parts: continue
        if args.output and Path(args.output) == file_path: continue
        
        # Outcome path logic
        if is_single_file and args.output:
            out_p = Path(args.output)
            audio_file = out_p if out_p.suffix else out_p / file_path.with_suffix(".wav").name
        else:
            out_root = Path(args.output) if args.output else Path("output")
            try: rel = file_path.relative_to(input_path)
            except: rel = file_path.name
            audio_file = out_root / Path(rel).with_suffix(".wav")
            
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
             content = read_file_content(file_path)
             if not content: continue
             cleaned = clean_markdown(content)
             if not cleaned: continue
             
             f_hash = get_file_hash(cleaned, args.model, voice)
             if cache.get(str(file_path)) == f_hash and audio_file.exists():
                 print(f"Skipping cached: {file_path.name}")
                 continue
                 
             work_items.append((cleaned, args.model, voice, str(audio_file), str(file_path), f_hash))
        except Exception as e:
             print(f"Skipping {file_path}: {e}")

    if not work_items: 
        print("Nothing to do.")
        return

    print(f"Processing {len(work_items)} items...")
    
    if args.parallel and args.model not in ["f5-tts", "xtts"]:
         from concurrent.futures import ProcessPoolExecutor
         with ProcessPoolExecutor() as executor:
             results = list(executor.map(process_single_item, [(w[0], w[1], w[2], w[3]) for w in work_items]))
    else:
         results = []
         for w in work_items:
             results.append(process_single_item((w[0], w[1], w[2], w[3])))
             print(f"Generated {Path(w[3]).name}")

    for i, (ok, err) in enumerate(results):
        if ok: cache[work_items[i][4]] = work_items[i][5]
        else: print(f"Failed {work_items[i][4]}: {err}")
        
    with open(CACHE_FILE, 'wb') as f: pickle.dump(cache, f)
    print("Done.")

if __name__ == "__main__":
    main()
