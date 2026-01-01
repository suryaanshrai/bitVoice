#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import pickle
import re
import logging
import soundfile as sf
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Tuple, Any

# Suppress some common warnings from libraries
warnings.filterwarnings("ignore")

# --- Logging Setup ---
logger = logging.getLogger("bitvoice")

# --- Configuration ---
@dataclass
class Settings:
    CACHE_DIR: str = "caches"
    MODELS_DIR: str = "models"
    CACHE_FILENAME: str = "cache.pkl"
    KOKORO_MODEL: str = "kokoro-v1.0.onnx"
    KOKORO_VOICES: str = "voices-v1.0.bin"
    PIPER_MODEL: str = "en_US-lessac-medium.onnx"
    PIPER_CONFIG: str = "en_US-lessac-medium.onnx.json"

    @property
    def cache_path(self) -> str:
        return os.path.join(self.CACHE_DIR, self.CACHE_FILENAME)

    @property
    def kokoro_model_path(self) -> str:
        return os.path.join(self.MODELS_DIR, self.KOKORO_MODEL)

    @property
    def kokoro_voices_path(self) -> str:
        return os.path.join(self.MODELS_DIR, self.KOKORO_VOICES)

    @property
    def piper_model_path(self) -> str:
        return os.path.join(self.MODELS_DIR, self.PIPER_MODEL)

    @property
    def piper_config_path(self) -> str:
        return os.path.join(self.MODELS_DIR, self.PIPER_CONFIG)

CONF = Settings()

# --- Utilities ---
# --- Utilities ---
def get_file_hash(content: str, model_name: str, voice_name: str) -> str:
    """Generate SHA256 hash for content and parameters."""
    full_string = f"{content}|{model_name}|{voice_name}"
    return hashlib.sha256(full_string.encode('utf-8')).hexdigest()

def clean_markdown(text: str) -> str:
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

# --- File Handlers ---
def _read_text_file(path: Path) -> Optional[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def _read_pdf_file(path: Path) -> Optional[str]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def _read_docx_file(path: Path) -> Optional[str]:
    from docx import Document
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def _read_epub_file(path: Path) -> Optional[str]:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    
    book = epub.read_epub(str(path))
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            chapters.append(soup.get_text())
    return "\n".join(chapters)

FILE_HANDLERS: Dict[str, Callable[[Path], Optional[str]]] = {
    '.md': _read_text_file,
    '.txt': _read_text_file,
    '.pdf': _read_pdf_file,
    '.docx': _read_docx_file,
    '.epub': _read_epub_file
}

def read_file_content(file_path: Any) -> Optional[str]:
    """Read text from supported file formats."""
    if isinstance(file_path, str): 
        file_path = Path(file_path)
    
    suffix = file_path.suffix.lower()
    handler = FILE_HANDLERS.get(suffix)
    
    if not handler:
        logger.warning(f"Unsupported format: {suffix}")
        return None
        
    try:
        return handler(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

# --- Engines ---
# --- Engines ---
class TTSEngine:
    def get_voices(self) -> List[str]: raise NotImplementedError
    def generate(self, text: str, voice: str, output_path: str) -> None: raise NotImplementedError

class KokoroEngine(TTSEngine):
    def __init__(self):
        if not os.path.exists(CONF.kokoro_model_path) or not os.path.exists(CONF.kokoro_voices_path):
             pass 
        try:
            from kokoro_onnx import Kokoro
            if os.path.exists(CONF.kokoro_model_path):
                self.kokoro = Kokoro(CONF.kokoro_model_path, CONF.kokoro_voices_path)
            else:
                 raise FileNotFoundError(f"Kokoro model not found at {CONF.kokoro_model_path}")
        except ImportError:
            raise ImportError("kokoro-onnx not installed.")
    
    def get_voices(self) -> List[str]: return self.kokoro.get_voices()
    def generate(self, text: str, voice: str, output_path: str) -> None:
        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        sf.write(output_path, samples, sample_rate)

class Pyttsx3Engine(TTSEngine):
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
    def get_voices(self) -> List[str]: return [v.name for v in self.engine.getProperty('voices')]
    def generate(self, text: str, voice: str, output_path: str) -> None:
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
        if not os.path.exists(CONF.piper_model_path): raise FileNotFoundError(f"Piper model not found at {CONF.piper_model_path}")
        try:
            from piper import PiperVoice
            self.voice = PiperVoice.load(CONF.piper_model_path, config_path=CONF.piper_config_path)
        except ImportError: raise ImportError("piper-tts not installed.")
    def get_voices(self) -> List[str]: return ["default"]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        with open(output_path, "wb") as wav_file:
            self.voice.synthesize(text, wav_file)

class GTTSEngine(TTSEngine):
    def __init__(self):
        from gtts import gTTS
        self.gTTS = gTTS
    def get_voices(self) -> List[str]: return ["en", "fr", "es"]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        tts = self.gTTS(text=text, lang=voice if voice in self.get_voices() else "en")
        tts.save(output_path)

class F5TTSEngine(TTSEngine):
    def __init__(self):
        from f5_tts.api import F5TTS
        from importlib.resources import files
        self.files = files
        self.f5tts = F5TTS()
    def get_voices(self) -> List[str]: return ["default"]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        ref_file = voice
        if voice == "default" or not os.path.exists(voice):
             ref_file = str(self.files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        self.f5tts.infer(ref_file=ref_file, ref_text="Ref text.", gen_text=text, file_wave=output_path)

class XTTSEngine(TTSEngine):
    def __init__(self):
        from TTS.api import TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    def get_voices(self) -> List[str]: return ["default"]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        speaker_wav = voice if os.path.exists(voice) else None
        if not speaker_wav: self.tts.tts_to_file(text=text, file_path=output_path, speaker=self.tts.speakers[0], language="en")
        else: self.tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language="en")

class FishSpeechEngine(TTSEngine):
    def get_voices(self) -> List[str]: return []
    def generate(self, text: str, voice: str, output_path: str) -> None: raise NotImplementedError("Fish Speech not manually configured.")

def get_engine(model_name: str) -> TTSEngine:
    if model_name == "kokoro": return KokoroEngine()
    elif model_name == "pyttsx3": return Pyttsx3Engine()
    elif model_name == "piper": return PiperEngine()
    elif model_name == "gtts": return GTTSEngine()
    elif model_name == "f5-tts": return F5TTSEngine()
    elif model_name == "xtts": return XTTSEngine()
    else: raise ValueError(f"Unknown model: {model_name}")

# --- Worker for Parallel ---
def process_single_item(item_data: Tuple[str, str, str, str]) -> Tuple[bool, Optional[str]]:
    text, model_name, voice_name, output_path = item_data
    try:
        engine = get_engine(model_name)
        engine.generate(text, voice_name, output_path)
        return True, None
    except Exception as e:
        logger.error(f"Error processing item: {e}")
        return False, str(e)

# --- BitVoice Library Class ---
class BitVoice:
    """
    Main entry point for using BitVoice as a library.
    Example:
      bv = BitVoice(model='kokoro', voice='af_heart')
      bv.convert_file('book.txt', 'audio.wav')
    """
    def __init__(self, model: str = "kokoro", voice: Optional[str] = None):
        self.model = model
        self.voice = voice
        self.engine: Optional[TTSEngine] = None
    
    def _init_engine(self) -> None:
        if not self.engine:
            self.engine = get_engine(self.model)
            if not self.voice:
                try:
                    voices = self.engine.get_voices()
                    self.voice = "af_heart" if self.model == "kokoro" else (voices[0] if voices else "default")
                except Exception as e:
                    logger.debug(f"Could not autoset voice: {e}")
                    self.voice = "default"
    
    def convert_text(self, text: str, output_path: str) -> None:
        self._init_engine()
        assert self.engine is not None
        self.engine.generate(text, self.voice or "default", output_path)

    def convert_file(self, input_file: str, output_path: str) -> None:
        text = read_file_content(input_file)
        if not text:
            raise ValueError(f"Could not read text from {input_file}")
        cleaned = clean_markdown(text)
        self.convert_text(cleaned, output_path)

# --- Installation Logic ---
def install_tool() -> None:
    """Install CLI wrapper and add to PATH."""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    
    if os.name == 'nt':
        wrapper_path = os.path.join(script_dir, "bitvoice.bat")
        with open(wrapper_path, "w") as f:
            f.write(f'@echo off\n"{sys.executable}" "{script_path}" %*')
        print(f"[SUCCESS] Created wrapper: {wrapper_path}")
        
        # Add to PATH via Registry
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_ALL_ACCESS)
            try:
                current_path, _ = winreg.QueryValueEx(key, 'Path')
            except FileNotFoundError:
                current_path = ""
            
            if script_dir.lower() not in current_path.lower().split(';'):
                new_path = f"{current_path};{script_dir}" if current_path else script_dir
                winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
                print(f"[SUCCESS] Added {script_dir} to User PATH.")
                print("Note: You may need to restart your terminal for changes to take effect.")
            else:
                 print(f"[INFO] {script_dir} is already in PATH.")
            winreg.CloseKey(key)
        except Exception as e:
            print(f"[ERROR] Could not update PATH registry: {e}")
            print(f"Please manually add {script_dir} to your environment variables.")

    else:
        wrapper_path = os.path.join(script_dir, "bitvoice")
        with open(wrapper_path, "w") as f:
            f.write(f'#!/bin/bash\nexec "{sys.executable}" "{script_path}" "$@"')
        os.chmod(wrapper_path, 0o755)
        print(f"[SUCCESS] Created wrapper: {wrapper_path}")
        
        # Add to PATH on Linux/Mac
        shell = os.environ.get("SHELL", "")
        home = os.path.expanduser("~")
        config_file = None
        
        if "zsh" in shell:
            config_file = os.path.join(home, ".zshrc")
        elif "bash" in shell:
            # On Linux, .bashrc is common. On Mac, .bash_profile might be used.
            if os.path.exists(os.path.join(home, ".bash_profile")):
                config_file = os.path.join(home, ".bash_profile")
            else:
                config_file = os.path.join(home, ".bashrc")
        else:
            # Fallback or other shells (profile)
            config_file = os.path.join(home, ".profile")

        export_line = f'export PATH="$PATH:{script_dir}"'
        
        if config_file:
            try:
                # Check if already exists
                content = ""
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        content = f.read()
                
                if script_dir not in content:
                    with open(config_file, "a") as f:
                        f.write(f"\n# Added by BitVoice\n{export_line}\n")
                    print(f"[SUCCESS] Added to PATH in {config_file}")
                    print(f"Run 'source {config_file}' or restart your terminal.")
                else:
                    print(f"[INFO] Already in PATH in {config_file}")
            except Exception as e:
                print(f"[ERROR] Could not update {config_file}: {e}")
                print(f"Please manually run: {export_line}")
        else:
             print(f"Could not detect shell config file. Please manually run: {export_line}")

def install_library_package() -> None:
    """Install the current directory as a pip package."""
    # Check venv
    in_venv = (sys.prefix != sys.base_prefix)
    if not in_venv:
        # We use print here as it is an interactive prompt
        print("Warning: You are NOT running in a virtual environment.")
        confirm = input("Do you want to install 'bitvoice' into your global python environment? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    print("Installing bitvoice as a library (editable mode)...")
    
    install_f5 = False
    f5_confirm = input("Do you want to install optional F5-TTS support (Heavy, ~3GB)? [y/N]: ")
    if f5_confirm.lower() == 'y': install_f5 = True

    import subprocess
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-e"]
        if install_f5:
             cmd.append(".[f5]")
        else:
             cmd.append(".")
             
        subprocess.check_call(cmd)
        print("\nSuccess! You can now use 'bitvoice' as a command or 'import bitvoice' in Python.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")

def install_f5_tts_deps() -> None:
    """Install heavy dependencies for F5-TTS using library extras."""
    print("Installing F5-TTS dependencies via extras...")
    import subprocess
    try:
        # We try to install via pip install .[f5]
        subprocess.check_call([sys.executable, "-m", "pip", "install", ".[f5]"])
        print("\n[SUCCESS] F5-TTS dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")

# --- Main CLI ---
def main() -> None:
    parser = argparse.ArgumentParser(description="BitVoice: Convert text/doc files to Speech.")
    parser.add_argument("--input", "-i", type=str, help="Input directory or single file.")
    parser.add_argument("--output", "-o", type=str, help="Output directory or filename.")
    parser.add_argument("--model", "-m", type=str, default="kokoro", help="TTS Model (kokoro, piper, etc).")
    parser.add_argument("--voice", "-v", type=str, help="Voice name.")
    parser.add_argument("--parallel", "-p", action="store_true", help="Enable parallel processing.")
    parser.add_argument("--install", action="store_true", help="Install CLI wrapper script (Legacy).")
    parser.add_argument("--install-library", action="store_true", help="Install as a Python library.")
    parser.add_argument("--install-f5-tts", action="store_true", help="Install F5-TTS dependencies (Heavy).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()

    # Configure Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    if args.install_f5_tts:
        install_f5_tts_deps()
        return

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
        logger.error(f"Error: {input_path} not found.")
        return
        
    is_single_file = input_path.is_file()
    supported_exts = {'.md', '.txt', '.pdf', '.docx', '.epub'}
    
    files_to_process: List[Path] = []
    if is_single_file:
         if input_path.suffix.lower() in supported_exts: files_to_process = [input_path]
         else: 
             logger.warning(f"Unknown extension {input_path.suffix}, trying anyway.")
             files_to_process = [input_path]
    else:
        for ext in supported_exts:
             files_to_process.extend(input_path.rglob(f"*{ext}"))
             
    if not files_to_process:
        logger.info("No supported files found.")
        return

    # Logic for processing
    try:
        dummy = get_engine(args.model) # Validate model
        # Get voices (try/except for robustness)
        try: 
            available_voices = dummy.get_voices()
        except Exception as e: 
            logger.debug(f"Could not get voices: {e}")
            available_voices = ["default"]
            
        voice = args.voice
        if not voice:
             if args.model == "kokoro": voice = "af_heart"
             elif args.model in ["piper", "f5-tts", "xtts"]: voice = "default"
             else: voice = "default"
        logger.info(f"Using Model: {args.model}, Voice: {voice}")
    except Exception as e:
        logger.error(f"Error initializing model {args.model}: {e}")
        return

    # Cache load
    if not os.path.exists(CONF.CACHE_DIR): os.makedirs(CONF.CACHE_DIR)
    cache: Dict[str, str] = {}
    if os.path.exists(CONF.cache_path):
        try:
            with open(CONF.cache_path, 'rb') as f:
                cache = pickle.load(f)
        except Exception as e:
            logger.debug(f"Could not load cache: {e}")

    work_items: List[Tuple[str, str, str, str, str, str]] = []
    for file_path in files_to_process:
        # Skip cache files and output files
        if CONF.CACHE_DIR in file_path.parts: continue
        if args.output and Path(args.output).resolve() == file_path.resolve(): continue
        
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
             
             f_hash = get_file_hash(cleaned, args.model, str(voice))
             if cache.get(str(file_path)) == f_hash and audio_file.exists():
                 logger.info(f"Skipping cached: {file_path.name}")
                 continue
                 
             work_items.append((cleaned, args.model, str(voice), str(audio_file), str(file_path), f_hash))
        except Exception as e:
             logger.error(f"Skipping {file_path}: {e}")

    if not work_items: 
        logger.info("Nothing to do.")
        return

    logger.info(f"Processing {len(work_items)} items...")
    
    if args.parallel and args.model not in ["f5-tts", "xtts"]:
         from concurrent.futures import ProcessPoolExecutor
         with ProcessPoolExecutor() as executor:
             results = list(executor.map(process_single_item, [(w[0], w[1], w[2], w[3]) for w in work_items]))
    else:
         results = []
         for w in work_items:
             results.append(process_single_item((w[0], w[1], w[2], w[3])))
             logger.info(f"Generated {Path(w[3]).name}")

    for i, (ok, err) in enumerate(results):
        if ok: cache[work_items[i][4]] = work_items[i][5]
        else: logger.error(f"Failed {work_items[i][4]}: {err}")
        
    with open(CONF.cache_path, 'wb') as f: pickle.dump(cache, f)
    logger.info("Done.")

if __name__ == "__main__":
    main()
