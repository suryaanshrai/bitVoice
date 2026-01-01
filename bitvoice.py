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
# --- Download Utils ---
def download_file(url: str, dest_path: str) -> None:
    import urllib.request
    
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:
                print(f"\rDownloading: {percent}%", end="")
        
        urllib.request.urlretrieve(url, dest_path, progress)
        print("\rDownload Complete!      ")
        logger.info(f"Downloaded {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(dest_path): os.remove(dest_path)
        raise

# --- Engines ---
class TTSEngine:
    def get_voices(self) -> List[Tuple[str, str]]: raise NotImplementedError
    def generate(self, text: str, voice: str, output_path: str) -> None: raise NotImplementedError

class KokoroEngine(TTSEngine):
    def __init__(self):
        # Check baked-in path first (Docker defaults)
        baked_model = "/app/models/kokoro-v1.0.onnx"
        baked_voices = "/app/models/voices-v1.0.bin"
        
        self.model_path = CONF.kokoro_model_path
        self.voices_path = CONF.kokoro_voices_path

        if os.path.exists(baked_model) and os.path.exists(baked_voices):
            logger.info(f"Using baked-in Kokoro model at {baked_model}")
            self.model_path = baked_model
            self.voices_path = baked_voices
        else:
             # Auto-download model if missing locally
             if not os.path.exists(self.model_path):
                  os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                  logger.info("Kokoro model not found. Downloading...")
                  download_file("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", self.model_path)

             if not os.path.exists(self.voices_path):
                  os.makedirs(os.path.dirname(self.voices_path), exist_ok=True)
                  logger.info("Kokoro voices not found. Downloading...")
                  download_file("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", self.voices_path)

        try:
            from kokoro_onnx import Kokoro
            if os.path.exists(self.model_path):
                self.kokoro = Kokoro(self.model_path, self.voices_path)
            else:
                 raise FileNotFoundError(f"Kokoro model not found at {self.model_path}")
        except ImportError:
            raise ImportError("kokoro-onnx not installed.")
    
    def get_voices(self) -> List[Tuple[str, str]]:
        # Hardcoded descriptions for standard Kokoro voices if available
        descriptions = {
            "af_heart": "American Female - Heart", "af_alloy": "American Female - Alloy",
            "af_aoede": "American Female - Aoede", "af_bella": "American Female - Bella",
            "af_jessica": "American Female - Jessica", "af_kore": "American Female - Kore",
            "af_murphy": "American Female - Murphy", "af_nicole": "American Female - Nicole",
            "af_river": "American Female - River", "af_sarah": "American Female - Sarah",
            "af_sky": "American Female - Sky", "am_adam": "American Male - Adam",
            "am_echo": "American Male - Echo", "am_eric": "American Male - Eric",
            "am_fenrir": "American Male - Fenrir", "am_liam": "American Male - Liam",
            "am_michael": "American Male - Michael", "am_onyx": "American Male - Onyx",
            "am_puck": "American Male - Puck", "bf_alice": "British Female - Alice",
            "bf_emma": "British Female - Emma", "bf_isabella": "British Female - Isabella",
            "bf_lily": "British Female - Lily", "bm_daniel": "British Male - Daniel",
            "bm_fable": "British Male - Fable", "bm_george": "British Male - George",
            "bm_lewis": "British Male - Lewis"
        }
        raw_voices = self.kokoro.get_voices()
        return [(v, descriptions.get(v, "Kokoro Voice")) for v in raw_voices]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        sf.write(output_path, samples, sample_rate)

class Pyttsx3Engine(TTSEngine):
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
    def get_voices(self) -> List[Tuple[str, str]]:
        return [(v.name, v.id) for v in self.engine.getProperty('voices')]
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
        # Auto-download model if missing
        if not os.path.exists(CONF.piper_model_path):
            os.makedirs(os.path.dirname(CONF.piper_model_path), exist_ok=True)
            logger.info("Piper model not found. Downloading default (en_US-lessac-medium)...")
            download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx", CONF.piper_model_path)
        
        if not os.path.exists(CONF.piper_config_path):
            os.makedirs(os.path.dirname(CONF.piper_config_path), exist_ok=True)
            download_file("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json", CONF.piper_config_path)

        if not os.path.exists(CONF.piper_model_path): raise FileNotFoundError(f"Piper model not found at {CONF.piper_model_path}")
        try:
            from piper import PiperVoice
            self.voice = PiperVoice.load(CONF.piper_model_path, config_path=CONF.piper_config_path)
        except ImportError: raise ImportError("piper-tts not installed.")
    def get_voices(self) -> List[Tuple[str, str]]: return [("default", "Default Piper Voice")]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        with open(output_path, "wb") as wav_file:
            self.voice.synthesize(text, wav_file)

class GTTSEngine(TTSEngine):
    def __init__(self):
        from gtts import gTTS
        self.gTTS = gTTS
    def get_voices(self) -> List[Tuple[str, str]]:
        return [("en", "English"), ("fr", "French"), ("es", "Spanish")]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        tts = self.gTTS(text=text, lang=voice if voice in self.get_voices() else "en")
        tts.save(output_path)

class F5TTSEngine(TTSEngine):
    def __init__(self):
        from f5_tts.api import F5TTS
        from importlib.resources import files
        self.files = files
        self.f5tts = F5TTS()
    def get_voices(self) -> List[Tuple[str, str]]: return [("default", "Reference Audio (F5-TTS)")]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        ref_file = voice
        if voice == "default" or not os.path.exists(voice):
             ref_file = str(self.files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        self.f5tts.infer(ref_file=ref_file, ref_text="Ref text.", gen_text=text, file_wave=output_path)

class XTTSEngine(TTSEngine):
    def __init__(self):
        from TTS.api import TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    def get_voices(self) -> List[Tuple[str, str]]: return [("default", "Default Speaker")]
    def generate(self, text: str, voice: str, output_path: str) -> None:
        speaker_wav = voice if os.path.exists(voice) else None
        if not speaker_wav: self.tts.tts_to_file(text=text, file_path=output_path, speaker=self.tts.speakers[0], language="en")
        else: self.tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language="en")

class FishSpeechEngine(TTSEngine):
    def get_voices(self) -> List[Tuple[str, str]]: return []
    def generate(self, text: str, voice: str, output_path: str) -> None: raise NotImplementedError("Fish Speech not manually configured.")

class MeloTTSEngine(TTSEngine):
    def __init__(self):
        try:
            from melo.api import TTS
        except ImportError:
            raise ImportError("MeloTTS not installed. Please install 'git+https://github.com/myshell-ai/MeloTTS.git'")
        # Initialize with English by default
        self.model = TTS(language='EN', device='auto')
        
    def get_voices(self) -> List[Tuple[str, str]]:
        if hasattr(self.model, 'hps') and hasattr(self.model.hps.data, 'spk2id'):
             return [(k, f"MeloTTS {k}") for k in self.model.hps.data.spk2id.keys()]
        return [("EN-US", "MeloTTS EN-US (Default)")]

    def generate(self, text: str, voice: str, output_path: str) -> None:
        speaker_id = 0
        if hasattr(self.model, 'hps') and hasattr(self.model.hps.data, 'spk2id'):
             speaker_id = self.model.hps.data.spk2id.get(voice, self.model.hps.data.spk2id.get('EN-US', 0))
        self.model.tts_to_file(text, speaker_id, output_path, speed=1.0)

class ChatterboxEngine(TTSEngine):
    def __init__(self):
        try:
            from chatterbox import Chatterbox
        except ImportError:
            raise ImportError("Chatterbox not installed.")
        self.cb = Chatterbox()

    def get_voices(self) -> List[Tuple[str, str]]:
        return [("default", "Chatterbox Default")]

    def generate(self, text: str, voice: str, output_path: str) -> None:
        # Check if voice is a file path (Voice Cloning)
        if voice and os.path.exists(voice) and os.path.isfile(voice):
            logger.info(f"Chatterbox: Cloning voice from {voice}")
            # chatterbox usually takes speaker_wav argument for cloning
            self.cb.tts(text, output_path, speaker_wav=voice)
        else:
            # Default synthesis
            self.cb.tts(text, output_path)


MODEL_INFO = {
    "kokoro": {"desc": "High-quality, lightweight neural TTS (~150MB)."},
    "piper": {"desc": "Fast, local neural TTS (Requires download)."},
    "pyttsx3": {"desc": "Offline, system-native TTS (No download)."},
    "gtts": {"desc": "Google Translate TTS (Online only)."},
    "f5-tts": {"desc": "F5-TTS model (Heavy, requires GPU/strong CPU)."},
    "xtts": {"desc": "XTTS v2 (Heavy, high quality)."},
    "melo": {"desc": "MeloTTS (High quality, multilingual)."},
    "chatterbox": {"desc": "Chatterbox TTS (Voice cloning capabilities)."}
}

def get_engine(model_name: str) -> TTSEngine:
    if model_name == "kokoro": return KokoroEngine()
    elif model_name == "pyttsx3": return Pyttsx3Engine()
    elif model_name == "piper": return PiperEngine()
    elif model_name == "gtts": return GTTSEngine()
    elif model_name == "f5-tts": return F5TTSEngine()
    elif model_name == "xtts": return XTTSEngine()
    elif model_name == "melo": return MeloTTSEngine()
    elif model_name == "chatterbox": return ChatterboxEngine()
    elif model_name in MODEL_INFO: raise ValueError(f"Model {model_name} defined but not implemented.")
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
                    # voices is now List[Tuple[str, str]]
                    self.voice = "af_heart" if self.model == "kokoro" else (voices[0][0] if voices else "default")
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





# --- Main CLI ---
def main() -> None:
    parser = argparse.ArgumentParser(description="BitVoice: Convert text/doc files to Speech.")
    parser.add_argument("--input", "-i", type=str, help="Input directory or single file.")
    parser.add_argument("--output", "-o", type=str, help="Output directory or filename.")
    parser.add_argument("--model", "-m", type=str, default="kokoro", help="TTS Model (kokoro, piper, etc).")
    parser.add_argument("--voice", "-v", type=str, help="Voice name.")
    parser.add_argument("--parallel", "-p", action="store_true", help="Enable parallel processing.")


    parser.add_argument("--model-list", action="store_true", help="List supported TTS models.")
    parser.add_argument("--voice-list", type=str, help="List voices for a specific model.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()

    if args.model_list:
        print("Supported Models:")
        for model in MODEL_INFO:
            print(f" - {model}: {MODEL_INFO[model]['desc']}")
        return

    if args.voice_list:
        model_name = args.voice_list
        print(f"Listing voices for {model_name}...")
        try:
            engine = get_engine(model_name)
            voices = engine.get_voices()
            print(f"Found {len(voices)} voices:")
            for v_id, v_desc in voices:
                print(f" - {v_id}: {v_desc}")
        except Exception as e:
            print(f"Error listing voices: {e}")
        return

    # Configure Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )




        
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
            available_voices_info = dummy.get_voices()
            available_voices = [v[0] for v in available_voices_info]
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
    
    results = [None] * len(work_items)
    total_items = len(work_items)

    if args.parallel and args.model not in ["f5-tts", "xtts"]:
         from concurrent.futures import ProcessPoolExecutor, as_completed
         with ProcessPoolExecutor() as executor:
             # Submit all tasks
             futures = {executor.submit(process_single_item, (w[0], w[1], w[2], w[3])): i for i, w in enumerate(work_items)}
             
             for future in as_completed(futures):
                 i = futures[future]
                 file_name = Path(work_items[i][4]).name
                 try:
                     res = future.result()
                     results[i] = res
                     status = "Success" if res[0] else "Failed"
                     logger.info(f"[{len([x for x in results if x])}/{total_items}] {status}: {file_name}")
                 except Exception as exc:
                     logger.error(f"[{len([x for x in results if x])}/{total_items}] Exception processing {file_name}: {exc}")
                     results[i] = (False, str(exc))
    else:
         for i, w in enumerate(work_items):
             file_name = Path(w[4]).name
             logger.info(f"[{i+1}/{total_items}] Switch Processing: {file_name}")
             res = process_single_item((w[0], w[1], w[2], w[3]))
             results[i] = res
             if res[0]:
                 logger.info(f"    -> Generated: {Path(w[3]).name}")
             else:
                 logger.error(f"    -> Failed: {res[1]}")

    logger.info("--- Summary ---")
    for i, res in enumerate(results):
        if not res: continue # Should not happen
        ok, err = res
        if ok: 
            cache[work_items[i][4]] = work_items[i][5]
        else: 
            logger.warning(f"Failed item: {Path(work_items[i][4]).name} - {err}")
        
    with open(CONF.cache_path, 'wb') as f: pickle.dump(cache, f)
    logger.info("Done.")

if __name__ == "__main__":
    main()
