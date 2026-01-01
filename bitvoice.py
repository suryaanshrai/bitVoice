#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import pickle
import re
import logging
import soundfile as sf
import torchaudio
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable, Tuple, Any

# Suppress some common warnings from libraries
warnings.filterwarnings("ignore")

# --- Configuration ---
# Ensure Hugging Face models (F5-TTS, Piper) use the persistent volume
# This must be set before importing transformers/diffusers or other HF libs
if os.path.exists("/app/models"):
    os.environ["HF_HOME"] = "/app/models/huggingface"


# --- Logging Setup ---
logger = logging.getLogger("bitvoice")

# --- Configuration ---
@dataclass
class Settings:
    CACHE_DIR: str = ".bitvoice_cache"
    # Logic to find models: 
    # 1. If /app/models exists (Docker), use that
    # 2. Else use ./models relative to CWD
    MODELS_DIR: str = "/app/models" if os.path.exists("/app/models") else "models"
    CACHE_FILENAME: str = "cache.pkl"
    PIPER_VOICE_DEFAULT: str = "en_US-lessac-medium"
    
    @property
    def piper_models_dir(self) -> str:
        return os.path.join(self.MODELS_DIR, "piper")

CONF = Settings()

# --- Utilities ---
# --- Utilities ---
def get_file_hash(content: str, model_name: str, voice_name: str) -> str:
    """Generate SHA256 hash for content and parameters."""
    full_string = f"{content}|{model_name}|{voice_name}"
    return hashlib.sha256(full_string.encode('utf-8')).hexdigest()

def clean_markdown(text: str, filename: Optional[str] = None) -> str:
    """Basic markdown cleaning for better TTS."""
    # Remove YAML frontmatter: --- ... ---
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
    
    # If filename is provided, prepend it as a header (without extension)
    if filename:
        # Strip extension if present
        name = os.path.splitext(filename)[0]
        text = f"# {name}\n\n{text}"

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

class PiperEngine(TTSEngine):
    def __init__(self):
        self.voices_dir = CONF.piper_models_dir
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir, exist_ok=True)
            
        # Check for default model
        self.default_voice_name = CONF.PIPER_VOICE_DEFAULT
        self.default_onnx = os.path.join(self.voices_dir, f"{self.default_voice_name}.onnx")
        self.default_json = os.path.join(self.voices_dir, f"{self.default_voice_name}.onnx.json")
        
        if not os.path.exists(self.default_onnx):
             logger.info(f"Piper model {self.default_voice_name} not found. Downloading...")
             # We can use piper.download_voices module mechanism or manual download. 
             # For robustness, we'll try to run the python module command.
             try:
                 import subprocess
                 # Downloading to current dir then moving? piping? 
                 # piper.download_voices downloads to CWD. We'll verify this behavior or support explicit paths if possible.
                 # Actually, let's use the list from huggingface or similar if we wanted, but subprocess is safer for the "official" way.
                 # However, to avoid side effects in CWD, we might change CWD temporarily or just rely on manual download URL if we knew it.
                 # Let's try downloading specifically:
                 base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
                 download_file(f"{base_url}/en_US-lessac-medium.onnx", self.default_onnx)
                 download_file(f"{base_url}/en_US-lessac-medium.onnx.json", self.default_json)
             except Exception as e:
                 logger.error(f"Failed to auto-download Piper model: {e}")

        try:
            from piper import PiperVoice
            # We don't load the voice here yet, we load it on generate or if a specific voice is requested
            # But for simplicity, we can load the default one to verify install
            pass 
        except ImportError:
            raise ImportError("piper-tts not installed.")
            
    def get_voices(self) -> List[Tuple[str, str]]:
        # List all .onnx files in the directory
        voices = []
        if os.path.exists(self.voices_dir):
            for f in os.listdir(self.voices_dir):
                if f.endswith(".onnx"):
                    name = f[:-5] # remove .onnx
                    voices.append((name, f"Piper Voice: {name}"))
        if not voices:
             voices.append((self.default_voice_name, "Default Piper Voice"))
        return voices

    def generate(self, text: str, voice: str, output_path: str) -> None:
        from piper import PiperVoice
        import wave
        
        voice_name = voice if voice and voice != "default" else self.default_voice_name
        onnx_path = os.path.join(self.voices_dir, f"{voice_name}.onnx")
        
        if not os.path.exists(onnx_path):
            # Fallback or error
            if voice_name == self.default_voice_name:
                 raise FileNotFoundError(f"Default model {onnx_path} missing.")
            else:
                 logger.warning(f"Voice {voice_name} not found, trying default.")
                 onnx_path = self.default_onnx

        # Use CUDA if available
        use_cuda = False
        try:
             import torch
             if torch.cuda.is_available(): use_cuda = True
        except: pass
        
        model = PiperVoice.load(onnx_path, use_cuda=use_cuda)
        
        with wave.open(output_path, "wb") as wav_file:
            model.synthesize_wav(text, wav_file)





class ChatterboxEngine(TTSEngine):
    def __init__(self):
        try:
            from chatterbox.tts import ChatterboxTTS
            import torch
        except ImportError:
            raise ImportError("chatterbox-tts (or torchaudio) not installed.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Chatterbox: Loading model on {device}")
        self.model = ChatterboxTTS.from_pretrained(device=device)

    def get_voices(self) -> List[Tuple[str, str]]:
        return [("default", "Chatterbox Default"), ("cloned", "Cloned Voice (provide path)")]

    def generate(self, text: str, voice: str, output_path: str) -> None:
        # Check if voice is a file path (Voice Cloning)
        if voice and os.path.exists(voice) and os.path.isfile(voice):
            logger.info(f"Chatterbox: Cloning voice from {voice}")
            wav = self.model.generate(text, audio_prompt_path=voice)
        else:
            # Default synthesis
            wav = self.model.generate(text)
            
        torchaudio.save(output_path, wav, self.model.sr)


class ChatterboxTurboEngine(TTSEngine):
    def __init__(self):
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            import torch
        except ImportError:
            raise ImportError("chatterbox-tts (or torchaudio) not installed.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Chatterbox Turbo: Loading model on {device}")
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)

    def get_voices(self) -> List[Tuple[str, str]]:
        return [("turbo", "Chatterbox Turbo (requires reference clip for best results)")]

    def generate(self, text: str, voice: str, output_path: str) -> None:
         # Turbo requires a reference clip often, or might work without? 
         # User code: wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
         # If no voice allowed, we might need a default reference or it fails.
         # For now, if voice is provided (path), usage it. 
         
         prompt_path = None
         if voice and os.path.exists(voice) and os.path.isfile(voice):
             prompt_path = voice
         
         if prompt_path:
              logger.info(f"Chatterbox Turbo: Using prompt {prompt_path}")
              wav = self.model.generate(text, audio_prompt_path=prompt_path)
         else:
              logger.warning("Chatterbox Turbo: No voice prompt provided. Creating simple generation (might vary).")
              # Try generating without prompt if library allows, or fail. 
              # Assuming generate() might take optional. If strictly required, we might fail.
              # Based on user snippet: "Generate audio (requires a reference clip for voice cloning)"
              # Let's try passing None or not passing it.
              try:
                  wav = self.model.generate(text)
              except Exception as e:
                  raise ValueError(f"Chatterbox Turbo likely requires a voice prompt file. Please specify one with -v. Error: {e}")
         
         torchaudio.save(output_path, wav, self.model.sr)


MODEL_INFO = {
    "piper": {"desc": "Fast, high-quality neural TTS (ONNX)."},
    "chatterbox": {"desc": "Chatterbox TTS (Voice cloning capabilities)."},
    "chatterbox-turbo": {"desc": "Chatterbox Turbo TTS (Fast + Cloning)."}
}

def get_engine(model_name: str) -> TTSEngine:
    if model_name == "piper": return PiperEngine()
    elif model_name == "chatterbox": return ChatterboxEngine()
    elif model_name == "chatterbox-turbo": return ChatterboxTurboEngine()
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
    def __init__(self, model: str = "piper", voice: Optional[str] = None):
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
                    self.voice = "en_US-lessac-medium" if self.model == "piper" else (voices[0][0] if voices else "default")
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
        cleaned = clean_markdown(text, filename=os.path.basename(input_file))
        self.convert_text(cleaned, output_path)





# --- Main CLI ---
def validate_in_cwd(path_str: str) -> Path:
    """Ensure path is within the current working directory."""
    try:
        p = Path(path_str).resolve()
        cwd = Path.cwd().resolve()
        if not p.is_relative_to(cwd):
             raise ValueError(f"Path '{path_str}' must be inside the current directory.")
        return p
    except Exception as e:
        raise ValueError(f"Invalid path '{path_str}': {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="BitVoice: Convert text/doc files to Speech.\nNOTE: Input and Output paths must be relative to the current directory.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", "-i", type=str, help="Input directory or single file (must be in current dir).")
    parser.add_argument("--output", "-o", type=str, help="Output directory or filename (must be in current dir).")
    parser.add_argument("--model", "-m", type=str, default="piper", help="TTS Model (piper, chatterbox, chatterbox-turbo).")
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

    try:
        # 1. Validate Input
        input_path = validate_in_cwd(args.input)
        if not input_path.exists():
            logger.error(f"Error: {input_path} not found.")
            return

        is_single_file = input_path.is_file()
        
        # 2. Determine Output Path & Behavior
        # Logic:
        # - Single file in, no output arg -> [filename].wav in CWD
        # - Single file in, output arg -> [output arg] as filename in CWD
        # - Dir in, no output arg -> ./output/ in CWD
        # - Dir in, output arg -> [output arg] as dir in CWD
        
        output_path_arg = None
        if args.output:
             output_path_arg = validate_in_cwd(args.output)

        output_root: Path
        
        if is_single_file:
            if output_path_arg:
                # User specified output filename
                output_root = output_path_arg
                if output_root.is_dir():
                     # Edge case: user gave a dir for a single file, append filename
                     output_root = output_root / input_path.with_suffix(".wav").name
            else:
                # Default: same basename, wav extension, in CWD
                output_root = Path.cwd() / input_path.with_suffix(".wav").name
        else:
             # Directory input
             if output_path_arg:
                 output_root = output_path_arg
             else:
                 output_root = Path.cwd() / "output"

        # 3. Determine Cache Location
        # Cache should be inside the input directory main folder
        if is_single_file:
            cache_root = input_path.parent
        else:
            cache_root = input_path
            
        CONF.CACHE_DIR = str(cache_root / ".bitvoice_cache")
        logger.debug(f"Cache directory set to: {CONF.CACHE_DIR}")

    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
        return

    # Gather files
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
             if args.model == "piper": voice = CONF.PIPER_VOICE_DEFAULT
             else: voice = "default"
        logger.info(f"Using Model: {args.model}, Voice: {voice}")
    except Exception as e:
        logger.error(f"Error initializing model {args.model}: {e}")
        return

    # Cache load
    if not os.path.exists(CONF.CACHE_DIR): os.makedirs(CONF.CACHE_DIR, exist_ok=True)
    cache: Dict[str, str] = {}
    if os.path.exists(CONF.cache_path):
        try:
            with open(CONF.cache_path, 'rb') as f:
                cache = pickle.load(f)
        except Exception as e:
            logger.debug(f"Could not load cache: {e}")

    work_items: List[Tuple[str, str, str, str, str, str]] = []
    for file_path in files_to_process:
        # Skip cache files and specific directory ignores
        # Specifically ignore .bitvoice_cache if it was picked up
        if ".bitvoice_cache" in file_path.parts: continue
        
        # Calculate Output Path
        if is_single_file:
            audio_file = output_root
        else:
            # Directory structure mirroring
            try: rel = file_path.relative_to(input_path)
            except: rel = file_path.name
            audio_file = output_root / Path(rel).with_suffix(".wav")
            
        # Avoid overwriting input file if something goes really wrong
        if audio_file.resolve() == file_path.resolve():
            logger.warning(f"Skipping {file_path.name}: Output would overwrite input.")
            continue

        audio_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
             content = read_file_content(file_path)
             if not content: continue
             cleaned = clean_markdown(content, filename=file_path.name)
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

    if args.parallel:
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
