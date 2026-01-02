import argparse
import os
import sys
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .config import CONF
from .engines import get_engine, MODEL_INFO, TTSEngine
from .utils.text import clean_markdown
from .utils.files import read_file_content, get_file_hash

logger = logging.getLogger("bitvoice")

# --- Parallel Worker Logic ---
_worker_engine: Optional[TTSEngine] = None

def init_worker(model_name: str):
    """Initialize the engine once per worker process."""
    global _worker_engine
    try:
        _worker_engine = get_engine(model_name)
    except Exception as e:
        logger.error(f"Worker init failed: {e}")

def process_single_item(item_data: Tuple[str, str, str, str, str, str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    text, model_name, voice_name, output_path, src_path, hash_val, config = item_data
    
    # If running in parallel, use the global worker engine.
    # If running sequentially (and this function is called directly), init local engine if needed.
    # ideally sequential loop should just hold one engine and call generate.
    
    # HOWEVER, for the pool map, we rely on _worker_engine being set by init_worker.
    # If it's None, we might be in sequential mode or something went wrong.
    
    engine = _worker_engine
    if engine is None:
        # Fallback for when not using pool initializer (e.g. debugging or sequential generic call)
        try:
            engine = get_engine(model_name)
        except Exception as e:
            return False, f"Engine init failed: {e}"

    try:
        engine.generate(text, voice_name, output_path, **config)
        return True, None
    except Exception as e:
        logger.error(f"Error processing item: {e}")
        return False, str(e)

# --- CLI Helpers ---
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

    # Chatterbox Config
    parser.add_argument("--cb-speed", type=float, help="Chatterbox: Speaking rate/flow (0.0-1.0).", default=None)
    parser.add_argument("--cb-temp", type=float, help="Chatterbox: Temperature (0.0-1.0).", default=None)
    parser.add_argument("--cb-exag", type=float, help="Chatterbox: Exaggeration (0.0-1.0).", default=None)
    
    args = parser.parse_args()

    # Logging config
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    if args.model_list:
        print("Supported Models:")
        for model in MODEL_INFO:
            print(f" - {model}: {MODEL_INFO[model]['desc']}")
        return

    if args.voice_list:
        print(f"Listing voices for {args.voice_list}...")
        try:
            engine = get_engine(args.voice_list)
            voices = engine.get_voices()
            print(f"Found {len(voices)} voices:")
            for v_id, v_desc in voices:
                print(f" - {v_id}: {v_desc}")
        except Exception as e:
            print(f"Error listing voices: {e}")
        return

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
        
        # 2. Determine Output Path
        output_path_arg = None
        if args.output:
             output_path_arg = validate_in_cwd(args.output)

        output_root: Path
        if is_single_file:
            if output_path_arg:
                output_root = output_path_arg
                if output_root.is_dir():
                     output_root = output_root / input_path.with_suffix(".wav").name
            else:
                output_root = Path.cwd() / input_path.with_suffix(".wav").name
        else:
             output_root = output_path_arg if output_path_arg else Path.cwd() / "output"

        # 3. Determine Cache Location
        cache_root = input_path.parent if is_single_file else input_path
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

    # Validate model once
    try:
        dummy = get_engine(args.model)
        try: 
            available_voices_info = dummy.get_voices()
            available_voices = [v[0] for v in available_voices_info]
        except Exception as e: 
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
        if ".bitvoice_cache" in file_path.parts: continue
        
        if is_single_file:
            audio_file = output_root
        else:
            try: rel = file_path.relative_to(input_path)
            except: rel = file_path.name
            audio_file = output_root / Path(rel).with_suffix(".wav")
            
        if audio_file.resolve() == file_path.resolve():
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
                 
             work_items.append((cleaned, args.model, str(voice), str(audio_file), str(file_path), f_hash, {
                 "speed": args.cb_speed,
                 "temperature": args.cb_temp,
                 "exaggeration": args.cb_exag
             }))
        except Exception as e:
             logger.error(f"Skipping {file_path}: {e}")

    if not work_items: 
        logger.info("Nothing to do.")
        return

    logger.info(f"Processing {len(work_items)} items...")
    
    results = [None] * len(work_items)
    
    if args.parallel:
         from concurrent.futures import ProcessPoolExecutor, as_completed
         # Use init_worker to load model once per process
         with ProcessPoolExecutor(initializer=init_worker, initargs=(args.model,)) as executor:
             futures = {executor.submit(process_single_item, w): i for i, w in enumerate(work_items)}
             for future in as_completed(futures):
                 i = futures[future]
                 file_name = Path(work_items[i][4]).name
                 try:
                     res = future.result()
                     results[i] = res
                     status = "Success" if res[0] else "Failed"
                     logger.info(f"[{len([x for x in results if x])}/{len(work_items)}] {status}: {file_name}")
                 except Exception as exc:
                     logger.error(f"Exception processing {file_name}: {exc}")
                     results[i] = (False, str(exc))
    else:
        # Sequential: Reuse the dummy engine we created earlier, or create one if we didn't assign it to dummy
        # Actually dummy was created inside local scope. Let's create one.
        main_engine = get_engine(args.model) 
        for i, w in enumerate(work_items):
             file_name = Path(w[4]).name
             logger.info(f"[{i+1}/{len(work_items)}] Processing: {file_name}")
             # We can manually call generate on the engine instance to reuse it
             try:
                 main_engine.generate(w[0], w[2], w[3], **w[6])
                 results[i] = (True, None)
                 logger.info(f"    -> Generated: {Path(w[3]).name}")
             except Exception as e:
                 logger.error(f"    -> Failed: {e}")
                 results[i] = (False, str(e))

    logger.info("--- Summary ---")
    for i, res in enumerate(results):
        if not res: continue
        ok, err = res
        if ok: 
            cache[work_items[i][4]] = work_items[i][5]
        else: 
            logger.warning(f"Failed item: {Path(work_items[i][4]).name} - {err}")
        
    with open(CONF.cache_path, 'wb') as f: pickle.dump(cache, f)
    logger.info("Done.")
