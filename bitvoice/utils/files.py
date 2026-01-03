import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Callable

logger = logging.getLogger("bitvoice")

# --- Hashing ---
def get_file_hash(content: str, model_name: str, voice_name: str) -> str:
    """Generate SHA256 hash for content and parameters."""
    full_string = f"{content}|{model_name}|{voice_name}"
    return hashlib.sha256(full_string.encode('utf-8')).hexdigest()

# --- File Reading ---
def _read_text_file(path: Path) -> Optional[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

FILE_HANDLERS: Dict[str, Callable[[Path], Optional[str]]] = {
    '.md': _read_text_file
}

def read_file_content(file_path: Path) -> Optional[str]:
    """Read text from supported file formats."""
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
