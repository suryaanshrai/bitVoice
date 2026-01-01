import os
import logging
from dataclasses import dataclass

# --- Logging Setup ---
logger = logging.getLogger("bitvoice")

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
    
    @property
    def cache_path(self) -> str:
        return os.path.join(self.CACHE_DIR, self.CACHE_FILENAME)

CONF = Settings()

# Ensure Hugging Face models use persistent volume in Docker
# This must be run before importing transformers/diffusers
if os.path.exists("/app/models"):
    os.environ["HF_HOME"] = "/app/models/huggingface"
