from typing import Dict
from .base import TTSEngine

from .chatterbox import ChatterboxEngine, ChatterboxTurboEngine

MODEL_INFO = {
    "chatterbox": {"desc": "Chatterbox TTS (Voice cloning capabilities)."},
    "chatterbox-turbo": {"desc": "Chatterbox Turbo TTS (Fast + Cloning)."}
}

def get_engine(model_name: str) -> TTSEngine:
    if model_name == "chatterbox": return ChatterboxEngine()
    elif model_name == "chatterbox-turbo": return ChatterboxTurboEngine()
    elif model_name in MODEL_INFO: raise ValueError(f"Model {model_name} defined but not implemented.")
    else: raise ValueError(f"Unknown model: {model_name}")
