import os
import wave
import logging
import torchaudio
from typing import List, Tuple
from .base import TTSEngine
from ..config import CONF
from ..utils.files import download_file

logger = logging.getLogger("bitvoice")

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
             try:
                 base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
                 download_file(f"{base_url}/en_US-lessac-medium.onnx", self.default_onnx)
                 download_file(f"{base_url}/en_US-lessac-medium.onnx.json", self.default_json)
             except Exception as e:
                 logger.error(f"Failed to auto-download Piper model: {e}")

        try:
            from piper import PiperVoice
            # Just test import
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

    def generate(self, text: str, voice: str, output_path: str, **kwargs) -> None:
        from piper import PiperVoice
        
        voice_name = voice if voice and voice != "default" else self.default_voice_name
        onnx_path = os.path.join(self.voices_dir, f"{voice_name}.onnx")
        
        if not os.path.exists(onnx_path):
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
        
        # Load model for each generation (inefficient, but consistent with original unless we cache)
        # Optimization: In parallel processing, we might load once.
        model = PiperVoice.load(onnx_path, use_cuda=use_cuda)
        
        with wave.open(output_path, "wb") as wav_file:
            model.synthesize_wav(text, wav_file)
