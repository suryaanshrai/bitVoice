import os
import torch
import torchaudio
import logging
from typing import List, Tuple
from .base import TTSEngine

logger = logging.getLogger("bitvoice")

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class ChatterboxEngine(TTSEngine):
    def __init__(self):
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            raise ImportError("chatterbox-tts (or torchaudio) not installed.")
        
        self.device = get_device()
        logger.info(f"Chatterbox: Loading model on {self.device}")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)

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
        except ImportError:
            raise ImportError("chatterbox-tts (or torchaudio) not installed.")
        
        self.device = get_device()
        logger.info(f"Chatterbox Turbo: Loading model on {self.device}")
        self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)

    def get_voices(self) -> List[Tuple[str, str]]:
        return [("turbo", "Chatterbox Turbo (requires reference clip for best results)")]

    def generate(self, text: str, voice: str, output_path: str) -> None:
         prompt_path = None
         if voice and os.path.exists(voice) and os.path.isfile(voice):
             prompt_path = voice
         
         if prompt_path:
              logger.info(f"Chatterbox Turbo: Using prompt {prompt_path}")
              wav = self.model.generate(text, audio_prompt_path=prompt_path)
         else:
              logger.warning("Chatterbox Turbo: No voice prompt provided. Creating simple generation (might vary).")
              try:
                  wav = self.model.generate(text)
              except Exception as e:
                  raise ValueError(f"Chatterbox Turbo likely requires a voice prompt file. Please specify one with -v. Error: {e}")
         
         torchaudio.save(output_path, wav, self.model.sr)
