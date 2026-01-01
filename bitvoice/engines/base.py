from typing import List, Tuple, Any
from abc import ABC, abstractmethod

class TTSEngine(ABC):
    @abstractmethod
    def get_voices(self) -> List[Tuple[str, str]]:
        """Return list of (voice_id, voice_description)."""
        pass

    @abstractmethod
    def generate(self, text: str, voice: str, output_path: str) -> None:
        """Generate audio from text."""
        pass
