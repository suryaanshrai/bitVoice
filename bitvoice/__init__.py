import os
import logging
from typing import Optional

from .engines import get_engine, TTSEngine
from .config import CONF
from .utils.files import read_file_content
from .utils.text import clean_markdown

logger = logging.getLogger("bitvoice")

class BitVoice:
    """
    Main entry point for using BitVoice as a library.
    Example:
      bv = BitVoice(model='kokoro', voice='af_heart')
      bv.convert_file('book.txt', 'audio.wav')
    """
    def __init__(self, model: str = "chatterbox", voice: Optional[str] = None):
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
                    self.voice = voices[0][0] if voices else "default"
                except Exception as e:
                    logger.debug(f"Could not autoset voice: {e}")
                    self.voice = "default"
    
    def convert_text(self, text: str, output_path: str) -> None:
        self._init_engine()
        assert self.engine is not None
        self.engine.generate(text, self.voice or "default", output_path)

    def convert_file(self, input_file: str, output_path: str) -> None:
        file_path = os.path.abspath(input_file) # Ensure absolute for reading if needed, though read_file_content handles Path
        from pathlib import Path
        p = Path(file_path)
        
        text = read_file_content(p)
        if not text:
            raise ValueError(f"Could not read text from {input_file}")
        cleaned = clean_markdown(text, filename=p.name)
        self.convert_text(cleaned, output_path)
