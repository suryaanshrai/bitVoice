import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from typing import Tuple
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitvoice import BitVoice
from bitvoice.cli import main

# --- Library Integration ---
@patch("bitvoice.get_engine")
def test_bitvoice_convert_text(mock_get_eng: MagicMock) -> None:
    mock_engine_instance = MagicMock()
    mock_engine_instance.get_voices.return_value = [("default", "desc")]
    mock_get_eng.return_value = mock_engine_instance
    
    bv = BitVoice(model="test_model", voice="test_voice")
    bv.convert_text("Hello World", "out.wav")
    
    mock_get_eng.assert_called_with("test_model")
    mock_engine_instance.generate.assert_called_with("Hello World", "test_voice", "out.wav")

@patch("bitvoice.get_engine")
def test_bitvoice_convert_file(mock_get_eng: MagicMock, temp_files: Tuple[Path, Path]) -> None:
    md_file, _ = temp_files
    mock_engine_instance = MagicMock()
    mock_engine_instance.get_voices.return_value = [("default", "desc")]
    mock_get_eng.return_value = mock_engine_instance
    
    bv = BitVoice(model="test_model")
    bv.convert_file(str(md_file), "out.wav")
    
    mock_engine_instance.generate.assert_called()
    args = mock_engine_instance.generate.call_args
    assert "Hello" in args[0][0]

# --- CLI Integration ---
@patch("bitvoice.cli.validate_in_cwd", side_effect=lambda x: Path(x))
@patch("bitvoice.cli.get_engine")
def test_main_cli_e2e(mock_get_eng: MagicMock, mock_validate: MagicMock, temp_files: Tuple[Path, Path]) -> None:
    md_file, _ = temp_files
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    mock_eng.get_voices.return_value = [("piper_voice", "desc")]
    
    # We construct argv with the temp file path
    with patch("sys.argv", ["bitvoice", "--input", str(md_file), "--model", "piper", "--output", str(md_file.parent / "out.wav")]):
        # We need to preserve pickle/cache
        with patch("pickle.dump"), patch("pickle.load"): 
             main()
             
    mock_get_eng.assert_called_with("piper")
    mock_eng.generate.assert_called()

def test_main_model_list() -> None:
    # Test --model-list flag
    with patch("sys.argv", ["bitvoice", "--model-list"]), \
         patch("bitvoice.cli.MODEL_INFO", {"test_model": {"desc": "Test Description"}}), \
         patch("builtins.print") as mock_print:
        main()
        # Verify it printed the model info
        mock_print.assert_any_call(" - test_model: Test Description")

@patch("bitvoice.cli.get_engine")
def test_main_voice_list(mock_get_eng: MagicMock) -> None:
    # Test --voice-list flag
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    mock_eng.get_voices.return_value = [("voice1", "Voice 1 Info")]
    
    with patch("sys.argv", ["bitvoice", "--voice-list", "test_model"]), \
         patch("builtins.print") as mock_print:
        main()
        
    mock_get_eng.assert_called_with("test_model")
    mock_print.assert_any_call(" - voice1: Voice 1 Info")
