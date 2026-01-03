import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Generator, Tuple

# Add parent directory to path so we can import bitvoice
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitvoice.utils.text import clean_markdown
from bitvoice.utils.files import get_file_hash, read_file_content
from bitvoice.cli import process_single_item
from bitvoice.config import CONF

# --- Unit Tests: Utilities ---
def test_get_file_hash() -> None:
    content = "test content"
    h1 = get_file_hash(content, "model1", "voice1")
    h2 = get_file_hash(content, "model1", "voice1")
    h3 = get_file_hash(content, "model2", "voice1")
    assert h1 == h2
    assert h1 != h3

# --- Unit Tests: File Reading (Mocked for Formats) ---
def test_read_md(temp_files: Tuple[Path, Path]) -> None:
    md, txt = temp_files
    assert "Hello" in (read_file_content(md) or "")
    assert "**bold**" in (read_file_content(md) or "")
    # txt file should return None as it is no longer supported
    assert read_file_content(txt) is None

# --- Unit Tests: Worker Logic ---
@patch("bitvoice.cli.get_engine")
def test_process_single_item_success(mock_get_eng: MagicMock) -> None:
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    
    item = ("text", "chatterbox", "voice", "out.wav", "src.txt", "hash", {})
    
    # Patch the global worker engine in cli to be None so it tries to load
    # Or set it to our mock
    with patch("bitvoice.cli._worker_engine", mock_eng):
        success, err = process_single_item(item)
    
    assert success is True
    assert err is None
    # Verify kwargs were passed if any, or just generate call
    mock_eng.generate.assert_called_with("text", "voice", "out.wav")

@patch("bitvoice.cli.get_engine")
def test_process_single_item_failure(mock_get_eng: MagicMock) -> None:
    # Scenario: _worker_engine is None, and get_engine fails
    mock_get_eng.side_effect = Exception("Engine Load Failed")
    
    item = ("text", "chatterbox", "voice", "out.wav", "src.txt", "hash", {})
    with patch("bitvoice.cli._worker_engine", None):
        success, err = process_single_item(item)
    
    assert success is False
    assert "Engine Load Failed" in (err or "")

# --- Unit Tests: Settings ---
def test_settings_paths() -> None:
    assert CONF.cache_path.endswith("cache.pkl")
