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
def test_read_md_txt(temp_files: Tuple[Path, Path]) -> None:
    md, txt = temp_files
    assert "Hello" in (read_file_content(md) or "")
    assert "**bold**" in (read_file_content(md) or "")
    assert "Simple text" in (read_file_content(txt) or "")

@patch("pypdf.PdfReader")
def test_read_pdf(mock_pdf_reader: MagicMock, tmp_path: Path) -> None:
    f = tmp_path / "test.pdf"
    f.touch()
    
    # Mock page content
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Page Content"
    
    mock_pdf_instance = mock_pdf_reader.return_value
    mock_pdf_instance.pages = [mock_page]
    
    content = read_file_content(f)
    assert content is not None
    assert "PDF Page Content" in content

@patch("docx.Document")
def test_read_docx(mock_document: MagicMock, tmp_path: Path) -> None:
    f = tmp_path / "test.docx"
    f.touch()
    
    mock_doc_instance = mock_document.return_value
    p1 = MagicMock()
    p1.text = "Docx Para 1"
    mock_doc_instance.paragraphs = [p1]
    
    content = read_file_content(f)
    assert content is not None
    assert "Docx Para 1" in content

def test_read_epub(tmp_path: Path) -> None:
    f = tmp_path / "test.epub"
    f.touch()
    
    # Create manual mocks
    mock_epub_module = MagicMock()
    mock_book = mock_epub_module.read_epub.return_value
    
    mock_item = MagicMock()
    mock_item.get_type.return_value = 9 
    mock_item.get_content.return_value = b"<html>Content</html>"
    mock_book.get_items.return_value = [mock_item]
    
    # Mock BS4
    mock_bs_module = MagicMock()
    mock_soup = mock_bs_module.BeautifulSoup.return_value
    mock_soup.get_text.return_value = "Parsed Epub Content"
    
    # Mock Parent ebooklib
    mock_ebooklib = MagicMock()
    mock_ebooklib.ITEM_DOCUMENT = 9
    mock_ebooklib.epub = mock_epub_module
    
    # Comprehensive sys.modules patch
    with patch.dict(sys.modules, {
        'ebooklib': mock_ebooklib, 
        'ebooklib.epub': mock_epub_module,
        'bs4': mock_bs_module
    }):
         content = read_file_content(f)
         assert content is not None
         assert "Parsed Epub Content" in content

def test_read_unsupported(tmp_path: Path) -> None:
    f = tmp_path / "test.xyz"
    f.touch()
    assert read_file_content(f) is None

# --- Unit Tests: Worker Logic ---
@patch("bitvoice.cli.get_engine")
def test_process_single_item_success(mock_get_eng: MagicMock) -> None:
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    
    item = ("text", "piper", "voice", "out.wav", "src.txt", "hash", {})
    
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
    
    item = ("text", "piper", "voice", "out.wav", "src.txt", "hash", {})
    with patch("bitvoice.cli._worker_engine", None):
        success, err = process_single_item(item)
    
    assert success is False
    assert "Engine Load Failed" in (err or "")

# --- Unit Tests: Settings ---
def test_settings_paths() -> None:
    assert CONF.cache_path.endswith("cache.pkl")
