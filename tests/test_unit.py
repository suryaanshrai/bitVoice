import pytest
import sys
from unittest.mock import MagicMock, patch
import bitvoice
from bitvoice import (
    clean_markdown,
    get_file_hash,
    read_file_content,
    get_engine,
    process_single_item
)

# --- Utilities ---
def test_clean_markdown():
    raw = "# Header\nSome **bold** and *italic* text.\n[Link](http://example.com) and `code`."
    expected = "Header\nSome bold and italic text.\nLink and code."
    assert clean_markdown(raw) == expected

def test_clean_markdown_images():
    raw = "![Image](img.png) Text."
    expected = "Text."
    assert clean_markdown(raw) == expected

def test_get_file_hash():
    content = "test content"
    h1 = get_file_hash(content, "model1", "voice1")
    h2 = get_file_hash(content, "model1", "voice1")
    h3 = get_file_hash(content, "model2", "voice1")
    assert h1 == h2
    assert h1 != h3

# --- File Reading ---
def test_read_md_txt(temp_files):
    md, txt = temp_files
    assert "Hello" in read_file_content(md)
    assert "**bold**" in read_file_content(md)
    assert "Simple text" in read_file_content(txt)

@patch("pypdf.PdfReader")
def test_read_pdf(mock_pdf_reader, tmp_path):
    f = tmp_path / "test.pdf"
    f.touch()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Page Content"
    mock_pdf_instance = mock_pdf_reader.return_value
    mock_pdf_instance.pages = [mock_page]
    assert "PDF Page Content" in read_file_content(f)

@patch("docx.Document")
def test_read_docx(mock_document, tmp_path):
    f = tmp_path / "test.docx"
    f.touch()
    mock_doc_instance = mock_document.return_value
    p1 = MagicMock()
    p1.text = "Docx Para 1"
    mock_doc_instance.paragraphs = [p1]
    assert "Docx Para 1" in read_file_content(f)

def test_read_epub(tmp_path):
    f = tmp_path / "test.epub"
    f.touch()
    
    mock_epub_module = MagicMock()
    mock_book = mock_epub_module.read_epub.return_value
    mock_item = MagicMock()
    mock_item.get_type.return_value = 9 
    mock_item.get_content.return_value = b"<html>Content</html>"
    mock_book.get_items.return_value = [mock_item]
    
    mock_bs_module = MagicMock()
    mock_soup = mock_bs_module.BeautifulSoup.return_value
    mock_soup.get_text.return_value = "Parsed Epub Content"
    
    mock_ebooklib = MagicMock()
    mock_ebooklib.ITEM_DOCUMENT = 9
    mock_ebooklib.epub = mock_epub_module
    
    with patch.dict(sys.modules, {
        'ebooklib': mock_ebooklib, 
        'ebooklib.epub': mock_epub_module,
        'bs4': mock_bs_module
    }):
         content = read_file_content(f)
         assert "Parsed Epub Content" in content

def test_read_unsupported(tmp_path):
    f = tmp_path / "test.xyz"
    f.touch()
    assert read_file_content(f) is None

# --- Logic & Mocks ---
@patch("bitvoice.KokoroEngine")
def test_get_engine_kokoro(mock_kokoro):
    engine = get_engine("kokoro")
    mock_kokoro.assert_called_once()

@patch("bitvoice.Pyttsx3Engine")
def test_get_engine_pyttsx3(mock_pyttsx3):
    engine = get_engine("pyttsx3")
    mock_pyttsx3.assert_called_once()

def test_get_engine_invalid():
    with pytest.raises(ValueError):
        get_engine("invalid_model")

# --- Worker Logic ---
@patch("bitvoice.get_engine")
def test_process_single_item_success(mock_get_eng):
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    success, err = process_single_item(("text", "kokoro", "voice", "out.wav"))
    assert success is True
    assert err is None
    mock_eng.generate.assert_called_with("text", "voice", "out.wav")

@patch("bitvoice.get_engine")
def test_process_single_item_failure(mock_get_eng):
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    mock_eng.generate.side_effect = Exception("Fail")
    success, err = process_single_item(("text", "kokoro", "voice", "out.wav"))
    assert success is False
    assert "Fail" in err
