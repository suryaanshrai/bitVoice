import pytest
import sys
import os

# Ensure bitvoice is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def temp_files(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello\n This is **bold**.", encoding="utf-8")
    
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Simple text.", encoding="utf-8")
    
    return md_file, txt_file
