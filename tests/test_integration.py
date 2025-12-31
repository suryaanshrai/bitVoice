import pytest
import sys
import os
from unittest.mock import MagicMock, patch, mock_open
import bitvoice
from bitvoice import BitVoice

# --- Library Integration ---
@patch("bitvoice.get_engine")
def test_bitvoice_convert_text(mock_get_gen):
    mock_engine_instance = MagicMock()
    mock_engine_instance.get_voices.return_value = ["default"]
    mock_get_gen.return_value = mock_engine_instance
    
    bv = BitVoice(model="test_model", voice="test_voice")
    bv.convert_text("Hello World", "out.wav")
    
    mock_get_gen.assert_called_with("test_model")
    mock_engine_instance.generate.assert_called_with("Hello World", "test_voice", "out.wav")

@patch("bitvoice.get_engine")
def test_bitvoice_convert_file(mock_get_gen, temp_files):
    md_file, _ = temp_files
    mock_engine_instance = MagicMock()
    mock_get_gen.return_value = mock_engine_instance
    
    bv = BitVoice(model="test_model")
    bv.convert_file(md_file, "out.wav")
    
    mock_engine_instance.generate.assert_called()
    args = mock_engine_instance.generate.call_args
    assert "Hello" in args[0][0]

# --- CLI Integration ---

@patch("sys.argv", ["bitvoice.py", "--install"])
@patch("bitvoice.install_tool")
def test_cli_install_flag(mock_install):
    # Fixed: Removed raises SystemExit because the function returns normally
    bitvoice.main()
    mock_install.assert_called_once()

@patch("sys.argv", ["bitvoice.py", "--install-library"])
@patch("bitvoice.install_library_package")
def test_cli_install_library_flag(mock_install_lib):
    bitvoice.main()
    mock_install_lib.assert_called_once()

@patch("sys.argv", ["bitvoice.py", "--install-f5-tts"])
@patch("bitvoice.install_f5_tts_deps")
def test_cli_install_f5_flag(mock_install_f5):
    bitvoice.main()
    mock_install_f5.assert_called_once()

@patch("subprocess.check_call")
@patch("sys.executable", "/usr/bin/python3")
@patch("builtins.input", return_value="y")
def test_install_library_package_function(mock_input, mock_check_call):
    with patch("sys.prefix", "/usr"), patch("sys.base_prefix", "/usr"):
        bitvoice.install_library_package()
        mock_check_call.assert_called_with(["/usr/bin/python3", "-m", "pip", "install", "-e", "."])

@patch("builtins.open", new_callable=mock_open)
@patch("os.chmod")
def test_install_tool_function(mock_chmod, mock_file):
    with patch("os.name", "posix"):
        bitvoice.install_tool()
        # In mock_open, we check if write was called
        # mock_file() returns the file handle
        mock_file().write.assert_called()
        args = mock_file().write.call_args[0][0]
        assert "#!/bin/bash" in args

@patch("sys.argv", ["bitvoice.py", "--input", "test.md", "--model", "kokoro", "--voice", "af_heart"])
@patch("bitvoice.Path.exists", return_value=True)
@patch("bitvoice.Path.is_file", return_value=True) # It IS a file
@patch("bitvoice.read_file_content", return_value="Test Content")
@patch("bitvoice.get_engine")
def test_main_cli_execution_flow(mock_get_eng, mock_read, mock_isfile, mock_exists):
    # Fixed Path issues by mocking Path object properties or ensure logic simpler
    # The previous error was ValueError: WindowsPath('.') has an empty name
    # This happens if code does Path(".") and expects it to be a file with stem.
    # In this test we pass "test.md", so Path("test.md").
    
    # We must properly mock Path so that input_path.suffix works
    # However, since we mock Path class in the code via `bitvoice.Path`, it might get messy.
    # Better to NOT mock Path class globally, but rely on real Path for string inputs.
    # But we want to avoid disk I/O.
    # Let's rely on `read_file_content` mock and `is_file`.
    
    # Actually, bitvoice.py uses `Path(args.input)`. If we mock `bitvoice.Path`, `Path("test.md")` returns a MagicMock.
    # MagicMock().suffix is another MagicMock.
    # We should let Path be real, but mock `.exists()` and `.is_file()` on the instance?
    # No, hard to patch instance methods of real classes globally easily without side effects.
    
    # Alternative strategy: Use real files (via tmp_path) and mock only the expensive parts (engine, read_file if needed).
    pass 

# Rewriting test_main_cli_execution_flow to use real Path but mocked engine
@patch("bitvoice.get_engine")
def test_main_cli_e2e(mock_get_eng, temp_files):
    md_file, _ = temp_files
    mock_eng = MagicMock()
    mock_get_eng.return_value = mock_eng
    mock_eng.get_voices.return_value = ["af_heart"]
    
    # We construct argv with the temp file path
    with patch("sys.argv", ["bitvoice.py", "--input", str(md_file), "--model", "kokoro", "--output", str(md_file.parent / "out.wav")]):
        # We need to preserve pickle/cache
        with patch("pickle.dump"), patch("pickle.load"): 
             # We want real file reading this time to test that flow
             bitvoice.main()
             
    mock_get_eng.assert_called_with("kokoro")
    mock_eng.generate.assert_called()
