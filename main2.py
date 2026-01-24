from datetime import datetime

start_time = datetime.now()

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import re
from markdown import markdown
from bs4 import BeautifulSoup
import torch
import unicodedata
import os, json, time
from pathlib import Path
import hashlib
import subprocess
import shutil


_SCRIPT_START_PERF = time.perf_counter()


def _format_duration_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs:0.1f}s"
    if minutes:
        return f"{minutes}m {secs:0.1f}s"
    return f"{secs:0.2f}s"


def _log_file_timing(*, rel_key: str, file_seconds: float, final_output: Path) -> None:
    overall_seconds = time.perf_counter() - _SCRIPT_START_PERF
    print(
        "\n\n"
        + f"TIMING | {rel_key} | file: {_format_duration_seconds(file_seconds)} | elapsed: {_format_duration_seconds(overall_seconds)} | output: {final_output}"
        + "\n\n"
    )


_MODEL: ChatterboxTTS | None = None


def get_model() -> ChatterboxTTS:
    global _MODEL
    if _MODEL is None:
        _MODEL = ChatterboxTTS.from_pretrained(device="cuda")
    return _MODEL


OUTPUT_DIR = "audios"
INPUT_DIR = "content"
CHUNK_LENGTH = 100

# Output compression settings
OUTPUT_AUDIO_EXT = ".mp3"
MP3_VBR_QUALITY = 6  # libmp3lame VBR quality: 0(best/largest) .. 9(worst/smallest)
MP3_RESAMPLE_HZ = 24000  # keep 24kHz to match MODEL.sr; set 44100 for max compatibility
DELETE_WAV_AFTER_COMPRESS = True
FFMPEG_BINARY = "ffmpeg"
# SETTINGS = {
#     "audio_prompt_path": "mysample.wav",
#     "exaggeration": 0.72, # [0.25 - 2]
#     "cfg_weight": 0.25, # [0.02 - 1]
#     "temperature": 0.15, # [0.05 - 5]
# }
SETTINGS = {
    "audio_prompt_path": "mysample.wav",
    "exaggeration": 0.675, # [0.25 - 2]
    "cfg_weight": 0.3, # [0.02 - 1]
    "temperature": 0.4, # [0.05 - 5]
}

HASH_JSON = "hashes.json"
HASHING = True

# Text cleanup controls
STRIP_EMOJIS = True
STRIP_OTHER_SYMBOLS = True

# Cache controls
# Hash the *source markdown* (normalized) instead of cleaned chunks so changes to the cleaner
# don't force a full regeneration.
CLEANING_VERSION = 2
REQUIRE_CLEANING_VERSION_MATCH_FOR_CACHE = False
ADOPT_EXISTING_MP3_WITHOUT_HASH_ENTRY = True


_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F1E6, 0x1F1FF),  # regional indicator symbols (flags)
    (0x1F300, 0x1F5FF),  # misc symbols & pictographs
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F680, 0x1F6FF),  # transport & map
    (0x1F700, 0x1F77F),  # alchemical symbols
    (0x1F780, 0x1F7FF),  # geometric extended
    (0x1F800, 0x1F8FF),  # supplemental arrows-c
    (0x1F900, 0x1F9FF),  # supplemental symbols & pictographs
    (0x1FA00, 0x1FA6F),  # chess symbols etc.
    (0x1FA70, 0x1FAFF),  # symbols & pictographs extended-a
    (0x2600, 0x26FF),    # misc symbols
    (0x2700, 0x27BF),    # dingbats
)

_EMOJI_SINGLETONS: set[int] = {
    0x200D,  # zero width joiner
    0x20E3,  # combining enclosing keycap
    0xFE0E,  # variation selector-15 (text)
    0xFE0F,  # variation selector-16 (emoji)
    0x200B,  # zero width space
    0x2060,  # word joiner
}


def _is_emoji_like(codepoint: int) -> bool:
    if codepoint in _EMOJI_SINGLETONS:
        return True
    # Skin tone modifiers
    if 0x1F3FB <= codepoint <= 0x1F3FF:
        return True
    return any(start <= codepoint <= end for start, end in _EMOJI_RANGES)


def sanitize_text_for_tts(text: str) -> str:
    """Remove emojis and other chars that tend to slow/derail TTS.

    Keeps normal letters, numbers, whitespace and punctuation.
    """
    if not text:
        return ""

    # Normalize compatibility forms (full-width, ligatures, etc.) to reduce weird chars.
    text = unicodedata.normalize("NFKC", text)

    cleaned_chars: list[str] = []
    for ch in text:
        cp = ord(ch)

        if STRIP_EMOJIS and _is_emoji_like(cp):
            continue

        cat = unicodedata.category(ch)

        # Drop control/format/private-use/surrogates/unassigned characters.
        # Keep standard whitespace; everything else in C* tends to be invisible trouble.
        if cat[0] == "C":
            if ch in ("\n", "\t", " "):
                cleaned_chars.append(ch)
            continue

        # Drop misc symbols (most emoji-like leftovers end up in So/Sk).
        if STRIP_OTHER_SYMBOLS and cat in ("So", "Sk"):
            continue

        cleaned_chars.append(ch)

    cleaned = "".join(cleaned_chars)
    # Normalize whitespace for chunking.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _load_hash_db(path: str) -> dict:
    """Load a versioned hash DB.

    Format v2:
    {
      "version": 2,
      "files": {
        "Blogs/foo.md": {
          "source_hash": "...",
          "tts_hash": "...",
          "cleaning_version": 2,
          "chunk_length": 150,
          "updated_at": "2026-01-24T12:34:56"
        }
      },
      "legacy": {"<old hash>": true}
    }

    Legacy format (v1) was: {"<hash>": true, ...}
    """
    if not HASHING:
        return {"version": 2, "files": {}, "legacy": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"version": 2, "files": {}, "legacy": {}}
    except json.JSONDecodeError:
        return {"version": 2, "files": {}, "legacy": {}}

    if not isinstance(data, dict):
        return {"version": 2, "files": {}, "legacy": {}}

    if "files" in data and isinstance(data.get("files"), dict):
        return {
            "version": int(data.get("version") or 2),
            "files": data.get("files") or {},
            "legacy": data.get("legacy") or {},
        }

    # Legacy v1: dict[str,bool]
    legacy = {k: v for k, v in data.items() if isinstance(k, str)}
    return {"version": 2, "files": {}, "legacy": legacy}


def _save_hash_db(path: str, data: dict) -> None:
    if not HASHING:
        return
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def _hash_text_chunks(chunks: list[str]) -> str:
    return hashlib.sha256(" ".join(chunks).encode("utf-8")).hexdigest()


def sanitize_text_for_hash(text: str) -> str:
    """Stable sanitization used for caching.

    Intentionally strips emoji-like chars and noisy symbols so the cache doesn't invalidate
    when only URLs/emojis/formatting change.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    cleaned_chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if _is_emoji_like(cp):
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "C":
            if ch in ("\n", "\t", " "):
                cleaned_chars.append(ch)
            continue
        if cat in ("So", "Sk"):
            continue
        cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _hash_source_markdown_for_cache(md_content: str) -> str:
    """Hash the *semantic* parts of markdown that affect speech."""
    if not md_content:
        return hashlib.sha256(b"").hexdigest()

    # Strip frontmatter.
    md = re.sub(r"^---\s*\n.*?\n---\s*\n", " ", md_content, flags=re.DOTALL)

    # Strip fenced code blocks (ignored by TTS anyway).
    md = re.sub(r"```[\s\S]*?```", " ", md)
    md = re.sub(r"~~~[\s\S]*?~~~", " ", md)

    # Flatten common markdown constructs so URL-only changes don't invalidate cache.
    md = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", md)  # images
    md = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", md)   # inline links
    md = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", md)    # reference links
    md = re.sub(r"\[([^\]]+)\]\[\]", r"\1", md)
    md = re.sub(r"^\s*\[[^\]]+\]:\s*\S+.*$", " ", md, flags=re.MULTILINE)  # ref defs

    # Remove raw URLs.
    md = re.sub(r"https?://\S+", " ", md)
    md = re.sub(r"www\.\S+", " ", md)

    md = sanitize_text_for_hash(md)
    return hashlib.sha256(md.encode("utf-8")).hexdigest()


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def compress_wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    *,
    vbr_quality: int = MP3_VBR_QUALITY,
    sample_rate_hz: int = MP3_RESAMPLE_HZ,
    ffmpeg_binary: str = FFMPEG_BINARY,
) -> bool:
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            winget_ffmpeg = Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
            if winget_ffmpeg.exists():
                ffmpeg = str(winget_ffmpeg)
    ffmpeg = ffmpeg or ffmpeg_binary
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-c:a",
        "libmp3lame",
        "-q:a",
        str(int(vbr_quality)),
        str(mp3_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            "ffmpeg not found. Install it and ensure it's on PATH (e.g. `winget install Gyan.FFmpeg`)."
        )
        return False

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if stderr:
            print(f"ffmpeg failed for {wav_path}: {stderr}")
        else:
            print(f"ffmpeg failed for {wav_path} (exit={result.returncode})")
        return False

    return _is_nonempty_file(mp3_path)


def remove_silence(audio, sample_rate, silence_threshold=0.01, min_silence_duration=0.5, keep_silence_duration=0.5):
    """
    Remove longer silences from audio while keeping natural pauses.
    
    Args:
        audio: torch tensor of audio (channels, samples)
        sample_rate: audio sample rate
        silence_threshold: amplitude threshold for silence detection (0.01 = -40dB)
        min_silence_duration: minimum silence duration to remove (in seconds)
        keep_silence_duration: duration of silence to keep as natural pause (in seconds)
    """
    # Convert to mono if stereo
    if audio.dim() > 1 and audio.shape[0] > 1:
        audio_mono = audio.mean(dim=0)
    else:
        audio_mono = audio.squeeze()
    
    # Calculate energy/amplitude
    energy = torch.abs(audio_mono)
    
    # Detect silence
    is_silent = energy < silence_threshold
    
    # Find silence segments
    min_silence_samples = int(min_silence_duration * sample_rate)
    keep_silence_samples = int(keep_silence_duration * sample_rate)
    
    segments_to_keep = []
    i = 0
    
    while i < len(is_silent):
        if not is_silent[i]:
            # Non-silent region - find where it ends
            start = i
            while i < len(is_silent) and not is_silent[i]:
                i += 1
            segments_to_keep.append((start, i))
        else:
            # Silent region - check duration
            silence_start = i
            while i < len(is_silent) and is_silent[i]:
                i += 1
            silence_duration = i - silence_start
            
            # If silence is long, keep only a short pause
            if silence_duration > min_silence_samples:
                # Add a short pause
                pause_end = min(silence_start + keep_silence_samples, i)
                if segments_to_keep:  # Only add pause if not at the start
                    segments_to_keep.append((silence_start, pause_end))
            else:
                # Keep short silences as-is
                segments_to_keep.append((silence_start, i))
    
    # Reconstruct audio
    if not segments_to_keep:
        return audio
    
    output_segments = [audio[..., start:end] for start, end in segments_to_keep]
    return torch.cat(output_segments, dim=-1)


def split_text(text, title=None):
    def _word_wrap(segment: str, max_len: int) -> list[str]:
        segment = re.sub(r"\s+", " ", (segment or "")).strip()
        if not segment:
            return []
        words = segment.split(" ")
        out: list[str] = []
        cur = ""
        for w in words:
            if not w:
                continue
            if not cur:
                cur = w
                continue
            if len(cur) + 1 + len(w) <= max_len:
                cur = f"{cur} {w}"
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)

        # If a single "word" is still too long, hard-split it.
        final: list[str] = []
        for part in out:
            if len(part) <= max_len:
                final.append(part)
            else:
                for i in range(0, len(part), max_len):
                    final.append(part[i : i + max_len])
        return final

    def _split_long_sentence(sentence: str, max_len: int) -> list[str]:
        sentence = re.sub(r"\s+", " ", (sentence or "")).strip()
        if not sentence:
            return []
        if len(sentence) <= max_len:
            return [sentence]

        # First try splitting by clause punctuation, keeping punctuation on the left.
        parts = [p.strip() for p in re.split(r"(?<=[,;:])\s*", sentence) if p.strip()]
        if len(parts) == 1:
            # Next, try splitting on long dashes (common in prose).
            parts = [p.strip() for p in re.split(r"(?<=\u2014)\s*", sentence) if p.strip()]

        out: list[str] = []
        for p in parts:
            if len(p) <= max_len:
                out.append(p)
            else:
                out.extend(_word_wrap(p, max_len))
        return out

    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return [title] if title else []

    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    segments: list[str] = []
    for s in sentences:
        segments.extend(_split_long_sentence(s, CHUNK_LENGTH))

    # Merge adjacent segments back up to CHUNK_LENGTH.
    result: list[str] = []
    current = ""
    for seg in segments:
        if not seg:
            continue
        if not current:
            current = seg
            continue
        if len(current) + 1 + len(seg) <= CHUNK_LENGTH:
            current = f"{current} {seg}"
        else:
            result.append(current)
            current = seg
    if current:
        result.append(current)

    if title:
        result = [title] + result
    return result


def generate_audio(text, settings):
    model = get_model()
    audio_segments = []

    for para in text:
        if para.strip():  
            print(f"\n\nGENERATING CHUNK of size {len(para)}. CHUNK content: {para}")            
            wav = model.generate(para, **settings)
            audio_segments.append(wav)
    
    combined_wav = torch.cat(audio_segments, dim=-1)
    
    # Remove longer silences while keeping natural pauses
    combined_wav = remove_silence(combined_wav, model.sr, min_silence_duration=1.0)
    
    return combined_wav


def _ensure_ends_with_pause(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text[-1] in ".?!:;":
        return text
    return text + "."


def _extract_tag_text(tag) -> str:
    return tag.get_text(separator=" ", strip=True)


def _convert_lists_for_tts(soup: BeautifulSoup) -> None:
    for lst in soup.find_all(["ul", "ol"]):
        items = lst.find_all("li", recursive=False)
        if not items:
            items = lst.find_all("li")

        rendered: list[str] = []
        ordered = lst.name == "ol"
        index = 1
        for li in items:
            item_text = _extract_tag_text(li)
            # Task lists often render checkboxes like "[x]" / "[ ]" in text.
            item_text = re.sub(r"^\[\s*[xX]?\s*\]\s*", "", item_text)
            item_text = item_text.strip("-•* \t")
            if not item_text:
                continue
            if ordered:
                rendered.append(f"{index}. {_ensure_ends_with_pause(item_text)}")
                index += 1
            else:
                rendered.append(_ensure_ends_with_pause(item_text))

        if rendered:
            lst.replace_with(" ".join(rendered) + " ")
        else:
            lst.decompose()


def _convert_tables_for_tts(soup: BeautifulSoup) -> None:
    for table in soup.find_all("table"):
        rows_text: list[str] = []
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            cell_text = [cell.get_text(separator=" ", strip=True) for cell in cells]
            row_text = ", ".join(filter(None, cell_text))
            if row_text:
                rows_text.append(row_text)

        if rows_text:
            table.replace_with("Table: " + ". ".join(rows_text) + ". ")
        else:
            table.decompose()


def _convert_structure_for_tts(soup: BeautifulSoup) -> None:
    # Drop non-content elements.
    for el in soup.find_all(["script", "style"]):
        el.decompose()

    # Remove code blocks entirely (TTS tends to struggle with them).
    for pre in soup.find_all("pre"):
        pre.decompose()

    # Inline code: keep its text.
    for code in soup.find_all("code"):
        code.replace_with(code.get_text(separator=" ", strip=True))

    # Images: keep alt text if present.
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt:
            img.replace_with(alt)
        else:
            img.decompose()

    # Links: keep anchor text, drop the URL.
    for a in soup.find_all("a"):
        a.replace_with(a.get_text(separator=" ", strip=True))

    # Headings: turn into sentence boundaries.
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        t = _extract_tag_text(h)
        if t:
            h.replace_with(_ensure_ends_with_pause(t) + " ")
        else:
            h.decompose()

    # Blockquotes: add a small hint so it sounds natural.
    for bq in soup.find_all("blockquote"):
        t = _extract_tag_text(bq)
        if t:
            bq.replace_with("Quote: " + _ensure_ends_with_pause(t) + " ")
        else:
            bq.decompose()

    # Horizontal rules: just a pause.
    for hr in soup.find_all("hr"):
        hr.replace_with(". ")

    # Inputs (often task list checkboxes) aren't helpful for TTS.
    for inp in soup.find_all("input"):
        inp.decompose()

    _convert_tables_for_tts(soup)
    _convert_lists_for_tts(soup)

def clean_md(md_content):
    md_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', md_content, flags=re.DOTALL)
    # Remove fenced code blocks early (robust even if markdown parsing/escaping behaves oddly).
    md_content = re.sub(r"```[\s\S]*?```", " ", md_content)
    md_content = re.sub(r"~~~[\s\S]*?~~~", " ", md_content)
    # Flatten reference-style links in case they survive parsing.
    md_content = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", md_content)
    md_content = re.sub(r"\[([^\]]+)\]\[\]", r"\1", md_content)
    html_content = markdown(md_content, extensions=['tables', 'fenced_code', 'sane_lists'])
    soup = BeautifulSoup(html_content, 'html.parser')
    
    _convert_structure_for_tts(soup)

    text = soup.get_text(separator=" ")
    # Drop common footnote/citation markers.
    text = re.sub(r"\[(\d+)\]", "", text)
    text = text.replace("↩", " ")
    return sanitize_text_for_tts(text)



def parse_dir():    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    hash_db = _load_hash_db(HASH_JSON)
    file_db = hash_db.get("files") if isinstance(hash_db.get("files"), dict) else {}
    
    # Find all markdown files recursively
    md_files = list(input_path.rglob("*.md"))
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check for generate-audio:true in frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, flags=re.DOTALL)
        if not frontmatter_match:
            continue
        
        frontmatter = frontmatter_match.group(1)
        if not re.search(r'generate-audio', frontmatter, re.IGNORECASE):
            continue
        
        print(f"Processing: {md_file}")
        
        # Clean the markdown content
        cleaned_text = clean_md(content)
        
        # Get the title (filename without .md extension)
        title = md_file.stem
        
        # Split the text into chunks
        chunks = split_text(cleaned_text, title=title)

        # Create output path maintaining directory structure
        relative_path = md_file.relative_to(input_path)
        wav_file = output_path / relative_path.with_suffix(".wav")
        mp3_file = output_path / relative_path.with_suffix(OUTPUT_AUDIO_EXT)

        rel_key = relative_path.as_posix()
        source_hash = _hash_source_markdown_for_cache(content)
        entry = file_db.get(rel_key) if isinstance(file_db, dict) else None
        mp3_exists = _is_nonempty_file(mp3_file)
        wav_exists = _is_nonempty_file(wav_file)

        if HASHING and isinstance(entry, dict):
            if entry.get("source_hash") == source_hash and mp3_exists:
                if (not REQUIRE_CLEANING_VERSION_MATCH_FOR_CACHE) or (
                    entry.get("cleaning_version") == CLEANING_VERSION
                ):
                    print(f"\n\nSkipping audio file generation for {rel_key} (cached)\n\n")
                    continue

            # Backwards-compat: previously generated outputs were WAV; compress them instead of regenerating.
            if entry.get("source_hash") == source_hash and (not mp3_exists) and wav_exists:
                print(
                    f"\n\nFound cached WAV for {rel_key}; compressing to MP3 instead of regenerating\n\n"
                )
                file_start_perf = time.perf_counter()
                ok = compress_wav_to_mp3(wav_file, mp3_file)
                if ok:
                    print(f"Saved MP3: {mp3_file}")
                    if DELETE_WAV_AFTER_COMPRESS:
                        try:
                            wav_file.unlink(missing_ok=True)
                            print(f"Deleted WAV: {wav_file}")
                        except OSError as e:
                            print(f"Failed to delete WAV {wav_file}: {e}")

                    _log_file_timing(
                        rel_key=rel_key,
                        file_seconds=time.perf_counter() - file_start_perf,
                        final_output=mp3_file,
                    )
                else:
                    print(f"MP3 compression failed; keeping WAV: {wav_file}")
                    _log_file_timing(
                        rel_key=rel_key,
                        file_seconds=time.perf_counter() - file_start_perf,
                        final_output=wav_file,
                    )
                continue

        # If an MP3 already exists but we don't have a hash entry yet, adopt it.
        if HASHING and ADOPT_EXISTING_MP3_WITHOUT_HASH_ENTRY and mp3_exists and not isinstance(entry, dict):
            print(f"\n\nFound existing MP3 for {rel_key}; adopting into cache without regenerating\n\n")
            file_db[rel_key] = {
                "source_hash": source_hash,
                "tts_hash": None,
                "cleaning_version": CLEANING_VERSION,
                "chunk_length": CHUNK_LENGTH,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "adopted": True,
            }
            hash_db["files"] = file_db
            _save_hash_db(HASH_JSON, hash_db)
            continue
        
        # Generate audio
        file_start_perf = time.perf_counter()
        combined_audio = generate_audio(text=chunks, settings=SETTINGS)
        
        if combined_audio is None:
            continue
        
        # Create parent directories if they don't exist
        wav_file.parent.mkdir(parents=True, exist_ok=True)

        # Save WAV (temporary) then compress to MP3
        ta.save(str(wav_file), combined_audio, get_model().sr)
        print(f"Saved WAV: {wav_file}")

        ok = compress_wav_to_mp3(wav_file, mp3_file)
        if ok:
            print(f"Saved MP3: {mp3_file}")
            if DELETE_WAV_AFTER_COMPRESS:
                try:
                    wav_file.unlink(missing_ok=True)
                    print(f"Deleted WAV: {wav_file}")
                except OSError as e:
                    print(f"Failed to delete WAV {wav_file}: {e}")

            if HASHING:
                file_db[rel_key] = {
                    "source_hash": source_hash,
                    "tts_hash": _hash_text_chunks(chunks),
                    "cleaning_version": CLEANING_VERSION,
                    "chunk_length": CHUNK_LENGTH,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                hash_db["files"] = file_db
                _save_hash_db(HASH_JSON, hash_db)

            _log_file_timing(
                rel_key=rel_key,
                file_seconds=time.perf_counter() - file_start_perf,
                final_output=mp3_file,
            )
        else:
            print(f"MP3 compression failed; keeping WAV: {wav_file}")

            _log_file_timing(
                rel_key=rel_key,
                file_seconds=time.perf_counter() - file_start_perf,
                final_output=wav_file,
            )

        print("Sleeping for 30 seconds for system cooldown")
        time.sleep(30)





if __name__ == "__main__":
    parse_dir()

    duration = datetime.now() - start_time
    print(f"\n\nTIME TAKEN TO EXECUTE: {duration}")