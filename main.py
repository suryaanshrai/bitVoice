from datetime import datetime

start_time = datetime.now()

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import re
from markdown import markdown
from bs4 import BeautifulSoup
import torch
import unicodedata


CHUNK_LENGTH = 150

STRIP_EMOJIS = True
STRIP_OTHER_SYMBOLS = True

_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F1E6, 0x1F1FF),
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FA6F),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
)

_EMOJI_SINGLETONS: set[int] = {
    0x200D,
    0x20E3,
    0xFE0E,
    0xFE0F,
    0x200B,
    0x2060,
}


def _is_emoji_like(codepoint: int) -> bool:
    if codepoint in _EMOJI_SINGLETONS:
        return True
    if 0x1F3FB <= codepoint <= 0x1F3FF:
        return True
    return any(start <= codepoint <= end for start, end in _EMOJI_RANGES)


def sanitize_text_for_tts(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    cleaned_chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if STRIP_EMOJIS and _is_emoji_like(cp):
            continue
        cat = unicodedata.category(ch)
        if cat[0] == "C":
            if ch in ("\n", "\t", " "):
                cleaned_chars.append(ch)
            continue
        if STRIP_OTHER_SYMBOLS and cat in ("So", "Sk"):
            continue
        cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
# SETTINGS = {
#     "audio_prompt_path": "mysample.wav",
#     "exaggeration": 0.65, # [0.25 - 2]
#     "cfg_weight": 0.05, # [0.02 - 1]
#     "temperature": 0.9, # [0.05 - 5]
# }

SETTINGS = {
    "audio_prompt_path": "mysample.wav",
    "exaggeration": 0.6, # [0.25 - 2]
    "cfg_weight": 0.02, # [0.02 - 1]
    # "temperature": 0.4, # [0.05 - 5]
}


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
        parts = [p.strip() for p in re.split(r"(?<=[,;:])\s*", sentence) if p.strip()]
        if len(parts) == 1:
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

def read_dir():
    pass


def generate_audio(text, dest, settings):
    model = ChatterboxTTS.from_pretrained(device="cuda")
    audio_segments = []

    for para in text:
        if para.strip():  
            print(f"\n\nGENERATING CHUNK of size {len(para)}. CHUNK content: {para}")            
            wav = model.generate(para, **settings)
            audio_segments.append(wav)
    
    combined_wav = torch.cat(audio_segments, dim=-1)
    ta.save(dest, combined_wav, model.sr)

def clean_md(md_content):
    md_content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', md_content, flags=re.DOTALL)
    md_content = re.sub(r"```[\s\S]*?```", " ", md_content)
    md_content = re.sub(r"~~~[\s\S]*?~~~", " ", md_content)
    md_content = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", md_content)
    md_content = re.sub(r"\[([^\]]+)\]\[\]", r"\1", md_content)
    html_content = markdown(md_content, extensions=['tables', 'fenced_code', 'sane_lists'])
    soup = BeautifulSoup(html_content, 'html.parser')

    for el in soup.find_all(["script", "style"]):
        el.decompose()
    for pre in soup.find_all("pre"):
        pre.decompose()
    for code in soup.find_all("code"):
        code.replace_with(code.get_text(separator=" ", strip=True))
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt:
            img.replace_with(alt)
        else:
            img.decompose()
    for a in soup.find_all("a"):
        a.replace_with(a.get_text(separator=" ", strip=True))
    for hr in soup.find_all("hr"):
        hr.replace_with(". ")
    for inp in soup.find_all("input"):
        inp.decompose()

    # Convert tables to speakable text
    for table in soup.find_all('table'):
        rows_text = []
        for row in table.find_all('tr'):
            cells = row.find_all(['th', 'td'])
            cell_text = [cell.get_text(separator=" ", strip=True) for cell in cells]
            row_text = ", ".join(filter(None, cell_text))
            if row_text:
                rows_text.append(row_text)
        if rows_text:
            table.replace_with("Table: " + ". ".join(rows_text) + ". ")
        else:
            table.decompose()

    # Convert lists into sentence-like text
    for lst in soup.find_all(["ul", "ol"]):
        items = lst.find_all("li", recursive=False) or lst.find_all("li")
        rendered = []
        ordered = lst.name == "ol"
        idx = 1
        for li in items:
            t = li.get_text(separator=" ", strip=True)
            t = re.sub(r"^\[\s*[xX]?\s*\]\s*", "", t).strip("-•* \t")
            if not t:
                continue
            if t[-1] not in ".?!:;":
                t += "."
            rendered.append(f"{idx}. {t}" if ordered else t)
            if ordered:
                idx += 1
        if rendered:
            lst.replace_with(" ".join(rendered) + " ")
        else:
            lst.decompose()

    # Headings + quotes as boundaries
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        t = h.get_text(separator=" ", strip=True)
        if t and t[-1] not in ".?!:;":
            t += "."
        h.replace_with((t + " ") if t else "")
    for bq in soup.find_all("blockquote"):
        t = bq.get_text(separator=" ", strip=True)
        if t and t[-1] not in ".?!:;":
            t += "."
        bq.replace_with(("Quote: " + t + " ") if t else "")

    text = soup.get_text(separator=" ")
    text = re.sub(r"\[(\d+)\]", "", text).replace("↩", " ")
    return sanitize_text_for_tts(text)


if __name__ == "__main__":
    with open("Minesweeper.md", "r") as f:
        content = f.read()
        poem = clean_md(content)

    chunks = split_text(poem)
    generate_audio(text=chunks[:3], dest="minesweeper.wav", settings=SETTINGS)

























    duration = datetime.now() - start_time
    print(f"\n\nTIME TAKEN TO EXECUTE: {duration}")