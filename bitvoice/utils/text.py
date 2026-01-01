import re
import os
from typing import Optional

def clean_markdown(text: str, filename: Optional[str] = None) -> str:
    """Basic markdown cleaning for better TTS."""
    if not text: return ""

    # Remove YAML frontmatter: --- ... ---
    text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
    
    # If filename is provided, prepend it as a header (without extension)
    if filename:
        # Strip extension if present
        name = os.path.splitext(filename)[0]
        text = f"# {name}\n\n{text}"

    # Remove images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove links: [text](url) -> text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove bold/italic: **text** or *text* -> text
    text = re.sub(r'(\*\*|__|(\*|_))(.*?)\1', r'\3', text)
    # Remove headers: # Header -> Header
    text = re.sub(r'^#+\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
    # Remove code blocks: ```code```
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code: `code`
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()
