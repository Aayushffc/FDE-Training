import re
from typing import List, Tuple


def find_boundary_split(text: str, target_pos: int, window: int = 50) -> int:
    """
    Search backward from target_pos within a search window for the best natural boundary.
    Hierarchy:
    1. Double newline (paragraph boundary: \\n\\n)
    2. Single newline (line break: \\n)
    3. Sentence boundary (. ! ?) followed by whitespace
    4. Word boundary (whitespace)
    """
    if target_pos >= len(text):
        return len(text)

    start_search = max(0, target_pos - window)
    candidate_region = text[start_search:target_pos]

    # 1. Paragraph boundary
    idx = candidate_region.rfind("\n\n")
    if idx != -1:
        return start_search + idx + 2

    # 2. Line break
    idx = candidate_region.rfind("\n")
    if idx != -1:
        return start_search + idx + 1

    # 3. Sentence boundary: [.!?] followed by space
    match = list(re.finditer(r"[.!?]\s", candidate_region))
    if match:
        return start_search + match[-1].end()

    # 4. Word boundary
    idx = candidate_region.rfind(" ")
    if idx != -1:
        return start_search + idx + 1

    return target_pos


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into individual sentences with character start/end offsets.
    Preserves exact positions in original cleaned text.
    """
    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\n+")
    sentences: List[Tuple[str, int, int]] = []

    last_end = 0
    for match in sentence_pattern.finditer(text):
        sent = text[last_end : match.start()].strip()
        if sent:
            sentences.append((sent, last_end, last_end + len(sent)))
        last_end = match.end()

    tail = text[last_end:].strip()
    if tail:
        sentences.append((tail, last_end, last_end + len(tail)))

    return sentences


def split_paragraphs(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into paragraphs delimited by double newlines.
    Returns (paragraph_text, char_start, char_end).
    """
    paragraphs: List[Tuple[str, int, int]] = []
    last_end = 0
    for match in re.finditer(r"\n\s*\n", text):
        p_text = text[last_end : match.start()].strip()
        if p_text:
            paragraphs.append((p_text, last_end, last_end + len(p_text)))
        last_end = match.end()

    tail = text[last_end:].strip()
    if tail:
        paragraphs.append((tail, last_end, last_end + len(tail)))

    return paragraphs
