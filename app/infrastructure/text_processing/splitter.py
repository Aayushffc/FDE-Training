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


def sliding_window_chunk_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[str, int, int]]:
    """
    Splits text using a fixed sliding character window.
    Returns list of (chunk_content, start_idx, end_idx).
    """
    chunks: List[Tuple[str, int, int]] = []
    step = chunk_size - chunk_overlap
    text_len = len(text)
    for start in range(0, text_len, step):
        end = min(start + chunk_size, text_len)
        sub = text[start:end].strip()
        if sub:
            chunks.append((sub, start, end))
        if end >= text_len:
            break
    return chunks


def hierarchical_chunk_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Hierarchically split text preserving semantic boundaries (paragraphs, newlines, sentences, words).
    Recursively descends separators: ["\\n\\n", "\\n", ". ", "? ", "! ", " "].
    Merges fragments with overlap up to chunk_size.
    """
    separators = ["\n\n", "\n", ". ", "? ", "! ", " "]

    def _split_text(txt: str, seps: List[str]) -> List[str]:
        if not seps or len(txt) <= chunk_size:
            if len(txt) <= chunk_size:
                return [txt]
            step = chunk_size - chunk_overlap
            return [txt[i : i + chunk_size] for i in range(0, len(txt), step)]

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep in txt:
            raw_parts = txt.split(sep)
            parts: List[str] = []
            for p in raw_parts:
                p_str = p.strip()
                if not p_str:
                    continue
                if sep in [". ", "? ", "! "]:
                    parts.append(p_str + sep.strip())
                else:
                    parts.append(p_str)
        else:
            return _split_text(txt, remaining_seps)

        good_parts: List[str] = []
        for part in parts:
            if len(part) <= chunk_size:
                good_parts.append(part)
            else:
                good_parts.extend(_split_text(part, remaining_seps))
        return good_parts

    atomic_parts = _split_text(text, separators)

    merged_chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for part in atomic_parts:
        added_len = len(part) + (1 if current_chunk else 0)
        if current_len + added_len <= chunk_size:
            current_chunk.append(part)
            current_len += added_len
        else:
            if current_chunk:
                merged_chunks.append(" ".join(current_chunk).strip())
                overlap_parts: List[str] = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) + (1 if overlap_parts else 0) <= chunk_overlap:
                        overlap_parts.insert(0, prev)
                        overlap_len += len(prev) + (1 if len(overlap_parts) > 1 else 0)
                    else:
                        break
                current_chunk = overlap_parts + [part]
                current_len = sum(len(p) for p in current_chunk) + max(0, len(current_chunk) - 1)
            else:
                current_chunk = [part]
                current_len = len(part)

    if current_chunk:
        final_sub = " ".join(current_chunk).strip()
        if not merged_chunks or final_sub != merged_chunks[-1]:
            merged_chunks.append(final_sub)

    return merged_chunks
