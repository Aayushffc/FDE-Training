import re


def clean_text(text: str) -> str:
    """
    Normalize and clean raw document text:
    - Replaces carriage returns and non-breaking spaces
    - Collapses repeated blank lines (>2 newlines to 2)
    - Strips trailing/leading spaces on lines
    """
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [line.strip() for line in normalized.split("\n")]
    joined = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk size and overlap adhere to operational bounds."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )
