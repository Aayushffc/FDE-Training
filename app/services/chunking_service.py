from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.infrastructure.text_processing import (
    clean_text as _clean_text,
    hierarchical_chunk_split,
    sliding_window_chunk_split,
    validate_chunk_params as _validate_chunk_params,
)


@dataclass
class ChunkItem:
    """Represents an extracted chunk with positional metadata."""

    chunk_index: int
    content: str
    char_start: int
    char_end: int
    char_count: int
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingService:
    """
    Production-grade text chunking engine supporting configurable chunk sizes,
    overlaps, boundary preservation (sentences, paragraphs, words), and metadata enrichment.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize and clean raw document text using infrastructure text processor."""
        return _clean_text(text)

    @staticmethod
    def validate_params(chunk_size: int, chunk_overlap: int) -> None:
        """Ensure chunk size and overlap adhere to operational bounds."""
        _validate_chunk_params(chunk_size, chunk_overlap)

    def split_by_characters(
        self,
        text: str,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        preserve_boundaries: bool = True,
    ) -> List[ChunkItem]:
        """
        Split text into character-bounded chunks with overlap.
        If preserve_boundaries is True, uses hierarchical separators (paragraph, sentence, word)
        to prevent cutting sentences or words across chunk boundaries.
        """
        self.validate_params(chunk_size, chunk_overlap)
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        if len(cleaned) <= chunk_size:
            return [
                ChunkItem(
                    chunk_index=0,
                    content=cleaned,
                    char_start=0,
                    char_end=len(cleaned),
                    char_count=len(cleaned),
                    word_count=len(cleaned.split()),
                    metadata={"total_chunks": 1},
                )
            ]

        if not preserve_boundaries:
            raw_tuples = sliding_window_chunk_split(cleaned, chunk_size, chunk_overlap)
            chunks = [
                ChunkItem(
                    chunk_index=i,
                    content=sub,
                    char_start=start,
                    char_end=end,
                    char_count=len(sub),
                    word_count=len(sub.split()),
                    metadata={"total_chunks": len(raw_tuples)},
                )
                for i, (sub, start, end) in enumerate(raw_tuples)
            ]
            return chunks

        # Hierarchical boundary preservation
        merged_chunks = hierarchical_chunk_split(cleaned, chunk_size, chunk_overlap)
        chunks: List[ChunkItem] = []
        cursor = 0
        total = len(merged_chunks)
        for i, text_block in enumerate(merged_chunks):
            found_pos = cleaned.find(text_block[:min(30, len(text_block))], cursor)
            if found_pos == -1:
                found_pos = cursor
            c_start = found_pos
            c_end = found_pos + len(text_block)
            cursor = max(cursor, c_start)
            chunks.append(
                ChunkItem(
                    chunk_index=i,
                    content=text_block,
                    char_start=c_start,
                    char_end=c_end,
                    char_count=len(text_block),
                    word_count=len(text_block.split()),
                    metadata={"total_chunks": total},
                )
            )

        return chunks

    def split_by_words(
        self,
        text: str,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
    ) -> List[ChunkItem]:
        """
        Split text into word-bounded chunks with word-level overlap.
        Useful for token-proportional chunking without word cutting.
        """
        self.validate_params(chunk_size, chunk_overlap)
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        words = cleaned.split()
        if not words:
            return []

        if len(words) <= chunk_size:
            content = " ".join(words)
            return [
                ChunkItem(
                    chunk_index=0,
                    content=content,
                    char_start=0,
                    char_end=len(content),
                    char_count=len(content),
                    word_count=len(words),
                    metadata={"total_chunks": 1},
                )
            ]

        chunks: List[ChunkItem] = []
        step = chunk_size - chunk_overlap
        idx = 0

        for i in range(0, len(words), step):
            slice_words = words[i : i + chunk_size]
            if not slice_words:
                break

            chunk_content = " ".join(slice_words)
            chunks.append(
                ChunkItem(
                    chunk_index=idx,
                    content=chunk_content,
                    char_start=0,
                    char_end=len(chunk_content),
                    char_count=len(chunk_content),
                    word_count=len(slice_words),
                    metadata={},
                )
            )
            idx += 1
            if i + chunk_size >= len(words):
                break

        total = len(chunks)
        for c in chunks:
            c.metadata["total_chunks"] = total

        return chunks

    def chunk_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        strategy: str = "character",
        preserve_boundaries: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        High-level chunking entry point that splits content and produces chunk dictionaries
        with merged metadata linking back to the parent document.
        """
        if strategy == "word":
            raw_chunks = self.split_by_words(
                text=content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            raw_chunks = self.split_by_characters(
                text=content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                preserve_boundaries=preserve_boundaries,
            )

        parent_meta = metadata or {}
        output: List[Dict[str, Any]] = []

        for item in raw_chunks:
            chunk_meta = {
                **parent_meta,
                "chunk_index": item.chunk_index,
                "total_chunks": item.metadata.get("total_chunks", len(raw_chunks)),
                "chunk_size": item.char_count,
                "word_count": item.word_count,
                "char_start": item.char_start,
                "char_end": item.char_end,
                "chunking_strategy": strategy,
                "chunk_size_config": chunk_size,
                "chunk_overlap_config": chunk_overlap,
            }
            output.append(
                {
                    "chunk_index": item.chunk_index,
                    "content": item.content,
                    "metadata": chunk_meta,
                    "char_count": item.char_count,
                    "word_count": item.word_count,
                }
            )

        return output
