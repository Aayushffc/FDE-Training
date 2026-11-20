import pytest

from app.services.chunking_service import ChunkItem, ChunkingService


class TestChunkingService:
    @pytest.fixture
    def chunker(self):
        return ChunkingService()

    def test_clean_text_normalizes_whitespace_and_newlines(self, chunker):
        raw = "  Hello \r\n world!  \n\n\n\n This is   a test. \xa0\n  "
        cleaned = chunker.clean_text(raw)
        assert "\r" not in cleaned
        assert "\xa0" not in cleaned
        assert "\n\n\n" not in cleaned
        assert cleaned.startswith("Hello")
        assert cleaned.endswith("test.")

    def test_clean_text_empty_input(self, chunker):
        assert chunker.clean_text("") == ""
        assert chunker.clean_text(None) == ""

    def test_validate_params_rejects_invalid_values(self, chunker):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            chunker.validate_params(chunk_size=0, chunk_overlap=0)

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            chunker.validate_params(chunk_size=100, chunk_overlap=-10)

        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than chunk_size"):
            chunker.validate_params(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than chunk_size"):
            chunker.validate_params(chunk_size=100, chunk_overlap=150)

    def test_split_by_characters_short_text_single_chunk(self, chunker):
        text = "This is a brief text under 400 characters."
        chunks = chunker.split_by_characters(text, chunk_size=400, chunk_overlap=80)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].content == text
        assert chunks[0].metadata["total_chunks"] == 1

    def test_split_by_characters_preserves_sentence_boundaries(self, chunker):
        sentence_1 = "First sentence outlines account creation."
        sentence_2 = "Second sentence details subscription billing plans."
        sentence_3 = "Third sentence clarifies technical troubleshooting."
        full_text = f"{sentence_1} {sentence_2} {sentence_3}"

        # Chunk size configured to accommodate roughly 1-2 sentences
        chunks = chunker.split_by_characters(
            full_text,
            chunk_size=75,
            chunk_overlap=25,
            preserve_boundaries=True,
        )

        assert len(chunks) >= 2
        # Verify chunks end on natural sentence boundary when possible
        for c in chunks[:-1]:
            assert c.content.endswith(".")

    def test_split_by_words(self, chunker):
        words = ["word" + str(i) for i in range(100)]
        text = " ".join(words)

        chunks = chunker.split_by_words(text, chunk_size=30, chunk_overlap=10)
        # Step is 20 words (30 - 10), 100 words -> chunks at 0..30, 20..50, 40..70, 60..90, 80..100 -> 5 chunks
        assert len(chunks) == 5
        assert chunks[0].word_count == 30
        assert chunks[1].word_count == 30
        assert chunks[-1].word_count <= 30
        assert chunks[0].chunk_index == 0
        assert chunks[-1].chunk_index == 4

    def test_chunk_document_enriches_metadata(self, chunker):
        content = (
            "Managing account security requires strong credentials. "
            "Always update your 2FA devices promptly. "
            "Keep emergency recovery codes stored safely."
        )
        parent_meta = {"category": "security", "source": "guide"}

        result = chunker.chunk_document(
            content=content,
            metadata=parent_meta,
            chunk_size=70,
            chunk_overlap=20,
            strategy="character",
        )

        assert len(result) >= 2
        for item in result:
            assert "category" in item["metadata"]
            assert item["metadata"]["category"] == "security"
            assert "chunk_index" in item["metadata"]
            assert "total_chunks" in item["metadata"]
            assert item["metadata"]["chunking_strategy"] == "character"
            assert item["metadata"]["chunk_size_config"] == 70
            assert item["metadata"]["chunk_overlap_config"] == 20
            assert item["char_count"] == len(item["content"])

    def test_chunk_document_empty_content(self, chunker):
        assert chunker.chunk_document("", metadata={"cat": "test"}) == []
        assert chunker.chunk_document("   \n\n   ") == []
