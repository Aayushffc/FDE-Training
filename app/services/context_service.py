from typing import Any, Dict, List, Optional, Tuple


class ContextEngineeringService:
    """
    Coordinates context engineering for RAG:
    - Deduplicates redundant or overlapping chunks
    - Enforces context window budget constraints (trimming)
    - Labels chunks with explicit source citations [Source 1], [Source 2]
    - Constructs system and user prompts with strict grounding and "I don't know" instructions
    """

    DEFAULT_MAX_CONTEXT_CHARS = 4000

    @staticmethod
    def deduplicate_chunks(items: List[Any], overlap_threshold: float = 0.80) -> List[Any]:
        """
        Deduplicates chunks by ID and content similarity.
        If a subsequent chunk shares >80% tokens with an already included chunk, it is omitted.
        """
        seen_ids = set()
        unique_items = []
        seen_token_sets: List[set] = []

        for item in items:
            item_id = str(getattr(item, "id", ""))
            if item_id and item_id in seen_ids:
                continue

            content = getattr(item, "content", "").strip()
            if not content:
                continue

            tokens = set(content.lower().split())
            if not tokens:
                continue

            # Check overlap against already admitted chunks
            is_redundant = False
            for prev_tokens in seen_token_sets:
                intersection = tokens & prev_tokens
                overlap_ratio = len(intersection) / min(len(tokens), len(prev_tokens))
                if overlap_ratio >= overlap_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                if item_id:
                    seen_ids.add(item_id)
                seen_token_sets.append(tokens)
                unique_items.append(item)

        return unique_items

    @classmethod
    def trim_context(
        cls,
        items: List[Any],
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> List[Any]:
        """
        Trims candidate chunks to fit strictly within the maximum character budget.
        Preserves higher-priority (earlier ranked) chunks.
        """
        admitted = []
        accumulated_chars = 0

        for item in items:
            content = getattr(item, "content", "")
            item_len = len(content) + 80  # Reserve overhead for source header labels
            if accumulated_chars + item_len <= max_chars:
                admitted.append(item)
                accumulated_chars += item_len
            else:
                # If budget is full, stop admitting lower-ranked chunks
                break

        return admitted

    @classmethod
    def assemble_labeled_context(
        cls,
        items: List[Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Builds a clearly labeled multi-source context block with citation anchors.
        Returns:
            context_text: Formatted string "[Source 1] ... [Source 2] ..."
            source_manifest: Structured list of source metadata for API response
        """
        blocks: List[str] = []
        manifest: List[Dict[str, Any]] = []

        for idx, item in enumerate(items, start=1):
            source_label = f"Source {idx}"
            chunk_id = str(getattr(item, "id", ""))
            doc_id = str(getattr(item, "document_id", ""))
            chunk_idx = getattr(item, "chunk_index", None)
            content = getattr(item, "content", "").strip()

            header = f"[{source_label}] (Document ID: {doc_id}, Chunk: {chunk_idx})"
            blocks.append(f"{header}\n{content}")

            manifest.append({
                "source_label": source_label,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "content": content,
                "similarity": getattr(item, "similarity", None),
                "rank": getattr(item, "rank", None),
                "rerank_score": getattr(item, "rerank_score", None),
            })

        context_text = "\n\n".join(blocks)
        return context_text, manifest

    @classmethod
    def build_rag_prompt(
        cls,
        question: str,
        items: List[Any],
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Assembles deduplicated, budget-trimmed, and source-labeled context
        and produces the system prompt and user prompt.
        """
        deduped = cls.deduplicate_chunks(items)
        trimmed = cls.trim_context(deduped, max_chars=max_context_chars)
        context_text, manifest = cls.assemble_labeled_context(trimmed)

        system_prompt = (
            "You are a helpful, strictly grounded AI support assistant.\n"
            "Your instructions:\n"
            "1. Answer the user's question using ONLY the provided context.\n"
            "2. If the answer is not available in the context, say exactly: 'I don't know based on the provided context.'\n"
            "3. Do NOT make assumptions, extrapolate, or use outside knowledge.\n"
            "4. Cite the source(s) supporting each statement using bracketed references (e.g. [Source 1], [Source 2]).\n"
            "5. Be concise, professional, and factual."
        )

        user_prompt = (
            f"CONTEXT:\n"
            f"{context_text if context_text else 'No relevant context found.'}\n\n"
            f"USER QUESTION:\n"
            f"{question.strip()}"
        )

        return system_prompt, user_prompt, manifest
