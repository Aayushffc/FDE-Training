"""
Chunking Experimentation Script
===============================
Evaluates 3 distinct chunking configurations on a comprehensive 1,000-word document:
- Config A (Fine-Grained): chunk_size=200, chunk_overlap=40
- Config B (Balanced):     chunk_size=400, chunk_overlap=80
- Config C (Contextual):   chunk_size=800, chunk_overlap=150

Analyzes chunk count, average size, boundary fidelity, and retrieval specificity.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunking_service import ChunkingService


def run_chunking_experiments() -> Dict[str, Any]:
    guide_path = Path(__file__).resolve().parent.parent / "data" / "account_payments_subscriptions_guide.json"
    with open(guide_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_content = data[0]["content"] if isinstance(data, list) else data["content"]
    total_chars = len(raw_content)
    total_words = len(raw_content.split())

    chunker = ChunkingService()

    configs = [
        {
            "name": "Config A (Fine-Grained)",
            "chunk_size": 200,
            "chunk_overlap": 40,
            "description": "Short chunks optimized for precise fact extraction and minimal noise.",
        },
        {
            "name": "Config B (Balanced)",
            "chunk_size": 400,
            "chunk_overlap": 80,
            "description": "Medium chunks balancing semantic coherence, sentence boundaries, and context.",
        },
        {
            "name": "Config C (Contextual)",
            "chunk_size": 800,
            "chunk_overlap": 150,
            "description": "Large chunks capturing complete multi-paragraph explanations and broad context.",
        },
    ]

    print("=" * 80)
    print("PHASE 1: DOCUMENT CHUNKING CONFIGURATION EXPERIMENT")
    print(f"Source Document Length: {total_chars} characters | {total_words} words")
    print("=" * 80)

    experiment_results: List[Dict[str, Any]] = []

    for cfg in configs:
        c_size = cfg["chunk_size"]
        c_overlap = cfg["chunk_overlap"]
        name = cfg["name"]

        chunks = chunker.chunk_document(
            content=raw_content,
            metadata=data.get("metadata", {}),
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            strategy="character",
            preserve_boundaries=True,
        )

        char_lens = [c["char_count"] for c in chunks]
        word_lens = [c["word_count"] for c in chunks]

        avg_chars = sum(char_lens) / len(char_lens) if char_lens else 0
        min_chars = min(char_lens) if char_lens else 0
        max_chars = max(char_lens) if char_lens else 0

        avg_words = sum(word_lens) / len(word_lens) if word_lens else 0
        min_words = min(word_lens) if word_lens else 0
        max_words = max(word_lens) if word_lens else 0

        print(f"\n--- {name} ---")
        print(f"Config: chunk_size={c_size}, chunk_overlap={c_overlap}")
        print(f"Total Chunks Generated: {len(chunks)}")
        print(f"Character Lengths: Avg = {avg_chars:.1f} | Min = {min_chars} | Max = {max_chars}")
        print(f"Word Lengths:      Avg = {avg_words:.1f} | Min = {min_words} | Max = {max_words}")
        print("Sample Chunk 0 Preview:")
        print(f"  \"{chunks[0]['content'][:120]}...\"")
        if len(chunks) > 1:
            print("Sample Chunk 1 Preview:")
            print(f"  \"{chunks[1]['content'][:120]}...\"")

        experiment_results.append({
            "name": name,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
            "total_chunks": len(chunks),
            "avg_chars": round(avg_chars, 1),
            "min_chars": min_chars,
            "max_chars": max_chars,
            "avg_words": round(avg_words, 1),
            "min_words": min_words,
            "max_words": max_words,
            "description": cfg["description"],
            "chunks_sample": [
                {"index": c["chunk_index"], "preview": c["content"][:100]}
                for c in chunks[:3]
            ],
        })

    return {
        "doc_total_chars": total_chars,
        "doc_total_words": total_words,
        "experiments": experiment_results,
    }


if __name__ == "__main__":
    run_chunking_experiments()
