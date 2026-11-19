"""
Document Seeding Script
=======================
Loads sample FAQ documents from data/sample_documents.json, generates
768-dimensional vector embeddings using Google Gemini (gemini-embedding-001),
and persists them into the PostgreSQL 'documents' table.

Usage:
    python scripts/seed_documents.py
    python scripts/seed_documents.py --file data/sample_documents.json --batch-size 10
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal
from app.services.ingestion_service import IngestionService


def seed_documents(
    file_path: str = "data/sample_documents.json",
    batch_size: int = 10,
    skip_duplicates: bool = True,
) -> None:
    data_file = Path(file_path)
    if not data_file.exists():
        print(f"Error: Sample data file '{data_file}' not found.")
        sys.exit(1)

    print(f"[*] Starting document seeding from '{data_file}'...")
    print(f"[*] Batch size: {batch_size}, Skip duplicates: {skip_duplicates}")

    db = SessionLocal()
    try:
        service = IngestionService(db=db)
        print(f"[*] Using embedding model: {service.model_name}")

        result = service.ingest_from_json_file(
            file_path=str(data_file),
            skip_duplicates=skip_duplicates,
            batch_size=batch_size,
        )

        print("\n" + "=" * 50)
        print("Document Seeding Complete!")
        print("=" * 50)
        print(f"Total dataset records: {result['total_records']}")
        print(f"Newly inserted:        {result['inserted']}")
        print(f"Skipped duplicates:    {result['skipped']}")

        # Fetch current total in DB
        docs, total_in_db = service.list_documents(limit=5)
        print(f"Total documents in DB: {total_in_db}")
        print("=" * 50)

    except Exception as exc:
        db.rollback()
        print(f"[!] Error during seeding: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed vector documents into PostgreSQL.")
    parser.add_argument(
        "--file",
        default="data/sample_documents.json",
        help="Path to sample documents JSON file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for Gemini embedding requests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not skip existing duplicates",
    )
    args = parser.parse_args()

    seed_documents(
        file_path=args.file,
        batch_size=args.batch_size,
        skip_duplicates=not args.force,
    )
