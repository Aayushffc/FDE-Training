import os
import sys
from pathlib import Path

import uvicorn


def _env_file() -> str | None:
    return ".env" if Path(".env").exists() else None


def dev() -> None:
    """Start the FastAPI development server with hot-reload."""
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        reload_dirs=["app"],
        env_file=_env_file(),
    )


def start() -> None:
    """Start the FastAPI server without hot-reload (production mode)."""
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        env_file=_env_file(),
    )


def _run_pytest(args: list[str]) -> None:
    import pytest

    raise SystemExit(pytest.main([*args, *sys.argv[1:]]))


def test() -> None:
    """Run all automated tests (unit, api, integration), excluding live smoke tests."""
    _run_pytest(["-o", "addopts=-v", "-m", "not live"])


def test_live() -> None:
    """Run only live API smoke tests."""
    _run_pytest(["-o", "addopts=-v", "-m", "live"])


def test_all() -> None:
    """Run the entire test suite, including live API smoke tests."""
    _run_pytest(["-o", "addopts=-v"])


def stream_client() -> None:
    """Run the live terminal HTTP streaming client against /v1/chat/stream."""
    import runpy

    script_path = Path("scripts") / "stream_client.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "stream_client.py"
    runpy.run_path(str(script_path), run_name="__main__")


def seed_chunks_cli() -> None:
    """Seed documents and document_chunks from data/chunk_data.json into PostgreSQL."""
    import runpy

    script_path = Path("scripts") / "seed_chunk_data.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "seed_chunk_data.py"
    runpy.run_path(str(script_path), run_name="__main__")


def seed_documents_cli() -> None:
    """Seed sample whole documents into PostgreSQL."""
    import runpy

    script_path = Path("scripts") / "seed_documents.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "seed_documents.py"
    runpy.run_path(str(script_path), run_name="__main__")


def seed_prompts_cli() -> None:
    """Seed prompt categories and prompt versions into PostgreSQL."""
    import runpy

    script_path = Path("scripts") / "seed_prompts.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "seed_prompts.py"
    runpy.run_path(str(script_path), run_name="__main__")


def evaluate_rag_cli() -> None:
    """Run RAG precision, recall, and refusal evaluation benchmarks."""
    import runpy

    script_path = Path("scripts") / "evaluate_rag.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_rag.py"
    runpy.run_path(str(script_path), run_name="__main__")
