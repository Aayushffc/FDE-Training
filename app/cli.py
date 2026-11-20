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


def run_experiments() -> None:
    """Run the customer support prompt engineering experiments (30 benchmark runs)."""
    import runpy

    script_path = Path("scripts") / "run_prompt_experiments.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_prompt_experiments.py"
    runpy.run_path(str(script_path), run_name="__main__")


def analyze_experiments() -> None:
    """Analyze and display prompt engineering benchmark metrics."""
    import runpy

    script_path = Path("scripts") / "analyze_results.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "analyze_results.py"
    runpy.run_path(str(script_path), run_name="__main__")


def stream_demo() -> None:
    """Run the live Server-Sent Events (SSE) token streaming demonstration."""
    import runpy

    script_path = Path("scripts") / "demonstrate_streaming.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "demonstrate_streaming.py"
    runpy.run_path(str(script_path), run_name="__main__")


def stream_client() -> None:
    """Run the live terminal HTTP streaming client against /v1/chat/stream."""
    import runpy

    script_path = Path("scripts") / "stream_client.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "stream_client.py"
    runpy.run_path(str(script_path), run_name="__main__")


def tool_demo() -> None:
    """Run the live LLM tool/function calling demonstration."""
    import runpy

    script_path = Path("scripts") / "demonstrate_tool_calling.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "demonstrate_tool_calling.py"
    runpy.run_path(str(script_path), run_name="__main__")


def seed_chunks_cli() -> None:
    """Seed documents and document_chunks into PostgreSQL with Gemini 768-dim embeddings."""
    import runpy

    script_path = Path("scripts") / "seed_chunks.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "seed_chunks.py"
    runpy.run_path(str(script_path), run_name="__main__")


def chunk_experiment_cli() -> None:
    """Run chunk size and overlap comparison experiments across 3 configurations."""
    import runpy

    script_path = Path("scripts") / "experiment_chunking.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "experiment_chunking.py"
    runpy.run_path(str(script_path), run_name="__main__")


def compare_chunks_cli() -> None:
    """Execute benchmark comparison of Keyword vs Vector vs Hybrid retrieval over chunks."""
    import runpy

    script_path = Path("scripts") / "compare_chunk_searches.py"
    if not script_path.exists():
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "compare_chunk_searches.py"
    runpy.run_path(str(script_path), run_name="__main__")



