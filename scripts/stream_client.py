"""
Terminal HTTP streaming client for POST /v1/chat/stream.
Connects directly to the running FastAPI server and prints tokens incrementally as they arrive.
"""

import argparse
import json
import sys
import urllib.request
from time import perf_counter


def stream_from_server(
    url: str = "http://127.0.0.1:8000/v1/chat/stream",
    prompt: str = "Explain why server-side streaming is important in LLM user interfaces in 3 sentences.",
    provider: str = "groq",
) -> None:
    print("=" * 65)
    print("  CONNECTING TO HTTP STREAMING ENDPOINT")
    print(f"  URL:      {url}")
    print(f"  Provider: {provider}")
    print(f"  Prompt:   \"{prompt}\"")
    print("=" * 65)
    print("\n[AI Streaming Output]:\n")

    payload = json.dumps({"message": prompt, "provider": provider}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = perf_counter()
    first_token_time = None
    chunks_count = 0

    try:
        with urllib.request.urlopen(req) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue

                raw_data = line[6:]
                if raw_data == "[DONE]":
                    break

                try:
                    data = json.loads(raw_data)
                    content = data.get("content", "")
                    if content:
                        chunks_count += 1
                        if first_token_time is None:
                            first_token_time = (perf_counter() - started_at) * 1000
                        sys.stdout.write(content)
                        sys.stdout.flush()
                except json.JSONDecodeError:
                    continue

        total_time_ms = (perf_counter() - started_at) * 1000
        print("\n\n" + "-" * 65)
        print(f"Stream Complete:")
        print(f"  Tokens Received:      {chunks_count}")
        print(f"  Time to First Token:  {first_token_time:.1f} ms" if first_token_time else "  Time to First Token: N/A")
        print(f"  Total Duration:       {total_time_ms:.1f} ms")
        print("=" * 65)

    except urllib.error.URLError as exc:
        print(f"\n[Error connecting to server]: {exc}")
        print("Make sure your FastAPI server is running with 'uv run dev' on http://127.0.0.1:8000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live HTTP token streaming client for /v1/chat/stream")
    parser.add_argument("prompt", nargs="?", default=None, help="User message sent to the LLM")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/stream", help="Streaming endpoint URL")
    parser.add_argument("--provider", default="groq", help="LLM provider (groq, openai, anthropic)")

    args = parser.parse_args()
    prompt = args.prompt
    if not prompt:
        prompt = "Explain why server-side streaming is important in LLM user interfaces in 3 sentences."

    stream_from_server(url=args.url, prompt=prompt, provider=args.provider)


if __name__ == "__main__":
    main()
