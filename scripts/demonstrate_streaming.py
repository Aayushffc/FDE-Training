import asyncio
import json
import sys
from time import perf_counter
from dotenv import load_dotenv

load_dotenv()

from app.modules.chat.schemas import ChatRequest
from app.modules.chat.service import default_chat_service


async def main():
    print("=" * 65)
    print("  FDE SPRINT LLM GATEWAY - STREAMING DEMONSTRATION")
    print("=" * 65)
    prompt = "Explain why server-side streaming is important in LLM user interfaces in 3 sentences."
    print(f"\nUser Prompt: \"{prompt}\"")
    print("Provider:    groq (openai/gpt-oss-120b)")
    print("Protocol:    Server-Sent Events (SSE) incremental token stream")
    print("-" * 65)
    print("Streaming Response: \n")

    request = ChatRequest(message=prompt, provider="groq")
    chunk_count = 0
    started_at = perf_counter()

    first_chunk_latency = None

    async for chunk_sse in default_chat_service.stream_sse(request):
        # Parse SSE line
        line = chunk_sse.strip()
        if not line or not line.startswith("data: "):
            continue

        raw_data = line[6:]
        if raw_data == "[DONE]":
            break

        data = json.loads(raw_data)
        content = data.get("content", "")
        if content:
            chunk_count += 1
            if first_chunk_latency is None:
                first_chunk_latency = (perf_counter() - started_at) * 1000
            sys.stdout.write(content)
            sys.stdout.flush()

    total_latency_ms = (perf_counter() - started_at) * 1000
    print("\n" + "-" * 65)
    print(f"Stream Summary:")
    print(f"  Chunks Received:        {chunk_count}")
    print(f"  Time to First Token:    {first_chunk_latency:.1f} ms" if first_chunk_latency else "  Time to First Token: N/A")
    print(f"  Total Stream Duration:  {total_latency_ms:.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
