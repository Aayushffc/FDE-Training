import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.modules.agents.schemas import AgentChatRequest
from app.modules.agents.service import default_agent_tool_service


async def demonstrate(query: str):
    print("=" * 70)
    print(f"STEP 1: USER REQUEST")
    print("=" * 70)
    print(f"Prompt: \"{query}\"")
    print(f"Target: Groq LLM with declared tool `get_order_status`")

    payload = AgentChatRequest(message=query, provider="groq")
    res = await default_agent_tool_service.execute_agent_chat(payload)

    print("\n" + "=" * 70)
    print(f"STEP 2 & 3: LLM TOOL SELECTION & BACKEND EXECUTION")
    print("=" * 70)
    if res.tool_calls:
        for idx, call in enumerate(res.tool_calls, start=1):
            print(f"Tool Call #{idx}:")
            print(f"  Tool Name:         {call.tool_name}")
            print(f"  Validated Args:    {json.dumps(call.arguments)}")
            print(f"  Backend Execution: {json.dumps(call.result, indent=4)}")
            print(f"  Execution Time:    {call.execution_latency_ms} ms")
    else:
        print("LLM answered directly without invoking any tools.")

    print("\n" + "=" * 70)
    print(f"STEP 4: FINAL SYNTHESIZED LLM RESPONSE")
    print("=" * 70)
    print(f"Model:               {res.model}")
    print(f"Total Turn Latency:  {res.total_latency_ms} ms")
    print(f"Estimated Cost:      ${res.estimated_cost_usd:.6f} USD")
    print(f"\nResponse Text:\n{res.reply}\n")


async def main():
    print("######################################################################")
    print("        FDE SPRINT LLM GATEWAY - TOOL CALLING DEMONSTRATION")
    print("######################################################################\n")

    # Scenario 1: Standard order lookup (ORD123)
    await demonstrate("What's the status of order ORD123?")

    # Scenario 2: Delivered order lookup (ORD-456)
    await demonstrate("Hi, can you track my package for order ORD-456?")


if __name__ == "__main__":
    asyncio.run(main())
