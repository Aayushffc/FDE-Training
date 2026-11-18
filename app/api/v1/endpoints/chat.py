from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.dependencies import get_chat_service
from app.modules.agents.schemas import AgentChatRequest, AgentChatResponse
from app.modules.agents.service import AgentToolService, get_agent_tool_service
from app.modules.agents.tools import AVAILABLE_TOOLS
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=None)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Handle chat completion requests.
    - If payload.stream is True: streams Server-Sent Events (SSE) chunks incrementally.
    - If payload.stream is False: returns the standard normalized ChatResponse JSON.
    """
    if payload.stream:
        return StreamingResponse(
            chat_service.stream_sse(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return await chat_service.generate_chat_completion(payload)


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Dedicated endpoint that returns incremental token chunks as Server-Sent Events (SSE)."""
    return StreamingResponse(
        chat_service.stream_sse(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/prompts")
async def list_available_prompts(
    chat_service: ChatService = Depends(get_chat_service),
) -> dict:
    """List available prompt categories and versions available for interactive testing."""
    return {
        "default_category": "customer_support",
        "available_customer_support_versions": chat_service.prompt_service.list_versions("customer_support"),
        "all_categories": chat_service.prompt_service.list_all(),
        "usage_example": {
            "message": "My order #12345 arrived damaged, please refund me.",
            "provider": "groq",
            "prompt_version": "v2",
        },
    }


@router.post("/tools", response_model=AgentChatResponse)
async def chat_with_tools(
    payload: AgentChatRequest,
    agent_service: AgentToolService = Depends(get_agent_tool_service),
) -> AgentChatResponse:
    """
    Interact with an LLM agent equipped with internal backend tools.
    The LLM decides whether to invoke backend tools (e.g., get_order_status),
    the backend validates and executes the function, and returns the synthesized answer.
    """
    return await agent_service.execute_agent_chat(payload)


@router.get("/tools")
async def list_available_agent_tools() -> dict:
    """List available backend tools declared for LLM function calling."""
    return {
        "available_tools": AVAILABLE_TOOLS,
        "mock_order_ids_for_testing": ["ORD123", "ORD-456", "ORD-789"],
        "example_prompt": "What is the status of my order ORD123?",
    }

