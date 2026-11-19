from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.v1.dependencies import get_chat_service
from app.modules.agents.schemas import AgentChatRequest, AgentChatResponse
from app.modules.agents.service import AgentToolService, get_agent_tool_service
from app.modules.agents.tools import AVAILABLE_TOOLS
from app.modules.chat.schemas import (
    ChatPromptVersionRequest,
    ChatPromptVersionResponse,
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from app.modules.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=None,
    summary="Standard conversational chat with AI",
    responses={
        200: {
            "description": "Conversational assistant completion using default system prompt",
            "content": {
                "application/json": {
                    "schema": ChatResponse.model_json_schema(),
                }
            },
        }
    },
)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Simple, direct conversational chat with the AI assistant.
    Uses the default system prompt to hold a normal conversation without sentiment analysis.
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


@router.get("/prompts", summary="List available prompts (alias)")
async def list_available_prompts_alias(
    chat_service: ChatService = Depends(get_chat_service),
) -> dict:
    """Convenience alias for /v1/prompts."""
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


@router.post(
    "/stream",
    summary="Stream tokens incrementally as Server-Sent Events (SSE)",
    responses={
        200: {
            "description": "Real-time token delta stream formatted as Server-Sent Events",
            "content": {
                "text/event-stream": {
                    "example": (
                        "data: {\"content\": \"Coffee \", \"is_complete\": false}\n\n"
                        "data: {\"content\": \"brews \", \"is_complete\": false}\n\n"
                        "data: {\"content\": \"and \", \"is_complete\": false}\n\n"
                        "data: {\"content\": \"syntax \", \"is_complete\": false}\n\n"
                        "data: {\"content\": \"flows.\", \"is_complete\": false}\n\n"
                        "data: {\"content\": \"\", \"is_complete\": true, \"provider\": \"groq\", \"latency_ms\": 284.1}\n\n"
                        "data: [DONE]\n\n"
                    )
                }
            },
        }
    },
)
async def chat_stream(
    payload: ChatStreamRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Dedicated endpoint for real-time token streaming.
    Streams Server-Sent Events (SSE) by default without requiring a boolean flag in the request body.
    """
    return StreamingResponse(
        chat_service.stream_sse(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/prompt-version",
    response_model=ChatPromptVersionResponse,
    summary="Test completions against specific versioned prompt templates",
    responses={
        200: {
            "description": "Completion generated using the selected prompt version",
            "content": {
                "application/json": {
                    "schema": ChatPromptVersionResponse.model_json_schema(),
                }
            },
        }
    },
)
async def chat_with_prompt_version(
    payload: ChatPromptVersionRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatPromptVersionResponse:
    """
    Execute chat completions using a specific versioned prompt template from the library.
    Allows comparing outputs across 'v1', 'v2', 'v3', or 'customer_support/v2'.
    """
    return await chat_service.generate_prompt_version_completion(payload)


@router.post(
    "/sentiment",
    response_model=SentimentAnalysisResponse,
    summary="Analyze customer ticket sentiment, urgency, churn risk, and entities",
    responses={
        200: {
            "description": "Structured sentiment analysis extraction",
            "content": {
                "application/json": {
                    "schema": SentimentAnalysisResponse.model_json_schema(),
                }
            },
        }
    },
)
async def analyze_sentiment(
    payload: SentimentAnalysisRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> SentimentAnalysisResponse:
    """
    Dedicated endpoint for customer support sentiment and ticket analysis.
    Extracts category, customer sentiment, urgency level, churn risk, entities, and recommended actions.
    """
    return await chat_service.analyze_sentiment(payload)


@router.post(
    "/tools",
    response_model=AgentChatResponse,
    summary="Autonomous LLM tool and function calling",
)
async def chat_with_tools(
    payload: AgentChatRequest,
    agent_service: AgentToolService = Depends(get_agent_tool_service),
) -> AgentChatResponse:
    """
    Interact with an LLM agent equipped with backend tools.
    The LLM determines if a tool is needed (e.g., get_order_status), validates parameters via Pydantic,
    executes the Python function, and returns a final synthesized response.
    """
    return await agent_service.execute_agent_chat(payload)


@router.get(
    "/tools",
    summary="List available backend tools declared for LLM function calling",
)
async def list_available_agent_tools() -> dict:
    """List available backend tools declared for LLM function calling."""
    return {
        "available_tools": AVAILABLE_TOOLS,
        "mock_order_ids_for_testing": ["ORD123", "ORD-456", "ORD-789"],
        "example_prompt": "What is the status of my order ORD123?",
    }
