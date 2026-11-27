import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.modules.agents.adapters.retrieval_adapter import RetrievalAdapter
from app.modules.agents.config import AgentConfig
from app.modules.agents.graph import build_agent_graph, run_agent_graph
from app.modules.agents.nodes import _trim_task_messages, normalize_openai_chat_completion
from app.modules.agents.schemas import AgentChatRequest
from app.modules.agents.service import AgentToolService
from app.schemas.rag import RAGResponse, RAGSourceItem


class TestResponseNormalizer:
    def test_normalizes_sdk_object_tool_calls(self) -> None:
        resp = SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_obj_1",
                                function=SimpleNamespace(
                                    name="get_order_status",
                                    arguments='{"order_id": "ORD123"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        out = normalize_openai_chat_completion(resp)
        assert out["tool_calls"] == [
            {
                "id": "call_obj_1",
                "name": "get_order_status",
                "arguments": '{"order_id": "ORD123"}',
            }
        ]

    def test_normalizes_dict_shaped_tool_calls(self) -> None:
        resp = SimpleNamespace(
            model="gpt-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_dict_123",
                                "function": {
                                    "name": "get_order_status",
                                    "arguments": '{"order_id": "ORD123"}',
                                },
                            }
                        ],
                    )
                )
            ],
            usage=None,
        )
        out = normalize_openai_chat_completion(resp)
        assert out["tool_calls"] == [
            {
                "id": "call_dict_123",
                "name": "get_order_status",
                "arguments": '{"order_id": "ORD123"}',
            }
        ]

    @pytest.mark.parametrize(
        "bad_response",
        [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(message=None)]),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[{"function": {"name": "get_order_status", "arguments": "{}"}}],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[{"id": "c1", "function": {"name": "", "arguments": "{}"}}],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[{"id": "c1", "function": {"name": "get_order_status"}}],
                        )
                    )
                ]
            ),
        ],
    )
    def test_malformed_provider_response_raises_value_error(self, bad_response) -> None:
        with pytest.raises(ValueError):
            normalize_openai_chat_completion(bad_response)


class TestMessageHistoryTrimming:
    def test_trim_keeps_system_user_and_atomic_tool_groups(self) -> None:
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{}"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c2", "content": "{}"},
        ]
        trimmed = _trim_task_messages(msgs, max_messages=4)
        assert trimmed[0]["role"] == "system"
        assert trimmed[1]["role"] == "user"
        assert [m.get("tool_call_id") or m["tool_calls"][0]["id"] for m in trimmed[2:]] == ["c2", "c2"]


class TestLangGraphAgentWorkflow:
    async def test_graph_direct_reply_without_tools(self) -> None:
        async def fake_llm(messages, tools):
            assert len(tools) == 2
            return {
                "model": "test-model",
                "content": "Hello! How can I help you today?",
                "tool_calls": [],
                "prompt_tokens": 12,
                "completion_tokens": 9,
            }

        state = await run_agent_graph(
            llm_caller=fake_llm,
            user_request="Hi there",
            system_prompt="You are a helpful support agent.",
            provider="groq",
            model="test-model",
        )

        assert state["execution_status"] == "completed"
        assert state["final_result"] == "Hello! How can I help you today?"
        assert state["tool_call_count"] == 0
        assert state["tool_results"] == []
        assert state["errors"] == []
        assert state["messages"][-1] == {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }

    async def test_graph_order_status_loop_preserves_tool_call_id_and_trace(self) -> None:
        observed_messages = []
        turn = {"count": 0}

        async def fake_llm(messages, tools):
            observed_messages.append(list(messages))
            turn["count"] += 1
            if turn["count"] == 1:
                return {
                    "model": "test-model",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_order_1",
                            "name": "get_order_status",
                            "arguments": '{"order_id": "ORD123"}',
                        }
                    ],
                    "prompt_tokens": 40,
                    "completion_tokens": 15,
                }
            return {
                "model": "test-model",
                "content": "Order ORD123 is Shipped via FedEx (FX-992817203).",
                "tool_calls": [],
                "prompt_tokens": 80,
                "completion_tokens": 20,
            }

        state = await run_agent_graph(
            llm_caller=fake_llm,
            user_request="Where is my order ORD123?",
            system_prompt="You are a helpful support agent.",
            task_id="task-xyz",
        )

        assert turn["count"] == 2
        assert state["execution_status"] == "completed"
        assert state["tool_call_count"] == 1
        assert len(state["tool_results"]) == 1
        assert state["tool_results"][0]["tool_call_id"] == "call_order_1"
        assert state["tool_results"][0]["tool_name"] == "get_order_status"
        assert state["tool_results"][0]["result"]["status"] == "Shipped"
        assert state["final_result"] == "Order ORD123 is Shipped via FedEx (FX-992817203)."

        second_turn_msgs = observed_messages[1]
        assert second_turn_msgs[-2]["role"] == "assistant"
        assert second_turn_msgs[-2]["tool_calls"][0]["id"] == "call_order_1"
        assert second_turn_msgs[-1]["role"] == "tool"
        assert second_turn_msgs[-1]["tool_call_id"] == "call_order_1"

        for entry in state["trace"]:
            assert entry["task_id"] == "task-xyz"
            assert "timestamp" in entry
            assert "duration_ms" in entry
        assert [entry["node"] for entry in state["trace"]] == ["llm_node", "tool_node", "llm_node"]
        assert state["trace"][1]["tool_call_id"] == "call_order_1"

    async def test_graph_multi_tool_order_and_rag_customer_support_demo(self) -> None:
        mock_rag_service = MagicMock()
        mock_rag_service.ask.return_value = RAGResponse(
            question="damaged keyboard replacement policy",
            answer="Damaged items can be replaced within 30 days of delivery. [Source 1]",
            sources=[
                RAGSourceItem(
                    source_label="Source 1",
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    chunk_index=0,
                    content="Damaged items qualify for free replacement within 30 days.",
                    relevance_score=0.94,
                )
            ],
            retrieval_strategy="hybrid",
            reranked=True,
            is_grounded=True,
            citations=["Source 1"],
            total_sources_evaluated=1,
        )
        retrieval_adapter = RetrievalAdapter(rag_service=mock_rag_service)

        turn = {"count": 0}

        async def fake_llm(messages, tools):
            turn["count"] += 1
            if turn["count"] == 1:
                return {
                    "model": "test-model",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "get_order_status",
                            "arguments": {"order_id": "ORD123"},
                        }
                    ],
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                }
            if turn["count"] == 2:
                return {
                    "model": "test-model",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "name": "search_knowledge",
                            "arguments": {
                                "query": "damaged keyboard replacement policy",
                                "top_k": 3,
                            },
                        }
                    ],
                    "prompt_tokens": 60,
                    "completion_tokens": 15,
                }
            return {
                "model": "test-model",
                "content": (
                    "Order ORD123 is Shipped via FedEx. Per our policy [Source 1], "
                    "damaged items qualify for a free replacement within 30 days."
                ),
                "tool_calls": [],
                "prompt_tokens": 90,
                "completion_tokens": 25,
            }

        state = await run_agent_graph(
            llm_caller=fake_llm,
            user_request="My order ORD123 keyboard arrived damaged. Can I replace it?",
            system_prompt="You are a customer support agent.",
            retrieval_adapter=retrieval_adapter,
        )

        assert state["execution_status"] == "completed"
        assert state["tool_call_count"] == 2
        assert [r["tool_call_id"] for r in state["tool_results"]] == ["call_1", "call_2"]
        assert [r["tool_name"] for r in state["tool_results"]] == [
            "get_order_status",
            "search_knowledge",
        ]
        assert state["tool_results"][1]["result"]["is_grounded"] is True
        assert state["tool_results"][1]["result"]["citations"] == ["Source 1"]
        assert "Shipped via FedEx" in (state["final_result"] or "")

    async def test_graph_enforces_max_tool_call_limit_and_appends_assistant_message(self) -> None:
        async def looping_llm(messages, tools):
            return {
                "model": "test-model",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_loop_{len(messages)}",
                        "name": "get_order_status",
                        "arguments": '{"order_id": "ORD123"}',
                    }
                ],
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

        state = await run_agent_graph(
            llm_caller=looping_llm,
            user_request="Keep checking order ORD123 forever",
            system_prompt="Support agent",
            config=AgentConfig(max_tool_calls=2, max_graph_steps=12),
        )

        assert state["execution_status"] == "limit_reached"
        assert state["tool_call_count"] == 2
        assert len(state["errors"]) >= 1
        assert state["messages"][-1]["role"] == "assistant"
        assert state["messages"][-1]["content"] == state["final_result"]

    async def test_graph_batch_tool_calls_respects_remaining_limit_without_incomplete_messages(self) -> None:
        turn = {"count": 0}

        async def batch_llm(messages, tools):
            turn["count"] += 1
            if turn["count"] == 1:
                return {
                    "model": "test-model",
                    "content": None,
                    "tool_calls": [
                        {"id": "batch_1", "name": "get_order_status", "arguments": '{"order_id": "ORD123"}'},
                        {"id": "batch_2", "name": "get_order_status", "arguments": '{"order_id": "ORD-456"}'},
                        {"id": "batch_3", "name": "get_order_status", "arguments": '{"order_id": "ORD-789"}'},
                    ],
                    "prompt_tokens": 20,
                    "completion_tokens": 15,
                }
            # Every requested tool call must have a matching tool response message
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert [m["tool_call_id"] for m in tool_msgs] == ["batch_1", "batch_2", "batch_3"]
            assert json.loads(tool_msgs[1]["content"])["code"] == "TOOL_CALL_LIMIT_REACHED"
            assert json.loads(tool_msgs[2]["content"])["code"] == "TOOL_CALL_LIMIT_REACHED"
            return {
                "model": "test-model",
                "content": "Checked ORD123; remaining lookups hit the limit.",
                "tool_calls": [],
                "prompt_tokens": 40,
                "completion_tokens": 10,
            }

        state = await run_agent_graph(
            llm_caller=batch_llm,
            user_request="Check 3 orders",
            system_prompt="Support agent",
            config=AgentConfig(max_tool_calls=1, max_graph_steps=12),
        )

        assert state["execution_status"] == "completed"
        assert state["tool_call_count"] == 1
        assert len(state["tool_results"]) == 1
        assert state["tool_results"][0]["tool_call_id"] == "batch_1"
        assert len(state["errors"]) == 2

    async def test_graph_sanitizes_validation_and_unexpected_tool_errors(self) -> None:
        turn = {"count": 0}

        async def fake_llm(messages, tools):
            turn["count"] += 1
            if turn["count"] == 1:
                return {
                    "model": "test-model",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad_json",
                            "name": "get_order_status",
                            "arguments": "{not-valid-json",
                        },
                        {
                            "id": "call_rag_boom",
                            "name": "search_knowledge",
                            "arguments": '{"query": "password reset"}',
                        },
                    ],
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                }

            err_1 = json.loads(messages[-2]["content"])
            err_2 = json.loads(messages[-1]["content"])
            assert err_1 == {
                "error": True,
                "code": "INVALID_TOOL_ARGUMENTS",
                "message": "Invalid arguments for 'get_order_status'. Please verify required parameters and try again.",
            }
            assert err_2 == {
                "error": True,
                "code": "TOOL_EXECUTION_FAILED",
                "message": "The requested operation could not be completed.",
            }
            # Ensure sensitive DB URI / internal exception details are NOT leaked to the model
            assert "secret-db-host" not in messages[-1]["content"]
            return {
                "model": "test-model",
                "content": "Please provide a valid order ID; knowledge search is temporarily unavailable.",
                "tool_calls": [],
                "prompt_tokens": 40,
                "completion_tokens": 12,
            }

        mock_adapter = MagicMock(spec=RetrievalAdapter)
        mock_adapter.asearch_knowledge = AsyncMock(
            side_effect=RuntimeError("postgresql://admin:secret@secret-db-host:5432/prod failed")
        )

        state = await run_agent_graph(
            llm_caller=fake_llm,
            user_request="Check my order and reset password",
            system_prompt="Support agent",
            retrieval_adapter=mock_adapter,
        )

        assert state["execution_status"] == "completed"
        assert len(state["errors"]) == 2

    async def test_graph_enforces_tool_timeout_seconds(self) -> None:
        turn = {"count": 0}

        async def slow_search(**kwargs):
            await asyncio.sleep(0.2)
            return {"answer": "late"}

        mock_adapter = MagicMock(spec=RetrievalAdapter)
        mock_adapter.asearch_knowledge = AsyncMock(side_effect=slow_search)

        async def fake_llm(messages, tools):
            turn["count"] += 1
            if turn["count"] == 1:
                return {
                    "model": "test-model",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_slow",
                            "name": "search_knowledge",
                            "arguments": '{"query": "refund policy"}',
                        }
                    ],
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            payload = json.loads(messages[-1]["content"])
            assert payload["code"] == "TOOL_TIMEOUT"
            return {
                "model": "test-model",
                "content": "Knowledge search timed out. Please try again shortly.",
                "tool_calls": [],
                "prompt_tokens": 20,
                "completion_tokens": 10,
            }

        state = await run_agent_graph(
            llm_caller=fake_llm,
            user_request="What is the refund policy?",
            system_prompt="Support agent",
            retrieval_adapter=mock_adapter,
            config=AgentConfig(tool_timeout_seconds=0.02),
        )

        assert state["execution_status"] == "completed"
        assert any("timed out" in err.lower() for err in state["errors"])

    async def test_graph_catches_recursion_limit_cleanly(self) -> None:
        async def looping_llm(messages, tools):
            return {
                "model": "test-model",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_rec_{len(messages)}",
                        "name": "get_order_status",
                        "arguments": '{"order_id": "ORD123"}',
                    }
                ],
                "prompt_tokens": 5,
                "completion_tokens": 5,
            }

        state = await run_agent_graph(
            llm_caller=looping_llm,
            user_request="Loop",
            system_prompt="Support",
            config=AgentConfig(max_tool_calls=10, max_graph_steps=3),
        )

        assert state["execution_status"] == "failed"
        assert any("max_graph_steps" in e for e in state["errors"])
        assert state["final_result"] is not None

    async def test_service_execute_agent_graph_and_chat_with_mocked_groq(
        self,
        mock_settings: Settings,
    ) -> None:
        tool_call = SimpleNamespace(
            id="call_svc_1",
            function=SimpleNamespace(
                name="get_order_status",
                arguments='{"order_id": "ORD-456"}',
            ),
        )
        first_resp = SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=15),
        )
        second_resp = SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[SimpleNamespace(message=SimpleNamespace(content="ORD-456 was delivered via UPS."))],
            usage=SimpleNamespace(prompt_tokens=90, completion_tokens=20),
        )

        mock_prompt_svc = MagicMock()
        mock_prompt_svc.get_system_prompt.return_value = "You are a support agent."

        service = AgentToolService(prompt_service=mock_prompt_svc)

        with patch("app.modules.agents.service.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = [first_resp, second_resp]

            chat_resp = await service.execute_agent_chat(
                AgentChatRequest(message="Status of ORD-456?", provider="groq")
            )

        assert chat_resp.execution_status == "completed"
        assert chat_resp.reply == "ORD-456 was delivered via UPS."
        assert len(chat_resp.tool_calls) == 1
        assert chat_resp.tool_calls[0].tool_call_id == "call_svc_1"
        assert chat_resp.tool_calls[0].result["status"] == "Delivered"

    async def test_service_rejects_unsupported_agent_provider(
        self,
        mock_settings: Settings,
    ) -> None:
        mock_prompt_svc = MagicMock()
        mock_prompt_svc.get_system_prompt.return_value = "You are a support agent."
        service = AgentToolService(prompt_service=mock_prompt_svc)

        with pytest.raises(LLMProviderError) as exc_info:
            await service.execute_agent_chat(
                AgentChatRequest(message="Status of ORD123?", provider="anthropic")
            )
        assert exc_info.value.status_code == 400

    def test_graph_compiles_cleanly(self) -> None:
        async def noop_llm(messages, tools):
            return {"model": "m", "content": "ok", "tool_calls": [], "prompt_tokens": 0, "completion_tokens": 0}

        compiled = build_agent_graph(noop_llm)
        assert compiled is not None
