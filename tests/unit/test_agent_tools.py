from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from app.core.exceptions import StructuredOutputValidationError
from app.modules.agents.adapters.retrieval_adapter import RetrievalAdapter
from app.modules.agents.schemas import OrderStatusArgs, SearchKnowledgeArgs
from app.modules.agents.tools import (
    AVAILABLE_TOOLS,
    dispatch_tool_call,
    execute_get_order_status,
    execute_search_knowledge,
)


class TestAgentToolsUnit:
    def test_order_status_args_valid(self) -> None:
        args = OrderStatusArgs(order_id="ORD123")
        assert args.order_id == "ORD123"

    def test_order_status_args_too_short(self) -> None:
        with pytest.raises(ValidationError):
            OrderStatusArgs(order_id="O")

    def test_execute_get_order_status_known_order(self) -> None:
        args = OrderStatusArgs(order_id="ORD123")
        details = execute_get_order_status(args)
        assert details.order_id == "ORD123"
        assert details.status == "Shipped"
        assert details.carrier == "FedEx"
        assert len(details.items) > 0

    def test_execute_get_order_status_case_insensitive(self) -> None:
        args = OrderStatusArgs(order_id="ord-456")
        details = execute_get_order_status(args)
        assert details.status == "Delivered"
        assert details.carrier == "UPS"

    def test_execute_get_order_status_unknown_order(self) -> None:
        args = OrderStatusArgs(order_id="ORD-UNKNOWN-999")
        details = execute_get_order_status(args)
        assert details.status == "NotFound"
        assert details.carrier is None
        assert details.items == []

    def test_dispatch_tool_call_valid_string_arguments(self) -> None:
        result, latency = dispatch_tool_call("get_order_status", '{"order_id": "ORD123"}')
        assert result["order_id"] == "ORD123"
        assert result["status"] == "Shipped"
        assert latency >= 0

    def test_dispatch_tool_call_valid_dict_arguments(self) -> None:
        result, latency = dispatch_tool_call("get_order_status", {"order_id": "ORD-789"})
        assert result["status"] == "Processing"
        assert result["carrier"] == "DHL Express"

    def test_dispatch_tool_call_invalid_arguments_raises(self) -> None:
        with pytest.raises(StructuredOutputValidationError):
            dispatch_tool_call("get_order_status", '{"order_id": ""}')

    def test_dispatch_tool_call_unknown_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch_tool_call("nonexistent_tool", {})

    def test_available_tools_contains_order_and_rag(self) -> None:
        names = [tool["function"]["name"] for tool in AVAILABLE_TOOLS]
        assert names == ["get_order_status", "search_knowledge"]

    def test_search_knowledge_args_validation(self) -> None:
        args = SearchKnowledgeArgs(query="What is the return policy?", top_k=3, strategy="hybrid")
        assert args.query == "What is the return policy?"
        assert args.top_k == 3
        with pytest.raises(ValidationError):
            SearchKnowledgeArgs(query="a")

    def test_dispatch_search_knowledge_delegates_to_retrieval_adapter(self) -> None:
        mock_adapter = MagicMock(spec=RetrievalAdapter)
        mock_adapter.search_knowledge.return_value = {
            "query": "refund policy",
            "answer": "Refunds are processed within 5 business days. [Source 1]",
            "is_grounded": True,
            "citations": ["Source 1"],
            "retrieval_strategy": "hybrid",
            "sources": [],
        }

        result, latency = dispatch_tool_call(
            "search_knowledge",
            '{"query": "refund policy", "top_k": 3}',
            retrieval_adapter=mock_adapter,
        )

        mock_adapter.search_knowledge.assert_called_once_with(
            query="refund policy",
            top_k=3,
            strategy="hybrid",
            rerank=True,
        )
        assert result["is_grounded"] is True
        assert result["citations"] == ["Source 1"]
        assert latency >= 0

    def test_execute_search_knowledge_helper(self) -> None:
        mock_adapter = MagicMock(spec=RetrievalAdapter)
        mock_adapter.search_knowledge.return_value = {"query": "lockout", "answer": "30 min", "is_grounded": True}
        out = execute_search_knowledge(
            SearchKnowledgeArgs(query="lockout", strategy="vector"),
            retrieval_adapter=mock_adapter,
        )
        assert out["answer"] == "30 min"
        mock_adapter.search_knowledge.assert_called_once_with(
            query="lockout",
            top_k=5,
            strategy="vector",
            rerank=True,
        )
