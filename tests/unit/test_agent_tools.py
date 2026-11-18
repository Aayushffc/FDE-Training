import pytest
from pydantic import ValidationError

from app.core.exceptions import StructuredOutputValidationError
from app.modules.agents.schemas import OrderStatusArgs
from app.modules.agents.tools import (
    dispatch_tool_call,
    execute_get_order_status,
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
