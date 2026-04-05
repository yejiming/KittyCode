"""Tests for provider conversion helpers in the LLM adapter."""

import threading
from types import SimpleNamespace

import pytest

from kittycode.llm import (
    _anthropic_message_to_response,
    _extract_system_message,
    _sleep_until_retry_or_cancel,
    _to_anthropic_messages,
    _to_anthropic_tools,
)
from kittycode.interrupts import CancellationRequested


def test_extract_system_message():
    system, conversation = _extract_system_message([
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ])

    assert system == "system prompt"
    assert conversation == [{"role": "user", "content": "hello"}]


def test_to_anthropic_tools():
    converted = _to_anthropic_tools([
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ])

    assert converted == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_to_anthropic_messages_converts_tool_flow():
    converted = _to_anthropic_messages([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tool-1", "content": "file contents"},
    ])

    assert converted == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "a.txt"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "file contents"}],
        },
    ]


def test_anthropic_message_to_response_calls_on_token():
    seen = []
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="tool_use", id="tool-2", name="grep", input={"pattern": "x"}),
            SimpleNamespace(type="text", text="world"),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=34),
    )

    response = _anthropic_message_to_response(message, on_token=seen.append)

    assert response.content == "hello world"
    assert [call.name for call in response.tool_calls] == ["grep"]
    assert response.tool_calls[0].arguments == {"pattern": "x"}
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 34
    assert seen == ["hello ", "world"]


def test_sleep_until_retry_or_cancel_raises_when_cancelled():
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(CancellationRequested):
        _sleep_until_retry_or_cancel(0.5, cancel_event)
