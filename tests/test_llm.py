"""Tests for provider conversion helpers in the LLM adapter."""

import logging
import threading
from io import StringIO
from types import SimpleNamespace

import pytest

from kittycode.llm import (
    LLM,
    _anthropic_message_to_response,
    _extract_system_message,
    _parse_openai_tool_calls,
    _openai_completion_to_response,
    _sleep_until_retry_or_cancel,
    _to_anthropic_messages,
    _to_anthropic_tools,
)
from kittycode.interrupts import CancellationRequested


def test_llm_keeps_provider_methods_on_class():
    assert hasattr(LLM, "_chat_anthropic")
    assert hasattr(LLM, "_call_with_retry")
    assert hasattr(LLM, "_call_anthropic_with_retry")


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


def test_parse_openai_tool_calls_logs_raw_tool_call_and_arguments():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("kittycode.llm")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
        parsed = _parse_openai_tool_calls(
            {
                0: {
                    "id": "call_1",
                    "name": "write_file",
                    "args": '{"file_path": "/tmp/a.txt", "content": ',
                }
            }
        )
    finally:
        logger.removeHandler(handler)

    assert [call.name for call in parsed] == ["write_file"]
    assert parsed[0].arguments == {}
    logs = stream.getvalue()
    assert "OpenAI raw tool call" in logs
    assert "write_file" in logs
    assert '"content": ' in logs


def test_openai_completion_to_response_reads_non_stream_tool_calls():
    response = _openai_completion_to_response(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="write_file",
                                    arguments='{"file_path": "/tmp/a.txt", "content": "hello"}',
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=34),
        )
    )

    assert response.content == "done"
    assert [call.name for call in response.tool_calls] == ["write_file"]
    assert response.tool_calls[0].arguments == {"file_path": "/tmp/a.txt", "content": "hello"}
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 34


def test_chat_openai_uses_non_stream_requests(monkeypatch):
    llm = LLM(model="gpt-4o", api_key="test-key", interface="openai")
    captured = {}

    def fake_call_with_retry(params, max_retries=3, cancel_event=None):
        captured.update(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )

    monkeypatch.setattr(llm, "_call_with_retry", fake_call_with_retry)

    response = llm._chat_openai(messages=[{"role": "user", "content": "hi"}], tools=[])

    assert captured["stream"] is False
    assert "stream_options" not in captured
    assert response.prompt_tokens == 1
    assert response.completion_tokens == 2


def test_chat_anthropic_uses_streaming_requests(monkeypatch):
    llm = LLM(model="claude-3-7-sonnet-latest", api_key="test-key", interface="anthropic")
    captured = {}

    final_message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello world"),
            SimpleNamespace(type="tool_use", id="tool_1", name="read_file", input={"path": "a.txt"}),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=22),
    )

    class FakeStream:
        def __enter__(self):
            self.text_stream = iter(["hello ", "world"])
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_final_message(self):
            return final_message

    def fake_stream(**params):
        captured.update(params)
        return FakeStream()

    llm.client = SimpleNamespace(
        messages=SimpleNamespace(
            stream=fake_stream,
            create=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("messages.create should not be used")),
        )
    )

    seen = []
    response = llm._chat_anthropic(
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hi"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        on_token=seen.append,
    )

    assert captured["model"] == "claude-3-7-sonnet-latest"
    assert captured["system"] == "system prompt"
    assert captured["max_tokens"] == 4096
    assert captured["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert captured["tools"] == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    assert seen == ["hello ", "world"]
    assert response.content == "hello world"
    assert [call.name for call in response.tool_calls] == ["read_file"]
    assert response.tool_calls[0].arguments == {"path": "a.txt"}
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 22
