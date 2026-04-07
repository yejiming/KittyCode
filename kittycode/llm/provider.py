"""LLM provider layer for OpenAI-compatible and Anthropic APIs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import anthropic
from anthropic import Anthropic
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from ..runtime.interrupts import CancellationRequested

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in self.tool_calls
            ]
        return message


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        interface: str = "openai",
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.interface = interface
        self.api_key = api_key
        self.base_url = base_url
        if interface == "anthropic":
            self.client = Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def clone(self):
        return LLM(
            model=self.model,
            api_key=self.api_key,
            interface=self.interface,
            base_url=self.base_url,
            **self.extra,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        cancel_event=None,
    ) -> LLMResponse:
        if self.interface == "anthropic":
            return self._chat_anthropic(messages, tools=tools, on_token=on_token, cancel_event=cancel_event)

        return self._chat_openai(messages, tools=tools, on_token=on_token, cancel_event=cancel_event)

    def _chat_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        cancel_event=None,
    ) -> LLMResponse:
        _raise_if_cancelled(cancel_event)
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        stream = self._call_openai_stream(params, cancel_event=cancel_event)
        _raise_if_cancelled(cancel_event)
        response = _openai_stream_to_response(stream, on_token=on_token, cancel_event=cancel_event)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        return response

    def _call_openai_stream(self, params: dict, cancel_event=None):
        stream_params = {**params, "stream_options": {"include_usage": True}}
        try:
            return self._call_with_retry(stream_params, cancel_event=cancel_event)
        except CancellationRequested:
            raise
        except Exception:
            return self._call_with_retry(params, cancel_event=cancel_event)

    def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
        cancel_event=None,
    ) -> LLMResponse:
        _raise_if_cancelled(cancel_event)
        system_message, conversation = _extract_system_message(messages)
        params: dict = {
            "model": self.model,
            "messages": _to_anthropic_messages(conversation),
            "max_tokens": int(self.extra.get("max_tokens", 4096)),
        }

        temperature = self.extra.get("temperature")
        if temperature is not None:
            params["temperature"] = temperature
        if system_message:
            params["system"] = system_message
        if tools:
            params["tools"] = _to_anthropic_tools(tools)

        message = self._call_anthropic_with_retry(params, on_token=on_token, cancel_event=cancel_event)
        _raise_if_cancelled(cancel_event)
        response = _anthropic_message_to_response(message)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        return response

    def _call_with_retry(self, params: dict, max_retries: int = 3, cancel_event=None):
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                _sleep_until_retry_or_cancel(2 ** attempt, cancel_event)
            except APIError as exc:
                if exc.status_code and exc.status_code >= 500 and attempt < max_retries - 1:
                    _sleep_until_retry_or_cancel(2 ** attempt, cancel_event)
                else:
                    raise

    def _call_anthropic_with_retry(self, params: dict, max_retries: int = 3, on_token=None, cancel_event=None):
        for attempt in range(max_retries):
            try:
                with self.client.messages.stream(**params) as stream:
                    return _consume_anthropic_stream(stream, on_token=on_token, cancel_event=cancel_event)
            except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError):
                if attempt == max_retries - 1:
                    raise
                _sleep_until_retry_or_cancel(2 ** attempt, cancel_event)
            except anthropic.APIError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code and status_code >= 500 and attempt < max_retries - 1:
                    _sleep_until_retry_or_cancel(2 ** attempt, cancel_event)
                else:
                    raise


def _parse_openai_tool_calls(tool_call_map: dict[int, dict]) -> list[ToolCall]:
    parsed_tool_calls: list[ToolCall] = []
    for index in sorted(tool_call_map):
        raw = tool_call_map[index]
        logger.debug("OpenAI raw tool call index=%s payload=%r", index, raw)
        try:
            args = json.loads(raw["args"])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "OpenAI tool call arguments could not be parsed as JSON; "
                "index=%s id=%r name=%r raw_arguments=%r error=%s",
                index,
                raw.get("id", ""),
                raw.get("name", ""),
                raw.get("args", ""),
                exc,
            )
            args = {}
        parsed_tool_calls.append(
            ToolCall(id=raw.get("id", ""), name=raw.get("name", ""), arguments=args)
        )
    return parsed_tool_calls


def _openai_stream_to_response(stream, on_token=None, cancel_event=None) -> LLMResponse:
    content_parts: list[str] = []
    tool_call_map: dict[int, dict] = {}
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in stream:
        _raise_if_cancelled(cancel_event)

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
            completion_tokens = getattr(usage, "completion_tokens", completion_tokens) or completion_tokens

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if content:
            content_parts.append(content)
            if on_token:
                on_token(content)

        for tool_call_delta in getattr(delta, "tool_calls", None) or []:
            index = getattr(tool_call_delta, "index", 0)
            raw = tool_call_map.setdefault(index, {"id": "", "name": "", "args": ""})

            tool_call_id = getattr(tool_call_delta, "id", None)
            if tool_call_id:
                raw["id"] = tool_call_id

            function = getattr(tool_call_delta, "function", None)
            if function is None:
                continue

            name = getattr(function, "name", None)
            if name:
                raw["name"] = name

            arguments = getattr(function, "arguments", None)
            if arguments:
                raw["args"] += arguments

    return LLMResponse(
        content="".join(content_parts),
        tool_calls=_parse_openai_tool_calls(tool_call_map),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _openai_completion_to_response(completion, on_token=None) -> LLMResponse:
    if not getattr(completion, "choices", None):
        return LLMResponse()

    message = completion.choices[0].message
    content = getattr(message, "content", "") or ""
    if content and on_token:
        on_token(content)

    tool_call_map: dict[int, dict] = {}
    for index, tool_call in enumerate(getattr(message, "tool_calls", []) or []):
        function = getattr(tool_call, "function", None)
        tool_call_map[index] = {
            "id": getattr(tool_call, "id", "") or "",
            "name": getattr(function, "name", "") if function else "",
            "args": getattr(function, "arguments", "") if function else "",
        }

    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return LLMResponse(
        content=content,
        tool_calls=_parse_openai_tool_calls(tool_call_map),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _extract_system_message(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    conversation: list[dict] = []

    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if content:
                system_parts.append(content)
            continue
        conversation.append(message)

    return "\n\n".join(system_parts), conversation


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    converted = []
    for tool in tools:
        function = tool.get("function", {})
        converted.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    converted: list[dict] = []
    index = 0

    while index < len(messages):
        message = messages[index]
        role = message.get("role")

        if role == "tool":
            tool_results = []
            while index < len(messages) and messages[index].get("role") == "tool":
                current = messages[index]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": current.get("tool_call_id", ""),
                        "content": current.get("content", "") or "",
                    }
                )
                index += 1
            converted.append({"role": "user", "content": tool_results})
            continue

        if role in {"user", "assistant"}:
            converted.append({"role": role, "content": _to_anthropic_content(message)})

        index += 1

    return converted


def _to_anthropic_content(message: dict) -> list[dict]:
    blocks: list[dict] = []

    content = message.get("content")
    if content:
        blocks.append({"type": "text", "text": str(content)})

    for tool_call in message.get("tool_calls", []) or []:
        function = tool_call.get("function", {})
        raw_args = function.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
        else:
            parsed_args = raw_args or {}

        blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "input": parsed_args,
            }
        )

    return blocks or [{"type": "text", "text": ""}]


def _anthropic_message_to_response(message, on_token=None) -> LLMResponse:
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in getattr(message, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "") or ""
            content_parts.append(text)
            if text and on_token:
                on_token(text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    arguments=getattr(block, "input", {}) or {},
                )
            )

    usage = getattr(message, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    return LLMResponse(
        content="".join(content_parts),
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _consume_anthropic_stream(stream, on_token=None, cancel_event=None):
    text_stream = getattr(stream, "text_stream", None)
    if text_stream is not None:
        for text in text_stream:
            _raise_if_cancelled(cancel_event)
            if text and on_token:
                on_token(text)

    _raise_if_cancelled(cancel_event)
    return stream.get_final_message()


def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise CancellationRequested()


def _sleep_until_retry_or_cancel(delay_seconds: float, cancel_event) -> None:
    if cancel_event is None:
        time.sleep(delay_seconds)
        return

    if cancel_event.wait(delay_seconds):
        raise CancellationRequested()
