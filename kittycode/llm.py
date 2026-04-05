"""LLM provider layer for OpenAI-compatible and Anthropic APIs."""

import json
import time
from dataclasses import dataclass, field

import anthropic
from anthropic import Anthropic
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from .interrupts import CancellationRequested


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

        try:
            params["stream_options"] = {"include_usage": True}
            stream = self._call_with_retry(params, cancel_event=cancel_event)
        except Exception:
            params.pop("stream_options", None)
            stream = self._call_with_retry(params, cancel_event=cancel_event)

        content_parts: list[str] = []
        tool_call_map: dict[int, dict] = {}
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in stream:
            _raise_if_cancelled(cancel_event)
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index
                    if index not in tool_call_map:
                        tool_call_map[index] = {"id": "", "name": "", "args": ""}
                    if tool_call_delta.id:
                        tool_call_map[index]["id"] = tool_call_delta.id
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_call_map[index]["name"] = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call_map[index]["args"] += tool_call_delta.function.arguments

        parsed_tool_calls: list[ToolCall] = []
        for index in sorted(tool_call_map):
            raw = tool_call_map[index]
            try:
                args = json.loads(raw["args"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            parsed_tool_calls.append(
                ToolCall(id=raw["id"], name=raw["name"], arguments=args)
            )

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

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

        message = self._call_anthropic_with_retry(params, cancel_event=cancel_event)
        _raise_if_cancelled(cancel_event)
        response = _anthropic_message_to_response(message, on_token=on_token)
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

    def _call_anthropic_with_retry(self, params: dict, max_retries: int = 3, cancel_event=None):
        for attempt in range(max_retries):
            try:
                return self.client.messages.create(**params)
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


def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise CancellationRequested()


def _sleep_until_retry_or_cancel(delay_seconds: float, cancel_event) -> None:
    if cancel_event is None:
        time.sleep(delay_seconds)
        return

    if cancel_event.wait(delay_seconds):
        raise CancellationRequested()
