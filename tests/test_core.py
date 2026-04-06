"""Tests for core modules: config, context, session, imports."""

import json
import types
import threading

import pytest

from kittycode import ALL_TOOLS, Agent, Config, LLM, __version__
from kittycode.context import ContextManager, estimate_tokens
from kittycode.llm import LLMResponse, ToolCall
import kittycode.session as session_module
from kittycode.session import list_sessions, load_session, save_session


def test_version():
    assert __version__ == "0.1.2"


def test_public_api_exports():
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert len(ALL_TOOLS) == 13


def test_config_from_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "interface": "openai",
        "api_key": "test-key",
        "model": "test-model",
        "base_url": "https://example.com/v1",
        "max_tokens": 1234,
        "temperature": 0.5,
        "max_context": 4567,
    }))

    config = Config.from_file(config_path)

    assert config.interface == "openai"
    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == "https://example.com/v1"
    assert config.max_tokens == 1234
    assert config.temperature == 0.5
    assert config.max_context_tokens == 4567


def test_config_defaults_when_file_missing(tmp_path):
    config = Config.from_file(tmp_path / "missing.json")
    assert config.interface == "openai"
    assert config.model == "gpt-4o"
    assert config.max_tokens == 4096
    assert config.temperature == 0.0


def test_config_does_not_read_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KITTYCODE_MODEL", "env-model")
    monkeypatch.setenv("KITTYCODE_API_KEY", "env-key")

    config = Config.from_file(tmp_path / "missing.json")

    assert config.model == "gpt-4o"
    assert config.api_key == ""


def test_config_supports_anthropic_interface(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "interface": "anthropic",
        "api_key": "anthropic-key",
        "model": "claude-3-7-sonnet-latest",
        "base_url": "https://api.anthropic.com",
    }))

    config = Config.from_file(config_path)

    assert config.interface == "anthropic"
    assert config.api_key == "anthropic-key"
    assert config.model == "claude-3-7-sonnet-latest"
    assert config.base_url == "https://api.anthropic.com"


def test_config_invalid_interface(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"interface": "invalid"}))

    with pytest.raises(ValueError, match="interface must be 'openai' or 'anthropic'"):
        Config.from_file(config_path)


def test_estimate_tokens():
    messages = [{"role": "user", "content": "hello world"}]
    tokens = estimate_tokens(messages)
    assert tokens > 0
    assert tokens < 100


def test_context_snip():
    context = ContextManager(max_tokens=3000)
    messages = [{"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000}]
    before = estimate_tokens(messages)
    context._snip_tool_outputs(messages)
    after = estimate_tokens(messages)
    assert after < before


def test_context_compress():
    context = ContextManager(max_tokens=2000)
    messages = []
    for index in range(20):
        messages.append({"role": "user", "content": f"msg {index} " + "a" * 200})
        messages.append({"role": "tool", "tool_call_id": f"t{index}", "content": "b" * 2000})
    before = estimate_tokens(messages)
    context.maybe_compress(messages, None)
    after = estimate_tokens(messages)
    assert after < before
    assert len(messages) < 40


def test_session_save_load(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    messages = [{"role": "user", "content": "test message"}]
    save_session(messages, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == messages
    assert loaded[1] == "test-model"


def test_session_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    assert load_session("nonexistent_session_id") is None


def test_list_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    sessions = list_sessions()
    assert isinstance(sessions, list)


class CancelAwareLLM:
    def __init__(self):
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("LLM chat should not be called after cancellation")


def test_agent_returns_interrupted_before_llm_call():
    llm = CancelAwareLLM()
    agent = Agent(llm=llm)
    cancel_event = threading.Event()
    cancel_event.set()

    response = agent.chat("stop now", cancel_event=cancel_event)

    assert response == "(interrupted)"
    assert llm.calls == 0


class SingleToolCallLLM:
    def __init__(self):
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return LLMResponse(
            tool_calls=[ToolCall(id="tool-1", name="fake_tool", arguments={"value": 1})]
        )


def test_agent_skips_tool_execution_when_cancelled_after_tool_announcement(monkeypatch):
    llm = SingleToolCallLLM()
    agent = Agent(llm=llm)
    executed = []

    class FakeTool:
        def execute(self, **kwargs):
            executed.append(kwargs)
            return "ran"

    monkeypatch.setattr("kittycode.agent.get_tool", lambda name: FakeTool())
    cancel_event = threading.Event()

    response = agent.chat(
        "run tool",
        cancel_event=cancel_event,
        on_tool=lambda *_args: cancel_event.set(),
    )

    assert response == "(interrupted)"
    assert executed == []


def test_agent_tool_forwards_cancel_event_to_sub_agent(monkeypatch):
    from kittycode.tools.agent import AgentTool

    captured = {}

    class FakeSubAgent:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def chat(self, task, cancel_event=None):
            captured["task"] = task
            captured["cancel_event"] = cancel_event
            return "(interrupted)" if cancel_event is not None and cancel_event.is_set() else "done"

    tool = AgentTool()
    tool._parent_agent = types.SimpleNamespace(
        llm=object(),
        tools=[tool],
        context=types.SimpleNamespace(max_tokens=1234),
    )
    cancel_event = threading.Event()
    cancel_event.set()

    monkeypatch.setattr("kittycode.agent.Agent", FakeSubAgent)

    result = tool.execute("stop sub-agent", cancel_event=cancel_event)

    assert result == "[Sub-agent completed]\n(interrupted)"
    assert captured["task"] == "stop sub-agent"
    assert captured["cancel_event"] is cancel_event
