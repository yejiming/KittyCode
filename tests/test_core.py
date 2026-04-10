"""Tests for core modules: config, context, session, imports."""

import json
import logging
import types
import threading
import importlib

import pytest

from kittycode import ALL_TOOLS, Agent, Config, LLM, __version__
from kittycode.agent import INTERRUPTED_TOOL_RESULT
from kittycode.context import ContextManager, estimate_tokens
from kittycode.config import PROVIDER_PRESETS, ProviderPreset, get_provider_preset
from kittycode.config.tui import (
    build_model_from_provider,
    load_config_tui_state,
    render_model_list,
    write_config_tui_state,
)
from kittycode.llm import LLMResponse, ToolCall
from kittycode.logging_utils import configure_logging
import kittycode.main as main_module
import kittycode.session as session_module
from kittycode.session import list_sessions, load_session, save_session


def test_public_api_exports():
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert len(ALL_TOOLS) == 13


def test_package_roots_reexport_from_dedicated_modules():
    config_pkg = importlib.import_module("kittycode.config")
    config_impl = importlib.import_module("kittycode.config.settings")
    llm_pkg = importlib.import_module("kittycode.llm")
    llm_impl = importlib.import_module("kittycode.llm.provider")
    prompt_pkg = importlib.import_module("kittycode.prompt")
    prompt_impl = importlib.import_module("kittycode.prompt.builder")
    skills_pkg = importlib.import_module("kittycode.skills")
    skills_impl = importlib.import_module("kittycode.skills.discovery")

    assert config_pkg.Config is config_impl.Config
    assert config_pkg.CONFIG_PATH == config_impl.CONFIG_PATH
    assert config_pkg.StoredModelConfig is config_impl.StoredModelConfig
    assert config_pkg.ProviderPreset is ProviderPreset
    assert config_pkg.PROVIDER_PRESETS is PROVIDER_PRESETS
    assert config_pkg.get_provider_preset is get_provider_preset

    assert llm_pkg.LLM is llm_impl.LLM
    assert llm_pkg.LLMResponse is llm_impl.LLMResponse
    assert llm_pkg.ToolCall is llm_impl.ToolCall
    assert llm_pkg._openai_stream_to_response is llm_impl._openai_stream_to_response

    assert prompt_pkg.system_prompt is prompt_impl.system_prompt
    assert prompt_pkg.user_prompt is prompt_impl.user_prompt
    assert prompt_pkg.AGENTS_DOC == prompt_impl.AGENTS_DOC

    assert skills_pkg.SkillDefinition is skills_impl.SkillDefinition
    assert skills_pkg.load_skills is skills_impl.load_skills
    assert skills_pkg.SKILLS_DIR == skills_impl.SKILLS_DIR


def test_config_from_file(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "openai",
                "provider": "openai",
                "api_key": "test-key",
                "model_name": "test-model",
                "base_url": "https://example.com/v1",
            }
        ],
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
    assert len(config.models) == 1
    assert config.models[0].interface == "openai"
    assert config.models[0].provider == "openai"
    assert config.models[0].api_key == "test-key"
    assert config.models[0].model_name == "test-model"
    assert config.models[0].base_url == "https://example.com/v1"


def test_config_defaults_when_file_missing(tmp_path):
    config = Config.from_file(tmp_path / "missing.json")
    assert config.interface == "openai"
    assert config.model == "gpt-4o"
    assert config.max_tokens == 32000
    assert config.temperature == 0.0
    assert config.max_context_tokens == 200000
    assert config.models == []


def test_config_supports_anthropic_provider(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "anthropic",
                "provider": "anthropic",
                "api_key": "anthropic-key",
                "model_name": "claude-3-7-sonnet-latest",
                "base_url": "https://api.anthropic.com",
            }
        ]
    }))

    config = Config.from_file(config_path)

    assert config.interface == "anthropic"
    assert config.api_key == "anthropic-key"
    assert config.model == "claude-3-7-sonnet-latest"
    assert config.base_url == "https://api.anthropic.com"
    assert config.models[0].provider == "anthropic"


def test_config_invalid_provider(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "openai",
                "provider": "invalid-provider",
                "api_key": "bad-key",
                "model_name": "bad-model",
                "base_url": "https://example.com",
            }
        ]
    }))

    config = Config.from_file(config_path)

    assert config.models[0].provider == "invalid-provider"
    assert config.interface == "openai"


def test_config_invalid_interface(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "invalid",
                "provider": "DeepSeek",
                "api_key": "bad-key",
                "model_name": "bad-model",
                "base_url": "https://example.com",
            }
        ]
    }))

    with pytest.raises(ValueError, match="models\\[0\\]\\.interface must be 'openai' or 'anthropic'"):
        Config.from_file(config_path)


def test_config_supports_multi_model_array_and_preserves_order(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "openai",
                "provider": "openai",
                "api_key": "sk-openai",
                "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
            },
            {
                "interface": "anthropic",
                "provider": "anthropic",
                "api_key": "sk-ant",
                "model_name": "claude-3-7-sonnet-latest",
                "base_url": "https://api.anthropic.com",
            },
        ],
    }))

    config = Config.from_file(config_path)

    assert [model.interface for model in config.models] == ["openai", "anthropic"]
    assert [model.provider for model in config.models] == ["openai", "anthropic"]
    assert config.interface == "openai"
    assert config.api_key == "sk-openai"
    assert config.model == "gpt-4o"
    assert config.base_url == "https://api.openai.com/v1"


def test_config_uses_phase_one_defaults_for_multi_model_shape(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "openai",
                "provider": "openai",
                "api_key": "sk-openai",
                "model_name": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
            }
        ]
    }))

    config = Config.from_file(config_path)

    assert config.max_tokens == 32000
    assert config.temperature == 0
    assert config.max_context_tokens == 200000


def test_config_requires_models_array(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"interface": "anthropic"}))

    with pytest.raises(ValueError, match="models is required"):
        Config.from_file(config_path)


def test_config_rejects_empty_models_list(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"models": []}))

    with pytest.raises(ValueError, match="models must contain at least one model"):
        Config.from_file(config_path)


def test_config_rejects_model_entry_missing_required_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "openai",
                "provider": "openai",
                "api_key": "sk-openai",
                "base_url": "https://api.openai.com/v1",
            }
        ]
    }))

    with pytest.raises(ValueError, match="models\\[0\\]\\.model_name is required"):
        Config.from_file(config_path)


def test_provider_presets_match_requested_catalog_exactly():
    assert [preset.provider for preset in PROVIDER_PRESETS] == [
        "DeepSeek",
        "Zhipu GLM",
        "Zhipu GLM en",
        "Bailian",
        "Kimi",
        "Kimi For Coding",
        "StepFun",
        "Minimax",
        "Minimax en",
        "DouBaoSeed",
        "Xiaomi MiMo",
        "ModelScope",
        "OpenRouter",
    ]


def test_provider_presets_expose_exact_default_base_urls_and_interfaces():
    expected = {
        "DeepSeek": ("https://api.deepseek.com/v1", "openai"),
        "Zhipu GLM": ("https://open.bigmodel.cn/api/paas/v4", "openai"),
        "Zhipu GLM en": ("https://api.z.ai/v1", "openai"),
        "Bailian": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "openai"),
        "Kimi": ("https://api.moonshot.cn/v1", "openai"),
        "Kimi For Coding": ("https://api.kimi.com/coding", "anthropic"),
        "StepFun": ("https://api.stepfun.ai/v1", "openai"),
        "Minimax": ("https://api.minimaxi.com/v1", "openai"),
        "Minimax en": ("https://platform.minimax.io", "openai"),
        "DouBaoSeed": ("https://ark.cn-beijing.volces.com/api/v3", "openai"),
        "Xiaomi MiMo": ("https://api.xiaomimimo.com/v1", "openai"),
        "ModelScope": ("https://api-inference.modelscope.cn/v1", "openai"),
        "OpenRouter": ("https://openrouter.ai/api/v1", "openai"),
    }

    actual = {
        preset.provider: (preset.base_url, preset.interface)
        for preset in PROVIDER_PRESETS
    }

    assert actual == expected


def test_get_provider_preset_returns_exact_preset_for_known_provider():
    preset = get_provider_preset("Kimi For Coding")

    assert preset == ProviderPreset(
        provider="Kimi For Coding",
        base_url="https://api.kimi.com/coding",
        interface="anthropic",
    )


def test_get_provider_preset_returns_none_for_unknown_provider():
    assert get_provider_preset("OpenAI") is None


def test_load_config_tui_state_reads_existing_models(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "models": [
            {
                "interface": "openai",
                "provider": "DeepSeek",
                "api_key": "sk-test",
                "model_name": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
            }
        ]
    }))

    state = load_config_tui_state(config_path)

    assert state.issue is None
    assert len(state.models) == 1
    assert state.models[0].provider == "DeepSeek"
    assert state.models[0].model_name == "deepseek-chat"


def test_load_config_tui_state_handles_missing_or_invalid_config(tmp_path):
    missing = load_config_tui_state(tmp_path / "missing.json")
    assert missing.issue is not None
    assert "Config file not found" in missing.issue

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{")
    invalid = load_config_tui_state(invalid_path)
    assert invalid.issue is not None
    assert "Config needs repair" in invalid.issue


def test_config_tui_builds_and_renders_model_list_and_writes_file(tmp_path):
    model = build_model_from_provider(
        "DeepSeek",
        api_key="sk-test",
        model_name="deepseek-chat",
    )

    assert model.interface == "openai"
    assert model.base_url == "https://api.deepseek.com/v1"

    rendered = render_model_list([model])
    assert "Provider" in rendered
    assert "Model" in rendered
    assert "Base URL" in rendered
    assert "DeepSeek" in rendered
    assert "deepseek-chat" in rendered

    state = load_config_tui_state(tmp_path / "missing.json")
    state.models.append(model)
    output_path = tmp_path / "config.json"
    write_config_tui_state(state, output_path)
    payload = json.loads(output_path.read_text())

    assert payload["models"][0] == {
        "interface": "openai",
        "provider": "DeepSeek",
        "api_key": "sk-test",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
    }


def test_config_activate_model_updates_runtime_fields():
    config = Config(
        interface="openai",
        model="gpt-4o",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        models=[
            importlib.import_module("kittycode.config.settings").StoredModelConfig(
                interface="openai",
                provider="OpenAI",
                api_key="sk-openai",
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
            ),
            importlib.import_module("kittycode.config.settings").StoredModelConfig(
                interface="anthropic",
                provider="Anthropic",
                api_key="sk-ant",
                model_name="claude-3-7-sonnet-latest",
                base_url="https://api.anthropic.com",
            ),
        ],
    )

    selected = config.activate_model(1)

    assert selected is config.models[1]
    assert config.interface == "anthropic"
    assert config.model == "claude-3-7-sonnet-latest"
    assert config.api_key == "sk-ant"
    assert config.base_url == "https://api.anthropic.com"


def test_config_active_model_index_returns_matching_entry_or_none():
    settings_module = importlib.import_module("kittycode.config.settings")
    config = Config(
        interface="openai",
        model="gpt-4o",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="OpenAI",
                api_key="sk-openai",
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
            ),
            settings_module.StoredModelConfig(
                interface="anthropic",
                provider="Anthropic",
                api_key="sk-ant",
                model_name="claude-3-7-sonnet-latest",
                base_url="https://api.anthropic.com",
            ),
        ],
    )

    assert config.active_model_index() == 0

    config.interface = "openai"
    config.model = "override-model"
    config.api_key = "override-key"
    config.base_url = "https://override.example/v1"

    assert config.active_model_index() is None


def test_config_activate_model_rejects_invalid_index():
    settings_module = importlib.import_module("kittycode.config.settings")
    config = Config(
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="OpenAI",
                api_key="sk-openai",
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
            )
        ]
    )

    with pytest.raises(ValueError, match="Invalid model index: 2"):
        config.activate_model(2)


def test_llm_reconfigure_rebuilds_client_and_preserves_counters(monkeypatch):
    provider_module = importlib.import_module("kittycode.llm.provider")
    constructed = []

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            constructed.append(("openai", api_key, base_url))

    class FakeAnthropic:
        def __init__(self, *, api_key, base_url):
            constructed.append(("anthropic", api_key, base_url))

    monkeypatch.setattr(provider_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(provider_module, "Anthropic", FakeAnthropic)

    llm = LLM(
        model="gpt-4o",
        api_key="sk-openai",
        interface="openai",
        base_url="https://api.openai.com/v1",
        max_tokens=2048,
    )
    llm.total_prompt_tokens = 11
    llm.total_completion_tokens = 7

    llm.reconfigure(
        model="claude-3-7-sonnet-latest",
        api_key="sk-ant",
        interface="anthropic",
        base_url="https://api.anthropic.com",
    )

    assert llm.model == "claude-3-7-sonnet-latest"
    assert llm.api_key == "sk-ant"
    assert llm.interface == "anthropic"
    assert llm.base_url == "https://api.anthropic.com"
    assert llm.total_prompt_tokens == 11
    assert llm.total_completion_tokens == 7
    assert llm.extra == {"max_tokens": 2048}
    assert constructed == [
        ("openai", "sk-openai", "https://api.openai.com/v1"),
        ("anthropic", "sk-ant", "https://api.anthropic.com"),
    ]


def test_config_write_persists_active_model_first(tmp_path):
    settings_module = importlib.import_module("kittycode.config.settings")
    config = Config(
        interface="openai",
        model="gpt-4o",
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        models=[
            settings_module.StoredModelConfig(
                interface="openai",
                provider="OpenAI",
                api_key="sk-openai",
                model_name="gpt-4o",
                base_url="https://api.openai.com/v1",
            ),
            settings_module.StoredModelConfig(
                interface="anthropic",
                provider="Anthropic",
                api_key="sk-ant",
                model_name="claude-3-7-sonnet-latest",
                base_url="https://api.anthropic.com",
            ),
        ],
    )

    config.activate_model(1)
    config_path = tmp_path / "config.json"
    config.write(config_path)

    payload = json.loads(config_path.read_text())
    assert payload["models"][0]["provider"] == "Anthropic"
    assert payload["models"][0]["model_name"] == "claude-3-7-sonnet-latest"
    assert payload["models"][1]["provider"] == "OpenAI"


def test_main_routes_config_flag_to_standalone_tui(monkeypatch):
    calls = []

    monkeypatch.setattr("sys.argv", ["kittycode", "--config"])
    monkeypatch.setattr(main_module, "run_config_tui", lambda: calls.append("config"))
    monkeypatch.setattr(main_module, "_cli_main", lambda: calls.append("cli"))

    assert main_module.main() == 0
    assert calls == ["config"]


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


def test_configure_logging_writes_to_default_log_file(tmp_path):
    log_path = configure_logging(base_dir=tmp_path)
    logger = logging.getLogger("kittycode.test")
    logger.warning("tool debug line")

    for handler in logging.getLogger("kittycode").handlers:
        handler.flush()

    assert log_path == tmp_path / "logs" / "kittycode.log"
    assert log_path.exists()
    assert "tool debug line" in log_path.read_text(encoding="utf-8")


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
    assert agent.messages[-2]["role"] == "assistant"
    assert agent.messages[-2]["tool_calls"][0]["id"] == "tool-1"
    assert agent.messages[-1] == {
        "role": "tool",
        "tool_call_id": "tool-1",
        "content": INTERRUPTED_TOOL_RESULT,
    }


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
