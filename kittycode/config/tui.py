"""Standalone TUI for editing KittyCode model configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit.shortcuts import button_dialog, input_dialog, message_dialog, radiolist_dialog

from .presets import PROVIDER_PRESETS, get_provider_preset
from .settings import CONFIG_PATH, Config

_DEFAULT_MAX_TOKENS = 32_000
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_CONTEXT = 200_000


@dataclass
class ConfigTUIModel:
    provider: str
    interface: str
    api_key: str
    model_name: str
    base_url: str


@dataclass
class ConfigTUIState:
    models: list[ConfigTUIModel] = field(default_factory=list)
    issue: str | None = None
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    max_context: int = _DEFAULT_MAX_CONTEXT


def _load_raw_defaults(config_path: Path) -> tuple[int, float, int]:
    if not config_path.exists():
        return (_DEFAULT_MAX_TOKENS, _DEFAULT_TEMPERATURE, _DEFAULT_MAX_CONTEXT)

    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return (_DEFAULT_MAX_TOKENS, _DEFAULT_TEMPERATURE, _DEFAULT_MAX_CONTEXT)

    if not isinstance(raw, dict):
        return (_DEFAULT_MAX_TOKENS, _DEFAULT_TEMPERATURE, _DEFAULT_MAX_CONTEXT)

    return (
        int(raw.get("max_tokens", _DEFAULT_MAX_TOKENS)),
        float(raw.get("temperature", _DEFAULT_TEMPERATURE)),
        int(raw.get("max_context", raw.get("max_context_tokens", _DEFAULT_MAX_CONTEXT))),
    )


def load_config_tui_state(config_path: Path | str | None = None) -> ConfigTUIState:
    path = Path(config_path).expanduser() if config_path is not None else CONFIG_PATH
    max_tokens, temperature, max_context = _load_raw_defaults(path)

    if not path.exists():
        return ConfigTUIState(
            issue=f"Config file not found: {path}\nUse this screen to create one.",
            max_tokens=max_tokens,
            temperature=temperature,
            max_context=max_context,
        )

    try:
        config = Config.from_file(path)
    except ValueError as exc:
        return ConfigTUIState(
            issue=f"Config needs repair:\n{exc}",
            max_tokens=max_tokens,
            temperature=temperature,
            max_context=max_context,
        )

    return ConfigTUIState(
        models=[
            ConfigTUIModel(
                provider=model.provider,
                interface=model.interface,
                api_key=model.api_key,
                model_name=model.model_name,
                base_url=model.base_url or "",
            )
            for model in config.models
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        max_context=config.max_context_tokens,
    )


def build_model_from_provider(
    provider: str,
    *,
    api_key: str = "",
    model_name: str = "",
    base_url: str | None = None,
) -> ConfigTUIModel:
    preset = get_provider_preset(provider)
    interface = preset.interface if preset is not None else "openai"
    resolved_base_url = base_url if base_url is not None else (preset.base_url if preset is not None else "")
    return ConfigTUIModel(
        provider=provider,
        interface=interface,
        api_key=api_key,
        model_name=model_name,
        base_url=resolved_base_url,
    )


def render_model_list(models: list[ConfigTUIModel]) -> str:
    provider_width = max(len("Provider"), *(len(model.provider) for model in models)) if models else len("Provider")
    model_width = max(len("Model"), *(len(model.model_name) for model in models)) if models else len("Model")
    base_width = max(len("Base URL"), *(len(model.base_url) for model in models)) if models else len("Base URL")

    header = f"{'Provider':<{provider_width}} | {'Model':<{model_width}} | {'Base URL':<{base_width}}"
    divider = f"{'-' * provider_width}-+-{'-' * model_width}-+-{'-' * base_width}"
    if not models:
        return "\n".join([header, divider, "(no models configured yet)"])

    rows = [
        f"{model.provider:<{provider_width}} | {model.model_name:<{model_width}} | {model.base_url:<{base_width}}"
        for model in models
    ]
    return "\n".join([header, divider, *rows])


def write_config_tui_state(state: ConfigTUIState, config_path: Path | str | None = None) -> None:
    path = Path(config_path).expanduser() if config_path is not None else CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": [
            {
                "interface": model.interface,
                "provider": model.provider,
                "api_key": model.api_key,
                "model_name": model.model_name,
                "base_url": model.base_url,
            }
            for model in state.models
        ],
        "max_tokens": state.max_tokens,
        "temperature": state.temperature,
        "max_context": state.max_context,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _select_provider(current_provider: str | None = None) -> str | None:
    values = [(preset.provider, preset.provider) for preset in PROVIDER_PRESETS]
    return radiolist_dialog(
        title="Select Provider",
        text="Choose a provider preset for this model.",
        values=values,
        default=current_provider or values[0][0],
    ).run()


def _prompt_text(title: str, label: str, default: str = "") -> str | None:
    return input_dialog(
        title=title,
        text=label,
        default=default,
    ).run()


def _edit_model(existing: ConfigTUIModel | None = None) -> ConfigTUIModel | None:
    selected_provider = _select_provider(existing.provider if existing is not None else None)
    if selected_provider is None:
        return None

    preset = get_provider_preset(selected_provider)
    api_key = _prompt_text("Model Config", "API KEY", existing.api_key if existing is not None else "")
    if api_key is None:
        return None
    model_name = _prompt_text("Model Config", "Model Name", existing.model_name if existing is not None else "")
    if model_name is None:
        return None
    default_base_url = existing.base_url if existing is not None and existing.provider == selected_provider else (
        preset.base_url if preset is not None else (existing.base_url if existing is not None else "")
    )
    base_url = _prompt_text("Model Config", "Base URL", default_base_url)
    if base_url is None:
        return None

    return build_model_from_provider(
        selected_provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
    )


def _select_model_index(models: list[ConfigTUIModel]) -> int | None:
    if not models:
        message_dialog(
            title="Edit Model",
            text="No configured models yet. Add one first.",
        ).run()
        return None

    values = [
        (index, f"{model.provider} | {model.model_name} | {model.base_url}")
        for index, model in enumerate(models)
    ]
    return radiolist_dialog(
        title="Edit Model",
        text="Choose an existing model to edit.",
        values=values,
        default=values[0][0],
    ).run()


def run_config_tui(config_path: Path | str | None = None) -> int:
    path = Path(config_path).expanduser() if config_path is not None else CONFIG_PATH
    state = load_config_tui_state(path)

    while True:
        body_parts = []
        if state.issue:
            body_parts.append(state.issue)
            body_parts.append("")
        body_parts.append(render_model_list(state.models))
        body_parts.append("")
        body_parts.append("Select an action below.")

        action = button_dialog(
            title="KittyCode Model Config",
            text="\n".join(body_parts),
            buttons=[
                ("Add Model", "add"),
                ("Edit Model", "edit"),
                ("Save & Exit", "save"),
                ("Quit", "quit"),
            ],
        ).run()

        if action in (None, "quit"):
            return 0

        if action == "add":
            updated = _edit_model()
            if updated is not None:
                state.models.append(updated)
                state.issue = None
            continue

        if action == "edit":
            index = _select_model_index(state.models)
            if index is None:
                continue
            updated = _edit_model(state.models[index])
            if updated is not None:
                state.models[index] = updated
                state.issue = None
            continue

        if action == "save":
            write_config_tui_state(state, path)
            message_dialog(
                title="Config Saved",
                text=f"Saved configuration to:\n{path}",
            ).run()
            return 0

    return 0
