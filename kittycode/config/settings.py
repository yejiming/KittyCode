"""Configuration loaded from ~/.kittycode/config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".kittycode" / "config.json"


def _pick(data: dict, *names: str, default=None):
    for name in names:
        value = data.get(name)
        if value not in (None, ""):
            return value
    return default


@dataclass
class Config:
    interface: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000

    @classmethod
    def from_file(cls, config_path: Path | str | None = None) -> "Config":
        path = Path(config_path).expanduser() if config_path is not None else CONFIG_PATH

        if not path.exists():
            return cls()

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a JSON object")

        interface = str(_pick(raw, "interface", "KITTYCODE_INTERFACE", default="openai")).lower()
        if interface not in {"openai", "anthropic"}:
            raise ValueError("interface must be 'openai' or 'anthropic'")

        default_model = "gpt-4o" if interface == "openai" else "claude-3-7-sonnet-latest"

        return cls(
            interface=interface,
            model=str(_pick(raw, "model", "KITTYCODE_MODEL", default=default_model)),
            api_key=str(_pick(raw, "api_key", "KITTYCODE_API_KEY", default="")),
            base_url=_pick(raw, "base_url", "KITTYCODE_BASE_URL"),
            max_tokens=int(_pick(raw, "max_tokens", "KITTYCODE_MAX_TOKENS", default=4096)),
            temperature=float(_pick(raw, "temperature", "KITTYCODE_TEMPERATURE", default=0.0)),
            max_context_tokens=int(
                _pick(
                    raw,
                    "max_context_tokens",
                    "max_context",
                    "KITTYCODE_MAX_CONTEXT",
                    default=128000,
                )
            ),
        )
