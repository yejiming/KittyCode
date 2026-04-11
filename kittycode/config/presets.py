"""Provider preset catalog for model configuration flows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    provider: str
    base_url: str
    interface: str


PROVIDER_PRESETS = (
    ProviderPreset("DeepSeek", "https://api.deepseek.com/v1", "openai"),
    ProviderPreset("Zhipu GLM", "https://open.bigmodel.cn/api/paas/v4", "openai"),
    ProviderPreset("Zhipu GLM en", "https://api.z.ai/v1", "openai"),
    ProviderPreset("Bailian", "https://dashscope.aliyuncs.com/compatible-mode/v1", "openai"),
    ProviderPreset("Kimi", "https://api.moonshot.cn/v1", "openai"),
    ProviderPreset("Kimi For Coding", "https://api.kimi.com/coding", "anthropic"),
    ProviderPreset("StepFun", "https://api.stepfun.ai/v1", "openai"),
    ProviderPreset("Minimax", "https://api.minimaxi.com/v1", "openai"),
    ProviderPreset("Minimax en", "https://platform.minimax.io", "openai"),
    ProviderPreset("DouBaoSeed", "https://ark.cn-beijing.volces.com/api/v3", "openai"),
    ProviderPreset("Xiaomi MiMo", "https://api.xiaomimimo.com/v1", "openai"),
    ProviderPreset("ModelScope", "https://api-inference.modelscope.cn/v1", "openai"),
    ProviderPreset("OpenRouter", "https://openrouter.ai/api/v1", "openai"),
    ProviderPreset("Ollama", "http://localhost:11434/v1", "openai")
)


def get_provider_preset(provider: str) -> ProviderPreset | None:
    for preset in PROVIDER_PRESETS:
        if preset.provider == provider:
            return preset
    return None
