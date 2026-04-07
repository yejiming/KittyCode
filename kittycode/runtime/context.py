"""Multi-layer context compression."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM


def _approx_tokens(text: str) -> int:
    """Rough token count for mixed English and Chinese content."""
    return len(text) // 3


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        if message.get("content"):
            total += _approx_tokens(message["content"])
        if message.get("tool_calls"):
            total += _approx_tokens(str(message["tool_calls"]))
    return total


class ContextManager:
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        self._snip_at = int(max_tokens * 0.50)
        self._summarize_at = int(max_tokens * 0.70)
        self._collapse_at = int(max_tokens * 0.90)

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """Apply compression layers as needed."""
        current = estimate_tokens(messages)
        compressed = False

        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)

        if current > self._summarize_at and len(messages) > 10:
            if self._summarize_old(messages, llm, keep_recent=8):
                compressed = True
                current = estimate_tokens(messages)

        if current > self._collapse_at and len(messages) > 4:
            self._hard_collapse(messages, llm)
            compressed = True

        return compressed

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        changed = False
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content", "")
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            message["content"] = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            changed = True
        return changed

    def _summarize_old(
        self,
        messages: list[dict],
        llm: LLM | None,
        keep_recent: int = 8,
    ) -> bool:
        if len(messages) <= keep_recent:
            return False

        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]
        summary = self._get_summary(old_messages, llm)

        messages.clear()
        messages.append(
            {
                "role": "user",
                "content": f"[Context compressed - conversation summary]\n{summary}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Got it, I have the context from our earlier conversation.",
            }
        )
        messages.extend(recent_messages)
        return True

    def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        tail = messages[-4:] if len(messages) > 4 else messages[-2:]
        summary = self._get_summary(messages[:-len(tail)], llm)

        messages.clear()
        messages.append({"role": "user", "content": f"[Hard context reset]\n{summary}"})
        messages.append(
            {"role": "assistant", "content": "Context restored. Continuing from where we left off."}
        )
        messages.extend(tail)

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        flat = self._flatten(messages)

        if llm:
            try:
                response = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return response.content
            except Exception:
                pass

        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "?")
            text = message.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        import re

        files_seen = set()
        errors = []

        for message in messages:
            text = message.get("content", "") or ""
            for match in re.finditer(r"[\w./\-]+\.\w{1,5}", text):
                files_seen.add(match.group())
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"