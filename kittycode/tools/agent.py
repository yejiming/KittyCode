"""Sub-agent spawning."""

import copy
from contextlib import nullcontext

from .base import Tool


class AgentTool(Tool):
    name = "agent"
    description = """
    Spawn a sub-agent to handle a complex sub-task independently.
    The sub-agent has its own context and tool access. Use this for:
    researching a codebase, implementing a multi-step change in isolation,
    or any task that would benefit from a fresh context window.
    """
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
        },
        "required": ["task"],
    }

    _parent_agent = None

    def execute(self, task: str, cancel_event=None) -> str:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"

        from ..agent import Agent

        parent = self._parent_agent
        sub_agent = Agent(
            llm=parent.llm.clone() if hasattr(parent.llm, "clone") else parent.llm,
            tools=[type(tool)() for tool in parent.tools if tool.name != "agent"],
            max_context_tokens=parent.context.max_tokens,
            max_rounds=20,
        )
        sub_agent.ask_user_handler = getattr(parent, "ask_user_handler", None)
        sub_agent.on_brief_message = getattr(parent, "on_brief_message", None)
        sub_agent.todos = copy.deepcopy(getattr(parent, "todos", []))
        sub_agent.brief_messages = copy.deepcopy(getattr(parent, "brief_messages", []))

        try:
            result = sub_agent.chat(task, cancel_event=cancel_event)
            with _parent_state_lock(parent):
                parent.todos = _merge_todos(getattr(parent, "todos", []), sub_agent.todos)
                parent.brief_messages = _merge_brief_messages(
                    getattr(parent, "brief_messages", []),
                    sub_agent.brief_messages,
                )
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return f"[Sub-agent completed]\n{result}"
        except Exception as exc:
            return f"Sub-agent error: {exc}"


def _parent_state_lock(parent):
    lock = getattr(parent, "_state_lock", None) or getattr(parent, "state_lock", None)
    return lock if lock is not None else nullcontext()


def _merge_todos(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = [copy.deepcopy(item) for item in existing]
    positions = {}

    for index, item in enumerate(merged):
        marker = _todo_marker(item)
        if marker is not None and marker not in positions:
            positions[marker] = index

    for item in incoming or []:
        copied = copy.deepcopy(item)
        marker = _todo_marker(copied)
        if marker is not None and marker in positions:
            merged[positions[marker]] = copied
            continue
        merged.append(copied)
        if marker is not None:
            positions[marker] = len(merged) - 1

    return merged


def _merge_brief_messages(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = [copy.deepcopy(item) for item in existing]
    seen = {_brief_marker(item) for item in merged}

    for item in incoming or []:
        copied = copy.deepcopy(item)
        marker = _brief_marker(copied)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(copied)

    return merged


def _todo_marker(item: dict) -> str | None:
    content = item.get("content")
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    return stripped.casefold()


def _brief_marker(item: dict):
    attachments = []
    for attachment in item.get("attachments") or []:
        if isinstance(attachment, dict):
            attachments.append(
                (
                    attachment.get("path"),
                    attachment.get("size"),
                    attachment.get("is_image"),
                )
            )
        else:
            attachments.append(attachment)
    return (
        item.get("message"),
        item.get("status"),
        tuple(attachments),
    )
