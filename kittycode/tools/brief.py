"""User-facing brief messages."""

from __future__ import annotations

from pathlib import Path

from .base import Tool

_VALID_STATUSES = {"normal", "proactive"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


class BriefTool(Tool):
    name = "brief"
    description = """
    Send a concise user-facing message and optional local file attachments.
    Use this for progress updates, blockers, or proactive status notifications.
    """
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to show the user",
            },
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional absolute or cwd-relative file paths to attach",
            },
            "status": {
                "type": "string",
                "description": "normal for replies, proactive for unsolicited updates",
            },
        },
        "required": ["message"],
    }

    _parent_agent = None

    def execute(
        self,
        message: str,
        attachments: list[str] | None = None,
        status: str = "normal",
    ) -> str:
        text = message.strip()
        if not text:
            return "Error: message is required"
        if status not in _VALID_STATUSES:
            return f"Error: invalid status {status!r}; expected one of {sorted(_VALID_STATUSES)}"

        resolved_attachments = []
        for raw_path in attachments or []:
            try:
                resolved_attachments.append(_resolve_attachment(raw_path))
            except ValueError as exc:
                return f"Error: {exc}"

        payload = {
            "message": text,
            "status": status,
            "attachments": resolved_attachments,
        }

        if self._parent_agent is not None:
            self._parent_agent.brief_messages.append(payload)
            callback = getattr(self._parent_agent, "on_brief_message", None)
            if callable(callback):
                callback(payload)
        else:
            messages = list(getattr(self, "_messages", []))
            messages.append(payload)
            self._messages = messages

        lines = [f"Sent brief message ({status}).", text]
        if resolved_attachments:
            lines.append("Attachments:")
            lines.extend(
                f"- {item['path']} ({item['size']} bytes, image={item['is_image']})"
                for item in resolved_attachments
            )
        return "\n".join(lines)


def _resolve_attachment(raw_path: str) -> dict:
    if not raw_path.strip():
        raise ValueError("attachment paths must be non-empty strings")

    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"attachment {raw_path!r} does not exist")
    if not path.is_file():
        raise ValueError(f"attachment {raw_path!r} is not a regular file")

    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "is_image": path.suffix.lower() in _IMAGE_SUFFIXES,
    }