"""Session persistence helpers."""

import json
import time
from pathlib import Path

SESSIONS_DIR = Path.home() / ".kittycode" / "sessions"


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """Save conversation to disk and return the session ID."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not session_id:
        session_id = f"session_{int(time.time())}"

    data = {
        "id": session_id,
        "model": model,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    path = SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return session_id


def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """Load a saved session and return (messages, model)."""
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    return data["messages"], data["model"]


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError):
            continue

        preview = ""
        for message in data.get("messages", []):
            if message.get("role") == "user" and message.get("content"):
                preview = message["content"][:80]
                break

        sessions.append(
            {
                "id": data.get("id", path.stem),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "preview": preview,
            }
        )

    return sessions[:20]