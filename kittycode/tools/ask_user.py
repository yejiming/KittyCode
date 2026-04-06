"""Interactive user questions."""

from __future__ import annotations

from .base import Tool


class AskUserTool(Tool):
    name = "ask_user"
    description = """
    Ask the user one or more questions during execution to clarify
    requirements, gather preferences, or make implementation decisions.
    """
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "Questions to ask the user (1-4 items)",
                "items": {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "string",
                            "description": "Short label for the question",
                        },
                        "question": {
                            "type": "string",
                            "description": "The full question to ask the user",
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": "Allow selecting multiple options",
                        },
                        "allowFreeformInput": {
                            "type": "boolean",
                            "description": "Allow free-form text in addition to fixed options",
                        },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "Display text for the option",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Optional explanation for the option",
                                    },
                                    "recommended": {
                                        "type": "boolean",
                                        "description": "Mark the option as the recommended default",
                                    },
                                },
                                "required": ["label"],
                            },
                        },
                    },
                    "required": ["header", "question"],
                },
            }
        },
        "required": ["questions"],
    }

    _parent_agent = None

    def execute(self, questions: list[dict]) -> str:
        try:
            normalized_questions = _normalize_questions(questions)
        except ValueError as exc:
            return f"Error: {exc}"

        handler = getattr(self._parent_agent, "ask_user_handler", None)
        if not callable(handler):
            return "Error: ask_user requires an interactive KittyCode session"

        answers = handler(normalized_questions) or {}
        if not answers:
            return "User declined to answer the questions."

        lines = ["User answers:"]
        for item in normalized_questions:
            answer = answers.get(item["question"], "") or "(no answer)"
            lines.append(f"- {item['header']}: {answer}")
            lines.append(f"  {item['question']}")
        return "\n".join(lines)


def _normalize_questions(questions: list[dict]) -> list[dict]:
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must contain at least one item")
    if len(questions) > 4:
        raise ValueError("questions may contain at most 4 items")

    normalized = []
    seen_questions: set[str] = set()

    for index, question in enumerate(questions, 1):
        if not isinstance(question, dict):
            raise ValueError(f"question #{index} must be an object")

        header = str(question.get("header", "")).strip()
        question_text = str(question.get("question", "")).strip()
        if not header:
            raise ValueError(f"question #{index} is missing header")
        if not question_text:
            raise ValueError(f"question #{index} is missing question text")
        if question_text in seen_questions:
            raise ValueError("question texts must be unique")
        seen_questions.add(question_text)

        raw_options = question.get("options") or []
        if raw_options and (not isinstance(raw_options, list) or len(raw_options) > 4):
            raise ValueError(f"question #{index} must have at most 4 options")

        options = []
        seen_labels: set[str] = set()
        for option_index, option in enumerate(raw_options, 1):
            if not isinstance(option, dict):
                raise ValueError(f"question #{index} option #{option_index} must be an object")
            label = str(option.get("label", "")).strip()
            if not label:
                raise ValueError(f"question #{index} option #{option_index} is missing label")
            if label.casefold() in seen_labels:
                raise ValueError(f"question #{index} contains duplicate option label {label!r}")
            seen_labels.add(label.casefold())
            options.append(
                {
                    "label": label,
                    "description": str(option.get("description", "")).strip(),
                    "recommended": bool(option.get("recommended", False)),
                }
            )

        normalized.append(
            {
                "header": header,
                "question": question_text,
                "options": options,
                "multiSelect": bool(question.get("multiSelect", False)),
                "allowFreeformInput": bool(question.get("allowFreeformInput", True)),
            }
        )

    return normalized