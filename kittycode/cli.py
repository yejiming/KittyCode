"""Interactive CLI for KittyCode."""

import argparse
from dataclasses import dataclass
import inspect
import math
import os
import queue
import select
import re
from types import SimpleNamespace
import sys
import threading
import textwrap
import time

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - not expected on macOS/Linux
    termios = None
    tty = None

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import TextArea
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import CONFIG_PATH, Config
from .llm import LLM
from .runtime.agent import Agent
from .runtime.interrupts import CancellationRequested
from .runtime.logging import configure_logging
from .runtime.session import list_sessions, load_session, save_session

console = Console()
_PIXEL_CAT_ART = (
    " /\\_/\\\\\n"
    "( o.o )___________\n"
    " > ^           __)\n"
    " /_ __ ___ ___/\n"
    "   \\_/   V \\_\\\n"
)

_BUILTIN_COMMANDS = {
    "/help": "Show this help",
    "/reset": "Clear conversation history",
    "/skills": "Show loaded local skills",
    "/model": "Switch model mid-conversation",
    "/tokens": "Show token usage",
    "/compact": "Compress conversation context",
    "/save": "Save session to disk",
    "/sessions": "List saved sessions",
    "/quit": "Exit KittyCode",
}

_INPUT_AREA_MIN_HEIGHT = 1
_INPUT_AREA_MAX_HEIGHT = 8
_INPUT_PROMPT_LABEL = ">"
_AUTHOR_NAME = "Jimmy Ye"

ROLE_STYLE = {
    "system": "class:history.system",
    "user": "class:history.user",
    "assistant": "class:history.assistant",
    "tool": "class:history.tool",
    "startup": "class:startup.text",
}

_APP_STYLE = Style.from_dict(
    {
        "history.system": "fg:#56b6c2",
        "history.user": "fg:#27cd96",
        "history.assistant": "fg:#d68786",
        "history.assistant.label": "bold",
        "history.tool": "fg:#56b6c2",
        "footer": "fg:#d68786",
        "footer.label": "fg:#d68786 bold",
        "history.markdown.heading1": "bold underline",
        "history.markdown.heading2": "bold",
        "history.markdown.heading3": "bold",
        "history.markdown.heading4": "bold",
        "history.markdown.heading5": "bold",
        "history.markdown.heading6": "bold",
        "history.markdown.list_marker": "fg:#95a88f",
        "history.markdown.quote": "fg:#93a2ab",
        "history.markdown.emphasis": "italic",
        "history.markdown.strong": "bold",
        "history.markdown.strike": "strike",
        "history.markdown.code": "fg:#b6a58e",
        "history.markdown.fence": "fg:#8d98a5",
        "history.markdown.codeblock": "fg:#9ea7b2",
        "input.rule": "fg:#6b7280",
        "startup.text": "fg:#6b7280",
        "startup.frame": "fg:#d68786",
        "startup.cat": "fg:#d68786",
    }
)


class HistoryStyleProcessor(Processor):
    def __init__(self, get_line_metadata):
        self._get_line_metadata = get_line_metadata

    def apply_transformation(self, transformation_input):
        line_metadata = self._get_line_metadata()
        line_no = transformation_input.lineno
        metadata = line_metadata[line_no] if line_no < len(line_metadata) else {}
        base_style = metadata.get("base_style", "") if isinstance(metadata, dict) else ""

        if not base_style:
            return Transformation(transformation_input.fragments)

        line_text = "".join(fragment[1] for fragment in transformation_input.fragments)
        if isinstance(metadata, dict) and metadata.get("label"):
            return Transformation(
                [(_merge_prompt_toolkit_styles(line_text and transformation_input.fragments[0][0], base_style, "class:history.assistant.label"), line_text)]
            )
        if isinstance(metadata, dict) and metadata.get("startup"):
            return Transformation(_style_startup_line(line_text, base_style))
        if isinstance(metadata, dict) and metadata.get("markdown"):
            return Transformation(_style_history_markdown_line(line_text, metadata))

        fragments = []
        for fragment in transformation_input.fragments:
            style = fragment[0]
            text = fragment[1]
            rest = fragment[2:]
            merged_style = _merge_prompt_toolkit_styles(style, base_style)
            fragments.append((merged_style, text, *rest))

        return Transformation(fragments)


class SlashCommandCompleter(Completer):
    def __init__(self, command_provider):
        self.command_provider = command_provider

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        typed = text.casefold()
        for command in sorted(self.command_provider()):
            if command.casefold().startswith(typed):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=_BUILTIN_COMMANDS.get(command, "Use this skill"),
                )


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="kittycode",
        description="Minimal AI coding agent. Supports OpenAI-compatible and Anthropic APIs.",
    )
    parser.add_argument("--config", action="store_true", help="Open the guided configuration setup")
    parser.add_argument("-m", "--model", help=argparse.SUPPRESS)
    parser.add_argument("--interface", choices=["openai", "anthropic"], help=argparse.SUPPRESS)
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    parser.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def _load_config() -> Config:
    try:
        return Config.from_file()
    except ValueError as exc:
        console.print(f"[red bold]Invalid config file:[/] {CONFIG_PATH}")
        console.print("Run `kittycode --config` to create or repair your model configuration.")
        console.print(str(exc))
        sys.exit(1)


def _apply_cli_overrides(config: Config, args) -> Config:
    if args.model:
        config.model = args.model
    if args.interface:
        config.interface = args.interface
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key
    return config


def _ensure_api_key(config: Config) -> None:
    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            f"Populate {CONFIG_PATH} with JSON such as:\n"
            "\n"
            "{\n"
            '  "interface": "openai",\n'
            '  "api_key": "sk-...",\n'
            '  "model": "gpt-4o",\n'
            '  "base_url": "https://api.openai.com/v1",\n'
            '  "max_tokens": 4096,\n'
            '  "temperature": 0,\n'
            '  "max_context": 128000\n'
            "}\n"
        )
        sys.exit(1)


def _create_agent(config: Config) -> Agent:
    llm = LLM(
        model=config.model,
        api_key=config.api_key,
        interface=config.interface,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return Agent(llm=llm, max_context_tokens=config.max_context_tokens)


def _resume_agent_session(agent: Agent, session_id: str) -> Agent:
    loaded = load_session(session_id)
    if loaded:
        agent.messages, _loaded_model = loaded
        console.print(f"[green]Resumed session: {session_id}[/green]")
        return agent

    console.print(f"[red]Session '{session_id}' not found.[/red]")
    sys.exit(1)


def _build_cli_runtime(args) -> tuple[Config, Agent]:
    config = _apply_cli_overrides(_load_config(), args)
    _ensure_api_key(config)
    agent = _create_agent(config)

    if args.resume:
        agent = _resume_agent_session(agent, args.resume)

    return config, agent


def main():
    configure_logging()
    args = _parse_args()
    config, agent = _build_cli_runtime(args)

    if args.prompt:
        _run_once(agent, args.prompt)
        return

    _repl(agent, config)


def _run_once(agent: Agent, prompt: str):
    streamed: list[str] = []
    assistant_stream = _MarkdownStreamRenderer(_emit_raw_terminal)

    def on_token(token):
        streamed.append(token)
        assistant_stream.write(token)

    def on_tool(name, kwargs):
        assistant_stream.finish()
        _show_tool_call(console, name, kwargs)

    def on_brief(payload):
        assistant_stream.finish()
        console.print(_render_brief_message(payload))
        for line in _render_brief_attachments(payload):
            console.print(line)

    agent.on_brief_message = on_brief
    agent.ask_user_handler = _non_interactive_ask_user

    response = agent.chat(prompt, on_token=on_token, on_tool=on_tool)
    assistant_stream.finish()
    if not streamed:
        _write_assistant_response(_emit_raw_terminal, response)


def _repl(agent: Agent, config: Config):
    input_reader = _build_input_reader(
        os.path.expanduser("~/.kittycode_history"),
        lambda: _slash_command_names(agent.skills),
        token_provider=lambda: (agent.llm.total_prompt_tokens, agent.llm.total_completion_tokens),
    )
    history_width = input_reader._history_render_width() if hasattr(input_reader, "_history_render_width") else console.size.width
    if hasattr(input_reader, "print_startup"):
        input_reader.print_startup(_render_startup_header(config, width=history_width))
    else:
        input_reader.print(_render_startup_header(config, width=history_width))
    pending_skill = None

    def handle_submit(raw_input: str) -> None:
        nonlocal agent, pending_skill

        user_input = raw_input.strip()
        if not user_input:
            return

        resolved_command, matches = _resolve_command_prefix(user_input, agent.skills)
        if resolved_command and resolved_command != user_input:
            user_input = resolved_command
        elif user_input.startswith("/") and matches and resolved_command is None and user_input not in matches:
            input_reader.print("[yellow]Matching commands:[/yellow] " + ", ".join(matches))
            return

        if user_input == "/quit":
            request_exit = getattr(input_reader, "request_exit", None)
            if callable(request_exit):
                request_exit()
            return
        if user_input == "/help":
            _show_help(input_reader)
            return
        if user_input == "/reset":
            agent.reset()
            pending_skill = None
            input_reader.clear_history()
            input_reader.print("[yellow]Conversation reset.[/yellow]")
            return
        if user_input == "/skills":
            _show_skills(agent.skills, input_reader)
            return
        if user_input == "/tokens":
            prompt_tokens = agent.llm.total_prompt_tokens
            completion_tokens = agent.llm.total_completion_tokens
            input_reader.print(
                f"Tokens used this session: [cyan]{prompt_tokens}[/cyan] prompt + "
                f"[cyan]{completion_tokens}[/cyan] completion = [bold]{prompt_tokens + completion_tokens}[/bold] total"
            )
            return
        if user_input == "/model":
            _run_model_selector(input_reader, agent, config)
            return
        if user_input.startswith("/model "):
            input_reader.print("[yellow]Use /model to open the selector.[/yellow]")
            return
        if user_input == "/compact":
            from .runtime.context import estimate_tokens

            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                input_reader.print(
                    f"[green]Compressed: {before} -> {after} tokens ({len(agent.messages)} messages)[/green]"
                )
            else:
                input_reader.print(
                    f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]"
                )
            return
        if user_input == "/save":
            session_id = save_session(agent.messages, config.model)
            input_reader.print(f"[green]Session saved: {session_id}[/green]")
            input_reader.print(f"Resume with: kittycode -r {session_id}")
            return
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                input_reader.print("[dim]No saved sessions.[/dim]")
            else:
                for session in sessions:
                    input_reader.print(
                        f"  [cyan]{session['id']}[/cyan] ({session['model']}, {session['saved_at']}) {session['preview']}"
                    )
            return

        skill_match = _match_skill_command(user_input, agent.skills)
        if skill_match is not None:
            skill, task = skill_match
            if task:
                user_input = _build_skill_request(skill, task)
            else:
                pending_skill = skill
                input_reader.print(f"[cyan]Selected skill:[/cyan] /{skill.name}")
                input_reader.print("[dim]Your next non-command message will use this skill.[/dim]")
                return
        elif pending_skill is not None and not user_input.startswith("/"):
            user_input = _build_skill_request(pending_skill, user_input)
            pending_skill = None

        streamed: list[str] = []
        assistant_stream = _MarkdownStreamRenderer(
            lambda text: None,
            console_obj=input_reader.rich_console,
            on_finish=input_reader.finalize_active_output,
            on_text=lambda text: input_reader.write_raw(text, kind="markdown"),
        )

        def on_token(token):
            streamed.append(token)
            assistant_stream.write(token)

        def on_tool(name, kwargs):
            assistant_stream.finish()
            _show_tool_call(input_reader, name, kwargs)

        def on_brief(payload):
            assistant_stream.finish()
            input_reader.print(_render_brief_message(payload))
            for line in _render_brief_attachments(payload):
                input_reader.print(line)

        cancel_event = threading.Event()
        input_reader.attach_cancel_event(cancel_event)

        try:
            response, interrupted, next_agent = _run_agent_with_escape_interrupt(
                agent,
                user_input,
                on_token=on_token,
                on_tool=on_tool,
                ask_user=lambda questions: _ask_user_questions(input_reader, questions),
                on_brief=on_brief,
                cancel_event=cancel_event,
                enable_tty_monitor=False,
            )
            agent = next_agent
            assistant_stream.finish()

            if interrupted or response == "(interrupted)":
                input_reader.print("[yellow]Interrupted.[/yellow]")
            elif not streamed:
                _write_assistant_response(input_reader.write, response)
        except KeyboardInterrupt:
            assistant_stream.finish()
            input_reader.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as exc:
            assistant_stream.finish()
            input_reader.print(f"\n[red]Error: {exc}[/red]")
        finally:
            input_reader.detach_cancel_event(cancel_event)

    try:
        if hasattr(input_reader, "run"):
            input_reader.run(handle_submit, message=_INPUT_PROMPT_LABEL)
        else:
            while True:
                raw_input = input_reader.prompt(_INPUT_PROMPT_LABEL)
                handle_submit(raw_input)
                if raw_input.strip() == "/quit":
                    break
    except (EOFError, KeyboardInterrupt):
        input_reader.print("\nBye!")
    

def _show_help(io=None):
    io = io or _build_input_reader(os.path.expanduser("~/.kittycode_history"), lambda: list(_BUILTIN_COMMANDS))
    io.print(
        Panel(
            "[bold]Commands:[/bold]\n"
            "  /help          Show this help\n"
            "  /reset         Clear conversation history\n"
            "  /skills        Show loaded local skills\n"
            "  /<skill name>  Use a loaded skill\n"
            "  /model         Switch model mid-conversation\n"
            "  /tokens        Show token usage\n"
            "  /compact       Compress conversation context\n"
            "  /save          Save session to disk\n"
            "  /sessions      List saved sessions\n"
            "  /quit          Exit KittyCode",
            title="KittyCode Help",
            border_style="dim",
        )
    )


def _show_skills(skills, io=None):
    io = io or _build_input_reader(os.path.expanduser("~/.kittycode_history"), lambda: list(_BUILTIN_COMMANDS))
    io.print(Panel(_format_skills(skills), title="KittyCode Skills", border_style="dim"))


def _format_model_choices(config: Config) -> tuple[list[str], str | None]:
    active_index = config.active_model_index()
    lines = ["Provider | Model"]

    for index, model in enumerate(config.models, 1):
        marker = "*" if active_index == index - 1 else " "
        lines.append(f"{index}. {marker} {model.provider} | {model.model_name}")

    notices: list[str] = []
    if active_index is None and config.models:
        notices.append("Current runtime is outside the configured model list.")
    if len(config.models) == 1:
        notices.append("Only one configured model is available, so switching is unavailable.")

    return lines, " ".join(notices) if notices else None


def _select_model_index(io, config: Config) -> int | None:
    lines, notice = _format_model_choices(config)
    io.print("\n".join(lines))
    if notice:
        io.print(f"[dim]{notice}[/dim]")

    if len(config.models) <= 1:
        return None

    while True:
        try:
            raw = io.prompt("Model >").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not raw:
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(config.models):
                return index

        io.print(f"[yellow]Choose a model number between 1 and {len(config.models)}.[/yellow]")


def _run_model_selector(io, agent: Agent, config: Config) -> None:
    selected_index = _select_model_index(io, config)
    if selected_index is None:
        return

    if config.active_model_index() == selected_index:
        selected = config.models[selected_index]
        io.print(f"[dim]Already using [cyan]{selected.provider}/{selected.model_name}[/cyan][/dim]")
        return

    selected = config.activate_model(selected_index)
    agent.llm.reconfigure(
        model=selected.model_name,
        api_key=selected.api_key,
        interface=selected.interface,
        base_url=selected.base_url,
    )
    config.write()
    io.print(f"Switched to [cyan]{selected.provider}/{selected.model_name}[/cyan]")


def _format_skills(skills) -> str:
    if not skills:
        return "No skills loaded from ~/.kittycode/skills"

    lines = []
    for index, skill in enumerate(skills, 1):
        lines.append(f"{index}. {skill.name}")
        lines.append(f"   /{skill.name}")
        lines.append(f"   {skill.description}")
        lines.append(f"   {skill.path}")
    return "\n".join(lines)


def _slash_command_names(skills) -> list[str]:
    names = list(_BUILTIN_COMMANDS)
    for skill in skills:
        command = _skill_command_name(skill)
        if command not in names:
            names.append(command)
    return names


def _skill_command_name(skill) -> str:
    return f"/{skill.name}"


def _resolve_command_prefix(user_input: str, skills) -> tuple[str | None, list[str]]:
    if not user_input.startswith("/"):
        return None, []

    typed = user_input.casefold()
    matches = [command for command in _slash_command_names(skills) if command.casefold().startswith(typed)]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _match_skill_command(user_input: str, skills):
    lowered = user_input.casefold()
    for skill in sorted(skills, key=lambda item: len(_skill_command_name(item)), reverse=True):
        command = _skill_command_name(skill)
        lowered_command = command.casefold()
        if lowered == lowered_command:
            return skill, ""
        if lowered.startswith(lowered_command + " "):
            return skill, user_input[len(command):].strip()
    return None


def _build_skill_request(skill, task: str) -> str:
    return (
        f'Use the local skill "{skill.name}" for this request.\n'
        f"Skill description: {skill.description}\n"
        f"Skill path: {skill.path}\n"
        "Before doing other work, read its SKILL.md and any related files under that path.\n\n"
        f"Task:\n{task}"
    )


def _render_brief_message(payload: dict):
    status = payload.get("status", "normal")
    title = "Proactive Update" if status == "proactive" else "User Update"
    border_style = "yellow" if status == "proactive" else "cyan"
    return Panel(Markdown(payload.get("message", "")), title=title, border_style=border_style)


def _render_brief_attachments(payload: dict) -> list[str]:
    lines = []
    for attachment in payload.get("attachments") or []:
        lines.append(
            "[dim]Attachment:[/dim] "
            f"{attachment['path']} ({attachment['size']} bytes, image={attachment['is_image']})"
        )
    return lines


def _non_interactive_ask_user(_questions):
    raise RuntimeError("ask_user is only available in interactive mode")


def _ask_user_questions(io, questions: list[dict]) -> dict[str, str]:
    answers: dict[str, str] = {}

    for item in questions:
        io.finish_live_tool_output()
        io.print(
            Panel(
                _format_question_prompt(item),
                title=item["header"],
                border_style="cyan",
            )
        )

        while True:
            try:
                raw = io.prompt(f"{item['header']} >").strip()
            except (EOFError, KeyboardInterrupt):
                return answers

            parsed = _parse_question_answer(raw, item)
            if parsed is not None:
                answers[item["question"]] = parsed
                break

            io.print(
                "[yellow]Invalid response. Use option numbers like 1 or 1,3, "
                "or enter text when free-form input is allowed.[/yellow]"
            )

    return answers


def _format_question_prompt(item: dict) -> str:
    lines = [item["question"]]
    options = item.get("options") or []

    if options:
        lines.append("")
        for index, option in enumerate(options, 1):
            line = f"{index}. {option['label']}"
            if option.get("recommended"):
                line += " (recommended)"
            if option.get("description"):
                line += f" - {option['description']}"
            lines.append(line)
        lines.append("")
        if item.get("multiSelect"):
            lines.append("Reply with one or more option numbers separated by commas.")
        else:
            lines.append("Reply with one option number.")
        if item.get("allowFreeformInput", True):
            lines.append("You can also enter free text.")
    else:
        lines.extend(["", "Reply with free text."])

    return "\n".join(lines)


def _parse_question_answer(raw: str, item: dict) -> str | None:
    text = raw.strip()
    if not text:
        return None

    options = item.get("options") or []
    if not options:
        return text

    multi_select = bool(item.get("multiSelect", False))
    allow_freeform = bool(item.get("allowFreeformInput", True))
    parts = [part.strip() for part in text.split(",")] if multi_select else [text]
    if not parts or any(not part for part in parts):
        return None

    answers: list[str] = []
    for part in parts:
        if part.isdigit():
            index = int(part) - 1
            if 0 <= index < len(options):
                answers.append(options[index]["label"])
                continue
            return None

        matched = next(
            (option["label"] for option in options if option["label"].casefold() == part.casefold()),
            None,
        )
        if matched is not None:
            answers.append(matched)
            continue

        if allow_freeform:
            answers.append(part)
            continue

        return None

    if not multi_select and len(answers) != 1:
        return None
    return ", ".join(_dedupe_strings(answers))


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        marker = value.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    summary = ", ".join(f"{key}={repr(value)[:40]}" for key, value in kwargs.items())
    return summary[:maxlen] + ("..." if len(summary) > maxlen else "")


def _format_tool_call(name: str, kwargs: dict) -> str:
    if name == "todo_write":
        return f"{name}({repr(kwargs)})"
    return f"{name}({_brief(kwargs)})"


def _format_tool_call_details(name: str, kwargs: dict) -> list[tuple[str, object]]:
    details: list[tuple[str, object]] = [("tool", name)]
    details.extend(_flatten_tool_call_arguments(kwargs))
    return details


def _flatten_tool_call_arguments(value, prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, dict):
        if not value:
            rows.append((prefix or "arguments", "{}"))
            return rows
        for key, nested_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_tool_call_arguments(nested_value, next_prefix))
        return rows
    if isinstance(value, list):
        if not value:
            rows.append((prefix or "arguments", "[]"))
            return rows
        for index, nested_value in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(_flatten_tool_call_arguments(nested_value, next_prefix))
        return rows
    rows.append((prefix or "arguments", value))
    return rows


def _render_tool_call_value(value) -> Text:
    if value is None:
        return Text("null", style="dim")
    if isinstance(value, bool):
        return Text("true" if value else "false", style="bold green" if value else "bold yellow")
    if isinstance(value, (int, float)):
        return Text(str(value), style="bold cyan")
    if isinstance(value, str) and value in {"{}", "[]"}:
        return Text(value, style="dim")
    if isinstance(value, str):
        return Text(value, style="green")
    return Text(str(value), style="white")


def _render_tool_call_details(name: str, kwargs: dict):
    details = _format_tool_call_details(name, kwargs)
    summary = Table.grid(expand=False, padding=(0, 1))
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column(style="white")

    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(style="bold cyan", no_wrap=True)
    body.add_column(style="white")
    for path, value in details[1:]:
        body.add_row(path, _render_tool_call_value(value))

    content = Group(
        summary,
        Text("Arguments", style="bold magenta"),
        body if details[1:] else Text("No arguments", style="dim"),
    )
    return Panel(
        content,
        title=f"Tool Call: {name}",
        border_style="dim cyan",
    )


def _show_tool_call(io, name: str, kwargs: dict) -> None:
    io.print(_render_tool_call_details(name, kwargs))


def _write_assistant_response(write, response: str) -> None:
    if not response:
        return
    visible_response = _filter_think_display_text(response)
    if not visible_response:
        return
    rendered = _render_markdown_to_plain_text(visible_response)
    write(rendered if rendered.endswith("\n") else f"{rendered}\n")


def _render_to_plain_text(value, width: int = 80) -> str:
    render_console = Console(
        force_terminal=False,
        color_system=None,
        width=max(width, 20),
        legacy_windows=False,
        highlight=False,
    )
    with render_console.capture() as capture:
        render_console.print(value)
    return capture.get().rstrip("\n")


def _render_markdown_to_plain_text(text: str, width: int | None = None) -> str:
    if not text:
        return ""
    terminal_width = console.size.width if width is None else width
    return _render_to_plain_text(_render_markdown(text), width=terminal_width)


def _filter_think_display_text(text: str) -> str:
    if not text:
        return ""

    markers = ("<think>", "</think>")
    hidden = False
    visible: list[str] = []
    index = 0

    while index < len(text):
        if text.startswith("</think>", index):
            hidden = False
            index += len("</think>")
            continue
        if text.startswith("<think>", index):
            hidden = not hidden
            index += len("<think>")
            continue

        remainder = text[index:]
        if remainder.startswith("<") and any(marker.startswith(remainder) for marker in markers):
            break

        if not hidden:
            visible.append(text[index])
        index += 1

    return "".join(visible)


def _render_markdown(text: str):
    renderables: list[object] = []
    lines = text.splitlines()
    markdown_buffer: list[str] = []
    index = 0

    while index < len(lines):
        table_block = _consume_markdown_table_block(lines, index)
        if table_block is None:
            markdown_buffer.append(lines[index])
            index += 1
            continue

        if markdown_buffer:
            renderables.append(Markdown("\n".join(markdown_buffer)))
            markdown_buffer = []

        renderables.append(_render_markdown_table(table_block))
        index += len(table_block)

    if markdown_buffer or not renderables:
        renderables.append(Markdown("\n".join(markdown_buffer)))

    return Group(*renderables)


def _consume_markdown_table_block(lines: list[str], start: int) -> list[str] | None:
    if start + 1 >= len(lines):
        return None
    if not _is_markdown_table_row(lines[start]) or not _is_markdown_table_delimiter(lines[start + 1]):
        return None

    block = [lines[start], lines[start + 1]]
    index = start + 2
    while index < len(lines) and _is_markdown_table_row(lines[index]):
        block.append(lines[index])
        index += 1
    return block


def _is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or "|" not in stripped:
        return False
    cells = _split_markdown_table_row(stripped)
    return len(cells) >= 2


def _is_markdown_table_delimiter(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = _split_markdown_table_row(stripped)
    if len(cells) < 2:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def _render_markdown_table(lines: list[str]) -> Table:
    header = _split_markdown_table_row(lines[0])
    alignments = [_parse_markdown_table_alignment(cell) for cell in _split_markdown_table_row(lines[1])]
    rows = [_split_markdown_table_row(line) for line in lines[2:]]

    column_count = max(len(header), len(alignments), *(len(row) for row in rows), 0)
    padded_header = _pad_table_cells(header, column_count)
    padded_alignments = _pad_table_cells(alignments, column_count, fill="left")
    padded_rows = [_pad_table_cells(row, column_count) for row in rows]

    table = Table(box=box.MINIMAL, show_edge=False, pad_edge=False)
    for index, title in enumerate(padded_header):
        table.add_column(_render_markdown_table_cell(title), justify=padded_alignments[index], no_wrap=False)
    for row in padded_rows:
        table.add_row(*[_render_markdown_table_cell(cell) for cell in row])
    return table


def _parse_markdown_table_alignment(cell: str) -> str:
    stripped = cell.strip()
    if stripped.startswith(":") and stripped.endswith(":"):
        return "center"
    if stripped.endswith(":"):
        return "right"
    return "left"


def _pad_table_cells(cells: list[str], size: int, fill: str = "") -> list[str]:
    return cells + [fill] * max(size - len(cells), 0)


def _render_markdown_table_cell(text: str):
    return Markdown(text) if text else Text("")


class _MarkdownStreamRenderer:
    """Incrementally stream plain text to the terminal/history."""

    def __init__(
        self,
        emit,
        render=None,
        refresh_interval: float = 0.05,
        now=None,
        terminal_width=None,
        live_factory=None,
        console_obj=None,
        on_finish=None,
        on_text=None,
    ):
        self.emit = emit
        self.refresh_interval = refresh_interval
        self._on_finish = on_finish
        self._on_text = on_text
        self._now = now or time.monotonic
        self._console = console_obj or console
        if terminal_width is None:
            self._terminal_width = lambda: self._console.size.width
        elif callable(terminal_width):
            self._terminal_width = terminal_width
        else:
            self._terminal_width = lambda: terminal_width
        self.render = render or (lambda text: _render_markdown(text))
        # live_factory accepted for backward compatibility but unused
        self._buffer = ""
        self._last_emit_at: float | None = None
        self._last_visible_text = ""

    def write(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        visible_text = _filter_think_display_text(self._buffer)
        if self._on_text is not None:
            if visible_text.startswith(self._last_visible_text):
                delta = visible_text[len(self._last_visible_text):]
                if delta:
                    self._on_text(delta)
            elif visible_text:
                self._on_text(visible_text)
        self._last_visible_text = visible_text
        current_time = self._now()
        if self._last_emit_at is not None and current_time - self._last_emit_at < self.refresh_interval:
            return
        self._render_current()
        self._last_emit_at = current_time

    def finish(self) -> None:
        if self._buffer and self._on_finish is not None:
            self._on_finish()
        self._reset()

    def _render_current(self) -> None:
        rendered = self.render(_filter_think_display_text(self._buffer))
        rendered_text = (
            rendered
            if isinstance(rendered, str)
            else _render_to_plain_text(rendered, width=self._terminal_width())
        )
        previous = getattr(self, "_last_emitted_text", "")
        if rendered_text.startswith(previous):
            delta = rendered_text[len(previous):]
            if delta:
                self.emit(delta)
        elif not previous and rendered_text:
            self.emit(rendered_text)
        self._last_emitted_text = rendered_text

    def _reset(self) -> None:
        self._buffer = ""
        self._last_emit_at = None
        self._last_emitted_text = ""
        self._last_visible_text = ""


def _render_startup_header(config: Config, width: int | None = None):
    width = width or console.size.width
    left_lines = _startup_left_box_lines(config)
    left_width = max(_visible_width(line) for line in left_lines)
    gap = 2
    min_right_width = 28

    if width < left_width + gap + min_right_width:
        return Text("\n".join(_startup_single_box_lines(config, width)))

    available_right = width - left_width - gap
    right_lines = _startup_right_box_lines(config, available_right, target_height=len(left_lines))
    lines = _merge_columns(left_lines, right_lines, left_width, gap)
    return Text("\n".join(lines))


def _last_line_start_offset(text: str) -> int:
    if not text:
        return 0
    return text.rfind("\n") + 1


def _startup_left_box_lines(config: Config) -> list[str]:
    content = _PIXEL_CAT_ART.splitlines() + [
        f"KittyCode v{__version__}",
        f"Model: {config.model}",
    ]
    inner_width = max(max(_visible_width(line) for line in content), 34)
    alignments = ["custom-left", "custom-left", "custom-left", "custom-left", "custom-left", "center", "center"]
    return _box_lines(content, inner_width=inner_width, align="center", line_alignments=alignments)


def _startup_single_box_lines(config: Config, total_width: int) -> list[str]:
    left_lines = _startup_left_box_lines(config)
    left_width = max(_visible_width(line) for line in left_lines)
    if total_width >= left_width:
        return left_lines
    return _startup_compact_box_lines(config, total_width)


def _startup_compact_box_lines(config: Config, total_width: int) -> list[str]:
    content = [
        f"KittyCode v{__version__}",
        f"Model: {config.model}",
    ]
    if total_width < 16:
        return _startup_minimal_lines(content, total_width)

    available_content_width = max(total_width - 4, 12)
    wrapped: list[str] = []
    for line in content:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=available_content_width,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
        )
    inner_width = min(max(max(_visible_width(line) for line in wrapped), 12), available_content_width)
    return _box_lines(wrapped, inner_width=inner_width, align="center")


def _startup_minimal_lines(content: list[str], total_width: int) -> list[str]:
    width = max(total_width, 1)
    lines: list[str] = []
    for line in content:
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for segment in wrapped:
            truncated = _truncate_to_width(segment, width)
            lines.append(_pad_visible(truncated, width, align="center"))
    return lines


def _startup_right_box_lines(config: Config, total_width: int, target_height: int | None = None) -> list[str]:
    box_chrome_width = 4
    available_content_width = max(total_width - box_chrome_width, 1)
    min_content_width = max(min(available_content_width, total_width // 2), 12)
    raw_content = [
        f"Interface: {config.interface}",
        f"Base: {config.base_url or 'default'}",
    ]
    startup_hint = "Type /help for commands, press Esc to interrupt a run, /quit to exit."
    raw_content.extend(
        textwrap.wrap(
            startup_hint,
            width=available_content_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )
    content: list[str] = []
    for line in raw_content:
        wrapped = textwrap.wrap(
            line,
            width=available_content_width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        content.extend(wrapped or [""])
    inner_width = min(max(max(_visible_width(line) for line in content), min_content_width), available_content_width)
    return _box_lines(content, inner_width=inner_width, align="left", target_height=target_height)


def _merge_columns(left_lines: list[str], right_lines: list[str], left_width: int, gap: int) -> list[str]:
    total_lines = max(len(left_lines), len(right_lines))
    merged: list[str] = []
    for index in range(total_lines):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        left_padding = " " * max(left_width - _visible_width(left), 0)
        if right:
            merged.append(f"{left}{left_padding}{' ' * gap}{right}")
        else:
            merged.append(left)
    return merged


def _box_lines(
    content_lines: list[str],
    inner_width: int,
    align: str,
    target_height: int | None = None,
    line_alignments: list[str] | None = None,
) -> list[str]:
    top = f"╭{'─' * (inner_width + 2)}╮"
    bottom = f"╰{'─' * (inner_width + 2)}╯"
    padded_content = list(content_lines)
    if target_height is not None:
        body_height = max(target_height - 2, len(padded_content))
        extra = max(body_height - len(padded_content), 0)
        top_pad = extra // 2
        bottom_pad = extra - top_pad
        padded_content = ([""] * top_pad) + padded_content + ([""] * bottom_pad)
    body = []
    for index, line in enumerate(padded_content):
        line_align = line_alignments[index] if line_alignments and index < len(line_alignments) else align
        padded = _pad_visible(line, inner_width, align=line_align)
        body.append(f"│ {padded} │")
    return [top, *body, bottom]


def _pad_visible(text: str, width: int, align: str = "left") -> str:
    visible = _visible_width(text)
    remaining = max(width - visible, 0)
    if align == "custom-left":
        left = 7
        right = remaining - left
        return f"{' ' * left}{text}{' ' * right}"
    if align == "center":
        left = remaining // 2
        right = remaining - left
        return f"{' ' * left}{text}{' ' * right}"
    return f"{text}{' ' * remaining}"


@dataclass
class _HistoryItem:
    role: str
    kind: str
    text: str


def _merge_prompt_toolkit_styles(*styles: str) -> str:
    class_names: list[str] = []
    inline_parts: list[str] = []

    for style in styles:
        if not style:
            continue
        for part in style.split():
            if part.startswith("class:"):
                for name in part[6:].split(","):
                    cleaned = name.strip()
                    if cleaned and cleaned not in class_names:
                        class_names.append(cleaned)
            else:
                inline_parts.append(part)

    merged: list[str] = []
    if class_names:
        merged.append(f"class:{','.join(class_names)}")
    merged.extend(inline_parts)
    return " ".join(merged)


def _style_inline_markdown(text: str, base_style: str, raw_text: str | None = None):
    inline_pattern = r"(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|~~[^~]+~~|(?<!\*)\*[^*]+\*(?!\*)|(?<!_)_[^_]+_(?!_))"
    use_raw_text = bool(raw_text and re.search(inline_pattern, raw_text))
    source = raw_text if use_raw_text else text
    parts = re.split(inline_pattern, source)
    fragments = []
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            display = part[1:-1] if use_raw_text else part
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:history.markdown.code"), display))
        elif (
            (part.startswith("**") and part.endswith("**"))
            or (part.startswith("__") and part.endswith("__"))
        ) and len(part) >= 4:
            display = part[2:-2] if use_raw_text else part
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:history.markdown.strong"), display))
        elif part.startswith("~~") and part.endswith("~~") and len(part) >= 4:
            display = part[2:-2] if use_raw_text else part
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:history.markdown.strike"), display))
        elif (
            (part.startswith("*") and part.endswith("*"))
            or (part.startswith("_") and part.endswith("_"))
        ) and len(part) >= 3:
            display = part[1:-1] if use_raw_text else part
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:history.markdown.emphasis"), display))
        else:
            fragments.append((base_style, part))
    return fragments


def _style_startup_line(text: str, base_style: str):
    if not text:
        return [(base_style, text)]

    for marker in ("╮  ╭", "│  │", "╯  ╰"):
        split_at = text.find(marker)
        if split_at >= 0:
            left = text[:split_at + 1]
            gap = text[split_at + 1:split_at + 3]
            right = text[split_at + 3:]
            return [
                *_style_startup_left_segment(left, base_style),
                (base_style, gap),
                *_style_startup_right_segment(right, base_style),
            ]

    return _style_startup_left_segment(text, base_style)


def _style_startup_left_segment(text: str, base_style: str):
    fragments = []
    frame_chars = set("╭╮╰╯─│")
    is_cat_line = any(marker in text for marker in ("/\\_/\\\\", "( o.o )", "> ^", "/_ __ ___ ___/", "\\_/   V \\_\\"))

    for char in text:
        if char in frame_chars:
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:startup.frame"), char))
        elif is_cat_line and char != " ":
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:startup.cat"), char))
        else:
            fragments.append((base_style, char))
    return fragments


def _style_startup_right_segment(text: str, base_style: str):
    if not text:
        return [(base_style, text)]

    fragments = []
    frame_chars = set("╭╮╰╯─│")
    for char in text:
        if char in frame_chars:
            fragments.append((_merge_prompt_toolkit_styles(base_style, "class:startup.frame"), char))
        else:
            fragments.append((base_style, char))
    return fragments


def _style_history_markdown_line(text: str, metadata: dict[str, object]):
    base_style = str(metadata.get("base_style", ""))
    markdown_kind = str(metadata.get("markdown_kind", ""))
    raw_text = metadata.get("raw_text")
    raw_line = str(raw_text) if isinstance(raw_text, str) else None

    if not text:
        return [(base_style, text)]
    if markdown_kind == "heading":
        level = 1
        if raw_line is not None:
            match = re.match(r"\s{0,3}(#{1,6})\s", raw_line)
            if match:
                level = len(match.group(1))
        return [(_merge_prompt_toolkit_styles(base_style, f"class:history.markdown.heading{level}"), text)]
    if markdown_kind == "quote":
        return [(_merge_prompt_toolkit_styles(base_style, "class:history.markdown.quote"), text)]
    if markdown_kind == "fence":
        return [(_merge_prompt_toolkit_styles(base_style, "class:history.markdown.fence"), text)]
    if markdown_kind == "codeblock":
        return [(_merge_prompt_toolkit_styles(base_style, "class:history.markdown.codeblock"), text)]
    if markdown_kind == "table":
        return _style_history_table_line(text, base_style, raw_line)
    if markdown_kind == "list":
        match = re.match(r"(\s*(?:[-*+]\s+|\d+\.\s+|•\s+))(.*)", text)
        if match:
            marker, remainder = match.groups()
            raw_remainder = None
            if raw_line is not None:
                raw_match = re.match(r"(\s*(?:[-*+]\s+|\d+\.\s+))(.*)", raw_line)
                if raw_match:
                    raw_remainder = raw_match.group(2)
            return [
                (_merge_prompt_toolkit_styles(base_style, "class:history.markdown.list_marker"), marker),
                *_style_inline_markdown(remainder, base_style, raw_text=raw_remainder),
            ]
    return _style_inline_markdown(text, base_style, raw_text=raw_line)


def _style_history_table_line(text: str, base_style: str, raw_line: str | None):
    if raw_line is None or _is_rendered_table_separator_line(text):
        return [(base_style, text)]

    rendered_cells = text.split("│")
    raw_cells = _split_markdown_table_row(raw_line)
    if len(rendered_cells) != len(raw_cells):
        return [(base_style, text)]

    fragments = []
    for index, cell_text in enumerate(rendered_cells):
        leading = len(cell_text) - len(cell_text.lstrip(" "))
        trailing = len(cell_text) - len(cell_text.rstrip(" "))
        stripped_text = cell_text.strip(" ")

        if leading:
            fragments.append((base_style, " " * leading))
        if stripped_text:
            fragments.extend(_style_inline_markdown(stripped_text, base_style, raw_text=raw_cells[index]))
        if trailing:
            fragments.append((base_style, " " * trailing))
        if index < len(rendered_cells) - 1:
            fragments.append((base_style, "│"))

    return fragments or [(base_style, text)]


def _is_rendered_table_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and all(char in "─┼" for char in stripped)


def _classify_markdown_line(line: str, in_fenced_block: bool) -> tuple[str, bool]:
    stripped = line.strip()
    if stripped.startswith("```"):
        return "fence", not in_fenced_block
    if in_fenced_block:
        return "codeblock", in_fenced_block
    if _is_markdown_table_row(line):
        return "table", in_fenced_block
    if re.match(r"\s{0,3}#{1,6}\s", line):
        return "heading", in_fenced_block
    if re.match(r"\s*(?:[-*+]\s+|\d+\.\s+)", line):
        return "list", in_fenced_block
    if re.match(r"\s*>\s?", line):
        return "quote", in_fenced_block
    return "markdown", in_fenced_block


def _classify_rendered_markdown_line(
    line: str,
    index: int,
    rendered_lines: list[str],
    raw_kinds: list[str],
    pending_heading_indexes: list[int],
    pending_codeblock_indexes: list[int],
) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if re.match(r"\s*•\s+", line):
        return "list"
    if line.lstrip().startswith("▌"):
        return "quote"

    previous_line = rendered_lines[index - 1] if index > 0 else ""
    next_line = rendered_lines[index + 1] if index + 1 < len(rendered_lines) else ""
    if pending_heading_indexes:
        first_heading = pending_heading_indexes[0]
        if index >= first_heading:
            pending_heading_indexes.pop(0)
            return "heading"

    if (
        pending_codeblock_indexes
        and line.startswith(" ")
        and not previous_line.strip()
        and not next_line.strip()
    ):
        pending_codeblock_indexes.pop(0)
        return "codeblock"

    if any(char in line for char in "│╷╵╶╴┼├┤┬┴╭╮╰╯"):
        return "table"
    if "codeblock" in raw_kinds and line.startswith(" "):
        return "codeblock"
    return "markdown"


def _build_history_line_metadata(item: _HistoryItem, rendered: str) -> list[dict[str, object]]:
    lines = rendered.split("\n") if rendered else [""]
    base_style = ROLE_STYLE.get(item.role, "")
    metadata: list[dict[str, object]] = []
    raw_kinds: list[str] = []
    raw_lines = item.text.split("\n") if item.text else []
    pending_heading_indexes: list[int] = []
    pending_codeblock_indexes: list[int] = []
    pending_inline_raw_lines: dict[str, list[str]] = {
        "markdown": [],
        "list": [],
        "quote": [],
        "heading": [],
    }
    assistant_label_present = bool(item.role == "assistant" and lines and lines[0] == "KittyCode")
    markdown_start_index = 1 if assistant_label_present else 0

    if item.role == "assistant" and item.kind == "markdown":
        in_fenced_block = False
        for raw_line in raw_lines:
            markdown_kind, in_fenced_block = _classify_markdown_line(raw_line, in_fenced_block)
            raw_kinds.append(markdown_kind)
            if raw_line.strip() and markdown_kind in pending_inline_raw_lines:
                pending_inline_raw_lines[markdown_kind].append(raw_line)

        nonempty_rendered_indexes = [
            index for index, line in enumerate(lines)
            if index >= markdown_start_index and line.strip()
        ]
        heading_count = raw_kinds.count("heading")
        codeblock_count = raw_kinds.count("codeblock")
        pending_heading_indexes = nonempty_rendered_indexes[:heading_count]
        pending_codeblock_indexes = nonempty_rendered_indexes[-codeblock_count:] if codeblock_count else []

    for index, line in enumerate(lines):
        line_metadata: dict[str, object] = {"base_style": base_style}
        if item.role == "startup":
            line_metadata["startup"] = True
        if assistant_label_present and index == 0:
            line_metadata["label"] = True
        if item.role == "assistant" and item.kind == "markdown" and index >= markdown_start_index:
            markdown_kind = _classify_rendered_markdown_line(
                line,
                index,
                lines,
                raw_kinds,
                pending_heading_indexes,
                pending_codeblock_indexes,
            )
            if markdown_kind:
                line_metadata["markdown"] = True
                line_metadata["markdown_kind"] = markdown_kind
                if markdown_kind in pending_inline_raw_lines and pending_inline_raw_lines[markdown_kind]:
                    line_metadata["raw_text"] = pending_inline_raw_lines[markdown_kind].pop(0)
        metadata.append(line_metadata)

    return metadata


def render_message_to_text(role: str, kind: str, text: str, width: int = 80) -> str:
    render_console = Console(
        force_terminal=False,
        color_system=None,
        width=max(width, 20),
        legacy_windows=False,
        highlight=False,
    )
    body = text or ""
    if role == "assistant":
        body = _filter_think_display_text(body)

    with render_console.capture() as capture:
        if role == "user":
            render_console.print(Text("> " + body))
        elif role == "assistant":
            if kind == "markdown":
                render_console.print(_render_markdown(body))
            else:
                render_console.print(Text(body))
        elif role == "tool":
            render_console.print("Tool Output")
            render_console.print(Text(body))
        else:
            if kind == "markdown":
                render_console.print(_render_markdown(body))
            else:
                render_console.print(Text(body))

    return capture.get().rstrip("\n")


class _ReadlineInput:
    def __init__(self, history_path: str, command_provider, token_provider=None):
        self.history_path = history_path
        self.command_provider = command_provider
        self.token_provider = token_provider or (lambda: (0, 0))

        self.history = FileHistory(history_path)
        self.completer = SlashCommandCompleter(command_provider)

        self._history_items: list[_HistoryItem] = []
        self._transient_output: _HistoryItem | None = None
        self._live_tool_output_lines: list[str] = []

        self._prompt_label = _INPUT_PROMPT_LABEL
        self._chat_prompt_label = _INPUT_PROMPT_LABEL
        self._input_top_rule_window: Window | None = None
        self._input_bottom_rule_window: Window | None = None

        self._submit_handler = None
        self._busy = False

        self._prompt_waiter: threading.Event | None = None
        self._prompt_result = ""

        self._active_cancel_event: threading.Event | None = None

        self._input_history: list[str] = []
        self._input_history_index = 0
        self._input_history_draft = ""

        self._ui_thread: threading.Thread | None = None
        self._input_area_height = _INPUT_AREA_MIN_HEIGHT

        self.history_buffer = Buffer(read_only=True)
        self._history_line_metadata: list[dict[str, object]] = [{}]
        self.history_window = Window(
            BufferControl(
                buffer=self.history_buffer,
                input_processors=[HistoryStyleProcessor(lambda: self._history_line_metadata)],
            ),
            wrap_lines=False,
            always_hide_cursor=True,
        )

        self.input_area = TextArea(
            height=self._input_area_height,
            prompt=self._input_prompt_fragments(),
            multiline=True,
            wrap_lines=True,
            completer=self.completer,
            history=self.history,
            complete_while_typing=True,
            accept_handler=self._on_accept,
        )
        self.input_buffer = self.input_area.buffer
        self.input_buffer.on_text_changed += self._handle_input_text_changed
        self.session = SimpleNamespace(completer=self.completer, history=self.history)

        self._input_top_rule_window = Window(
            content=FormattedTextControl(self._render_input_rule_fragments, focusable=False, show_cursor=False),
            height=1,
            dont_extend_height=True,
        )
        self._input_bottom_rule_window = Window(
            content=FormattedTextControl(self._render_input_rule_fragments, focusable=False, show_cursor=False),
            height=1,
            dont_extend_height=True,
        )
        self.footer_window = Window(
            content=FormattedTextControl(self._render_footer_fragments, focusable=False, show_cursor=False),
            height=1,
            dont_extend_height=True,
        )

        self.layout = Layout(
            FloatContainer(
                content=HSplit(
                    [
                        self.history_window,
                        self._input_top_rule_window,
                        self.input_area,
                        self._input_bottom_rule_window,
                        self.footer_window,
                    ]
                ),
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=8),
                        attach_to_window=self.input_area.window,
                        allow_cover_cursor=False,
                    )
                ],
            ),
            focused_element=self.input_area,
        )

        self.application = Application(
            layout=self.layout,
            key_bindings=self._build_key_bindings(),
            full_screen=True,
            mouse_support=True,
            style=_APP_STYLE,
        )

        self.rich_console = Console(
            file=_PromptToolkitOutputFile(self),
            force_terminal=True,
        )

    # -------------------------
    # lifecycle
    # -------------------------
    def run(self, on_submit, message: str = _INPUT_PROMPT_LABEL) -> None:
        self._submit_handler = on_submit
        self._chat_prompt_label = message
        self._set_prompt_label(message)
        self._load_input_history()
        self._sync_history_buffer_ui()

        self._ui_thread = threading.current_thread()
        try:
            self.application.run()
        finally:
            self._ui_thread = None

    def prompt(self, message: str) -> str:
        """
        仅用于 ask_user 这类“在当前 app 内同步追问一次”的场景。
        不会重新 run 一个新的 Application。
        """
        if self._ui_thread is None:
            self._set_prompt_label(message)
            result = self.application.run()
            if isinstance(result, str) and result:
                self.history.append_string(result)
                self._load_input_history()
            return result

        waiter = threading.Event()
        self._prompt_result = ""

        def begin_prompt():
            self._finalize_transient_output_ui()
            self._load_input_history()
            self._prompt_waiter = waiter
            self._set_prompt_label(message)
            self.input_area.buffer.set_document(Document(""), bypass_readonly=True)
            self.layout.focus(self.input_area)
            self._sync_history_buffer_ui()

        self._call_in_ui_thread(begin_prompt, wait=False)
        waiter.wait()
        return self._prompt_result

    def request_exit(self) -> None:
        self._call_in_ui_thread(lambda: self.application.exit(), wait=False)

    def attach_cancel_event(self, cancel_event: threading.Event) -> None:
        self._active_cancel_event = cancel_event

    def detach_cancel_event(self, cancel_event: threading.Event) -> None:
        if self._active_cancel_event is cancel_event:
            self._active_cancel_event = None

    # -------------------------
    # thread-safe wrappers
    # -------------------------
    def print(self, value):
        self._call_in_ui_thread(lambda: self._print_ui(value), wait=False)

    def print_startup(self, value):
        self._call_in_ui_thread(lambda: self._print_startup_ui(value), wait=False)

    def write(self, text: str):
        self._call_in_ui_thread(lambda: self._write_ui(text), wait=False)

    def write_raw(self, text: str, role: str = "assistant", kind: str = "plain"):
        self._call_in_ui_thread(lambda: self._write_raw_ui(text, role=role, kind=kind), wait=False)

    def finalize_active_output(self):
        self._call_in_ui_thread(self._finalize_transient_output_ui, wait=False)

    def start_live_tool_output(self):
        self._call_in_ui_thread(self._start_live_tool_output_ui, wait=False)

    def append_live_tool_output(self, text: str):
        self._call_in_ui_thread(lambda: self._append_live_tool_output_ui(text), wait=False)

    def finish_live_tool_output(self):
        self._call_in_ui_thread(self._finish_live_tool_output_ui, wait=False)

    def clear_history(self) -> None:
        self._call_in_ui_thread(self._clear_history_ui, wait=False)

    # -------------------------
    # accept / submit
    # -------------------------
    def _on_accept(self, buff) -> None:
        text = buff.text.strip()

        # ask_user 同步提问模式
        if self._prompt_waiter is not None:
            self.history.append_string(text)
            self._prompt_result = text
            buff.text = ""

            waiter = self._prompt_waiter
            self._prompt_waiter = None

            self._set_prompt_label(self._chat_prompt_label)
            self.layout.focus(self.input_area)
            self._sync_history_buffer_ui()

            waiter.set()
            return

        if not text:
            buff.text = ""
            self.application.invalidate()
            return

        if self._busy:
            buff.text = ""
            self._print_ui("[yellow]A run is still in progress.[/yellow]")
            return

        self.history.append_string(text)
        self._load_input_history()
        self._commit_prompt_input_ui(text)

        buff.text = ""
        self.layout.focus(self.input_area)
        self._sync_history_buffer_ui()

        if self._submit_handler is not None:
            self._busy = True
            thread = threading.Thread(
                target=self._run_submit,
                args=(text,),
                daemon=True,
            )
            thread.start()

    def _run_submit(self, text: str) -> None:
        try:
            if self._submit_handler is not None:
                self._submit_handler(text)
        finally:
            self._call_in_ui_thread(self._mark_idle_ui, wait=False)

    def _mark_idle_ui(self) -> None:
        self._busy = False
        self.layout.focus(self.input_area)
        self.application.invalidate()

    # -------------------------
    # key bindings
    # -------------------------
    def _build_key_bindings(self) -> KeyBindings:
        key_bindings = KeyBindings()

        @key_bindings.add("c-c")
        @key_bindings.add("c-d")
        def _exit(event):
            event.app.exit(exception=EOFError())

        @key_bindings.add("up", filter=has_focus(self.input_area), eager=True)
        def _history_previous(_event):
            self._navigate_completion_or_input_history(-1)

        @key_bindings.add("down", filter=has_focus(self.input_area), eager=True)
        def _history_next(_event):
            self._navigate_completion_or_input_history(1)

        @key_bindings.add("enter", filter=has_focus(self.input_area))
        def _submit_message(_event):
            self._submit_current_input()

        @key_bindings.add("c-j", filter=has_focus(self.input_area))
        def _insert_newline(_event):
            self._insert_input_newline()

        @key_bindings.add("escape")
        def _interrupt(_event):
            if self._active_cancel_event is not None and not self._active_cancel_event.is_set():
                self._active_cancel_event.set()
                self._print_ui("[yellow]Interrupt requested...[/yellow]")

        return key_bindings

    # -------------------------
    # prompt / history helpers
    # -------------------------
    def _input_prompt_fragments(self):
        return FormattedText([("bold", f"{self._prompt_label} ")])

    def _render_input_rule_fragments(self):
        width = self._window_render_width(None, fallback_padding=0)
        return [("class:input.rule", "─" * max(width, 1))]

    def _set_prompt_label(self, message: str) -> None:
        self._prompt_label = _INPUT_PROMPT_LABEL
        self.input_area.prompt = self._input_prompt_fragments()
        self._update_input_area_height_ui()
        self.application.invalidate()

    def _load_input_history(self) -> None:
        if not getattr(self.history, "_loaded", False):
            self.history._loaded_strings = list(self.history.load_history_strings())
            self.history._loaded = True
        self._input_history = list(self.history.get_strings())
        self._input_history_index = len(self._input_history)
        self._input_history_draft = ""

    def _navigate_input_history(self, delta: int) -> None:
        if not self._input_history:
            return

        current_text = self.input_area.buffer.text

        if self._input_history_index == len(self._input_history):
            self._input_history_draft = current_text

        next_index = self._input_history_index + delta
        next_index = max(0, min(len(self._input_history), next_index))
        if next_index == self._input_history_index:
            return

        self._input_history_index = next_index
        if self._input_history_index == len(self._input_history):
            next_text = self._input_history_draft
        else:
            next_text = self._input_history[self._input_history_index]

        self.input_area.buffer.set_document(
            Document(next_text, cursor_position=len(next_text)),
            bypass_readonly=True,
        )

    def _handle_input_text_changed(self, _buffer) -> None:
        self._update_input_area_height_ui()

    def _submit_current_input(self) -> None:
        self.input_buffer.validate_and_handle()

    def _insert_input_newline(self) -> None:
        current = self.input_buffer.document
        cursor = current.cursor_position
        updated = f"{current.text[:cursor]}\n{current.text[cursor:]}"
        self.input_buffer.set_document(
            Document(updated, cursor_position=cursor + 1),
            bypass_readonly=True,
        )

    def _update_input_area_height_ui(self) -> None:
        next_height = _input_area_height_for_text(
            self.input_buffer.text,
            content_width=self._input_area_content_width(),
            prompt_width=get_cwidth(f"{self._prompt_label} "),
        )
        if next_height == self._input_area_height:
            return

        self._input_area_height = next_height
        self.input_area.window.height = Dimension(
            min=next_height,
            preferred=next_height,
            max=next_height,
        )
        self.application.invalidate()

    def _input_area_content_width(self) -> int:
        return self._window_render_width(self.input_area.window, fallback_padding=6)

    def _render_footer_fragments(self):
        read_tokens, write_tokens = self.token_provider()
        width = self._footer_width()
        text = _compose_footer_line(
            width=width,
            left=f"KittyCode v{__version__}",
            center=f"Author: {_AUTHOR_NAME}",
            right=f"Read: {read_tokens}  Write: {write_tokens}",
        )
        return [("class:footer", text)]

    def _footer_width(self) -> int:
        try:
            return max(self.application.output.get_size().columns, 20)
        except Exception:
            return max(console.size.width, 20)

    def _navigate_completion_or_input_history(self, delta: int) -> None:
        if self.input_buffer.complete_state is not None:
            if delta < 0:
                self.input_buffer.complete_previous()
            else:
                self.input_buffer.complete_next()
            return

        self._navigate_input_history(delta)

    # -------------------------
    # UI-thread implementations
    # -------------------------
    def _print_ui(self, value):
        self._finalize_transient_output_ui()
        self._append_history_item_ui(
            "system",
            "plain",
            _render_to_plain_text(value, width=self._history_render_width()),
        )

    def _print_startup_ui(self, value):
        self._finalize_transient_output_ui()
        self._append_history_item_ui(
            "startup",
            "plain",
            _render_to_plain_text(value, width=self._history_render_width()),
        )

    def _write_ui(self, text: str):
        self._finalize_transient_output_ui()
        self._append_history_item_ui("assistant", "plain", _normalize_output_text(text))

    def _write_raw_ui(self, text: str, role: str = "assistant", kind: str = "plain"):
        plain = _normalize_output_text(text)
        if not plain:
            return

        if self._transient_output is None:
            self._transient_output = _HistoryItem(role=role, kind=kind, text=plain)
        elif self._transient_output.role == role and self._transient_output.kind == kind:
            self._transient_output.text += plain
        else:
            self._finalize_transient_output_ui()
            self._transient_output = _HistoryItem(role=role, kind=kind, text=plain)

        self._sync_history_buffer_ui()

    def _start_live_tool_output_ui(self):
        self._live_tool_output_lines = []
        if self._transient_output is not None and self._transient_output.role == "tool":
            self._transient_output = None
            self._sync_history_buffer_ui()

    def _append_live_tool_output_ui(self, text: str):
        if not text:
            return

        if self._transient_output is not None and self._transient_output.role != "tool":
            self._finalize_transient_output_ui()

        chunks = text.splitlines() or [text]
        self._live_tool_output_lines.extend(chunks)
        self._transient_output = _HistoryItem(
            role="tool",
            kind="plain",
            text="\n".join(self._live_tool_output_lines),
        )
        self._sync_history_buffer_ui()

    def _finish_live_tool_output_ui(self):
        self._finalize_transient_output_ui()
        self._live_tool_output_lines = []

    def _clear_history_ui(self) -> None:
        self._history_items = []
        self._transient_output = None
        self._live_tool_output_lines = []
        self._sync_history_buffer_ui()

    def _commit_prompt_input_ui(self, text: str) -> None:
        self._append_history_item_ui("user", "plain", text.strip())

    def _commit_prompt_input(self, text: str) -> None:
        self._commit_prompt_input_ui(text)

    def _append_history_item_ui(self, role: str, kind: str, text: str) -> None:
        cleaned = text.strip("\n")
        if not cleaned:
            return
        self._history_items.append(_HistoryItem(role=role, kind=kind, text=cleaned))
        self._sync_history_buffer_ui()

    def _finalize_transient_output_ui(self) -> None:
        if self._transient_output is None:
            return

        if not self._transient_output.text.strip():
            self._transient_output = None
            return

        self._history_items.append(
            _HistoryItem(
                role=self._transient_output.role,
                kind=self._transient_output.kind,
                text=self._transient_output.text.rstrip("\n"),
            )
        )
        self._transient_output = None
        self._sync_history_buffer_ui()

    def _sync_history_buffer_ui(self) -> None:
        items = list(self._history_items)
        if self._transient_output is not None and self._transient_output.text:
            items.append(self._transient_output)

        width = self._history_render_width()
        parts: list[str] = []
        line_metadata: list[dict[str, object]] = []

        for index, item in enumerate(items):
            if not item.text:
                continue

            rendered = render_message_to_text(
                item.role,
                item.kind,
                item.text,
                width=width,
            )
            if index > 0:
                parts.append("\n\n")
                line_metadata.append({})

            parts.append(rendered)
            line_metadata.extend(_build_history_line_metadata(item, rendered))

        text = "".join(parts)
        self._history_line_metadata = line_metadata or [{}]

        self.history_buffer.set_document(
            Document(text=text, cursor_position=_last_line_start_offset(text)),
            bypass_readonly=True,
        )
        self.application.invalidate()

    def _history_render_width(self) -> int:
        return self._window_render_width(self.history_window, fallback_padding=4)

    def _window_render_width(self, window, fallback_padding: int) -> int:
        if window is not None:
            render_info = getattr(window, "render_info", None)
            if render_info is not None:
                window_width = getattr(render_info, "window_width", 0)
                if window_width:
                    return max(window_width, 1)
        try:
            return max(self.application.output.get_size().columns - fallback_padding, 1)
        except Exception:
            return max(console.size.width - fallback_padding, 1)

    # -------------------------
    # scheduler
    # -------------------------
    def _call_in_ui_thread(self, func, wait: bool = False):
        loop = getattr(self.application, "loop", None)

        if self._ui_thread is None or threading.current_thread() is self._ui_thread or loop is None:
            return func()

        if not wait:
            loop.call_soon_threadsafe(func)
            return None

        done = threading.Event()
        result: dict[str, object] = {}

        def runner():
            try:
                result["value"] = func()
            except BaseException as exc:
                result["exc"] = exc
            finally:
                done.set()

        loop.call_soon_threadsafe(runner)
        done.wait()

        if "exc" in result:
            raise result["exc"]  # type: ignore[misc]

        return result.get("value")


class _PromptToolkitOutputFile:
    encoding = "utf-8"

    def __init__(self, output):
        self.output = output

    def write(self, text: str) -> int:
        self.output.write_raw(text)
        return len(text)

    def flush(self) -> None:
        flush = getattr(self.output, "flush", None)
        if flush is not None:
            flush()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return sys.stdout.fileno()


def _build_input_reader(history_path: str, command_provider, token_provider=None):
    return _ReadlineInput(history_path, command_provider, token_provider=token_provider)


def _normalize_output_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "")


def _emit_raw_terminal(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


def _visible_width(text: str) -> int:
    lines = text.splitlines()
    if not lines:
        return 0
    return max(get_cwidth(line) for line in lines)


def _truncate_to_width(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if get_cwidth(text) <= width:
        return text
    if width == 1:
        return text[:1]

    current = ""
    for char in text:
        if get_cwidth(current + char + "…") > width:
            break
        current += char
    return f"{current}…"


def _compose_footer_line(width: int, left: str, center: str, right: str) -> str:
    width = max(width, 20)
    line = [" "] * width

    def place(text: str, start: int) -> None:
        for index, char in enumerate(text):
            position = start + index
            if 0 <= position < width:
                line[position] = char

    left_budget = max(min(get_cwidth(left), width // 4), min(width // 5, 12))
    right_budget = max(min(get_cwidth(right), width // 2), min(width // 3, 24))
    left_text = _truncate_to_width(left, max(left_budget, 1))
    right_text = _truncate_to_width(right, max(right_budget, 1))

    left_end = min(len(left_text), width)
    right_start = max(width - len(right_text), left_end + 1)
    if right_start + len(right_text) > width:
        right_text = _truncate_to_width(right_text, max(width - left_end - 1, 1))
        right_start = max(width - len(right_text), left_end + 1)

    center_room = max(right_start - left_end - 2, 1)
    center_text = _truncate_to_width(center, center_room)

    place(left_text, 0)
    place(right_text, right_start)
    center_start = max(left_end + 1 + (center_room - len(center_text)) // 2, 0)
    place(center_text, center_start)
    return "".join(line)


def _wrapped_visual_line_count(text: str, content_width: int, prompt_width: int = 0) -> int:
    if content_width <= 0:
        return 1

    logical_lines = text.split("\n")
    total = 0

    for index, line in enumerate(logical_lines):
        line_width = get_cwidth(line)
        first_line_width = max(1, content_width - prompt_width) if index == 0 else content_width
        wrapped_lines = 1
        if line_width > first_line_width:
            wrapped_lines += math.ceil((line_width - first_line_width) / content_width)
        total += wrapped_lines

    return max(total, 1)


def _input_area_height_for_text(
    text: str,
    content_width: int,
    prompt_width: int,
    min_height: int = _INPUT_AREA_MIN_HEIGHT,
    max_height: int = _INPUT_AREA_MAX_HEIGHT,
) -> int:
    visual_lines = _wrapped_visual_line_count(text, content_width, prompt_width)
    return max(min_height, min(max_height, visual_lines))


def _render_live_tool_output(lines: list[str], max_lines: int = 7, pad_to: int | None = None) -> str:
    if not lines:
        return "\r\x1b[2K"
    return f"\r\x1b[2K\x1b[90m{lines[-1]}\x1b[0m"


def _render_tool_output_summary(lines: list[str], max_lines: int = 7) -> str:
    visible = lines[-max_lines:]
    return "".join(f"\r\x1b[2K\x1b[90m{line}\x1b[0m\n" for line in visible)


class _LiveToolOutputRenderer:
    def __init__(self, emit, max_lines: int = 7, refresh_interval: float = 0.1, now=None):
        self.emit = emit
        self.max_lines = max_lines
        self.refresh_interval = refresh_interval
        self._now = now or time.monotonic
        self.lines: list[str] = []
        self.rendered = False
        self._last_emit_at: float | None = None

    def start(self):
        self.lines = []
        self.rendered = False
        self._last_emit_at = None

    def append(self, text: str):
        chunks = text.splitlines() or [text]
        self.lines.extend(chunks)
        current_time = self._now()
        if self._last_emit_at is not None and current_time - self._last_emit_at < self.refresh_interval:
            return
        self.emit(_render_live_tool_output(self.lines, self.max_lines))
        self.rendered = True
        self._last_emit_at = current_time

    def finish(self):
        if self.rendered:
            self.emit("\r\x1b[2K")
        if self.lines:
            self.emit(_render_tool_output_summary(self.lines, self.max_lines))
        self.start()


def _run_agent_with_escape_interrupt(
    agent: Agent,
    user_input: str,
    on_token=None,
    on_tool=None,
    on_tool_output=None,
    ask_user=None,
    on_brief=None,
    cancel_event: threading.Event | None = None,
    enable_tty_monitor: bool = True,
):
    cancel_event = cancel_event or threading.Event()
    callback_gate = threading.Event()
    callback_gate.set()
    run = agent.begin_run() if hasattr(agent, "begin_run") else None
    pending_questions: queue.Queue[tuple[list[dict], dict, threading.Event]] = queue.Queue()
    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def _guard(callback):
        if callback is None:
            return None

        def wrapped(*args, **kwargs):
            if callback_gate.is_set():
                callback(*args, **kwargs)

        return wrapped

    def worker():
        try:
            if run is not None and hasattr(agent, "activate_run"):
                with agent.activate_run(run):
                    agent.ask_user_handler = _make_ask_user_bridge(
                        pending_questions,
                        cancel_event,
                        ask_user,
                    )
                    agent.on_brief_message = _guard(on_brief)
                    chat_kwargs = {
                        "on_token": _guard(on_token),
                        "on_tool": _guard(on_tool),
                        "cancel_event": cancel_event,
                    }
                    if "on_tool_output" in inspect.signature(agent.chat).parameters:
                        chat_kwargs["on_tool_output"] = _guard(on_tool_output)

                    result["response"] = agent.chat(
                        user_input,
                        **chat_kwargs,
                    )
            else:
                chat_kwargs = {
                    "on_token": _guard(on_token),
                    "on_tool": _guard(on_tool),
                    "cancel_event": cancel_event,
                }
                if "on_tool_output" in inspect.signature(agent.chat).parameters:
                    chat_kwargs["on_tool_output"] = _guard(on_tool_output)

                result["response"] = agent.chat(
                    user_input,
                    **chat_kwargs,
                )
        except BaseException as exc:  # pragma: no cover
            error["exc"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    interrupted = False
    monitor = _create_escape_monitor(cancel_event) if enable_tty_monitor else None

    if monitor is None:
        while thread.is_alive():
            _service_user_question_requests(pending_questions, ask_user, cancel_event)
            if cancel_event.is_set():
                interrupted = True
                break
            thread.join(0.05)
    else:
        with monitor:
            while thread.is_alive():
                _service_user_question_requests(pending_questions, ask_user, cancel_event)
                if cancel_event.is_set() or monitor.poll(0.05):
                    interrupted = True
                    break
                thread.join(0.05)

    if interrupted and thread.is_alive():
        interrupted_run = None
        if run is not None and hasattr(agent, "snapshot_run") and hasattr(agent, "commit_run"):
            interrupted_run = agent.snapshot_run(run, repair_messages=True)
        callback_gate.clear()
        _cancel_pending_user_question_requests(pending_questions)
        if interrupted_run is not None:
            agent.commit_run(interrupted_run)
        return "(interrupted)", True, agent

    _service_user_question_requests(pending_questions, ask_user, cancel_event)
    thread.join()
    callback_gate.clear()

    if "exc" in error:
        raise error["exc"]

    if run is not None and hasattr(agent, "commit_run"):
        agent.commit_run(run)

    was_interrupted = interrupted or cancel_event.is_set() or result.get("response") == "(interrupted)"
    return result.get("response", ""), was_interrupted, agent


def _make_ask_user_bridge(pending_questions, cancel_event, ask_user):
    if ask_user is None:
        return _non_interactive_ask_user

    def bridge(questions):
        payload: dict[str, object] = {}
        done = threading.Event()
        pending_questions.put((questions, payload, done))
        while not done.wait(0.05):
            if cancel_event.is_set():
                raise CancellationRequested()
        error = payload.get("error")
        if error is not None:
            raise error
        return payload.get("answers", {})

    return bridge


def _service_user_question_requests(pending_questions, ask_user, cancel_event) -> None:
    if ask_user is None:
        _cancel_pending_user_question_requests(pending_questions)
        return

    while True:
        try:
            questions, payload, done = pending_questions.get_nowait()
        except queue.Empty:
            return

        if cancel_event.is_set():
            payload["error"] = CancellationRequested()
            done.set()
            continue

        try:
            payload["answers"] = ask_user(questions)
        except (EOFError, KeyboardInterrupt):
            cancel_event.set()
            payload["error"] = CancellationRequested()
        except Exception as exc:  # pragma: no cover - defensive transport
            payload["error"] = exc
        finally:
            done.set()


def _cancel_pending_user_question_requests(pending_questions) -> None:
    while True:
        try:
            _questions, payload, done = pending_questions.get_nowait()
        except queue.Empty:
            return
        payload["error"] = CancellationRequested()
        done.set()


def _create_escape_monitor(cancel_event, stream=None):
    stream = stream or sys.stdin
    if termios is None or tty is None:
        return None
    if not hasattr(stream, "isatty") or not stream.isatty():
        return None
    return _EscapeMonitor(stream, cancel_event)


class _EscapeMonitor:
    def __init__(self, stream, cancel_event):
        self.stream = stream
        self.cancel_event = cancel_event
        self.fd = stream.fileno()
        self._original_mode = None

    def __enter__(self):
        self._original_mode = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original_mode is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._original_mode)

    def poll(self, timeout: float) -> bool:
        ready, _, _ = select.select([self.stream], [], [], timeout)
        if not ready:
            return False
        chars = os.read(self.fd, 1)
        if chars == b"\x1b":
            self.cancel_event.set()
            return True
        return False
