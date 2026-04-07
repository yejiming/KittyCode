"""Interactive CLI for KittyCode."""

import argparse
import inspect
import os
import queue
import select
import re
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
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession, print_formatted_text
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .agent import Agent
from .config import CONFIG_PATH, Config
from .interrupts import CancellationRequested
from .llm import LLM
from .logging_utils import configure_logging
from .session import list_sessions, load_session, save_session

console = Console()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ACCENT = "\x1b[38;5;81m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

_ANSI_COLORS = {
    "RESET": "\x1b[0m",
    "OUTLINE": "\x1b[38;5;240m",
    "FUR": "\x1b[38;5;215m",
    "EYE": "\x1b[38;5;255m",
    "NOSE": "\x1b[38;5;218m"
}

_PIXEL_CAT_ART = (
    f"{_ANSI_COLORS['OUTLINE']} /\\_/\\{_ANSI_COLORS['RESET']}\n"
    f"{_ANSI_COLORS['OUTLINE']}({_ANSI_COLORS['FUR']} {_ANSI_COLORS['EYE']}o.o{_ANSI_COLORS['FUR']} {_ANSI_COLORS['OUTLINE']}){_ANSI_COLORS['FUR']}___________{_ANSI_COLORS['RESET']}\n"
    f"{_ANSI_COLORS['OUTLINE']} {_ANSI_COLORS['FUR']}>{_ANSI_COLORS['OUTLINE']} {_ANSI_COLORS['NOSE']}^{_ANSI_COLORS['OUTLINE']}           __){_ANSI_COLORS['RESET']}\n"
    f"{_ANSI_COLORS['OUTLINE']} /_{_ANSI_COLORS['FUR']} __ ___ ___{_ANSI_COLORS['OUTLINE']}/{_ANSI_COLORS['RESET']}\n"
    f"   {_ANSI_COLORS['OUTLINE']}\\_{_ANSI_COLORS['FUR']}/   V {_ANSI_COLORS['OUTLINE']}\\_\\{_ANSI_COLORS['RESET']}\n"
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
    parser.add_argument("-m", "--model", help="Model name (default: value from ~/.kittycode/config.json)")
    parser.add_argument("--interface", choices=["openai", "anthropic"], help="Interface type (default: value from ~/.kittycode/config.json)")
    parser.add_argument("--base-url", help="API base URL (default: value from ~/.kittycode/config.json)")
    parser.add_argument("--api-key", help="API key (default: value from ~/.kittycode/config.json)")
    parser.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    parser.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


def main():
    configure_logging()
    args = _parse_args()
    try:
        config = Config.from_file()
    except ValueError as exc:
        console.print(f"[red bold]Invalid config file:[/] {CONFIG_PATH}")
        console.print(str(exc))
        sys.exit(1)

    if args.model:
        config.model = args.model
    if args.interface:
        config.interface = args.interface
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

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

    llm = LLM(
        model=config.model,
        api_key=config.api_key,
        interface=config.interface,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, _loaded_model = loaded
            console.print(f"[green]Resumed session: {args.resume}[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    if args.prompt:
        _run_once(agent, args.prompt)
        return

    _repl(agent, config)


def _run_once(agent: Agent, prompt: str):
    streamed: list[str] = []
    live_tool_output = _LiveToolOutputRenderer(_emit_raw_terminal)

    def on_token(token):
        streamed.append(token)
        print(token, end="", flush=True)

    def on_tool(name, kwargs):
        live_tool_output.finish()
        console.print(f"\n[dim]> {_format_tool_call(name, kwargs)}[/dim]")

    def on_tool_output(name, text):
        if name == "bash":
            live_tool_output.append(text)

    def on_brief(payload):
        live_tool_output.finish()
        console.print(_render_brief_message(payload))
        for line in _render_brief_attachments(payload):
            console.print(line)

    agent.on_brief_message = on_brief
    agent.ask_user_handler = _non_interactive_ask_user

    response = agent.chat(prompt, on_token=on_token, on_tool=on_tool, on_tool_output=on_tool_output)
    live_tool_output.finish()
    if streamed:
        print()
    else:
        console.print(Markdown(response))


def _repl(agent: Agent, config: Config):
    input_reader = _build_input_reader(
        os.path.expanduser("~/.kittycode_history"),
        lambda: _slash_command_names(agent.skills),
    )
    input_reader.print(_render_startup_header(config, width=console.size.width))
    pending_skill = None

    while True:
        try:
            user_input = input_reader.prompt("You >").strip()
        except (EOFError, KeyboardInterrupt):
            input_reader.print("\nBye!")
            break

        if not user_input:
            continue

        resolved_command, matches = _resolve_command_prefix(user_input, agent.skills)
        if resolved_command and resolved_command != user_input:
            user_input = resolved_command
        elif user_input.startswith("/") and matches and resolved_command is None and user_input not in matches:
            input_reader.print("[yellow]Matching commands:[/yellow] " + ", ".join(matches))
            continue

        if user_input == "/quit":
            break
        if user_input == "/help":
            _show_help(input_reader)
            continue
        if user_input == "/reset":
            agent.reset()
            pending_skill = None
            input_reader.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/skills":
            _show_skills(agent.skills, input_reader)
            continue
        if user_input == "/tokens":
            prompt_tokens = agent.llm.total_prompt_tokens
            completion_tokens = agent.llm.total_completion_tokens
            input_reader.print(
                f"Tokens used this session: [cyan]{prompt_tokens}[/cyan] prompt + "
                f"[cyan]{completion_tokens}[/cyan] completion = [bold]{prompt_tokens + completion_tokens}[/bold] total"
            )
            continue
        if user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                input_reader.print(f"Switched to [cyan]{new_model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens

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
            continue
        if user_input == "/save":
            session_id = save_session(agent.messages, config.model)
            input_reader.print(f"[green]Session saved: {session_id}[/green]")
            input_reader.print(f"Resume with: kittycode -r {session_id}")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                input_reader.print("[dim]No saved sessions.[/dim]")
            else:
                for session in sessions:
                    input_reader.print(
                        f"  [cyan]{session['id']}[/cyan] ({session['model']}, {session['saved_at']}) {session['preview']}"
                    )
            continue

        skill_match = _match_skill_command(user_input, agent.skills)
        if skill_match is not None:
            skill, task = skill_match
            if task:
                user_input = _build_skill_request(skill, task)
            else:
                pending_skill = skill
                input_reader.print(f"[cyan]Selected skill:[/cyan] /{skill.name}")
                input_reader.print("[dim]Your next non-command message will use this skill.[/dim]")
                continue
        elif pending_skill is not None and not user_input.startswith("/"):
            user_input = _build_skill_request(pending_skill, user_input)
            pending_skill = None

        streamed: list[str] = []

        def on_token(token):
            streamed.append(token)
            input_reader.write(token)

        def on_tool(name, kwargs):
            input_reader.finish_live_tool_output()
            input_reader.print(f"\n[dim]> {_format_tool_call(name, kwargs)}[/dim]")

        def on_tool_output(name, text):
            if name == "bash":
                input_reader.append_live_tool_output(text)

        def on_brief(payload):
            input_reader.finish_live_tool_output()
            input_reader.print(_render_brief_message(payload))
            for line in _render_brief_attachments(payload):
                input_reader.print(line)

        try:
            response, interrupted, next_agent = _run_agent_with_escape_interrupt(
                agent,
                user_input,
                on_token=on_token,
                on_tool=on_tool,
                on_tool_output=on_tool_output,
                ask_user=lambda questions: _ask_user_questions(input_reader, questions),
                on_brief=on_brief,
            )
            agent = next_agent
            input_reader.finish_live_tool_output()
            if streamed:
                input_reader.write("\n")
            if interrupted or response == "(interrupted)":
                input_reader.print("[yellow]Interrupted.[/yellow]")
            elif not streamed:
                input_reader.print(Markdown(response))
        except KeyboardInterrupt:
            input_reader.finish_live_tool_output()
            input_reader.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as exc:
            input_reader.finish_live_tool_output()
            input_reader.print(f"\n[red]Error: {exc}[/red]")


def _show_help(io=None):
    io = io or _build_input_reader(os.path.expanduser("~/.kittycode_history"), lambda: list(_BUILTIN_COMMANDS))
    io.print(
        Panel(
            "[bold]Commands:[/bold]\n"
            "  /help          Show this help\n"
            "  /reset         Clear conversation history\n"
            "  /skills        Show loaded local skills\n"
            "  /<skill name>  Use a loaded skill\n"
            "  /model <name>  Switch model mid-conversation\n"
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


def _brief(kwargs: dict, maxlen: int = 400) -> str:
    summary = ", ".join(f"{key}={repr(value)[:200]}" for key, value in kwargs.items())
    return summary[:maxlen] + ("..." if len(summary) > maxlen else "")


def _format_tool_call(name: str, kwargs: dict) -> str:
    if name == "todo_write":
        return f"{name}({repr(kwargs)})"
    return f"{name}({_brief(kwargs)})"


def _render_startup_header(config: Config, width: int | None = None):
    width = width or console.size.width
    left_lines = _startup_left_box_lines(config)
    left_width = max(_visible_width(line) for line in left_lines)
    gap = 2
    min_right_width = 28

    available_right = width - left_width - gap
    if available_right < min_right_width:
        return Text.from_ansi("\n".join(left_lines))

    right_lines = _startup_right_box_lines(config, available_right, target_height=len(left_lines))
    lines = _merge_columns(left_lines, right_lines, left_width, gap)
    return Text.from_ansi("\n".join(lines))


def _pixel_cat_banner() -> str:
    return "\n".join(
        _PIXEL_CAT_ART.splitlines() + [f"{_ACCENT}{_BOLD}KittyCode{_RESET} v{__version__}"]
    )


def _startup_left_box_lines(config: Config) -> list[str]:
    content = _PIXEL_CAT_ART.splitlines() + [
        f"{_ACCENT}{_BOLD}KittyCode{_RESET} v{__version__}",
        f"{_BOLD}Model:{_RESET} {config.model}",
    ]
    inner_width = max(max(_visible_width(line) for line in content), 34)
    alignments = ["custom-left", "custom-left", "custom-left", "custom-left", "custom-left", "center", "center"]
    return _box_lines(content, inner_width=inner_width, align="center", line_alignments=alignments)


def _startup_right_box_lines(config: Config, total_width: int, target_height: int | None = None) -> list[str]:
    content_width = max(total_width - 4, 24)
    content = [
        f"{_BOLD}Interface:{_RESET} {config.interface}",
        f"{_BOLD}Base:{_RESET} {config.base_url or 'default'}",
    ]
    startup_hint = "Type /help for commands, press Esc to interrupt a run, /quit to exit."
    bold_tokens = {
        "/help": f"{_BOLD}/help{_RESET}",
        "Esc": f"{_BOLD}Esc{_RESET}",
        "/quit": f"{_BOLD}/quit{_RESET}",
    }
    content.extend(
        line
        for line in (
            _bold_startup_hint_tokens(wrapped_line, bold_tokens)
            for wrapped_line in textwrap.wrap(
                startup_hint,
                width=max(content_width, 1),
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    )
    inner_width = min(max(max(_visible_width(line) for line in content), 24), content_width)
    return _box_lines(content, inner_width=inner_width, align="left", target_height=target_height)


def _bold_startup_hint_tokens(line: str, bold_tokens: dict[str, str]) -> str:
    for token, styled_token in bold_tokens.items():
        line = line.replace(token, styled_token)
    return line


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
    top = f"{_ACCENT}╭{'─' * (inner_width + 2)}╮{_RESET}"
    bottom = f"{_ACCENT}╰{'─' * (inner_width + 2)}╯{_RESET}"
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
        body.append(f"{_ACCENT}│{_RESET} {padded} {_ACCENT}│{_RESET}")
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


class _ReadlineInput:
    def __init__(self, history_path: str, command_provider):
        self.history_path = history_path
        self.command_provider = command_provider
        self.session = PromptSession(
            history=FileHistory(history_path),
            completer=SlashCommandCompleter(command_provider),
            complete_while_typing=True,
        )
        self._live_tool_output = _LiveToolOutputRenderer(_emit_raw_terminal)

    def prompt(self, message: str) -> str:
        return self.session.prompt(f"{message} ", complete_while_typing=True)

    def print(self, value):
        self._emit_ansi(_render_as_ansi(value))

    def write(self, text: str):
        self._emit_ansi(text)

    def start_live_tool_output(self):
        self._live_tool_output.start()

    def append_live_tool_output(self, text: str):
        self._live_tool_output.append(text)

    def finish_live_tool_output(self):
        self._live_tool_output.finish()

    @staticmethod
    def _emit_ansi(text: str):
        print_formatted_text(ANSI(text), end="", flush=True)


def _build_input_reader(history_path: str, command_provider):
    return _ReadlineInput(history_path, command_provider)


def _emit_raw_terminal(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


def _render_as_ansi(value) -> str:
    with console.capture() as capture:
        console.print(value)
    return capture.get()


def _visible_width(text: str) -> int:
    lines = text.splitlines()
    if not lines:
        return 0
    return max(len(_ANSI_RE.sub("", line)) for line in lines)


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
):
    cancel_event = threading.Event()
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
        except BaseException as exc:  # pragma: no cover - surfaced in main thread
            error["exc"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    interrupted = False
    monitor = _create_escape_monitor(cancel_event)
    if monitor is None:
        while thread.is_alive():
            _service_user_question_requests(pending_questions, ask_user, cancel_event)
            thread.join(0.05)
    else:
        with monitor:
            while thread.is_alive():
                _service_user_question_requests(pending_questions, ask_user, cancel_event)
                if monitor.poll(0.05):
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
