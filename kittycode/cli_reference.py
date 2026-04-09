import asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from prompt_toolkit import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.layout import Layout
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import HSplit, Window, ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea


# =========================
# 渲染辅助
# =========================
def _get_console_width() -> int:
    try:
        return max(20, get_app().output.get_size().columns - 8)
    except Exception:
        return 80


def render_message_to_text(role: str, kind: str, text: str) -> str:
    """
    只渲染单条消息，返回纯文本。
    这里仍然使用 Rich 做 Markdown/终端排版，
    但不再把 ANSI 转给 prompt_toolkit。
    """
    console = Console(
        force_terminal=False,
        color_system=None,
        width=_get_console_width(),
        legacy_windows=False,
        highlight=False,
    )

    with console.capture() as capture:
        if role == "system":
            console.print(text)
        elif role == "user":
            console.print(f"You: {text}")
        elif role == "bot":
            console.print("Bot:")
            if kind == "markdown":
                console.print(Markdown(text or ""))
            else:
                console.print(Text(text))
        else:
            console.print(text)

    return capture.get().rstrip("\n")


ROLE_STYLE = {
    "system": "class:history.system",
    "user": "class:history.user",
    "bot": "class:history.bot",
    "plain": "",
}




COMMAND_HINTS = {
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


COMMAND_HINT_HEIGHT = 5


def _get_current_command_token(text: str) -> str:
    if not text.startswith("/"):
        return ""

    parts = text.split()
    token = parts[-1] if parts else text
    return token if token.startswith("/") else ""


def _get_matching_commands(token: str) -> list[tuple[str, str]]:
    if not token:
        return []

    return [
        (command, description)
        for command, description in COMMAND_HINTS.items()
        if command.startswith(token)
    ]


def should_show_command_hints() -> bool:
    try:
        token = _get_current_command_token(input_area.buffer.document.text_before_cursor)
    except Exception:
        return False

    return bool(token and _get_matching_commands(token))


def _get_visible_command_matches() -> list[tuple[str, str]]:
    if not command_matches:
        return []

    start = max(0, command_selected_index - (COMMAND_HINT_HEIGHT - 1))
    max_start = max(0, len(command_matches) - COMMAND_HINT_HEIGHT)
    start = min(start, max_start)
    end = start + COMMAND_HINT_HEIGHT
    return command_matches[start:end]


def sync_command_hint_buffer() -> None:
    global command_matches, command_selected_index, command_hint_line_styles

    token = _get_current_command_token(input_area.buffer.document.text_before_cursor)
    matches = _get_matching_commands(token)

    previous_command = None
    if command_matches and 0 <= command_selected_index < len(command_matches):
        previous_command = command_matches[command_selected_index][0]

    command_matches = matches

    if not command_matches:
        command_selected_index = 0
    elif previous_command is not None:
        matched_index = next((i for i, (cmd, _) in enumerate(command_matches) if cmd == previous_command), None)
        command_selected_index = matched_index if matched_index is not None else 0
    else:
        command_selected_index = 0

    visible_matches = _get_visible_command_matches()
    lines = []
    command_hint_line_styles = []

    for command, description in visible_matches:
        lines.append(f"{command:<10} {description}")
        is_selected = (
            command_matches
            and 0 <= command_selected_index < len(command_matches)
            and command_matches[command_selected_index][0] == command
        )
        command_hint_line_styles.append("class:command.selected" if is_selected else "class:command.item")

    while len(lines) < COMMAND_HINT_HEIGHT:
        lines.append("")
        command_hint_line_styles.append("")

    command_hint_buffer.set_document(
        Document(text="\n".join(lines), cursor_position=0),
        bypass_readonly=True,
    )

    try:
        app = get_app()
        if app.layout.has_focus(command_hint_control) and not command_matches:
            app.layout.focus(input_area)
        app.invalidate()
    except Exception:
        pass


def move_command_selection(delta: int) -> None:
    global command_selected_index

    if not command_matches:
        return

    command_selected_index = (command_selected_index + delta) % len(command_matches)
    visible_matches = _get_visible_command_matches()

    lines = []
    command_hint_line_styles.clear()
    selected_command = command_matches[command_selected_index][0]

    for command, description in visible_matches:
        lines.append(f"{command:<10} {description}")
        command_hint_line_styles.append("class:command.selected" if command == selected_command else "class:command.item")

    while len(lines) < COMMAND_HINT_HEIGHT:
        lines.append("")
        command_hint_line_styles.append("")

    command_hint_buffer.set_document(
        Document(text="\n".join(lines), cursor_position=0),
        bypass_readonly=True,
    )

    try:
        get_app().invalidate()
    except Exception:
        pass


def apply_selected_command() -> None:
    if not command_matches:
        return

    command = command_matches[command_selected_index][0]
    input_area.buffer.set_document(Document(command, cursor_position=len(command)), bypass_readonly=True)
    sync_command_hint_buffer()

    try:
        get_app().layout.focus(input_area)
    except Exception:
        pass


# =========================
# 历史缓存
# =========================
history_items: list[tuple[str, str]] = []  # (rendered_text, style_str)

streaming_message = {
    "active": False,
    "role": "",
    "kind": "",
    "text": "",
}

history_line_styles: list[str] = []
command_hint_line_styles: list[str] = []
command_matches: list[tuple[str, str]] = []
command_selected_index = 0
auto_follow = True


def _line_count(text: str) -> int:
    return Document(text).line_count


def rebuild_streaming_text() -> tuple[str, str] | None:
    if not streaming_message["active"]:
        return None

    rendered = render_message_to_text(
        streaming_message["role"],
        streaming_message["kind"],
        streaming_message["text"],
    )
    return rendered, ROLE_STYLE.get(streaming_message["role"], "")


def sync_history_buffer() -> None:
    """
    把缓存中的历史消息 + 当前流式消息，同步到 Buffer。
    """
    global history_line_styles

    old_cursor = history_buffer.cursor_position
    parts: list[str] = []
    line_styles: list[str] = []

    items = list(history_items)
    streaming_item = rebuild_streaming_text()
    if streaming_item is not None:
        items.append(streaming_item)

    for idx, (text, style_str) in enumerate(items):
        if idx > 0:
            parts.append("\n")
            line_styles.append("")

        parts.append(text)
        line_styles.extend([style_str] * _line_count(text))

    full_text = "".join(parts)
    history_line_styles = line_styles or [""]

    new_cursor = len(full_text) if auto_follow else min(old_cursor, len(full_text))
    history_buffer.set_document(
        Document(text=full_text, cursor_position=new_cursor),
        bypass_readonly=True,
    )

    try:
        get_app().invalidate()
    except Exception:
        pass


def append_final_message(role: str, kind: str, text: str) -> None:
    rendered = render_message_to_text(role, kind, text)
    history_items.append((rendered, ROLE_STYLE.get(role, "")))
    sync_history_buffer()


def start_stream_message(role: str, kind: str) -> None:
    streaming_message["active"] = True
    streaming_message["role"] = role
    streaming_message["kind"] = kind
    streaming_message["text"] = ""
    sync_history_buffer()


def update_stream_message(text: str) -> None:
    streaming_message["text"] = text
    sync_history_buffer()


def finish_stream_message() -> None:
    if not streaming_message["active"]:
        return

    rendered = render_message_to_text(
        streaming_message["role"],
        streaming_message["kind"],
        streaming_message["text"],
    )
    history_items.append((rendered, ROLE_STYLE.get(streaming_message["role"], "")))

    streaming_message["active"] = False
    streaming_message["role"] = ""
    streaming_message["kind"] = ""
    streaming_message["text"] = ""

    sync_history_buffer()


# =========================
# 历史显示区（BufferControl）
# =========================
class CommandHintStyleProcessor(Processor):
    def apply_transformation(self, transformation_input):
        line_no = transformation_input.lineno
        base_style = command_hint_line_styles[line_no] if line_no < len(command_hint_line_styles) else ""

        if not base_style:
            return Transformation(transformation_input.fragments)

        fragments = []
        for fragment in transformation_input.fragments:
            style = fragment[0]
            text = fragment[1]
            rest = fragment[2:]
            merged_style = f"{style} {base_style}".strip()
            fragments.append((merged_style, text, *rest))

        return Transformation(fragments)


class HistoryStyleProcessor(Processor):
    def apply_transformation(self, transformation_input):
        line_no = transformation_input.lineno
        base_style = history_line_styles[line_no] if line_no < len(history_line_styles) else ""

        if not base_style:
            return Transformation(transformation_input.fragments)

        fragments = []
        for fragment in transformation_input.fragments:
            style = fragment[0]
            text = fragment[1]
            rest = fragment[2:]
            merged_style = f"{style} {base_style}".strip()
            fragments.append((merged_style, text, *rest))

        return Transformation(fragments)


history_buffer = Buffer(
    document=Document("", 0),
    read_only=True,
    multiline=True,
)

history_control = BufferControl(
    buffer=history_buffer,
    focusable=True,
    focus_on_click=True,
    input_processors=[HistoryStyleProcessor()],
)

history_window = Window(
    content=history_control,
    wrap_lines=False,
    always_hide_cursor=True,
)


command_hint_buffer = Buffer(
    document=Document("", 0),
    read_only=True,
    multiline=True,
)

command_hint_control = BufferControl(
    buffer=command_hint_buffer,
    focusable=True,
    focus_on_click=True,
    input_processors=[CommandHintStyleProcessor()],
)

command_hint_window = Window(
    content=command_hint_control,
    height=COMMAND_HINT_HEIGHT,
    wrap_lines=False,
    always_hide_cursor=True,
)


def scroll_history_to_bottom() -> None:
    global auto_follow
    auto_follow = True
    sync_history_buffer()


def stop_auto_follow() -> None:
    global auto_follow
    auto_follow = False


# 初始化 system 消息
append_final_message(
    "system",
    "plain",
    (
        "History panel ready.\n"
        "Tab: switch focus between history/input\n"
        "PgUp/PgDn / Mouse wheel: scroll history\n"
        "End: jump to bottom and resume auto-follow\n"
        "Ctrl-Q: quit"
    ),
)


# =========================
# 流式输出
# =========================
async def stream_history_markdown(role: str, content: str, delay: float = 0.04) -> None:
    start_stream_message(role, "markdown")

    rendered = ""
    for ch in content:
        rendered += ch
        update_stream_message(rendered)
        await asyncio.sleep(delay)

    finish_stream_message()


# =========================
# 输入框
# =========================
def on_accept(buff):
    text = buff.text.strip()
    if not text:
        buff.text = ""
        sync_command_hint_buffer()
        return

    scroll_history_to_bottom()
    append_final_message("user", "plain", text)

    md = (
        "You typed:\n"
        f"{text}\n"
        "- 历史区改成了 BufferControl\n"
        "- 只会重新渲染最后一条 streaming message\n"
        "- 超过一屏后，新的消息仍然能跟随到底部\n"
    )

    get_app().create_background_task(
        stream_history_markdown("bot", md)
    )

    buff.text = ""
    sync_command_hint_buffer()


input_area = TextArea(
    height=3,
    prompt=">>> ",
    multiline=False,
    wrap_lines=False,
    accept_handler=on_accept,
)

def _on_input_buffer_changed(_=None) -> None:
    sync_command_hint_buffer()

    try:
        app = get_app()
        if app.layout.has_focus(input_area) and input_area.buffer.document.text_before_cursor == "/" and command_matches:
            app.layout.focus(command_hint_control)
            app.invalidate()
    except Exception:
        pass


input_area.buffer.on_text_changed += _on_input_buffer_changed
input_area.buffer.on_cursor_position_changed += lambda _: sync_command_hint_buffer()


# =========================
# 快捷键
# =========================
kb = KeyBindings()


@kb.add("c-q")
@kb.add("c-c")
def _(event):
    event.app.exit()


@kb.add("up", filter=has_focus(command_hint_control))
def _(event):
    move_command_selection(-1)


@kb.add("down", filter=has_focus(command_hint_control))
def _(event):
    move_command_selection(1)


@kb.add("tab", filter=has_focus(command_hint_control))
@kb.add("enter", filter=has_focus(command_hint_control))
def _(event):
    apply_selected_command()


@kb.add("escape", filter=has_focus(command_hint_control))
def _(event):
    event.app.layout.focus(input_area)


@kb.add("backspace", filter=has_focus(command_hint_control))
def _(event):
    input_area.buffer.delete_before_cursor(count=1)
    sync_command_hint_buffer()
    if not command_matches:
        event.app.layout.focus(input_area)


@kb.add(Keys.Any, filter=has_focus(command_hint_control))
def _(event):
    text = event.data
    if text:
        input_area.buffer.insert_text(text)
        sync_command_hint_buffer()


# =========================
# 布局
# =========================
root = HSplit([
    Frame(history_window, title="History"),
    HSplit([
        ConditionalContainer(
            content=Frame(command_hint_window, title="Commands"),
            filter=Condition(lambda: should_show_command_hints()),
        ),
        Frame(input_area, title="Input"),
    ]),
])

style = Style.from_dict({
    "history.system": "fg:#666666",
    "history.user": "fg:#22c55e",
    "history.bot": "fg:#06b6d4",
    "command.item": "fg:#cbd5e1",
    "command.selected": "bg:#1e293b fg:#ffffff",
})

app = Application(
    layout=Layout(root, focused_element=input_area),
    key_bindings=kb,
    full_screen=False,
    mouse_support=True,
    enable_page_navigation_bindings=False,
    style=style,
)

if __name__ == "__main__":
    scroll_history_to_bottom()
    app.run()