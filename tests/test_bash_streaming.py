"""Tests for streaming bash output into the CLI."""

from kittycode.cli import (
    _ReadlineInput,
    _LiveToolOutputRenderer,
    _render_live_tool_output,
    _render_tool_output_summary,
)
from kittycode.tools import get_tool


def test_bash_streams_output_lines_to_callback():
    bash = get_tool("bash")
    seen = []

    result = bash.execute(
        command="python3 -c \"print('one'); print('two')\"",
        stream_callback=seen.append,
    )

    assert seen == ["one", "two"]
    assert "one" in result
    assert "two" in result


def test_render_live_tool_output_shows_latest_line_in_gray():
    rendered = _render_live_tool_output([f"line {index}" for index in range(10)])

    assert rendered == "\r\x1b[2K\x1b[90mline 9\x1b[0m"


def test_render_tool_output_summary_keeps_last_seven_lines_in_gray():
    rendered = _render_tool_output_summary([f"line {index}" for index in range(10)])

    assert rendered.count("\x1b[2K") == 7
    assert rendered.count("\x1b[90m") == 7
    assert "line 0" not in rendered
    assert "line 1" not in rendered
    assert "line 2" not in rendered
    for index in range(3, 10):
        assert f"line {index}" in rendered


def test_live_tool_output_renderer_overwrites_single_status_line():
    emitted = []
    clock = iter([0.0, 0.2])
    renderer = _LiveToolOutputRenderer(emitted.append, now=lambda: next(clock))

    renderer.append("one")
    renderer.append("two")

    assert len(emitted) == 2
    assert emitted[0] == "\r\x1b[2K\x1b[90mone\x1b[0m"
    assert emitted[1] == "\r\x1b[2K\x1b[90mtwo\x1b[0m"


def test_live_tool_output_renderer_throttles_high_frequency_updates():
    emitted = []
    clock_values = iter([0.0, 0.01, 0.02, 0.20])
    renderer = _LiveToolOutputRenderer(emitted.append, refresh_interval=0.1, now=lambda: next(clock_values))

    renderer.append("one")
    renderer.append("two")
    renderer.append("three")
    renderer.append("four")

    assert emitted == [
        "\r\x1b[2K\x1b[90mone\x1b[0m",
        "\r\x1b[2K\x1b[90mfour\x1b[0m",
    ]


def test_live_tool_output_renderer_finish_prints_last_seven_line_summary():
    emitted = []
    renderer = _LiveToolOutputRenderer(emitted.append)

    for index in range(10):
        renderer.append(f"line {index}")
    renderer.finish()

    assert emitted[-2] == "\r\x1b[2K"
    assert emitted[-1].count("\x1b[2K") == 7
    assert "line 2" not in emitted[-1]
    for index in range(3, 10):
        assert f"line {index}" in emitted[-1]


def test_readline_input_live_output_renders_in_shared_transcript_while_running():
    reader = _ReadlineInput("/tmp/kittycode-history", lambda: {"/help"})

    reader.start_live_tool_output()
    reader.append_live_tool_output("hello")

    assert reader._transient_output is not None
    assert reader._transient_output.role == "tool"
    assert reader._transient_output.text == "hello"
    assert reader.history_buffer.text == "Tool Output\nhello"


def test_readline_input_finish_live_output_keeps_plain_summary_in_history():
    reader = _ReadlineInput("/tmp/kittycode-history", lambda: {"/help"})

    reader.start_live_tool_output()
    reader.append_live_tool_output("one\ntwo")
    reader.finish_live_tool_output()

    assert reader._transient_output is None
    assert "one" in reader.history_buffer.text
    assert "two" in reader.history_buffer.text
