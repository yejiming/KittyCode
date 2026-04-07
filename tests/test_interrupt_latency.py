"""Regression tests for Esc interrupt responsiveness."""

import threading
import time

from kittycode.agent import INTERRUPTED_TOOL_RESULT
from kittycode.cli import _run_agent_with_escape_interrupt


class BlockingAgent:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def fork(self):
        return self

    def chat(self, _user_input, on_token=None, on_tool=None, cancel_event=None):
        self.started.set()
        self.release.wait()
        return "(interrupted)" if cancel_event is not None and cancel_event.is_set() else "done"


class ImmediateInterruptMonitor:
    def __init__(self, cancel_event, agent):
        self.cancel_event = cancel_event
        self.agent = agent
        self.triggered = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def poll(self, timeout: float) -> bool:
        self.agent.started.wait(timeout)
        if not self.triggered:
            self.cancel_event.set()
            self.triggered = True
            return True
        return False


class NeverInterruptMonitor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @staticmethod
    def poll(timeout: float) -> bool:
        time.sleep(min(timeout, 0.01))
        return False


def test_run_agent_with_escape_interrupt_returns_without_waiting_for_blocked_worker(monkeypatch):
    agent = BlockingAgent()

    def fake_create_escape_monitor(cancel_event, stream=None):
        return ImmediateInterruptMonitor(cancel_event, agent)

    releaser = threading.Timer(0.2, agent.release.set)
    monkeypatch.setattr("kittycode.cli._create_escape_monitor", fake_create_escape_monitor)

    try:
        releaser.start()
        start = time.perf_counter()
        response, interrupted, returned_agent = _run_agent_with_escape_interrupt(agent, "hello")
        elapsed = time.perf_counter() - start
    finally:
        releaser.cancel()
        agent.release.set()

    assert interrupted is True
    assert response == "(interrupted)"
    assert returned_agent is agent
    assert elapsed < 0.1


class ScriptedRunAgent:
    def __init__(self, label, started=None, release=None, emitted=None):
        self.label = label
        self.started = started or threading.Event()
        self.release = release or threading.Event()
        self.emitted = emitted if emitted is not None else []

    def fork(self):
        return self

    def chat(self, _user_input, on_token=None, on_tool=None, cancel_event=None):
        self.started.set()
        self.release.wait()
        if on_token is not None:
            on_token(self.label)
        self.emitted.append(self.label)
        return self.label


class ForkingAgent:
    def __init__(self, workers):
        self._workers = list(workers)
        self.fork_count = 0

    def fork(self):
        worker = self._workers[self.fork_count]
        self.fork_count += 1
        return worker


def test_interrupted_run_does_not_emit_late_output_or_replace_agent(monkeypatch):
    old_started = threading.Event()
    old_release = threading.Event()
    new_release = threading.Event()
    old_worker = ScriptedRunAgent("old", started=old_started, release=old_release)
    new_worker = ScriptedRunAgent("new", release=new_release)
    agent = ForkingAgent([old_worker, new_worker])
    seen = []
    monitor_calls = 0

    def fake_create_escape_monitor(cancel_event, stream=None):
        nonlocal monitor_calls
        monitor_calls += 1
        if monitor_calls == 1:
            return ImmediateInterruptMonitor(cancel_event, old_worker)
        return NeverInterruptMonitor()

    monkeypatch.setattr("kittycode.cli._create_escape_monitor", fake_create_escape_monitor)
    new_releaser = threading.Timer(0.05, new_release.set)

    try:
        response, interrupted, returned_agent = _run_agent_with_escape_interrupt(
            agent,
            "first",
            on_token=seen.append,
        )
        assert interrupted is True
        assert response == "(interrupted)"
        assert returned_agent is agent

        new_releaser.start()
        second_response, second_interrupted, second_agent = _run_agent_with_escape_interrupt(
            returned_agent,
            "second",
            on_token=seen.append,
        )

        old_release.set()
        time.sleep(0.05)
        time.sleep(0.05)
    finally:
        new_releaser.cancel()
        old_release.set()
        new_release.set()

    assert second_interrupted is False
    assert second_response == "new"
    assert second_agent is new_worker
    assert seen == ["new"]


class MessageSnapshotWorker:
    def __init__(self, started=None, release=None):
        self.started = started or threading.Event()
        self.release = release or threading.Event()
        self.messages = []
        self.todos = []
        self.brief_messages = []

    def chat(self, _user_input, on_token=None, on_tool=None, cancel_event=None):
        self.messages.append({"role": "user", "content": _user_input})
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\":\"sleep 10\"}"},
                    }
                ],
            }
        )
        self.todos.append({"content": "keep interrupted todo"})
        self.brief_messages.append({"message": "keep interrupted brief", "attachments": []})
        self.started.set()
        self.release.wait()
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "late tool result",
            }
        )
        self.messages.append({"role": "assistant", "content": "late message"})
        if on_token is not None:
            on_token("late token")
        return "(interrupted)" if cancel_event is not None and cancel_event.is_set() else "done"


class SnapshotParentAgent:
    def __init__(self, worker):
        self.worker = worker
        self.messages = [{"role": "assistant", "content": "existing context"}]
        self.todos = []
        self.brief_messages = []

    def fork(self):
        self.worker.messages = list(self.messages)
        self.worker.todos = list(self.todos)
        self.worker.brief_messages = list(self.brief_messages)
        return self.worker


def test_interrupted_run_merges_partial_tool_call_messages_back_with_interrupt_tool_result(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    worker = MessageSnapshotWorker(started=started, release=release)
    agent = SnapshotParentAgent(worker)
    seen = []

    def fake_create_escape_monitor(cancel_event, stream=None):
        return ImmediateInterruptMonitor(cancel_event, worker)

    monkeypatch.setattr("kittycode.cli._create_escape_monitor", fake_create_escape_monitor)

    try:
        response, interrupted, returned_agent = _run_agent_with_escape_interrupt(
            agent,
            "hello",
            on_token=seen.append,
        )

        assert interrupted is True
        assert response == "(interrupted)"
        assert returned_agent is agent
        assert returned_agent.messages == [
            {"role": "assistant", "content": "existing context"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{\"command\":\"sleep 10\"}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": INTERRUPTED_TOOL_RESULT,
            },
        ]
        assert returned_agent.todos == [{"content": "keep interrupted todo"}]
        assert returned_agent.brief_messages == [{"message": "keep interrupted brief", "attachments": []}]

        release.set()
        time.sleep(0.05)
        time.sleep(0.05)
    finally:
        release.set()

    assert returned_agent.messages == [
        {"role": "assistant", "content": "existing context"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"sleep 10\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": INTERRUPTED_TOOL_RESULT,
        },
    ]
    assert returned_agent.todos == [{"content": "keep interrupted todo"}]
    assert returned_agent.brief_messages == [{"message": "keep interrupted brief", "attachments": []}]
    assert seen == []
