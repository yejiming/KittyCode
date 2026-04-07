"""Regression tests for Esc interrupt responsiveness."""

import copy
import threading
import time
from contextlib import contextmanager

from kittycode.agent import INTERRUPTED_TOOL_RESULT, repair_incomplete_tool_calls
from kittycode.cli import _run_agent_with_escape_interrupt


class RunAwareAgent:
    def __init__(self):
        self._messages = []
        self._todos = []
        self._brief_messages = []
        self._run_local = threading.local()

    def _current_run_state(self):
        return getattr(self._run_local, "state", None)

    @property
    def messages(self):
        state = self._current_run_state()
        return state["messages"] if state is not None else self._messages

    @messages.setter
    def messages(self, value):
        state = self._current_run_state()
        if state is not None:
            state["messages"] = value
        else:
            self._messages = value

    @property
    def todos(self):
        state = self._current_run_state()
        return state["todos"] if state is not None else self._todos

    @todos.setter
    def todos(self, value):
        state = self._current_run_state()
        if state is not None:
            state["todos"] = value
        else:
            self._todos = value

    @property
    def brief_messages(self):
        state = self._current_run_state()
        return state["brief_messages"] if state is not None else self._brief_messages

    @brief_messages.setter
    def brief_messages(self, value):
        state = self._current_run_state()
        if state is not None:
            state["brief_messages"] = value
        else:
            self._brief_messages = value

    def begin_run(self):
        return {
            "messages": copy.deepcopy(self._messages),
            "todos": copy.deepcopy(self._todos),
            "brief_messages": copy.deepcopy(self._brief_messages),
        }

    @contextmanager
    def activate_run(self, run_state):
        previous_state = self._current_run_state()
        self._run_local.state = run_state
        try:
            yield
        finally:
            if previous_state is None:
                try:
                    del self._run_local.state
                except AttributeError:
                    pass
            else:
                self._run_local.state = previous_state

    def commit_run(self, run_state):
        self._messages = copy.deepcopy(run_state["messages"])
        self._todos = copy.deepcopy(run_state["todos"])
        self._brief_messages = copy.deepcopy(run_state["brief_messages"])

    def snapshot_run(self, run_state, repair_messages: bool = False):
        snapshot = {
            "messages": copy.deepcopy(run_state["messages"]),
            "todos": copy.deepcopy(run_state["todos"]),
            "brief_messages": copy.deepcopy(run_state["brief_messages"]),
        }
        if "plan" in run_state:
            snapshot["plan"] = run_state["plan"]
        if repair_messages:
            repair_incomplete_tool_calls(snapshot["messages"])
        return snapshot


class BlockingAgent(RunAwareAgent):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

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


class SequencedRunAgent(RunAwareAgent):
    def __init__(self, plans):
        super().__init__()
        self._plans = list(plans)
        self._plan_index = 0
        self._plan_lock = threading.Lock()

    def begin_run(self):
        run_state = super().begin_run()
        with self._plan_lock:
            run_state["plan"] = self._plans[self._plan_index]
            self.started = run_state["plan"]["started"]
            self._plan_index += 1
        return run_state

    def chat(self, _user_input, on_token=None, on_tool=None, cancel_event=None):
        plan = self._current_run_state()["plan"]
        plan["started"].set()
        plan["release"].wait()
        self.messages.append({"role": "assistant", "content": plan["label"]})
        if on_token is not None:
            on_token(plan["label"])
        return "(interrupted)" if cancel_event is not None and cancel_event.is_set() else plan["label"]


def test_interrupted_run_does_not_emit_late_output_or_replace_agent(monkeypatch):
    old_started = threading.Event()
    old_release = threading.Event()
    new_release = threading.Event()
    agent = SequencedRunAgent(
        [
            {"label": "old", "started": old_started, "release": old_release},
            {"label": "new", "started": threading.Event(), "release": new_release},
        ]
    )
    seen = []
    monitor_calls = 0

    def fake_create_escape_monitor(cancel_event, stream=None):
        nonlocal monitor_calls
        monitor_calls += 1
        if monitor_calls == 1:
            return ImmediateInterruptMonitor(cancel_event, agent)
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
    assert second_agent is agent
    assert second_agent.messages == [{"role": "assistant", "content": "new"}]
    assert seen == ["new"]


class MessageSnapshotAgent(RunAwareAgent):
    def __init__(self, started=None, release=None):
        super().__init__()
        self.started = started or threading.Event()
        self.release = release or threading.Event()
        self.messages = [{"role": "assistant", "content": "existing context"}]

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


def test_interrupted_run_commits_partial_tool_call_snapshot_without_late_mutations(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    agent = MessageSnapshotAgent(started=started, release=release)
    seen = []

    def fake_create_escape_monitor(cancel_event, stream=None):
        return ImmediateInterruptMonitor(cancel_event, agent)

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
