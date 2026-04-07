"""Core agent loop.

This is the heart of KittyCode.

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text, which means it is
done working and ready to report back.
"""

import concurrent.futures
import copy
import inspect
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..llm import LLM
from ..prompt import system_prompt, user_prompt
from ..skills import load_skills
from ..tools import create_tool_instances, get_tool
from ..tools.base import Tool
from .context import ContextManager
from .interrupts import CancellationRequested


INTERRUPTED_TOOL_RESULT = "Error: tool execution interrupted before a result was produced."


@dataclass
class AgentState:
    messages: list[dict] = field(default_factory=list)
    todos: list[dict] = field(default_factory=list)
    brief_messages: list[dict] = field(default_factory=list)


@dataclass
class AgentRun:
    state: AgentState
    ask_user_handler: object = None
    on_brief_message: object = None


def repair_incomplete_tool_calls(messages: list[dict]) -> None:
    """Append synthetic tool results for assistant tool calls that never received one."""
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or not tool_calls:
            index += 1
            continue

        expected_ids = [tool_call.get("id") for tool_call in tool_calls if tool_call.get("id")]
        insert_at = index + 1
        seen_ids: list[str] = []

        while insert_at < len(messages) and messages[insert_at].get("role") == "tool":
            tool_call_id = messages[insert_at].get("tool_call_id")
            if tool_call_id:
                seen_ids.append(tool_call_id)
            insert_at += 1

        missing_ids = [tool_call_id for tool_call_id in expected_ids if tool_call_id not in seen_ids]
        if missing_ids:
            messages[insert_at:insert_at] = [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": INTERRUPTED_TOOL_RESULT,
                }
                for tool_call_id in missing_ids
            ]
            insert_at += len(missing_ids)

        index = insert_at


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
    ):
        self.llm = llm
        self.tools = list(tools) if tools is not None else create_tool_instances()
        self._state = AgentState()
        self._state_lock = threading.Lock()
        self._run_local = threading.local()
        self._ask_user_handler = None
        self._on_brief_message = None
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self.skills = []
        self._system = ""
        self._initialize_prompt_state()

        for tool in self.tools:
            tool.bind_agent(self)

    def _current_run(self) -> AgentRun | None:
        return getattr(self._run_local, "run", None)

    def _current_state(self) -> AgentState:
        current_run = self._current_run()
        if current_run is not None:
            return current_run.state
        return self._state

    @property
    def messages(self) -> list[dict]:
        return self._current_state().messages

    @messages.setter
    def messages(self, value: list[dict]) -> None:
        current_run = self._current_run()
        if current_run is not None:
            current_run.state.messages = value
            return
        with self._state_lock:
            self._state.messages = value

    @property
    def todos(self) -> list[dict]:
        return self._current_state().todos

    @todos.setter
    def todos(self, value: list[dict]) -> None:
        current_run = self._current_run()
        if current_run is not None:
            current_run.state.todos = value
            return
        with self._state_lock:
            self._state.todos = value

    @property
    def brief_messages(self) -> list[dict]:
        return self._current_state().brief_messages

    @brief_messages.setter
    def brief_messages(self, value: list[dict]) -> None:
        current_run = self._current_run()
        if current_run is not None:
            current_run.state.brief_messages = value
            return
        with self._state_lock:
            self._state.brief_messages = value

    @property
    def ask_user_handler(self):
        current_run = self._current_run()
        if current_run is not None:
            return current_run.ask_user_handler
        return self._ask_user_handler

    @ask_user_handler.setter
    def ask_user_handler(self, value) -> None:
        current_run = self._current_run()
        if current_run is not None:
            current_run.ask_user_handler = value
            return
        self._ask_user_handler = value

    @property
    def on_brief_message(self):
        current_run = self._current_run()
        if current_run is not None:
            return current_run.on_brief_message
        return self._on_brief_message

    @on_brief_message.setter
    def on_brief_message(self, value) -> None:
        current_run = self._current_run()
        if current_run is not None:
            current_run.on_brief_message = value
            return
        self._on_brief_message = value

    def begin_run(self) -> AgentRun:
        with self._state_lock:
            return AgentRun(
                state=AgentState(
                    messages=copy.deepcopy(self._state.messages),
                    todos=copy.deepcopy(self._state.todos),
                    brief_messages=copy.deepcopy(self._state.brief_messages),
                ),
                ask_user_handler=self._ask_user_handler,
                on_brief_message=self._on_brief_message,
            )

    def snapshot_run(self, run: AgentRun, repair_messages: bool = False) -> AgentRun:
        snapshot = AgentRun(
            state=AgentState(
                messages=copy.deepcopy(run.state.messages),
                todos=copy.deepcopy(run.state.todos),
                brief_messages=copy.deepcopy(run.state.brief_messages),
            ),
            ask_user_handler=run.ask_user_handler,
            on_brief_message=run.on_brief_message,
        )
        if repair_messages:
            repair_incomplete_tool_calls(snapshot.state.messages)
        return snapshot

    @contextmanager
    def activate_run(self, run: AgentRun):
        previous_run = self._current_run()
        self._run_local.run = run
        try:
            yield
        finally:
            if previous_run is None:
                try:
                    del self._run_local.run
                except AttributeError:
                    pass
            else:
                self._run_local.run = previous_run

    def commit_run(self, run: AgentRun) -> None:
        with self._state_lock:
            self._state = AgentState(
                messages=copy.deepcopy(run.state.messages),
                todos=copy.deepcopy(run.state.todos),
                brief_messages=copy.deepcopy(run.state.brief_messages),
            )

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _build_user_message(self, user_input: str) -> str:
        return user_prompt(user_input, self.skills, self.todos)

    def _tool_schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools]

    def fork(self):
        worker = Agent(
            llm=self.llm.clone() if hasattr(self.llm, "clone") else self.llm,
            tools=[type(tool)() for tool in self.tools],
            max_context_tokens=self.context.max_tokens,
            max_rounds=self.max_rounds,
        )
        worker.messages = copy.deepcopy(self.messages)
        worker.todos = copy.deepcopy(self.todos)
        worker.brief_messages = copy.deepcopy(self.brief_messages)
        worker.ask_user_handler = self.ask_user_handler
        worker.on_brief_message = self.on_brief_message
        worker.skills = list(self.skills)
        worker._system = self._system
        return worker

    def chat(self, user_input: str, on_token=None, on_tool=None, on_tool_output=None, cancel_event=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.messages.append({"role": "user", "content": self._build_user_message(user_input)})
        self.context.maybe_compress(self.messages, self.llm)

        try:
            self._raise_if_cancelled(cancel_event)

            for _ in range(self.max_rounds):
                self._raise_if_cancelled(cancel_event)
                response = self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas(),
                    on_token=on_token,
                    cancel_event=cancel_event,
                )

                if not response.tool_calls:
                    self.messages.append(response.message)
                    return response.content

                self.messages.append(response.message)

                if len(response.tool_calls) == 1:
                    tool_call = response.tool_calls[0]
                    if on_tool:
                        on_tool(tool_call.name, tool_call.arguments)
                    self._raise_if_cancelled(cancel_event)
                    result = self._exec_tool(tool_call, cancel_event=cancel_event, on_output=on_tool_output)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                else:
                    results = self._exec_tools_parallel(
                        response.tool_calls,
                        on_tool=on_tool,
                        on_tool_output=on_tool_output,
                        cancel_event=cancel_event,
                    )
                    for tool_call, result in zip(response.tool_calls, results):
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )

                self.context.maybe_compress(self.messages, self.llm)

            return "(reached maximum tool-call rounds)"
        except CancellationRequested:
            repair_incomplete_tool_calls(self.messages)
            return "(interrupted)"

    def _exec_tool(self, tool_call, cancel_event=None, on_output=None) -> str:
        """Execute a single tool call and return the result string."""
        self._raise_if_cancelled(cancel_event)
        tool = get_tool(tool_call.name, self.tools)
        if tool is None:
            return f"Error: unknown tool '{tool_call.name}'"
        try:
            execute_kwargs = dict(tool_call.arguments)
            parameters = inspect.signature(tool.execute).parameters
            if on_output is not None and "stream_callback" in parameters:
                execute_kwargs["stream_callback"] = lambda text: on_output(tool_call.name, text)
            if cancel_event is not None and "cancel_event" in parameters:
                execute_kwargs["cancel_event"] = cancel_event
            return tool.execute(**execute_kwargs)
        except TypeError as exc:
            return f"Error: bad arguments for {tool_call.name}: {exc}"
        except Exception as exc:
            return f"Error executing {tool_call.name}: {exc}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None, on_tool_output=None, cancel_event=None) -> list[str]:
        """Run multiple tool calls concurrently using threads."""
        for tool_call in tool_calls:
            if on_tool:
                on_tool(tool_call.name, tool_call.arguments)
        self._raise_if_cancelled(cancel_event)

        current_run = self._current_run()

        def run_tool(tool_call):
            if current_run is None:
                return self._exec_tool(tool_call, cancel_event=cancel_event, on_output=on_tool_output)
            with self.activate_run(current_run):
                return self._exec_tool(tool_call, cancel_event=cancel_event, on_output=on_tool_output)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run_tool, tool_call) for tool_call in tool_calls]
            return [future.result() for future in futures]

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()

    def _initialize_prompt_state(self) -> None:
        self.skills = load_skills(force_reload=True)
        self._system = system_prompt(self.tools)

    def refresh_skills(self, force_reload: bool = False):
        """Compatibility shim for startup-only skill loading.

        Skills and the system prompt are initialized once when the Agent is
        created and remain fixed for the lifetime of the process.
        """
        return list(self.skills)

    @staticmethod
    def _raise_if_cancelled(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise CancellationRequested()
