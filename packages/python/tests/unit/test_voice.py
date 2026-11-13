import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.manifest import read_manifest
from rfnry_engine.models import ModelChoice, ToolCall, ToolSpec, Usage
from rfnry_engine.records import (
    Event,
    Run,
    Spoken,
    SpokenLine,
    ToolCallStarted,
    ToolResultReady,
    Voice,
    VoiceStart,
)
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.tools import Result, Tool, Tools

from .fakes import Scripted, text_reply
from .trees import provided, write_folder

VOICE: dict[str, Any] = {"kind": "voice", "model": "live", "instructions": "voice.md"}
FILES = {"voice.md": "Speak plainly.\n"}
GENERATION: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "voice.md"}
AREA = json.dumps(
    {"type": "choice", "instructions": "Which area?", "options": {"other": None, "billing": None}}
)
UNFIT = cast(Any, Scripted())
SAID = Spoken(
    transcript=[
        SpokenLine(speaker="person", text="Hello"),
        SpokenLine(speaker="assistant", text="Hi there"),
    ],
    model="live-1",
)
PID = "import os\ndef run(args, call):\n    return {'pid': os.getpid(), 'run_id': call['run_id']}\n"
SLOW_ASYNC = (
    "import asyncio\n"
    "from pathlib import Path\n"
    "async def run(args, call):\n"
    "    Path(args['began']).touch()\n"
    "    try:\n"
    "        await asyncio.sleep(60)\n"
    "    except asyncio.CancelledError:\n"
    "        Path(args['cancelled']).touch()\n"
    "        raise\n"
)
ENDED = r"^engine: the session of agent 'voice' has ended; a tool call cannot start after it$"


class Live:
    model = "declared-1"

    async def open(self, *, tools: list[ToolSpec], instructions: str | None) -> object:
        raise AssertionError("the engine hands the model to the session and never calls it")


class Recording:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


class Slow:
    def __init__(self) -> None:
        self.began = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, args: dict[str, Any]) -> Result:
        self.began.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return "never"


def engine_of(
    root: Path, live: Any | None = None, refinery: Recording | Refinery | None = None
) -> Engine:
    return Engine(root, providers=provided({"live": live or Live()}), refinery=refinery)


def tool(name: str, body: Any = None) -> Tool:
    async def answers(args: dict[str, Any]) -> Result:
        return "done"

    return Tool(name, f"The {name} tool", body=body or answers)


def with_tools(root: Path, *names: str, **entry: Any) -> Path:
    return write_folder(
        root, {"voice": {**VOICE, "tools": {"allow": list(names)}, **entry}}, files=FILES
    )


async def until(check: Callable[[], bool], seconds: float = 3.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


async def spoke(start: VoiceStart) -> Spoken:
    return SAID


async def never(start: VoiceStart) -> Spoken:
    raise AssertionError("the session must not open")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("files", {"allow": ["voice.md"]}),
        ("skills", {"allow": ["s/*"]}),
        ("orchestration", {"max_depth": 1}),
        ("workspace", True),
        ("questions", ["q/*.json"]),
        ("max_model_calls", 3),
    ],
)
def test_a_voice_entry_takes_only_its_keys(tmp_path: Path, key: str, value: object) -> None:
    write_folder(tmp_path, {"voice": {**VOICE, key: value}}, files=FILES)
    with pytest.raises(
        ConfigurationError, match=rf"agents\.voice\.{key} in .*a voice agent takes only kind"
    ):
        read_manifest(tmp_path)


def test_a_voice_mode_takes_only_its_keys(tmp_path: Path) -> None:
    write_folder(
        tmp_path, {"voice": {**VOICE, "modes": {"chat": {"workspace": True}}}}, files=FILES
    )
    with pytest.raises(ConfigurationError, match=r"agents\.voice\.modes\.chat\.workspace in "):
        read_manifest(tmp_path)


async def test_a_voice_agent_needs_a_model_with_open(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    needs = r"^engine: agent 'voice' is a voice agent, which needs a live model, and model 'live' has no open\(\)$"
    with pytest.raises(ConfigurationError, match=needs):
        await engine_of(tmp_path, Scripted()).talk("voice", "acme", never)
    with pytest.raises(ConfigurationError, match=needs):
        await engine_of(tmp_path).talk("voice", "acme", never, models={"live": UNFIT})


async def test_a_run_s_models_takes_a_built_live_model_or_a_choice(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    built, backup, seen = Live(), Live(), list[VoiceStart]()

    async def session(start: VoiceStart) -> Spoken:
        seen.append(start)
        return SAID

    engine = Engine(
        tmp_path, providers=provided({"live": Live(), "backup": backup, "unused": Live()})
    )
    await engine.talk("voice", "acme", session, models={"live": built})
    await engine.talk("voice", "acme", session, models={"live": ModelChoice(provider="backup")})
    assert [start.model for start in seen] == [built, backup]


async def test_a_shared_mapping_passes_for_a_generation_run(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE, "writer": GENERATION}, files=FILES)
    writer = Scripted(text_reply("Done"))
    engine = Engine(tmp_path, providers=provided({"live": Live(), "m": Scripted()}))
    models: dict[str, Any] = {"live": Live(), "m": writer}
    run = await engine.run("writer", "acme", "Go", models=models)
    assert run.kind == "generation" and writer.calls != []


@pytest.mark.parametrize("via", ["delegate_to", "route"])
def test_delegate_to_and_route_cannot_name_a_voice_agent(tmp_path: Path, via: str) -> None:
    lead: dict[str, Any] = (
        {**GENERATION, "orchestration": {"delegate_to": ["voice"]}}
        if via == "delegate_to"
        else {
            "kind": "decision",
            "model": "j",
            "instructions": "voice.md",
            "questions": ["q/*.json"],
            "orchestration": {"route": {"by": "area", "to": {"billing": "voice"}}},
        }
    )
    write_folder(tmp_path, {"lead": lead, "voice": VOICE}, files={**FILES, "q/area.json": AREA})
    with pytest.raises(
        ConfigurationError,
        match=r"^compiled: agent 'lead' reaches 'voice', a voice agent; only engine\.talk runs one",
    ):
        load_folder(tmp_path)


async def test_run_and_tool_refuse_a_voice_agent(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    engine = engine_of(tmp_path)
    refused = r"^engine: agent 'voice' is a voice agent; engine\.talk runs it$"
    with pytest.raises(InputError, match=refused):
        await engine.run("voice", "acme", "Hello")
    with pytest.raises(InputError, match=refused):
        engine.tool("voice", scope="acme")


async def test_a_voice_agent_with_only_modes_is_refused_as_a_voice_agent(tmp_path: Path) -> None:
    entry = {
        "kind": "voice",
        "model": "live",
        "modes": {"chat": {"instructions": "voice.md"}},
    }
    write_folder(tmp_path, {"voice": entry}, files=FILES)
    with pytest.raises(InputError, match=r"engine\.talk runs it"):
        await engine_of(tmp_path).run("voice", "acme", "Hello")


async def test_talk_refuses_another_kind(tmp_path: Path) -> None:
    write_folder(tmp_path, {"writer": GENERATION}, files=FILES)
    engine = Engine(tmp_path, providers=provided({"m": Scripted()}))
    with pytest.raises(
        InputError,
        match=r"^engine: agent 'writer' is a generation agent; engine\.run runs it, and engine\.talk only a voice agent$",
    ):
        await engine.talk("writer", "acme", never)


async def test_talk_needs_a_scope_and_an_open_engine(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    engine = engine_of(tmp_path)
    with pytest.raises(InputError, match=r"^engine: scope is required$"):
        await engine.talk("voice", "", never)
    await engine.close()
    with pytest.raises(InputError, match="this engine was closed"):
        await engine.talk("voice", "acme", never)


async def test_the_session_gets_its_modes_instructions_model_and_tools(tmp_path: Path) -> None:
    entry = {
        **VOICE,
        "tools": {"allow": ["consult"]},
        "modes": {"duplex": {"instructions": "duplex.md", "tools": {"allow": ["hang_up"]}}},
    }
    write_folder(tmp_path, {"voice": entry}, files={**FILES, "duplex.md": "Talk both ways.\n"})
    live, seen = Live(), list[VoiceStart]()

    async def session(start: VoiceStart) -> Spoken:
        seen.append(start)
        return SAID.model_copy(update={"usage": Usage(input=3, output=4)})

    voice = await engine_of(tmp_path, live).talk(
        "voice",
        "acme",
        session,
        mode="duplex",
        context="Today is Monday.",
        tools=Tools([tool("consult"), tool("hang_up"), tool("not_picked")]),
        tags={"call": "c-1"},
    )
    [start] = seen
    assert start.model is live and start.mode == "duplex" and start.run_id == voice.run_id
    assert start.instructions == "Talk both ways.\n\n<<<CONTEXT>>>\nToday is Monday."
    assert sorted(start.tools.names) == ["consult", "hang_up"]
    assert {d["name"] for d in start.tools.definitions()} == {"consult", "hang_up"}
    assert isinstance(voice, Voice) and voice.kind == "voice" and voice.parent_run_id is None
    assert (voice.agent, voice.scope, voice.mode) == ("voice", "acme", "duplex")
    assert voice.tags == {"call": "c-1"} and voice.context == "Today is Monday."
    assert voice.instructions == start.instructions
    assert [spec.name for spec in voice.tools] == ["consult", "hang_up"]
    assert voice.transcript == SAID.transcript and voice.steps == []
    assert voice.model == "live-1" and voice.usage == Usage(input=3, output=4)
    assert voice.error is None and voice.duration_ms >= 0
    assert voice.fingerprint.questions is None and voice.fingerprint.agent


async def test_the_instructions_are_the_file_alone_without_context(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    voice = await engine_of(tmp_path).talk("voice", "acme", spoke)
    assert voice.instructions == "Speak plainly."


async def test_the_model_is_what_the_session_reports_else_the_adapters(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)

    async def unnamed(start: VoiceStart) -> Spoken:
        return Spoken(transcript=[])

    voice = await engine_of(tmp_path).talk("voice", "acme", unnamed)
    assert voice.model == "declared-1" and voice.usage is None


async def test_a_bare_name_the_run_does_not_give_fails_before_the_session_opens(
    tmp_path: Path,
) -> None:
    with_tools(tmp_path, "consult")
    with pytest.raises(ConfigurationError, match=r"lists 'consult' in tools\.allow"):
        await engine_of(tmp_path).talk("voice", "acme", never)


@pytest.mark.parametrize("name", ["Read", "Load", "Delegate", "Shell", "Write"])
def test_a_tool_a_section_grants_is_a_load_error(tmp_path: Path, name: str) -> None:
    with_tools(tmp_path, name)
    with pytest.raises(
        ConfigurationError,
        match=rf"lists '{name}' in tools\.allow, which a voice agent has no section to grant; remove the name",
    ):
        load_folder(tmp_path)


@pytest.mark.parametrize("name", ["Output", "History"])
def test_a_tool_the_run_decides_is_a_load_error(tmp_path: Path, name: str) -> None:
    with_tools(tmp_path, name)
    with pytest.raises(ConfigurationError, match=f"lists '{name}' in tools.allow"):
        load_folder(tmp_path)


def test_nothing_is_granted_and_deny_drops_a_declared_tool(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {
            "voice": {
                **VOICE,
                "tools": {"allow": ["who", "consult"]},
                "modes": {"quiet": {"tools": {"deny": ["who"]}}},
            }
        },
        tools={"who": {"module": "tools/who.py", "description": "Who"}},
        files={**FILES, "tools/who.py": PID},
    )
    spec = load_folder(tmp_path)["voice"]
    assert spec.granted == frozenset() and spec.keep == ["consult", "who"]
    assert spec.modes["quiet"].keep == ["consult"] and spec.modes["quiet"].folder_tools == {}


async def test_a_tool_call_passes_approve_and_becomes_a_step(tmp_path: Path) -> None:
    with_tools(tmp_path, "allowed", "refused")
    ran, asked = list[str](), list[tuple[str, str, dict[str, Any]]]()

    async def allowed(args: dict[str, Any]) -> Result:
        ran.append("allowed")
        return {"kept": [1, 2]}

    async def refused(args: dict[str, Any]) -> Result:
        ran.append("refused")
        return "never"

    async def gate(agent: str, call: ToolCall) -> bool:
        asked.append((agent, call.name, call.args))
        return call.name == "allowed"

    async def session(start: VoiceStart) -> Spoken:
        assert await start.tools.dispatch("allowed", {"a": 1}) == {"kept": [1, 2]}
        with pytest.raises(InputError, match=r"^engine: NotApproved: refused was not approved$"):
            await start.tools.dispatch("refused", {"b": 2})
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice",
        "acme",
        session,
        tools=Tools([tool("allowed", allowed), tool("refused", refused)]),
        approve=gate,
    )
    assert ran == ["allowed"]
    assert asked == [("voice", "allowed", {"a": 1}), ("voice", "refused", {"b": 2})]
    assert [(s.tool, s.args, s.is_error, s.approved) for s in voice.steps] == [
        ("allowed", {"a": 1}, False, True),
        ("refused", {"b": 2}, True, False),
    ]
    assert voice.steps[0].result == '{"kept": [1, 2]}'


async def test_a_call_that_passes_its_deadline_is_an_error_and_a_step(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {"voice": {**VOICE, "tools": {"allow": ["slow"], "timeout": 0.05}}},
        files=FILES,
    )

    async def session(start: VoiceStart) -> Spoken:
        with pytest.raises(InputError, match=r"TimeoutError: engine: slow timed out after 0\.05s"):
            await start.tools.dispatch("slow", {})
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("slow", Slow())])
    )
    [step] = voice.steps
    assert step.is_error and step.stopped is False and "timed out" in str(step.result)


async def test_a_tool_that_raises_is_an_error_to_the_session(tmp_path: Path) -> None:
    with_tools(tmp_path, "broken")

    async def broken(args: dict[str, Any]) -> Result:
        raise ValueError("no such order")

    async def session(start: VoiceStart) -> Spoken:
        with pytest.raises(InputError, match=r"^engine: ValueError: no such order$"):
            await start.tools.dispatch("broken", {})
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("broken", broken)])
    )
    assert voice.steps[0].result == "ValueError: no such order" and voice.steps[0].is_error


async def test_max_parallel_caps_the_calls_of_a_session(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {"voice": {**VOICE, "tools": {"allow": ["held"], "max_parallel": 1}}},
        files=FILES,
    )
    running, most = 0, 0

    async def held(args: dict[str, Any]) -> Result:
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.01)
        running -= 1
        return "done"

    async def session(start: VoiceStart) -> Spoken:
        await asyncio.gather(*(start.tools.dispatch("held", {}) for _ in range(3)))
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("held", held)])
    )
    assert most == 1 and len(voice.steps) == 3


async def test_on_event_gets_the_tool_call_events(tmp_path: Path) -> None:
    with_tools(tmp_path, "consult")
    events: list[Event] = []

    async def watch(event: Event) -> None:
        events.append(event)

    async def session(start: VoiceStart) -> Spoken:
        await start.tools.dispatch("consult", {"q": 1})
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("consult")]), on_event=watch
    )
    [opened, ready] = events
    assert isinstance(opened, ToolCallStarted) and isinstance(ready, ToolResultReady)
    assert opened.run_id == ready.run_id == voice.run_id and opened.agent == "voice"
    assert opened.tool == "consult" and opened.args == {"q": 1}
    assert opened.call_id == ready.call_id and ready.step == voice.steps[0]


async def test_a_folder_tool_runs_in_its_worker(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {"voice": {**VOICE, "worker": "live_tools", "tools": {"allow": ["who"]}}},
        workers={"live_tools": {"processes": 1}},
        tools={"who": {"module": "tools/who.py", "description": "Who runs this"}},
        files={**FILES, "tools/who.py": PID},
    )
    answers = list[Any]()

    async def session(start: VoiceStart) -> Spoken:
        answers.append(await start.tools.dispatch("who", {}))
        return SAID

    async with engine_of(tmp_path) as engine:
        voice = await engine.talk("voice", "acme", session)
    [answer] = answers
    assert answer["pid"] != os.getpid() and answer["run_id"] == voice.run_id
    assert [(step.tool, step.is_error) for step in voice.steps] == [("who", False)]


async def test_a_cancelled_tool_call_is_cancelled_and_keeps_its_step(tmp_path: Path) -> None:
    with_tools(tmp_path, "slow", "quick")
    slow = Slow()
    events: list[Event] = []
    recording = Recording()

    async def watch(event: Event) -> None:
        events.append(event)

    async def session(start: VoiceStart) -> Spoken:
        call = asyncio.create_task(start.tools.dispatch("slow", {"q": "first"}))
        await slow.began.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert await start.tools.dispatch("quick", {}) == "done"
        return SAID

    voice = await engine_of(tmp_path, refinery=recording).talk(
        "voice",
        "acme",
        session,
        tools=Tools([tool("slow", slow), tool("quick")]),
        on_event=watch,
    )
    assert slow.cancelled.is_set()
    assert voice.error is None and voice.transcript == SAID.transcript
    [kept, done] = voice.steps
    assert (kept.tool, kept.args, kept.is_error, kept.stopped) == (
        "slow",
        {"q": "first"},
        True,
        True,
    )
    assert kept.result == "stopped by the user"
    assert (done.tool, done.is_error, done.stopped) == ("quick", False, False)
    assert recording.runs == [voice]
    ready = [event for event in events if isinstance(event, ToolResultReady)]
    assert [event.step.stopped for event in ready] == [True, False]


async def test_a_call_still_running_when_the_session_ends_is_cancelled_and_kept(
    tmp_path: Path,
) -> None:
    with_tools(tmp_path, "slow")
    slow = Slow()
    left: list[asyncio.Task[Any]] = []

    async def session(start: VoiceStart) -> Spoken:
        left.append(asyncio.create_task(start.tools.dispatch("slow", {})))
        await slow.began.wait()
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("slow", slow)])
    )
    assert slow.cancelled.is_set() and [step.stopped for step in voice.steps] == [True]
    with pytest.raises(asyncio.CancelledError):
        await left[0]


async def test_a_call_cancelled_in_its_result_handler_keeps_one_step(tmp_path: Path) -> None:
    with_tools(tmp_path, "quick")
    in_handler = asyncio.Event()
    ready: list[ToolResultReady] = []

    async def watch(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            ready.append(event)
            if len(ready) == 1:
                in_handler.set()
                await asyncio.Event().wait()

    async def session(start: VoiceStart) -> Spoken:
        call = asyncio.create_task(start.tools.dispatch("quick", {}))
        await in_handler.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("quick")]), on_event=watch
    )
    assert [(s.tool, s.is_error, s.stopped, s.result) for s in voice.steps] == [
        ("quick", False, False, "done")
    ]
    assert len(ready) == 1


async def test_a_cancelled_call_cancels_an_async_tool_in_its_worker(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {"voice": {**VOICE, "tools": {"allow": ["slow"]}}},
        tools={"slow": {"module": "tools/slow.py", "description": "Slow"}},
        files={**FILES, "tools/slow.py": SLOW_ASYNC},
    )
    began, cancelled = tmp_path / "began", tmp_path / "cancelled"

    async def session(start: VoiceStart) -> Spoken:
        call = asyncio.create_task(
            start.tools.dispatch("slow", {"began": str(began), "cancelled": str(cancelled)})
        )
        await until(began.exists, 10.0)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        await until(cancelled.exists)
        return SAID

    async with engine_of(tmp_path) as engine:
        voice = await engine.talk("voice", "acme", session)
    [step] = voice.steps
    assert step.tool == "slow" and step.stopped and step.is_error and cancelled.exists()


async def test_a_cancelled_call_keeps_what_approve_answered(tmp_path: Path) -> None:
    with_tools(tmp_path, "slow")
    slow = Slow()

    async def gate(agent: str, call: ToolCall) -> bool:
        return True

    async def session(start: VoiceStart) -> Spoken:
        call = asyncio.create_task(start.tools.dispatch("slow", {}))
        await slow.began.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("slow", slow)]), approve=gate
    )
    assert [(s.stopped, s.approved) for s in voice.steps] == [(True, True)]


@pytest.mark.parametrize("fails", ["raises", "answers no bool"])
async def test_an_approve_that_fails_leaves_an_error_step_and_reaches_the_session(
    tmp_path: Path, fails: str
) -> None:
    with_tools(tmp_path, "quick")
    ran = list[str]()

    async def quick(args: dict[str, Any]) -> Result:
        ran.append("quick")
        return "done"

    async def gate(agent: str, call: ToolCall) -> bool:
        if fails == "raises":
            raise RuntimeError("the gate is down")
        return cast(bool, "yes")

    expected: type[Exception] = RuntimeError if fails == "raises" else ConfigurationError
    written = (
        "RuntimeError: the gate is down"
        if fails == "raises"
        else "ConfigurationError: engine: approve must return True or False; got 'yes'"
    )

    async def session(start: VoiceStart) -> Spoken:
        with pytest.raises(expected):
            await start.tools.dispatch("quick", {"a": 1})
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("quick", quick)]), approve=gate
    )
    assert ran == [] and voice.error is None
    assert [(s.tool, s.args, s.is_error, s.stopped, s.approved, s.result) for s in voice.steps] == [
        ("quick", {"a": 1}, True, False, None, written)
    ]


async def test_a_result_handler_that_raises_reaches_the_session_and_leaves_one_step(
    tmp_path: Path,
) -> None:
    with_tools(tmp_path, "quick")

    async def watch(event: Event) -> None:
        if isinstance(event, ToolResultReady):
            raise RuntimeError("the handler broke")

    async def session(start: VoiceStart) -> Spoken:
        with pytest.raises(RuntimeError, match="the handler broke"):
            await start.tools.dispatch("quick", {})
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("quick")]), on_event=watch
    )
    assert [(s.tool, s.is_error, s.result) for s in voice.steps] == [("quick", False, "done")]


async def test_a_handler_that_raises_before_the_tool_leaves_an_error_step(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with_tools(tmp_path, "quick")

    async def watch(event: Event) -> None:
        raise RuntimeError("the handler broke")

    async def session(start: VoiceStart) -> Spoken:
        with pytest.raises(RuntimeError, match="the handler broke"):
            await start.tools.dispatch("quick", {})
        return SAID

    with caplog.at_level(logging.ERROR, logger="rfnry_engine.engine"):
        voice = await engine_of(tmp_path).talk(
            "voice", "acme", session, tools=Tools([tool("quick")]), on_event=watch
        )
    assert [(s.tool, s.is_error, s.result) for s in voice.steps] == [
        ("quick", True, "RuntimeError: the handler broke")
    ]
    assert "on_event failed on the last step" in caplog.text


async def test_a_call_after_the_session_ended_raises_and_runs_nothing(tmp_path: Path) -> None:
    with_tools(tmp_path, "quick")
    ran = list[str]()
    kept, pending = list[VoiceStart](), list[asyncio.Future[Any]]()

    async def quick(args: dict[str, Any]) -> Result:
        ran.append("quick")
        return "done"

    async def session(start: VoiceStart) -> Spoken:
        kept.append(start)
        pending.append(asyncio.ensure_future(start.tools.dispatch("quick", {})))
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("quick", quick)])
    )
    with pytest.raises(InputError, match=ENDED):
        await pending[0]
    with pytest.raises(InputError, match=ENDED):
        await kept[0].tools.dispatch("quick", {})
    assert ran == [] and voice.steps == []


async def test_a_tool_the_agent_was_not_offered_is_refused_before_approve_as_an_error_step(
    tmp_path: Path,
) -> None:
    with_tools(tmp_path, "quick")
    asked = list[str]()

    async def gate(agent: str, call: ToolCall) -> bool:
        asked.append(call.name)
        return True

    async def session(start: VoiceStart) -> Spoken:
        with pytest.raises(InputError, match=r"^engine: InputError: tools: unknown tool 'ghost'$"):
            await start.tools.dispatch("ghost", {"a": 1})
        assert await start.tools.only("quick").dispatch("quick", {}) == "done"
        return SAID

    voice = await engine_of(tmp_path).talk(
        "voice", "acme", session, tools=Tools([tool("quick")]), approve=gate
    )
    assert asked == ["quick"]
    assert [(s.tool, s.args, s.is_error, s.result, s.approved) for s in voice.steps] == [
        ("ghost", {"a": 1}, True, "InputError: tools: unknown tool 'ghost'", None),
        ("quick", {}, False, "done", True),
    ]


async def test_a_failed_session_is_recorded_and_raises(tmp_path: Path) -> None:
    with_tools(tmp_path, "quick")
    recording = Recording()

    async def session(start: VoiceStart) -> Spoken:
        await start.tools.dispatch("quick", {})
        raise RuntimeError("the line dropped")

    with pytest.raises(RuntimeError, match="the line dropped"):
        await engine_of(tmp_path, refinery=recording).talk(
            "voice", "acme", session, tools=Tools([tool("quick")])
        )
    [run] = recording.runs
    assert isinstance(run, Voice) and run.error == "RuntimeError: the line dropped"
    assert run.transcript == [] and [step.tool for step in run.steps] == ["quick"]


@pytest.mark.parametrize(
    "returned",
    ["not a transcript", {"transcript": []}, SimpleNamespace(transcript=[])],
    ids=["text", "dict", "lookalike"],
)
async def test_a_session_that_returns_no_spoken_is_a_failed_session(
    tmp_path: Path, returned: object
) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    recording = Recording()

    async def session(start: VoiceStart) -> Spoken:
        return cast(Spoken, returned)

    with pytest.raises(InputError, match=r"^engine: the session of agent 'voice' returned "):
        await engine_of(tmp_path, refinery=recording).talk("voice", "acme", session)
    [run] = recording.runs
    assert isinstance(run, Voice) and run.error is not None and run.transcript == []


async def test_a_cancelled_session_is_not_recorded(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    recording = Recording()
    entered = asyncio.Event()

    async def session(start: VoiceStart) -> Spoken:
        entered.set()
        await asyncio.Event().wait()
        return SAID

    talking = asyncio.create_task(
        engine_of(tmp_path, refinery=recording).talk("voice", "acme", session)
    )
    await entered.wait()
    talking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await talking
    assert recording.runs == []


async def test_record_false_records_nothing(tmp_path: Path) -> None:
    with_tools(tmp_path, "consult")
    recording = Recording()
    consult_runs = Recording()
    other = tmp_path / "other"
    other.mkdir()
    write_folder(
        other, {"writer": {**GENERATION, "description": "Writes"}}, files={"voice.md": "Write.\n"}
    )
    consulted = Engine(
        other,
        providers=provided({"m": Scripted(text_reply("Consulted"))}),
        refinery=consult_runs,
    )
    consult = consulted.tool("writer", scope="acme", name="consult")

    async def session(start: VoiceStart) -> Spoken:
        await start.tools.dispatch("consult", {"input": "Look it up"})
        return SAID

    async def failing(start: VoiceStart) -> Spoken:
        raise RuntimeError("the line dropped")

    engine = engine_of(tmp_path, refinery=recording)
    voice = await engine.talk("voice", "acme", session, tools=consult, record=False)
    with pytest.raises(RuntimeError):
        await engine.talk("voice", "acme", failing, tools=consult, record=False)
    assert voice.error is None and recording.runs == [] and consult_runs.runs == []


async def test_the_record_keeps_its_own_transcript(tmp_path: Path) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    lines = [SpokenLine(speaker="person", text="Hello")]

    async def session(start: VoiceStart) -> Spoken:
        return Spoken(transcript=lines)

    voice = await engine_of(tmp_path).talk("voice", "acme", session)
    lines[0].text = "changed"
    assert voice.transcript[0].text == "Hello"


async def test_a_recorded_talk_is_a_row_of_kind_voice(tmp_path: Path, storage: Storage) -> None:
    write_folder(tmp_path, {"voice": VOICE}, files=FILES)
    engine = engine_of(tmp_path, refinery=Refinery(storage))
    voice = await engine.talk("voice", "acme", spoke)
    row = await storage.get_run("acme", voice.run_id)
    assert row is not None and row.kind == "voice" and row.agent == "voice"
    assert row.body["transcript"] == SAID.model_dump(mode="json")["transcript"]
    assert row.body["kind"] == "voice" and row.parent_run_id is None
