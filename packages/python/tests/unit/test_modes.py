import json
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, NotFoundError
from rfnry_engine.models import Reply, ToolCall
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.export import export_key
from rfnry_engine.refinery.labels import Replay, Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Answering, MemoryBucket, Scripted, delegating, text_reply
from .trees import write_folder

RUN = "def run(args, call):\n    return 'ran'\n"
FILLS = "def run(input, call):\n    return {'options': {'a': 'A', 'b': 'B'}}\n"
FILES = {
    "entry.md": "Entry instructions.\n",
    "chat.md": "Chat instructions.\n",
    "email.md": "Email instructions.\n",
    "helper.md": "Helper entry.\n",
    "helper-chat.md": "Helper chat.\n",
    "skills/tone/SKILL.md": "---\ndescription: Keep it warm\n---\nBe warm.\n",
    "skills/escalate/SKILL.md": "---\ndescription: Hand it over\n---\nHand it over.\n",
    "tools/lookup.py": RUN,
    "tools/refund.py": RUN,
}
TOOLS = {
    "lookup": {"module": "tools/lookup.py", "description": "Looks up"},
    "refund": {"module": "tools/refund.py", "description": "Refunds"},
}
AGENT: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "entry.md"}
TWO_MODES = {"chat": {"instructions": "chat.md"}, "email": {"instructions": "email.md"}}
ONLY_MODES = {"a": {"kind": "generation", "model": "m", "modes": TWO_MODES}}
SINCE = datetime(2026, 1, 1, tzinfo=UTC)


def _check(instructions: str) -> str:
    return json.dumps({"type": "check", "instructions": instructions})


AREA = json.dumps(
    {"type": "choice", "instructions": "Which area?", "options": {"other": None, "billing": None}}
)
ASKED = {
    "t.md": "Sort.\n",
    "shared/urgent.json": _check("Is it urgent?"),
    "chat/area.json": _check("Chat: which area?"),
    "email/area.json": _check("Email: which area?"),
}
SORTER: dict[str, Any] = {
    "kind": "decision",
    "model": "j",
    "instructions": "t.md",
    "questions": ["shared/*.json"],
    "modes": {"chat": {"questions": ["chat/*.json"]}, "email": {"questions": ["email/*.json"]}},
}


def _names(tools: list[Any]) -> set[str]:
    return {tool.name for tool in tools}


def _lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode().splitlines()]


def _folders(keys: list[str]) -> list[str]:
    return [key.split("/")[1] for key in keys]


def _calling(name: str, **args: Any) -> Reply:
    return Reply(tool_calls=[ToolCall(id="c1", name=name, args=args)], stop_reason="tool_use")


async def _replayed(refinery: Refinery, candidate: Engine, agent: str) -> list[Replay]:
    async with aclosing(refinery.replay("acme", candidate, agent=agent)) as pairs:
        return [pair async for pair in pairs]


class _Kept:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


async def test_a_mode_overlays_the_entry(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **AGENT,
                "tools": {"include": ["lookup"]},
                "skills": {"include": ["skills/tone"]},
                "modes": {
                    "chat": {
                        "instructions": "chat.md",
                        "tools": {"include": ["refund"]},
                        "skills": {"include": ["skills/escalate"]},
                    }
                },
            }
        },
        tools=TOOLS,
        files=FILES,
    )
    model = Scripted(text_reply("On the entry."), text_reply("In the mode."))
    async with Engine(folder, models={"m": model}) as engine:
        plain = await engine.run("a", "acme", "Hi")
        moded = await engine.run("a", "acme", "Hi", mode="chat")
    (entry_system, _, entry_tools), (mode_system, _, mode_tools) = model.calls
    assert isinstance(plain, Generation) and isinstance(moded, Generation)
    assert (plain.mode, moded.mode) == (None, "chat")
    assert "Entry instructions." in entry_system and "Chat instructions." not in entry_system
    assert "Chat instructions." in mode_system and "Entry instructions." not in mode_system
    assert "lookup" in _names(entry_tools) and "refund" not in _names(entry_tools)
    assert {"lookup", "refund"} <= _names(mode_tools)
    assert "skills/tone/SKILL.md — Keep it warm" in entry_system
    assert "skills/escalate/SKILL.md" not in entry_system
    assert "skills/tone/SKILL.md — Keep it warm" in mode_system
    assert "skills/escalate/SKILL.md — Hand it over" in mode_system


async def test_a_tool_only_a_mode_picks_runs_in_a_worker(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**AGENT, "modes": {"chat": {"tools": {"include": ["refund"]}}}}},
        tools={"refund": TOOLS["refund"]},
        files=FILES,
    )
    model = Scripted(_calling("refund"), text_reply("Done."))
    async with Engine(folder, models={"m": model}) as engine:
        run = await engine.run("a", "acme", "Hi", mode="chat")
    assert isinstance(run, Generation)
    assert [(step.tool, step.result, step.is_error) for step in run.steps] == [
        ("refund", "ran", False)
    ]


async def test_a_script_only_a_mode_asks_fills_its_question(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "t": {
                "kind": "decision",
                "model": "j",
                "instructions": "t.md",
                "questions": ["shared/*.json"],
                "modes": {"chat": {"questions": ["chat/*.json"]}},
            }
        },
        files={
            "t.md": "Sort.\n",
            "shared/urgent.json": _check("Is it urgent?"),
            "chat/order.json": json.dumps({"type": "choice", "instructions": "Which order?"}),
            "chat/order.py": FILLS,
        },
    )
    async with Engine(folder, models={"j": Answering()}) as engine:
        run = await engine.run("t", "acme", "Hi", mode="chat")
    assert isinstance(run, Decision)
    assert run.questions["order"].model_dump()["options"] == {"a": "A", "b": "B"}


async def test_exclude_wins_over_a_modes_include(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {
                **AGENT,
                "tools": {"include": ["lookup"], "exclude": ["refund"]},
                "skills": {"include": ["skills/tone"], "exclude": ["escalate"]},
                "modes": {
                    "chat": {
                        "tools": {"include": ["refund"]},
                        "skills": {"include": ["skills/escalate"]},
                    }
                },
            },
            "other": {**AGENT, "tools": {"include": ["refund"]}},
        },
        tools=TOOLS,
        files=FILES,
    )
    model = Scripted(text_reply("Done."))
    async with Engine(folder, models={"m": model}) as engine:
        await engine.run("a", "acme", "Hi", mode="chat")
        surface = EngineTools(engine, agent="a", scope="acme", mode="chat")
        with pytest.raises(InputError) as refused:
            await surface.dispatch("Read", {"path": "skills/escalate/SKILL.md"})
    system, _, tools = model.calls[0]
    assert "lookup" in _names(tools) and "refund" not in _names(tools)
    assert "- lookup:" in system and "- refund:" not in system
    assert "skills/tone/SKILL.md" in system and "skills/escalate/SKILL.md" not in system
    assert "refund" not in surface
    assert "Hand it over" not in str(refused.value)


async def test_an_unknown_mode_is_notfound_before_any_model_call(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": {**AGENT, "modes": TWO_MODES}}, files=FILES)
    model = Scripted()
    with pytest.raises(
        NotFoundError, match=r"^engine: agent 'a' has no mode 'ghost'; its modes are chat, email$"
    ):
        await Engine(folder, models={"m": model}).run("a", "acme", "Hi", mode="ghost")
    assert model.calls == []


async def test_an_agent_without_modes_has_no_mode_to_select(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": AGENT}, files=FILES)
    model = Scripted()
    with pytest.raises(
        NotFoundError, match=r"^engine: agent 'a' has no mode 'chat'; it has no modes$"
    ):
        await Engine(folder, models={"m": model}).run("a", "acme", "Hi", mode="chat")
    assert model.calls == []


async def test_an_agent_of_the_old_loader_has_no_modes(tmp_path: Path) -> None:
    place = tmp_path / "a" / "AGENT.md"
    place.parent.mkdir()
    place.write_text(
        "---\nname: a\nkind: generation\nmodel: m\ndescription: Works\n---\nWork.\n",
        encoding="utf-8",
    )
    model = Scripted()
    with pytest.raises(
        NotFoundError, match=r"^engine: agent 'a' has no mode 'chat'; it has no modes$"
    ):
        await Engine(tmp_path, models={"m": model}).run("a", "acme", "Hi", mode="chat")
    assert model.calls == []


async def test_an_agent_with_only_modes_needs_one(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, ONLY_MODES, files=FILES)
    model = Scripted()
    engine = Engine(folder, models={"m": model})
    needs = r"^engine: agent 'a' needs a mode; its modes are chat, email$"
    with pytest.raises(InputError, match=needs):
        await engine.run("a", "acme", "Hi")
    with pytest.raises(InputError, match=needs):
        engine.tool("a", scope="acme")
    with pytest.raises(InputError, match=needs):
        EngineTools(engine, agent="a", scope="acme")
    assert model.calls == []


async def test_an_agent_with_only_modes_runs_in_one(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, ONLY_MODES, files=FILES)
    model = Scripted(text_reply("Done."))
    run = await Engine(folder, models={"m": model}).run("a", "acme", "Hi", mode="email")
    assert run.mode == "email" and "Email instructions." in model.calls[0][0]


async def test_delegated_routed_and_engine_tool_runs_take_no_mode(tmp_path: Path) -> None:
    helper_modes = {"chat": {"instructions": "helper-chat.md"}}
    folder = write_folder(
        tmp_path,
        {
            "helper": {
                **AGENT,
                "instructions": "helper.md",
                "description": "Helps",
                "modes": helper_modes,
            },
            "lead": {
                **AGENT,
                "orchestration": {"delegate_to": ["helper"]},
                "modes": {"chat": {"instructions": "chat.md"}},
            },
            "triage": {
                "kind": "decision",
                "model": "j",
                "instructions": "entry.md",
                "questions": ["q/*.json"],
                "orchestration": {"route": {"by": "area", "to": {"billing": "helper"}}},
                "modes": {"chat": {"instructions": "chat.md"}},
            },
        },
        files={**FILES, "q/area.json": AREA},
    )
    model = Scripted(
        delegating("helper"),
        text_reply("Helped."),
        text_reply("Done."),
        text_reply("Routed."),
        text_reply("By tool."),
    )
    kept = _Kept()
    async with Engine(folder, models={"m": model, "j": Answering()}, refinery=kept) as engine:
        led = await engine.run("lead", "acme", "Hi", mode="chat")
        routing = await engine.run("triage", "acme", "Hi", mode="chat")
        answered = await engine.tool("helper", scope="acme").dispatch("helper", {"input": "Hi"})
    assert isinstance(led, Generation) and isinstance(routing, Decision)
    assert led.mode == "chat" and routing.mode == "chat"
    delegated, routed = led.children[0], routing.routed_to
    assert delegated.mode is None and routed is not None and routed.mode is None
    assert isinstance(delegated, Generation) and isinstance(routed, Generation)
    assert "Helper entry." in delegated.system and "Helper chat." not in delegated.system
    assert "Helper entry." in routed.system and "Helper chat." not in routed.system
    assert answered == "By tool."
    assert kept.runs[-1].agent == "helper" and kept.runs[-1].mode is None
    assert "Helper entry." in model.calls[-1][0]


def test_each_mode_is_compiled_at_load(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path, {"a": {**AGENT, "modes": {"bad": {"instructions": "ghost.md"}}}}, files=FILES
    )
    with pytest.raises(
        ConfigurationError,
        match=r"^compiled: mode 'bad' of agent 'a' cannot read its instructions 'ghost\.md'",
    ):
        Engine(folder, models={"m": Scripted()})


def test_an_agent_named_like_a_mode_of_another_is_refused(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**AGENT, "modes": {"b": {"description": "A mode"}}}, "a.b": AGENT},
        files=FILES,
    )
    with pytest.raises(
        ConfigurationError,
        match=r"^compiled: agent 'a\.b' and mode 'b' of agent 'a' would export under one key, 'a\.b'; rename one$",
    ):
        Engine(folder, models={"m": Scripted()})


def test_two_modes_of_two_agents_exporting_under_one_key_are_refused(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "a": {**AGENT, "modes": {"b.c": {"description": "A mode"}}},
            "a.b": {**AGENT, "modes": {"c": {"description": "A mode"}}},
        },
        files=FILES,
    )
    with pytest.raises(
        ConfigurationError,
        match=r"^compiled: mode 'b\.c' of agent 'a' and mode 'c' of agent 'a\.b' would export under one key, 'a\.b\.c'; rename one$",
    ):
        Engine(folder, models={"m": Scripted()})


async def test_the_fingerprint_differs_by_mode(tmp_path: Path) -> None:
    alike = {"chat": {"description": "Alike"}, "email": {"description": "Alike"}}
    folder = write_folder(tmp_path, {"a": {**AGENT, "modes": alike}}, files=FILES)
    model = Scripted(text_reply("1"), text_reply("2"), text_reply("3"))
    async with Engine(folder, models={"m": model}) as engine:
        runs = [await engine.run("a", "acme", "Hi", mode=mode) for mode in (None, "chat", "email")]
    assert len({run.fingerprint.agent for run in runs}) == 3


async def test_a_decision_agents_modes_ask_their_own_questions(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"t": SORTER}, files=ASKED)
    model = Answering()
    async with Engine(folder, models={"j": model}) as engine:
        plain = await engine.run("t", "acme", "Hi")
        chat = await engine.run("t", "acme", "Hi", mode="chat")
        email = await engine.run("t", "acme", "Hi", mode="email")
    assert isinstance(plain, Decision)
    assert isinstance(chat, Decision) and isinstance(email, Decision)
    assert (plain.mode, chat.mode, email.mode) == (None, "chat", "email")
    assert [sorted(asked) for _, asked in model.calls] == [
        ["urgent"],
        ["area", "urgent"],
        ["area", "urgent"],
    ]
    assert chat.questions["area"].model_dump()["instructions"] == "Chat: which area?"
    assert email.questions["area"].model_dump()["instructions"] == "Email: which area?"


def test_the_export_key_separates_by_mode() -> None:
    assert export_key("acme", "t", "decision", "S") == "acme/t/decision/S.jsonl"
    assert export_key("acme", "t", "decision", "S", "chat") == "acme/t.chat/decision/S.jsonl"


async def test_the_mode_is_recorded_and_exported_apart(tmp_path: Path, storage: Storage) -> None:
    folder = write_folder(tmp_path, {"t": SORTER}, files=ASKED)
    refinery = Refinery(storage)
    async with Engine(folder, models={"j": Answering()}, refinery=refinery) as engine:
        plain = await engine.run("t", "acme", "Hi")
        chat = await engine.run("t", "acme", "Hi", mode="chat")
        email = await engine.run("t", "acme", "Hi", mode="email")
    assert (await refinery.run("acme", plain.run_id)).mode is None
    assert (await refinery.run("acme", chat.run_id)).mode == "chat"
    assert (await refinery.run("acme", email.run_id)).mode == "email"
    bucket = MemoryBucket()
    exported = await refinery.export("acme", bucket)
    assert _folders(exported.keys) == ["t", "t.chat", "t.email"]
    assert exported.rows == {"decision": 5}
    rows = {key.split("/")[1]: _lines(bucket.objects[key]) for key in exported.keys}
    assert {row["mode"] for row in rows["t"]} == {None}
    assert {row["mode"] for row in rows["t.chat"]} == {"chat"}
    assert {row["mode"] for row in rows["t.email"]} == {"email"}
    wording = {
        folder: next(r["question"]["instructions"] for r in held if r["question_id"] == "area")
        for folder, held in rows.items()
        if folder != "t"
    }
    assert wording == {"t.chat": "Chat: which area?", "t.email": "Email: which area?"}


async def test_exporting_one_mode_keeps_only_that_mode(tmp_path: Path, storage: Storage) -> None:
    folder = write_folder(tmp_path, {"t": SORTER}, files=ASKED)
    refinery = Refinery(storage)
    async with Engine(folder, models={"j": Answering()}, refinery=refinery) as engine:
        await engine.run("t", "acme", "Hi")
        chat = await engine.run("t", "acme", "Hi", mode="chat")
        await engine.run("t", "acme", "Hi", mode="email")
    bucket = MemoryBucket()
    exported = await refinery.export("acme", bucket, mode="chat")
    assert _folders(exported.keys) == ["t.chat"] and exported.rows == {"decision": 2}
    assert list(bucket.objects) == exported.keys
    stored = await storage.get_run("acme", chat.run_id)
    assert stored is not None and exported.watermark == stored.recorded
    nothing = await refinery.export("acme", MemoryBucket(), since=SINCE, mode="ghost")
    assert (nothing.keys, nothing.rows, nothing.watermark) == ([], {}, SINCE)


async def test_every_dataset_carries_the_mode(tmp_path: Path, storage: Storage) -> None:
    folder = write_folder(tmp_path, {"a": {**AGENT, "modes": TWO_MODES}}, files=FILES)
    refinery = Refinery(storage)
    model = Scripted(text_reply("No."))
    async with Engine(folder, models={"m": model}, refinery=refinery) as engine:
        run = await engine.run("a", "acme", "Hi", mode="chat")
    await refinery.review("acme", run.run_id, Review(correction="Yes."))
    bucket = MemoryBucket()
    exported = await refinery.export("acme", bucket)
    assert sorted(key.split("/")[2] for key in exported.keys) == ["generation", "preference"]
    assert set(_folders(exported.keys)) == {"a.chat"}
    for key in exported.keys:
        assert [row["mode"] for row in _lines(bucket.objects[key])] == ["chat"]


async def test_replay_uses_the_recorded_mode(tmp_path: Path, storage: Storage) -> None:
    folder = write_folder(tmp_path, {"a": {**AGENT, "modes": TWO_MODES}}, files=FILES)
    refinery = Refinery(storage)
    model = Scripted(text_reply("Recorded."))
    async with Engine(folder, models={"m": model}, refinery=refinery) as engine:
        await engine.run("a", "acme", "Hi", mode="chat")
    candidate_model = Scripted(text_reply("Replayed."))
    async with Engine(folder, models={"m": candidate_model}) as candidate:
        [pair] = await _replayed(refinery, candidate, "a")
    assert pair.error is None and pair.candidate is not None
    assert pair.recorded.mode == "chat" and pair.candidate.mode == "chat"
    assert "Chat instructions." in candidate_model.calls[0][0]


async def test_replay_of_a_decision_uses_the_recorded_mode(
    tmp_path: Path, storage: Storage
) -> None:
    folder = write_folder(tmp_path, {"t": SORTER}, files=ASKED)
    refinery = Refinery(storage)
    async with Engine(folder, models={"j": Answering()}, refinery=refinery) as engine:
        await engine.run("t", "acme", "Hi", mode="email")
    model = Answering()
    async with Engine(folder, models={"j": model}) as candidate:
        [pair] = await _replayed(refinery, candidate, "t")
    assert pair.error is None and pair.candidate is not None
    assert pair.candidate.mode == "email"
    assert sorted(model.calls[0][1]) == ["area", "urgent"]


async def test_replay_of_a_routed_run_uses_the_top_runs_mode(
    tmp_path: Path, storage: Storage
) -> None:
    folder = write_folder(
        tmp_path,
        {
            "helper": {
                **AGENT,
                "instructions": "helper.md",
                "modes": {"chat": {"instructions": "helper-chat.md"}},
            },
            "triage": {
                "kind": "decision",
                "model": "j",
                "instructions": "entry.md",
                "questions": ["q/*.json"],
                "orchestration": {"route": {"by": "area", "to": {"billing": "helper"}}},
                "modes": {"chat": {"instructions": "chat.md"}},
            },
        },
        files={**FILES, "q/area.json": AREA},
    )
    refinery = Refinery(storage)
    recorded = Scripted(text_reply("Routed."))
    async with Engine(
        folder, models={"m": recorded, "j": Answering()}, refinery=refinery
    ) as engine:
        await engine.run("triage", "acme", "Hi", mode="chat")
    replaying = Scripted(text_reply("Routed again."))
    async with Engine(folder, models={"m": replaying, "j": Answering()}) as candidate:
        [pair] = await _replayed(refinery, candidate, "triage")
    assert pair.error is None and isinstance(pair.candidate, Decision)
    assert pair.candidate.mode == "chat"
    assert pair.candidate.routed_to is not None and pair.candidate.routed_to.mode is None


async def test_replay_of_a_mode_that_is_gone_says_so(tmp_path: Path, storage: Storage) -> None:
    folder = write_folder(tmp_path, {"a": {**AGENT, "modes": TWO_MODES}}, files=FILES)
    refinery = Refinery(storage)
    model = Scripted(text_reply("Recorded."))
    async with Engine(folder, models={"m": model}, refinery=refinery) as engine:
        recorded = await engine.run("a", "acme", "Hi", mode="chat")
    (tmp_path / "later").mkdir()
    later = write_folder(
        tmp_path / "later",
        {"a": {**AGENT, "modes": {"email": {"instructions": "email.md"}}}},
        files=FILES,
    )
    untouched = Scripted()
    async with Engine(later, models={"m": untouched}) as candidate:
        [pair] = await _replayed(refinery, candidate, "a")
    assert pair.candidate is None and pair.error == (
        f"InputError: refinery: run {recorded.run_id!r} ran in mode 'chat', "
        "which agent 'a' of the candidate does not have"
    )
    assert untouched.calls == []
