import asyncio
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import (
    Answer,
    CheckAnswer,
    Image,
    Message,
    Reply,
    ScoreAnswer,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Answering, Fixed, Scripted, delegating, text_reply
from .trees import provided, write_folder

PNG = b"\x89PNG\r\n\x1a\ndiagram"
TEAM = '{"type": "choice", "instructions": "Which?", "options": {"other": "Other", "writing": "Writing"}}'
UNFIT: list[Any] = [
    [("strong", 1)],
    {"": Scripted()},
    {" strong": Scripted()},
    {1: Scripted()},
    {"strong": object()},
]
JUDGEMENT: dict[str, Answer] = {
    "grounded": CheckAnswer(probability=0.9),
    "done": ScoreAnswer(score=2, probabilities=[0.0, 0.1, 0.9]),
}
TRIAGE: dict[str, Answer] = {
    "urgent": CheckAnswer(probability=0.6),
    "severity": ScoreAnswer(score=1, probabilities=[0.2, 0.5, 0.2, 0.1]),
}
TAUGHT = Reply(
    tool_calls=[
        ToolCall(
            id="t1",
            name="Output",
            args={"answers": {"urgent": True, "severity": 3}, "rationale": "An outage is urgent."},
        )
    ],
    stop_reason="tool_use",
)
LOAD = Reply(
    tool_calls=[ToolCall(id="l1", name="Load", args={"path": "lead/diagram.png"})],
    stop_reason="tool_use",
)


class Pictures(Scripted):
    media_types = {"image/*"}


class Documents(Scripted):
    media_types = {"application/pdf"}


class Seeing(Answering):
    media_types = {"image/*"}


class Named(Scripted):
    def __init__(self, name: str, *replies: Reply) -> None:
        super().__init__(*replies)
        self.model = name


class NamedDecider(Answering):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.model = name


class Lockstep:
    def __init__(self, name: str, step: asyncio.Barrier, *replies: Reply) -> None:
        self.name, self.step = name, step
        self.replies = list(replies)
        self.calls = 0

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        await self.step.wait()
        self.calls += 1
        return self.replies.pop(0).model_copy(update={"model": self.name})


LEAD: dict[str, Any] = {
    "kind": "generation",
    "model": "strong",
    "description": "Works",
    "instructions": "lead.md",
}
WRITER: dict[str, Any] = {
    "kind": "generation",
    "model": "fast",
    "description": "Works",
    "instructions": "writer.md",
}
BODIES = {"lead.md": "Work.\n", "writer.md": "Work.\n"}


def pair(root: Path) -> Path:
    return write_folder(
        root,
        {"lead": {**LEAD, "orchestration": {"delegate_to": ["writer"]}}, "writer": WRITER},
        files=BODIES,
    )


def crew(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {**LEAD, "orchestration": {"delegate_to": ["triage", "writer"]}},
            "triage": {
                "kind": "decision",
                "model": "judge",
                "description": "Works",
                "instructions": "triage.md",
                "questions": ["triage/*.json"],
                "orchestration": {"route": {"by": "team", "to": {"writing": "writer"}}},
            },
            "writer": WRITER,
        },
        files={**BODIES, "triage.md": "", "triage/team.json": TEAM},
    )


def with_picture(root: Path) -> Path:
    (root / "lead").mkdir(parents=True, exist_ok=True)
    (root / "lead/diagram.png").write_bytes(PNG)
    return write_folder(
        root,
        {
            "lead": {
                **LEAD,
                "files": {"allow": ["lead/diagram.png"]},
                "orchestration": {"delegate_to": ["writer"]},
            },
            "writer": WRITER,
            "grader": {
                "kind": "decision",
                "model": "judge",
                "description": "Works",
                "instructions": "grader.md",
                "questions": ["grader/*.json"],
            },
        },
        files={
            **BODIES,
            "grader.md": "",
            "grader/urgent.json": '{"type": "check", "instructions": "Urgent?"}',
        },
    )


def with_a_mode(root: Path) -> Path:
    return write_folder(
        root,
        {
            "a": {
                "kind": "generation",
                "model": "m",
                "instructions": "a.md",
                "modes": {"quick": {"model": "q"}},
            }
        },
        files={"a.md": "Work.\n"},
    )


async def test_run_models_win_over_the_engines_by_key(tmp_path: Path) -> None:
    kept, fast = Scripted(text_reply("the engine's")), Scripted(text_reply("drafted"))
    engine = Engine(pair(tmp_path), providers=provided({"strong": kept, "fast": fast}))
    brought = Scripted(delegating("writer"), text_reply("the run's", model="run-1"))
    run = await engine.run("lead", "acme", "x", models={"strong": brought})
    assert isinstance(run, Generation)
    assert (run.reply, run.model) == ("the run's", "run-1")
    assert kept.calls == [] and len(brought.calls) == 2
    assert run.children[0].model is None and len(fast.calls) == 1
    afterwards = await engine.run("lead", "acme", "x")
    assert isinstance(afterwards, Generation)
    assert afterwards.reply == "the engine's" and len(brought.calls) == 2


async def test_an_engine_given_no_models_runs_on_the_runs(tmp_path: Path) -> None:
    run = await Engine(pair(tmp_path), providers=provided({"strong": None, "fast": None})).run(
        "writer", "acme", "x", models={"fast": Scripted(text_reply("done"))}
    )
    assert isinstance(run, Generation) and run.reply == "done"


async def test_delegated_and_routed_runs_use_the_callers_models(tmp_path: Path) -> None:
    engine_strong, engine_judge, engine_fast = Scripted(), Answering(), Scripted()
    engine = Engine(
        crew(tmp_path),
        providers=provided({"strong": engine_strong, "judge": engine_judge, "fast": engine_fast}),
    )
    lead = Scripted(
        delegating("writer"),
        delegating("triage", call_id="d2"),
        text_reply("done", model="lead-1"),
    )
    judge = NamedDecider("judge-1")
    fast = Scripted(text_reply("delegated", model="fast-1"), text_reply("routed", model="fast-2"))
    run = await engine.run(
        "lead", "acme", "x", models={"strong": lead, "judge": judge, "fast": fast}
    )
    assert isinstance(run, Generation)
    assert [step.is_error for step in run.steps] == [False, False]
    delegated, routing = run.children
    assert isinstance(delegated, Generation) and isinstance(routing, Decision)
    assert (run.model, delegated.model, routing.model) == ("lead-1", "fast-1", "judge-1")
    assert routing.routed_to is not None and routing.routed_to.model == "fast-2"
    assert len(lead.calls) == 3 and len(judge.calls) == 1 and len(fast.calls) == 2
    assert (engine_strong.calls, engine_judge.calls, engine_fast.calls) == ([], [], [])


async def test_concurrent_runs_keep_their_own_models(tmp_path: Path) -> None:
    step = asyncio.Barrier(2)

    def own(name: str) -> dict[str, Any]:
        return {
            "strong": Lockstep(f"lead-{name}", step, delegating("writer", name), text_reply(name)),
            "fast": Lockstep(f"writer-{name}", step, text_reply(f"drafted {name}")),
        }

    engine_strong, engine_fast = Scripted(), Scripted()
    engine = Engine(
        pair(tmp_path), providers=provided({"strong": engine_strong, "fast": engine_fast})
    )
    one, other = own("one"), own("other")
    async with asyncio.timeout(5):
        runs = await asyncio.gather(
            engine.run("lead", "acme", "x", models=one),
            engine.run("lead", "acme", "x", models=other),
        )
    for run, name in zip(runs, ("one", "other"), strict=True):
        assert isinstance(run, Generation)
        assert (run.reply, run.model) == (name, f"lead-{name}")
        assert [m.model for m in run.conversation if m.role == "assistant"] == [f"lead-{name}"] * 2
        child = run.children[0]
        assert isinstance(child, Generation)
        assert (child.reply, child.model) == (f"drafted {name}", f"writer-{name}")
        assert not run.steps[0].is_error
    assert [adapter.calls for adapter in (*one.values(), *other.values())] == [2, 1, 2, 1]
    assert (engine_strong.calls, engine_fast.calls) == ([], [])


async def test_the_kind_is_checked_at_run_start(tmp_path: Path) -> None:
    engine = Engine(
        crew(tmp_path), providers=provided({"strong": None, "fast": None, "judge": None})
    )
    decider = Answering()
    with pytest.raises(
        ConfigurationError,
        match=r"^engine: agent 'lead' is a generation agent, and model 'strong' has no generate\(\)$",
    ):
        await engine.run("lead", "acme", "x", models={"strong": decider})
    assert decider.calls == []
    generator = Scripted()
    with pytest.raises(
        ConfigurationError,
        match=r"^engine: agent 'triage' is a decision agent, and model 'judge' has no decide\(\)$",
    ):
        await engine.run("triage", "acme", "x", models={"judge": generator})
    assert generator.calls == []


async def test_the_kind_is_checked_for_the_runs_agent_alone(tmp_path: Path) -> None:
    run = await Engine(
        crew(tmp_path), providers=provided({"strong": None, "fast": None, "judge": None})
    ).run(
        "lead",
        "acme",
        "x",
        models={"strong": Scripted(text_reply("ok")), "judge": Scripted(), "fast": Answering()},
    )
    assert isinstance(run, Generation) and run.reply == "ok"


async def test_a_delegated_or_routed_child_is_checked_at_its_own_start(tmp_path: Path) -> None:
    engine = Engine(
        crew(tmp_path), providers=provided({"strong": None, "fast": None, "judge": None})
    )
    lead = Scripted(delegating("triage"), text_reply("noted"))
    run = await engine.run("lead", "acme", "x", models={"strong": lead, "judge": Scripted()})
    assert isinstance(run, Generation)
    assert run.reply == "noted" and run.steps[0].is_error
    assert "agent 'triage' is a decision agent, and model 'judge' has no decide()" in str(
        run.steps[0].result
    )
    judge = Answering()
    with pytest.raises(
        ConfigurationError, match="agent 'writer' is a generation agent, and model 'fast' has no"
    ):
        await engine.run("triage", "acme", "x", models={"judge": judge, "fast": Answering()})
    assert len(judge.calls) == 1


async def test_run_model_records_what_the_reply_reports(tmp_path: Path) -> None:
    root = with_picture(tmp_path)
    engine = Engine(
        root,
        providers=provided(
            {"strong": None, "fast": Named("engine-1"), "judge": NamedDecider("engine-2")}
        ),
    )
    reported = await engine.run(
        "writer",
        "acme",
        "x",
        models={"fast": Named("declared-1", text_reply("a", model="reported-1"))},
    )
    assert reported.model == "reported-1"
    declared = await engine.run(
        "writer", "acme", "x", models={"fast": Named("declared-2", text_reply("b"))}
    )
    assert declared.model == "declared-2"
    decided = await engine.run("grader", "acme", "x", models={"judge": NamedDecider("declared-3")})
    assert decided.model == "declared-3"


async def test_media_types_follow_the_runs_model(tmp_path: Path) -> None:
    root = with_picture(tmp_path)
    taken = await Engine(
        root, providers=provided({"strong": Documents(), "fast": None, "judge": None})
    ).run("lead", "acme", "x", models={"strong": Pictures(LOAD, text_reply("seen"))})
    assert isinstance(taken, Generation) and not taken.steps[0].is_error
    refused = await Engine(
        root, providers=provided({"strong": Pictures(), "fast": None, "judge": None})
    ).run("lead", "acme", "x", models={"strong": Documents(LOAD, text_reply("no"))})
    assert isinstance(refused, Generation) and refused.steps[0].is_error
    assert "does not take image/png" in str(refused.steps[0].result)
    picture = [Image(media_type="image/png", data=PNG), Text(text="What is it?")]
    seen = await Engine(
        root, providers=provided({"strong": None, "fast": None, "judge": Answering()})
    ).run("grader", "acme", picture, models={"judge": Seeing()})
    assert isinstance(seen, Decision) and seen.state == picture
    with pytest.raises(InputError, match="takes text only; got image/png"):
        await Engine(
            root, providers=provided({"strong": None, "fast": None, "judge": Seeing()})
        ).run("grader", "acme", picture, models={"judge": Answering()})


async def test_replay_takes_models(tmp_path: Path, storage: Storage) -> None:
    root = with_picture(tmp_path)
    refinery = Refinery(storage)
    production = Engine(
        root,
        refinery=refinery,
        providers=provided(
            {"strong": None, "fast": Scripted(text_reply("recorded")), "judge": Answering()}
        ),
    )
    await production.run("writer", "acme", "Draft it")
    await production.run("grader", "acme", "Is it urgent?")
    kept, kept_judge = Scripted(), Answering()
    candidate = Engine(
        root, providers=provided({"strong": None, "fast": kept, "judge": kept_judge})
    )
    fast = Scripted(text_reply("replayed", model="run-1"))
    judge = NamedDecider("run-2")
    generated = [
        pair
        async for pair in refinery.replay("acme", candidate, agent="writer", models={"fast": fast})
    ]
    decided = [
        pair
        async for pair in refinery.replay(
            "acme", candidate, agent="grader", models={"judge": judge}
        )
    ]
    assert [pair.error for pair in (*generated, *decided)] == [None, None]
    assert generated[0].candidate is not None and generated[0].candidate.model == "run-1"
    assert decided[0].candidate is not None and decided[0].candidate.model == "run-2"
    assert kept.calls == [] and kept_judge.calls == []


async def test_label_takes_models(fixtures: Path, storage: Storage) -> None:
    refinery = Refinery(storage)
    production = Engine(
        fixtures,
        refinery=refinery,
        providers=provided({"strong": Scripted(text_reply("Refunded.")), "judge": Fixed(TRIAGE)}),
    )
    generated = await production.run("billing", "acme", "Charged twice.")
    decided = await production.run("triage", "acme", {"message": "It is down"})
    engine = Engine(fixtures, providers=provided({"strong": None, "judge": None}))
    bare = await refinery.label("acme", engine, judge="judge", teacher="teacher")
    assert (bare.judged, bare.taught) == (0, 0)
    assert set(bare.errors) == {generated.run_id, decided.run_id}
    assert "has no decide()" in bare.errors[generated.run_id]
    judge, teacher = Fixed(JUDGEMENT), Scripted(TAUGHT)
    labeled = await refinery.label(
        "acme",
        engine,
        judge="judge",
        teacher="teacher",
        models={"strong": teacher, "judge": judge},
    )
    assert (labeled.judged, labeled.taught, labeled.errors) == (1, 1, {})
    assert len(judge.calls) == 1 and len(teacher.calls) == 1
    assert [row.run_id for row in await storage.labels("acme", source="judge")] == [
        generated.run_id
    ]
    assert [row.run_id for row in await storage.labels("acme", source="teacher")] == [
        decided.run_id
    ]


async def test_a_model_only_a_mode_names_can_be_given_to_the_engine_or_the_run(
    tmp_path: Path,
) -> None:
    folder = with_a_mode(tmp_path)
    on_engine = await Engine(
        folder, providers=provided({"m": Scripted(), "q": Scripted(text_reply("engine"))})
    ).run("a", "acme", "x", mode="quick")
    assert isinstance(on_engine, Generation) and on_engine.reply == "engine"
    on_run = await Engine(folder, providers=provided({"m": Scripted(), "q": Scripted()})).run(
        "a", "acme", "x", mode="quick", models={"q": Scripted(text_reply("run"))}
    )
    assert isinstance(on_run, Generation) and on_run.reply == "run"


async def test_a_mode_on_a_model_of_the_other_kind_is_refused(tmp_path: Path) -> None:
    engine = Engine(with_a_mode(tmp_path), providers=provided({"m": Scripted(), "q": Answering()}))
    with pytest.raises(
        ConfigurationError, match=r"agent 'a' is a generation agent, and model 'q' has no generate"
    ):
        await engine.run("a", "acme", "x", mode="quick")


async def test_a_run_refuses_a_model_no_agent_runs_on(tmp_path: Path) -> None:
    llm = Scripted()
    engine = Engine(pair(tmp_path), providers=provided({"strong": llm, "fast": None}))
    with pytest.raises(
        ConfigurationError,
        match="^engine: no agent runs on model 'fsat'; the agents of rfnry.yaml run on fast, strong$",
    ):
        await engine.run("lead", "acme", "x", models={"fsat": Scripted()})
    assert llm.calls == []


@pytest.mark.parametrize("models", UNFIT)
async def test_models_that_do_not_fit_are_refused_where_they_are_given(
    tmp_path: Path, storage: Storage, models: Any
) -> None:
    engine = Engine(pair(tmp_path), providers=provided({"strong": None, "fast": None}))
    with pytest.raises(ConfigurationError, match="^engine: "):
        await engine.run("lead", "acme", "x", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        engine.tool("lead", scope="acme", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        EngineTools(engine, agent="lead", scope="acme", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        Refinery(storage).replay("acme", engine, agent="lead", models=models)
    with pytest.raises(ConfigurationError, match="^engine: "):
        await Refinery(storage).label("acme", engine, models=models)


async def test_engine_tool_models_reach_its_run_and_what_it_delegates(tmp_path: Path) -> None:
    engine = Engine(pair(tmp_path), providers=provided({"strong": Scripted(), "fast": Scripted()}))
    lead = Scripted(delegating("writer"), text_reply("done"))
    fast = Scripted(text_reply("drafted"))
    tools = engine.tool("lead", scope="acme", models={"strong": lead, "fast": fast})
    assert await tools.dispatch("lead", {"input": "x"}) == "done"
    assert len(lead.calls) == 2 and len(fast.calls) == 1


async def test_a_run_lends_no_models_to_an_engine_tool_it_calls(tmp_path: Path) -> None:
    engine = Engine(
        write_folder(
            tmp_path,
            {"lead": {**LEAD, "tools": {"allow": ["consult"]}}, "writer": WRITER},
            runtime_tools=["consult"],
            files=BODIES,
        ),
        providers=provided({"strong": None, "fast": None}),
    )
    consult = engine.tool("writer", scope="acme", name="consult")
    lead = Scripted(
        Reply(
            tool_calls=[ToolCall(id="c1", name="consult", args={"input": "x"})],
            stop_reason="tool_use",
        ),
        text_reply("noted"),
    )
    fast = Scripted()
    run = await engine.run(
        "lead", "acme", "x", tools=consult, models={"strong": lead, "fast": fast}
    )
    assert isinstance(run, Generation) and run.steps[0].is_error
    assert "agent 'writer' is a generation agent, and model 'fast' has no generate()" in str(
        run.steps[0].result
    )
    assert fast.calls == []


async def test_engine_tools_follow_the_models_given(tmp_path: Path) -> None:
    engine = Engine(
        with_picture(tmp_path),
        providers=provided({"strong": Documents(), "fast": None, "judge": None}),
    )
    loaded = await EngineTools(
        engine, agent="lead", scope="acme", models={"strong": Pictures()}
    ).dispatch("Load", {"path": "lead/diagram.png"})
    assert loaded
    with pytest.raises(InputError, match="does not take image/png"):
        await EngineTools(engine, agent="lead", scope="acme").dispatch(
            "Load", {"path": "lead/diagram.png"}
        )
    fast = Scripted(text_reply("drafted"))
    delegated = await EngineTools(
        engine, agent="lead", scope="acme", models={"fast": fast}
    ).dispatch("Delegate", {"agent": "writer", "input": "x"})
    assert delegated == "drafted" and len(fast.calls) == 1
