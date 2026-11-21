import ast
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

import rfnry_engine
from rfnry_engine import Engine, Generation, Message, Reply, ToolCall, ToolSpec
from rfnry_engine.models import Answer, CheckAnswer, ChoiceAnswer, Decided, Usage

from .fakes import Crew, until
from .test_docs_examples import ILLUSTRATIVE, PYTHON, YAML, first_line
from .trees import provided, write_folder

COPY = "class PerRun:"
BACKGROUND = "jobs = {}"
TEAM = 'team = await engine.run("lead", scope="acme", input=goal)'


def calls(*made: tuple[str, str, dict[str, Any]]) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=i, name=name, args=args) for i, name, args in made],
        stop_reason="tool_use",
    )


async def execute(start: str, engine: Engine, **given: Any) -> dict[str, Any]:
    assert start in ILLUSTRATIVE
    block = next(block for block in PYTHON if start in block.splitlines())
    namespace: dict[str, Any] = {
        **{name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__},
        "asyncio": asyncio,
        "engine": engine,
        **given,
    }
    code = compile(block, "engine.md", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    await eval(code, namespace)
    return namespace


class Workers:
    def __init__(self, lead: list[Reply], workers: dict[str, list[Reply]]) -> None:
        self.lead, self.workers = lead, workers

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if "Lead." in system:
            return self.lead.pop(0)
        return self.workers[str(messages[0].content)].pop(0)


WORKSPACE: dict[str, Any] = {"kind": "generation", "model": "strong", "workspace": True}


def lead_and_worker(root: Path, **orchestration: Any) -> Path:
    root.mkdir()
    return write_folder(
        root,
        {
            "lead": {
                **WORKSPACE,
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["worker"], **orchestration},
            },
            "worker": {**WORKSPACE, "instructions": "worker.md", "tools": {"allow": ["claim"]}},
        },
        runtime_tools=["claim"],
        files={"lead.md": "Lead.\n", "worker.md": "Work.\n"},
    )


def delegates(*inputs: str, agent: str = "worker", **extra: Any) -> Reply:
    return calls(
        *(
            (f"d{i}", "Delegate", {"agent": agent, "input": text, **extra})
            for i, text in enumerate(inputs)
        )
    )


def done(text: str) -> Reply:
    return Reply(text=text, stop_reason="end_turn")


async def test_the_copy_each_recipe_keeps_each_worker_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    gates = {"a": asyncio.Event(), "b": asyncio.Event()}

    def writing(i: int, name: str) -> list[Reply | asyncio.Event]:
        write = calls((f"w{i}", "Write", {"path": "guide.md", "content": f"from worker {i}"}))
        return [write, gates[name], done(f"done {i}")]

    model = Crew(
        [delegates("a", "b", background=True), done("waiting"), done("waiting"), done("merged")],
        {"a": writing(1, "a"), "b": writing(2, "b")},
    )
    root = lead_and_worker(tmp_path / "agents", background=True)
    engine = Engine(root, providers=provided({"strong": model}))
    work = asyncio.create_task(execute(COPY, engine, message="go"))
    await until(lambda: len(model.calls["lead"]) == 2, "the lead's second call")
    gates["b"].set()
    await until(lambda: len(model.calls["lead"]) == 3, "the lead hearing b end")
    gates["a"].set()
    namespace = await asyncio.wait_for(work, 5)
    run, backend = namespace["run"], namespace["backend"]
    assert isinstance(run, Generation) and run.reply == "merged"
    assert sorted(p.read_text() for p in tmp_path.glob("work/*/guide.md")) == [
        "from worker 1",
        "from worker 2",
    ]
    started = [step.child_run_id for step in run.steps[:2]]
    assert [child.run_id for child in run.children] == started[::-1]
    assert backend.merged == [child.run_id for child in run.children]
    assert {backend.parent[child.run_id] for child in run.children} == {run.run_id}


async def test_the_claim_tool_gives_an_item_to_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    take = calls(("c", "claim", {"item": "t1"}))
    model = Workers(
        [delegates("a", "b"), done("done")], {"a": [take, done("a")], "b": [take, done("b")]}
    )
    engine = Engine(lead_and_worker(tmp_path / "agents"), providers=provided({"strong": model}))
    namespace = await execute(COPY, engine, message="go")
    run = namespace["run"]
    results = [json.loads(str(child.steps[0].result)) for child in run.children]
    assert sorted(r["claimed"] for r in results) == [False, True]
    assert {r["by"] for r in results} == {namespace["claims"]["t1"]}
    assert namespace["claims"]["t1"] in {child.run_id for child in run.children}
    assert sorted(namespace["backend"].merged) == sorted(child.run_id for child in run.children)


class Boss:
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        if "Support." in system:
            return done("all fixed")
        if len(messages) == 1:
            return calls(("s", "start", {"input": "It is down"}))
        last = str(messages[-1].content)
        if "running" in last and "alse" in last:
            return done(last)
        return calls((f"k{len(messages)}", "check", {"job": "0"}))


async def test_the_background_recipe_returns_a_run_that_links_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agents").mkdir()
    root = write_folder(
        tmp_path / "agents",
        {
            "boss": {
                "kind": "generation",
                "model": "strong",
                "instructions": "boss.md",
                "tools": {"allow": ["start", "check"]},
            },
            "support": {"kind": "generation", "model": "strong", "instructions": "support.md"},
        },
        runtime_tools=["start", "check"],
        files={"boss.md": "Boss.\n", "support.md": "Support.\n"},
    )
    engine = Engine(root, providers=provided({"strong": Boss()}))
    namespace = await execute(BACKGROUND, engine, message="go")
    outer = namespace["outer"]
    finished = await namespace["jobs"]["0"]
    assert isinstance(finished, Generation) and finished.reply == "all fixed"
    assert finished.call is not None
    assert finished.call.run_id == outer.run_id
    assert finished.call.tool == "start"
    assert json.loads(str(outer.steps[-1].result)) == {"running": False, "reply": "all fixed"}


async def test_the_background_team_recipe_runs_as_written(tmp_path: Path) -> None:
    written = next(block for block in YAML if "captain:" in block.splitlines())
    root = write_folder(
        tmp_path,
        yaml.safe_load(written),
        files={
            "lead/instructions.md": "Lead.\n",
            "captain/instructions.md": "Captain.\n",
            "prover/instructions.md": "Prove.\n",
        },
    )
    proofs = {f"lemma {n}": asyncio.Event() for n in range(1, 5)}

    def captaining(first: int, verdict: str) -> list[Reply | asyncio.Event]:
        part = delegates(f"lemma {first}", f"lemma {first + 1}", agent="prover", background=True)
        return [part, done("waiting"), done("waiting"), done(verdict)]

    model = Crew(
        [
            delegates("lemmas 1 and 2", "lemmas 3 and 4", agent="captain", background=True),
            done("waiting"),
            done("waiting"),
            done("Lemmas 1 to 4 hold."),
        ],
        {
            "lemmas 1 and 2": captaining(1, "1 and 2 hold"),
            "lemmas 3 and 4": captaining(3, "3 and 4 hold"),
            **{name: [gate, done(f"{name} holds")] for name, gate in proofs.items()},
        },
    )
    engine = Engine(root, providers=provided({"strong": model}))
    work = asyncio.create_task(execute(TEAM, engine, goal="Prove lemmas 1 to 4."))

    def made(who: str, n: int) -> Any:
        return lambda: len(model.calls[who]) == n

    await until(made("lead", 2), "the lead's second call")
    await until(made("lemmas 1 and 2", 2), "the first captain's second call")
    await until(made("lemmas 3 and 4", 2), "the second captain's second call")
    for proof, heard in [
        ("lemma 1", made("lemmas 1 and 2", 3)),
        ("lemma 2", made("lead", 3)),
        ("lemma 3", made("lemmas 3 and 4", 3)),
    ]:
        proofs[proof].set()
        await until(heard, f"the end of {proof} heard")
    proofs["lemma 4"].set()
    team = (await asyncio.wait_for(work, 5))["team"]
    assert isinstance(team, Generation) and team.reply == "Lemmas 1 to 4 hold."
    assert [m.notice.agent for m in team.conversation if m.notice] == ["captain", "captain"]
    captains = [captain for captain in team.children if isinstance(captain, Generation)]
    assert [len(captain.children) for captain in captains] == [2, 2]
    assert [
        (
            captain.reply,
            [prover.reply for prover in captain.children if isinstance(prover, Generation)],
        )
        for captain in captains
    ] == [
        ("1 and 2 hold", ["lemma 1 holds", "lemma 2 holds"]),
        ("3 and 4 hold", ["lemma 3 holds", "lemma 4 holds"]),
    ]
    assert model.calls["lead"][3][-1].content == "[helper d2 · captain · finished]\n3 and 4 hold"


class Counting:
    def __init__(self, lead: list[Reply]) -> None:
        self.lead = lead

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        reply = self.lead.pop(0) if "Lead." in system else done("ok")
        return reply.model_copy(update={"usage": Usage(input=100)})

    async def decide(self, state: Any, questions: Mapping[str, Any]) -> Decided:
        answers: dict[str, Answer] = {
            qid: (
                ChoiceAnswer(choice="go", probabilities={"go": 0.9, "stay": 0.1})
                if question.type == "choice"
                else CheckAnswer(probability=0.9)
            )
            for qid, question in questions.items()
        }
        return Decided(answers=answers, usage=Usage(input=10))


async def test_spent_sums_delegated_routed_and_decision_runs(tmp_path: Path) -> None:
    root = write_folder(
        tmp_path,
        {
            "lead": {
                "kind": "generation",
                "model": "strong",
                "instructions": "lead.md",
                "orchestration": {
                    "delegate_to": ["worker", "router", "judge"],
                    "background": True,
                },
            },
            "worker": {"kind": "generation", "model": "strong", "instructions": "worker.md"},
            "router": {
                "kind": "decision",
                "model": "judge",
                "instructions": "router.md",
                "questions": ["router/*.json"],
                "orchestration": {"route": {"by": "pick", "to": {"go": "worker"}}},
            },
            "judge": {
                "kind": "decision",
                "model": "judge",
                "instructions": "judge.md",
                "questions": ["judge/*.json"],
            },
        },
        files={
            "lead.md": "Lead.\n",
            "worker.md": "Work.\n",
            "router.md": "",
            "router/pick.json": '{"type": "choice", "instructions": "Where to?", "options": {"go": "Go on", "stay": "Stay"}}',
            "judge.md": "",
            "judge/ok.json": '{"type": "check", "instructions": "Is it fine?"}',
        },
    )
    delegating = calls(
        ("d0", "Delegate", {"agent": "worker", "input": "a"}),
        ("d1", "Delegate", {"agent": "router", "input": "b"}),
        ("d2", "Delegate", {"agent": "judge", "input": "c"}),
        ("d3", "Delegate", {"agent": "worker", "input": "e", "background": True}),
    )
    waiting = calls(("j", "Job", {"id": "d1", "wait": 5}))
    model = Counting([delegating, waiting, done("merged")])
    engine = Engine(root, providers=provided({"strong": model, "judge": model}))
    run = await engine.run("lead", "acme", "go")
    assert isinstance(run, Generation)
    block = next(b for b in PYTHON if first_line(b) == "def spent(run):")
    namespace: dict[str, Any] = {name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__}
    exec(block.split("\n\n")[0], namespace)
    helper = next(step.child_run_id for step in run.steps if "background" in step.args)
    assert helper in {child.run_id for child in run.children}
    usage, made = namespace["spent"](run)
    assert made == 8
    assert usage.input == 6 * 100 + 2 * 10
