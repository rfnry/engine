import ast
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import rfnry_engine
from rfnry_engine import Engine, Generation, Message, Reply, ToolCall, ToolSpec
from rfnry_engine.models import Answer, CheckAnswer, ChoiceAnswer, Decided, Usage

from .test_docs_examples import ILLUSTRATIVE, PYTHON, first_line

COPY = (
    "from rfnry_engine import Local, RunStarted, Tool, ToolResultReady, Tools, Workspace, "
    "current_call"
)
BACKGROUND = "jobs = {}"


def agents(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def calls(*made: tuple[str, str, dict[str, Any]]) -> Reply:
    return Reply(
        tool_calls=[ToolCall(id=i, name=name, args=args) for i, name, args in made],
        stop_reason="tool_use",
    )


async def execute(start: str, engine: Engine, **given: Any) -> dict[str, Any]:
    assert start in ILLUSTRATIVE
    block = next(block for block in PYTHON if first_line(block) == start)
    namespace: dict[str, Any] = {
        **{name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__},
        "Path": Path,
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


LEAD = {
    "lead/AGENT.md": "---\nkind: generation\nagents: [worker]\nworkspace: true\n---\nLead.\n",
    "worker/AGENT.md": "---\nkind: generation\nworkspace: true\n---\nWork.\n",
}


def delegates(*inputs: str) -> Reply:
    return calls(
        *(
            (f"d{i}", "Delegate", {"agent": "worker", "input": text})
            for i, text in enumerate(inputs)
        )
    )


def done(text: str) -> Reply:
    return Reply(text=text, stop_reason="end_turn")


async def test_the_copy_each_recipe_keeps_each_worker_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def writing(i: int) -> list[Reply]:
        write = calls((f"w{i}", "Write", {"path": "guide.md", "content": f"from worker {i}"}))
        return [write, done(f"done {i}")]

    model = Workers([delegates("a", "b"), done("merged")], {"a": writing(1), "b": writing(2)})
    engine = Engine(agents(tmp_path / "agents", LEAD), generation=model)
    namespace = await execute(COPY, engine, task="go")
    run, backend = namespace["run"], namespace["backend"]
    assert isinstance(run, Generation)
    assert sorted(p.read_text() for p in tmp_path.glob("work/*/guide.md")) == [
        "from worker 1",
        "from worker 2",
    ]
    assert set(backend.merged) == {child.run_id for child in run.children}
    assert {backend.parent[child.run_id] for child in run.children} == {run.run_id}


async def test_the_claim_tool_gives_a_task_to_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    take = calls(("c", "claim", {"task": "t1"}))
    model = Workers(
        [delegates("a", "b"), done("done")], {"a": [take, done("a")], "b": [take, done("b")]}
    )
    engine = Engine(agents(tmp_path / "agents", LEAD), generation=model)
    namespace = await execute(COPY, engine, task="go")
    run = namespace["run"]
    results = [json.loads(str(child.steps[0].result)) for child in run.children]
    assert sorted(r["claimed"] for r in results) == [False, True]
    assert {r["by"] for r in results} == {namespace["claims"]["t1"]}
    assert namespace["claims"]["t1"] in {child.run_id for child in run.children}


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
    files = {
        "boss/AGENT.md": "---\nkind: generation\n---\nBoss.\n",
        "support/AGENT.md": "---\nkind: generation\n---\nSupport.\n",
    }
    engine = Engine(agents(tmp_path / "agents", files), generation=Boss())
    namespace = await execute(BACKGROUND, engine, task="go")
    outer = namespace["outer"]
    finished = await namespace["jobs"]["0"]
    assert isinstance(finished, Generation) and finished.reply == "all fixed"
    assert finished.call is not None
    assert finished.call.run_id == outer.run_id
    assert finished.call.tool == "start"
    assert json.loads(str(outer.steps[-1].result)) == {"running": False, "reply": "all fixed"}


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
    files = {
        "lead/AGENT.md": "---\nkind: generation\nagents: [worker, router, judge]\n---\nLead.\n",
        "worker/AGENT.md": "---\nkind: generation\n---\nWork.\n",
        "router/AGENT.md": "---\nkind: decision\nroute:\n  by: pick\n  to: {go: worker}\n---\n",
        "router/questions/pick.md": (
            "---\ntype: choice\noptions: {go: Go on, stay: Stay}\n---\nWhere to?\n"
        ),
        "judge/AGENT.md": "---\nkind: decision\n---\n",
        "judge/questions/ok.md": "---\ntype: check\n---\nIs it fine?\n",
    }
    delegating = calls(
        ("d0", "Delegate", {"agent": "worker", "input": "a"}),
        ("d1", "Delegate", {"agent": "router", "input": "b"}),
        ("d2", "Delegate", {"agent": "judge", "input": "c"}),
    )
    model = Counting([delegating, done("merged")])
    engine = Engine(agents(tmp_path, files), generation=model, decision=model)
    run = await engine.run("lead", "acme", "go")
    block = next(b for b in PYTHON if first_line(b) == "def spent(run):")
    namespace: dict[str, Any] = {name: getattr(rfnry_engine, name) for name in rfnry_engine.__all__}
    exec(block.split("\nteam, alone")[0], namespace)
    usage, made = namespace["spent"](run)
    assert made == 6
    assert usage.input == 4 * 100 + 2 * 10
