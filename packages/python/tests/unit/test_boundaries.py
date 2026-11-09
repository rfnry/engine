import asyncio
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from rfnry_engine.engine import Engine
from rfnry_engine.models import (
    Answer,
    Check,
    Choice,
    Image,
    Message,
    Part,
    Question,
    QuestionLike,
    Reply,
    Score,
    State,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import Decision, Event, Generation, RunStarted
from rfnry_engine.refinery.labels import Review
from rfnry_engine.refinery.postgres import PostgresStorage
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import LabelRow, RunRow, Storage
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tools.tools import Tool, Tools
from rfnry_engine.tools.workspace import Local, Process, Workspace

from .fakes import Answering, Desk, Scripted, answer_to, ask, delegating, lead, sorter, text_reply
from .test_engine_workspace import make_tree, use

GENERATION = {
    "g/AGENT.md": "---\nname: g\nkind: generation\nmodel: strong\ndescription: G\npreload: true\n---\nYou are g.\n",
    "g/kb.md": "the preloaded knowledge\n",
}
SCHEMA: dict[str, Any] = {"type": "object", "properties": {"q": {"type": "object"}}}


def _tree(root: Path, files: Mapping[str, str] = GENERATION) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def _scribble(value: Any) -> None:
    pending = [value]
    while pending:
        held = pending.pop()
        inner = held.values() if isinstance(held, dict) else held
        pending.extend(item for item in inner if isinstance(item, dict | list))
        if isinstance(held, dict):
            held["MEDDLED"] = True
        else:
            held.append("MEDDLED")


def _deface(content: object) -> None:
    for part in content if isinstance(content, list) else []:
        if isinstance(part, Text):
            part.text = "MEDDLED"
        else:
            part.media_type = "meddled/x"


def _containers(model: object) -> list[Any]:
    return [value for value in vars(model).values() if isinstance(value, dict | list)]


def _tool_use(*calls: ToolCall, **more: Any) -> Reply:
    return Reply(tool_calls=list(calls), stop_reason="tool_use", **more)


class Meddler:
    def __init__(self, *replies: Reply, meddle: bool) -> None:
        self.replies = list(replies)
        self.meddle = meddle
        self.seen: list[tuple[list[Message], list[ToolSpec]]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.seen.append(
            ([m.model_copy(deep=True) for m in messages], [t.model_copy(deep=True) for t in tools])
        )
        if self.meddle:
            for message in messages:
                _deface(message.content)
                for call in message.tool_calls:
                    _scribble(call.args)
                message.tool_calls.append(ToolCall(id="meddled", name="find"))
                if message.reasoning is not None:
                    _scribble(message.reasoning)
            messages.append(Message(role="user", content="MEDDLED"))
            for tool in tools:
                _scribble(tool.input_schema)
            tools.append(ToolSpec(name="meddled", description="", input_schema={}))
        return self.replies.pop(0)


async def _converse(root: Path, meddle: bool) -> tuple[Generation, Meddler]:
    llm = Meddler(
        _tool_use(
            ToolCall(id="c1", name="find", args={"q": {"deep": [1]}}),
            reasoning={"blocks": [{"text": "think"}]},
        ),
        _tool_use(ToolCall(id="c2", name="find", args={"q": {"deep": [2]}})),
        text_reply("done"),
        meddle=meddle,
    )

    async def find(args: dict[str, Any]) -> list[Part]:
        return [Text(text="found"), Image(media_type="image/png", url="https://x/f.png")]

    run = await Engine(root, models={"strong": llm}).run(
        "g",
        "acme",
        [Text(text="look it up"), Image(media_type="image/png", url="https://x/a.png")],
        tools=Tools([Tool("find", "Find", body=find, input_schema=copy.deepcopy(SCHEMA))]),
    )
    assert isinstance(run, Generation) and run.reply == "done"
    return run, llm


def _recorded(run: Generation) -> dict[str, Any]:
    return {
        **run.model_dump(
            include={"input", "system", "preload", "tools", "messages", "conversation"}
        ),
        "steps": [(step.tool, step.args, step.result) for step in run.steps],
    }


async def test_what_the_generation_adapter_changes_reaches_neither_the_record_nor_the_next_call(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    plain, kept = await _converse(root, meddle=False)
    meddled, meddler = await _converse(root, meddle=True)
    assert plain.preload and plain.conversation[1].reasoning is not None
    assert len(kept.seen) == 3 and meddler.seen == kept.seen
    assert _recorded(meddled) == _recorded(plain)


async def test_what_consumer_code_handed_a_run_and_changes_later_leaves_its_record_alone(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    given = [Text(text="look it up"), Image(media_type="image/png", url="https://x/a.png")]
    returned: list[Part] = [Text(text="found")]
    schema = copy.deepcopy(SCHEMA)
    replies = [
        _tool_use(
            ToolCall(id="c1", name="find", args={"q": {"deep": [1]}}),
            reasoning={"blocks": [{"text": "think"}]},
            settings={"effort": {"levels": ["high"]}},
        ),
        text_reply("first"),
        _tool_use(
            ToolCall(id="c2", name="find", args={"q": {"deep": [2]}}),
            reasoning={"blocks": [{"text": "again"}]},
        ),
        text_reply("done"),
    ]

    async def find(args: dict[str, Any]) -> list[Part]:
        return returned

    tools = Tools([Tool("find", "Find", body=find, input_schema=schema)])
    taken: list[Part] = [Text(text="also this")]
    inbox: asyncio.Queue[Any] = asyncio.Queue()
    engine = Engine(root, models={"strong": Scripted(*replies)})
    first = await engine.run("g", "acme", [Text(text="earlier")], tools=tools)
    assert isinstance(first, Generation)
    inbox.put_nowait(taken)
    run = await engine.run("g", "acme", given, history=first.conversation, tools=tools, inbox=inbox)
    assert isinstance(run, Generation) and run.reply == "done"
    assert run.conversation[1].reasoning is not None and run.conversation[-2].content == taken
    held = run.model_dump()
    for content in (given, returned, taken):
        _deface(content)
    _scribble(schema)
    for reply in replies:
        for call in reply.tool_calls:
            _scribble(call.args)
        for value in (reply.reasoning, reply.settings):
            if value is not None:
                _scribble(value)
    for message in first.conversation:
        _deface(message.content)
        for call in message.tool_calls:
            _scribble(call.args)
        for value in (message.reasoning, message.settings):
            if value is not None:
                _scribble(value)
    assert run.model_dump() == held


class Judging:
    media_types = {"image/*"}

    def __init__(self) -> None:
        self.meddle = True
        self.seen: list[tuple[State, dict[str, QuestionLike]]] = []

    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> dict[str, Answer]:
        self.seen.append(
            (copy.deepcopy(state), {qid: copy.deepcopy(q) for qid, q in questions.items()})
        )
        answers = {qid: answer_to(question) for qid, question in questions.items()}
        if self.meddle:
            if isinstance(state, list):
                _deface(state)
            if not isinstance(state, str):
                _scribble(state)
            for question in questions.values():
                for value in _containers(question):
                    _scribble(value)
                cast(Any, question).instructions = "MEDDLED"
            cast(dict[str, QuestionLike], questions)["meddled"] = Check(instructions="MEDDLED")
        return answers


@pytest.mark.parametrize(
    "given",
    [
        [Text(text="the state"), Image(media_type="image/png", url="https://x/s.png")],
        {"ticket": {"lines": ["charged twice"]}},
    ],
    ids=["parts", "json"],
)
async def test_what_the_decision_adapter_changes_reaches_neither_the_record_the_input_nor_the_tree(
    tmp_path: Path, given: State
) -> None:
    sorter(tmp_path)
    asked = Engine(tmp_path, models={"judge": Answering()})._agents["sorter"].questions
    judge = Judging()
    engine = Engine(tmp_path, models={"judge": judge})
    held = copy.deepcopy(given)
    first = await engine.run("sorter", "tenant-a", given)
    judge.meddle = False
    second = await engine.run("sorter", "tenant-b", copy.deepcopy(held))
    assert isinstance(first, Decision) and isinstance(second, Decision)
    assert given == held and first.state == second.state == held
    assert first.questions == second.questions == asked
    assert judge.seen[1] == judge.seen[0] == (held, asked)


async def test_what_consumer_code_handed_a_decision_and_changes_later_leaves_its_record_alone(
    tmp_path: Path,
) -> None:
    judge = sorter(tmp_path)
    given = {"ticket": {"lines": ["charged twice"]}}
    run = await Engine(tmp_path, models={"judge": judge}).run("sorter", "acme", given)
    held = run.model_dump()
    _scribble(given)
    assert not callable(judge.answers)
    for answer in judge.answers.values():
        for value in _containers(answer):
            _scribble(value)
    assert run.model_dump() == held


async def test_a_handler_cannot_change_the_call_an_event_names(tmp_path: Path) -> None:
    refused: list[bool] = []

    async def watch(event: Event) -> None:
        if isinstance(event, RunStarted) and event.call is not None:
            try:
                event.call.scope = "other"
            except ValidationError:
                refused.append(True)

    llm = Scripted(delegating("writer"), text_reply("written"), text_reply("ok"))
    run = await Engine(lead(tmp_path), models={"strong": llm, "judge": Answering()}).run(
        "lead", "acme", "x", on_event=watch
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert refused == [True] and child.call is not None and child.call.scope == "acme"


class Scribbling(PostgresStorage):
    async def put_runs(self, scope: str, rows: list[RunRow]) -> None:
        await super().put_runs(scope, rows)
        for row in rows:
            _scribble(row.body)


async def test_a_storage_that_changes_the_rows_it_is_handed_changes_no_record(
    fixtures: Path, storage: Storage, pool: Any
) -> None:
    async def lookup(args: dict[str, Any]) -> list[Part]:
        return [Text(text="found")]

    llm = Scripted(
        _tool_use(ToolCall(id="c1", name="lookup", args={"order": {"lines": [1]}})),
        text_reply("done"),
    )
    refinery = Refinery(Scribbling(pool))
    run = await Engine(fixtures, refinery=refinery, models={"strong": llm}).run(
        "billing",
        "acme",
        {"ask": {"why": ["charged"]}},
        tools=Tools([Tool("lookup", "Look", body=lookup)]),
    )
    assert isinstance(run, Generation) and run.reply == "done"
    assert await refinery.run("acme", run.run_id) == run


async def test_what_the_engine_s_accessors_hand_out_cannot_change_the_engine(
    fixtures: Path,
) -> None:
    llm = Scripted(text_reply("ok"))
    engine = Engine(fixtures, models={"strong": llm, "judge": Answering()})
    before = copy.deepcopy(engine.agents)
    info = engine.agents["support"]
    info.agents.append("billing")
    info.reads.append("billing")
    info.tools.append("Write")
    info.skills["escalate"] = "MEDDLED"
    assert engine.agents == before
    with pytest.raises(TypeError):
        cast(dict[str, Any], engine.models)["strong"] = Scripted()
    assert engine.models["strong"] is llm


async def test_a_backend_that_changes_the_env_it_is_handed_changes_no_later_command(
    tmp_path: Path,
) -> None:
    tree = make_tree(tmp_path)
    room = tmp_path / "work"
    room.mkdir()
    seen: list[dict[str, str]] = []

    class Changing(Local):
        async def spawn(self, command: str, env: Mapping[str, str]) -> Process:
            seen.append(dict(env))
            cast(dict[str, str], env).clear()
            return await super().spawn(command, env)

    llm = Scripted(
        use(ToolCall(id="s1", name="Shell", args={"command": "true"})),
        use(ToolCall(id="s2", name="Shell", args={"command": "true"})),
        text_reply("ok"),
    )
    work = Workspace(Changing(room), scope="acme", tree=str(tree), bridge=False)
    run = await Engine(tree, models={"strong": llm}).run("coder", "acme", "x", workspace=work)
    assert isinstance(run, Generation) and not any(step.is_error for step in run.steps)
    assert len(seen) == 2 and seen[1] == seen[0] and seen[0]["RFNRY_TREE"] == str(tree)


async def test_a_folder_tool_that_changes_its_args_leaves_the_record_as_the_model_sent_it(
    tmp_path: Path,
) -> None:
    root = _tree(
        tmp_path,
        {
            **GENERATION,
            "g/tools/edit.json": json.dumps({"description": "Edit", "input_schema": SCHEMA}),
            "g/tools/edit.py": "def run(args, call):\n    args['q']['deep'].append(2)\n    args['added'] = True\n    return args\n",
        },
    )
    sent = {"q": {"deep": [1]}}
    llm = Scripted(
        _tool_use(ToolCall(id="c1", name="edit", args=copy.deepcopy(sent))), text_reply("done")
    )
    async with Engine(root, models={"strong": llm}) as engine:
        run = await engine.run("g", "acme", "x")
    assert isinstance(run, Generation) and run.reply == "done"
    [step] = run.steps
    assert step.args == sent and json.loads(str(step.result)) == {
        "q": {"deep": [1, 2]},
        "added": True,
    }
    assert llm.calls[1][1][-2].tool_calls[0].args == sent


async def test_what_engine_tools_hand_out_changes_no_other_instance_and_no_run(
    tmp_path: Path,
) -> None:
    root = _tree(
        tmp_path,
        {
            "g/AGENT.md": "---\nname: g\nkind: generation\nmodel: strong\ndescription: G\nagents: [sorter]\n---\nYou are g.\n",
            "g/tools/edit.json": json.dumps({"description": "Edit", "input_schema": SCHEMA}),
            "g/tools/edit.py": "def run(args, call):\n    return 'edited'\n",
        },
    )
    sorter(root)
    llm = Scripted(text_reply("ok"))
    async with Engine(root, models={"strong": llm, "judge": Answering()}) as engine:
        given = EngineTools(engine, agent="g", scope="acme").definitions()
        consult = engine.tool("g", scope="acme").definitions()
        pristine = copy.deepcopy([given, consult])
        assert {"Read", "edit", "Delegate"} <= {definition["name"] for definition in given}
        for definition in [*given, *consult]:
            _scribble(definition["input_schema"])
        again = EngineTools(engine, agent="g", scope="acme").definitions()
        assert [again, engine.tool("g", scope="acme").definitions()] == pristine
        await engine.run("g", "acme", "x")
    [(_, _, offered)] = llm.calls
    assert [tool.model_dump() for tool in offered] == pristine[0]


class Keeping(PostgresStorage):
    def __init__(self, pool: Any) -> None:
        super().__init__(pool)
        self.kept: dict[tuple[str, str], Any] = {}

    def _keep[R: RunRow | LabelRow](self, row: R) -> R:
        named = row.label_id if isinstance(row, LabelRow) else row.run_id
        return cast(R, self.kept.setdefault((type(row).__name__, named), row))

    async def run_tree(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[RunRow]:
        rows = await super().run_tree(scope, run_ids=run_ids, tags=tags)
        return [self._keep(row) for row in rows]

    async def put_label(self, scope: str, row: LabelRow) -> LabelRow:
        return self._keep(await super().put_label(scope, row))


async def test_what_the_refinery_reads_back_shares_nothing_with_the_rows_its_storage_holds(
    fixtures: Path, storage: Storage, pool: Any
) -> None:
    async def lookup(args: dict[str, Any]) -> str:
        return "found"

    keeping = Keeping(pool)
    refinery = Refinery(keeping)
    llm = Scripted(
        _tool_use(ToolCall(id="c1", name="lookup", args={"order": {"lines": [1]}})),
        text_reply("done"),
    )
    engine = Engine(fixtures, refinery=refinery, models={"strong": llm, "judge": Desk()})
    run = await engine.run(
        "billing",
        "acme",
        {"ask": {"why": ["charged"]}},
        tools=Tools([Tool("lookup", "Look", body=lookup)]),
    )
    decided = await engine.run("triage", "acme", "It is down")
    first = await refinery.run("acme", run.run_id)
    label = await refinery.review(
        "acme", decided.run_id, Review(correction={"urgent": {"why": ["it is down"]}})
    )
    held = copy.deepcopy([row.body for row in keeping.kept.values()])
    assert isinstance(first, Generation) and label.review is not None
    assert isinstance(label.review.correction, dict)
    _scribble(first.steps[0].args)
    _scribble(first.input)
    _scribble(label.review.correction)
    assert [row.body for row in keeping.kept.values()] == held
    assert await refinery.run("acme", run.run_id) == run


GRADER = {
    "grader/AGENT.md": "---\nname: grader\nkind: decision\nmodel: judge\ndescription: Grades\n---\n",
    "grader/questions/urgent.json": '{"type": "check", "instructions": "Urgent?"}',
    "lead/AGENT.md": "---\nname: lead\nkind: generation\nmodel: strong\ndescription: L\nagents: [grader]\n---\nLead.\n",
}


async def test_a_question_asked_at_run_time_is_the_record_s_own(tmp_path: Path) -> None:
    class Kept:
        def __init__(self) -> None:
            self.runs: list[Any] = []

        async def record(self, run: Any) -> None:
            self.runs.append(run)

    area = Choice(
        instructions="Which area?", options={"returns": None, "delivery": {"what": ["Parcels"]}}
    )
    kept = Kept()
    engine = Engine(
        _tree(tmp_path, GRADER), refinery=kept, models={"judge": Answering(), "strong": Scripted()}
    )
    await EngineTools(engine, agent="lead", scope="acme").dispatch(
        "Delegate", {"agent": "grader", "input": "x", "questions": {"area": area}}
    )
    await ask(engine, "grader", "acme", "x", questions={"area": area})
    assert len(kept.runs) == 2
    held = [run.model_dump() for run in kept.runs]
    assert all(isinstance(run, Decision) and run.runtime == ["area"] for run in kept.runs)
    _scribble(area.options)
    area.instructions = "MEDDLED"
    assert [run.model_dump() for run in kept.runs] == held


@pytest.mark.parametrize(
    "asked",
    [
        Choice(instructions="Which area?", options={"returns": None, "delivery": ["Parcels"]}),
        Score(instructions="How bad?", levels=["mild", {"severe": ["down"]}]),
        Check(instructions="Is it down?", yes="It is"),
    ],
    ids=["choice", "score", "check"],
)
async def test_an_agent_used_as_a_tool_takes_a_question_object_as_it_takes_its_dict(
    tmp_path: Path, asked: Question
) -> None:
    root = _tree(tmp_path, GRADER)
    judge = Answering()
    consult = Engine(root, models={"judge": judge, "strong": Scripted()}).tool(
        "grader", scope="acme"
    )
    by_object = await consult.dispatch("grader", {"input": "x", "questions": {"asked": asked}})
    by_dict = await consult.dispatch(
        "grader", {"input": "x", "questions": {"asked": asked.model_dump()}}
    )
    assert by_object == by_dict and isinstance(by_object, dict)
    assert set(by_object["answers"]) == {"asked", "urgent"}
    assert [questions["asked"] for _, questions in judge.calls] == [asked, asked]


class Seeing(Answering):
    media_types = {"image/*"}


@pytest.mark.parametrize(
    "given",
    [
        {"ticket": {"lines": ["charged twice"]}},
        [Text(text="charged twice"), Image(media_type="image/png", url="https://x/t.png")],
    ],
    ids=["json", "parts"],
)
async def test_a_routed_run_s_input_is_its_own_and_not_its_router_s_state(
    tmp_path: Path, given: State
) -> None:
    root = _tree(
        tmp_path,
        {
            "desk/AGENT.md": "---\nname: desk\nkind: decision\nmodel: judge\nroute: {by: team, to: {b: writer}}\n---\n",
            "desk/questions/team.json": '{"type": "choice", "instructions": "Which?", "options": {"a": null, "b": null}}',
            "writer/AGENT.md": "---\nname: writer\nkind: generation\nmodel: strong\ndescription: W\n---\nWrite.\n",
        },
    )
    engine = Engine(root, models={"judge": Seeing(), "strong": Scripted(text_reply("w"))})
    run = await engine.run("desk", "acme", copy.deepcopy(given))
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    child = run.routed_to
    assert run.state == given and child.input == given
    for held in (child.input, run.state):
        _deface(held)
        _scribble(held)
        assert (run.state == given) is (held is child.input)
