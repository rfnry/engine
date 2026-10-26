import base64
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError, ModelError, WorkerError
from rfnry_engine.models import Check, Choice, Image, Question, QuestionLike, Score, Text
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import load_tree

from .fakes import Answering, Scripted, delegating, text_reply

ORDER = (
    "def run(input, call):\n"
    "    return {'options': {call['scope'] + '-1': 'First', call['scope'] + '-2': 'Second'}}\n"
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, kind: str = "generation", model: str = "strong", extra: str = "") -> str:
    return f"---\nname: {name}\nkind: {kind}\nmodel: {model}\n{extra}---\nWork.\n"


def matcher(root: Path, script: str = ORDER, extra: str = "") -> Path:
    write(
        root / "matcher/AGENT.md", agent_md("matcher", kind="decision", model="judge", extra=extra)
    )
    write(
        root / "matcher/questions/order.json", '{"type": "choice", "instructions": "Which order?"}'
    )
    write(root / "matcher/questions/order.py", script)
    return root


class Recorded:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


class Seeing(Answering):
    media_types = {"image/*"}


def options(questions: Mapping[str, QuestionLike], qid: str = "order") -> list[str]:
    question = questions[qid]
    assert isinstance(question, Choice)
    return list(question.options)


def sent(run: Run, key: str) -> object:
    assert isinstance(run, Decision)
    question = run.questions["order"]
    assert isinstance(question, Choice)
    return json.loads(str(question.options[key]))


async def test_a_question_script_fills_options_per_tenant(tmp_path: Path) -> None:
    write(tmp_path / "matcher/AGENT.md", agent_md("matcher", kind="decision", model="judge"))
    write(
        tmp_path / "matcher/questions/order.json",
        '{"type": "choice", "instructions": "Which order?"}',
    )
    write(
        tmp_path / "matcher/questions/order.py",
        "def run(input, call):\n    return {'options': {call['scope'] + '-1': 'First', call['scope'] + '-2': 'Second'}}\n",
    )
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge}) as engine:
        run = await engine.run("matcher", "acme", "my desk")
        other = await engine.run("matcher", "globex", "my desk")
    assert isinstance(run, Decision) and isinstance(other, Decision)
    assert options(run.questions) == ["acme-1", "acme-2"]
    assert options(judge.calls[0][1]) == ["acme-1", "acme-2"]
    assert options(other.questions) == ["globex-1", "globex-2"]


async def test_a_fixed_json_question_is_asked_on_every_run(tmp_path: Path) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    write(
        tmp_path / "triage/questions/urgent.json",
        '{"type": "check", "instructions": "Urgent?", "yes": {"what": "Down"}}',
    )
    write(
        tmp_path / "triage/questions/severity.json",
        '{"type": "score", "instructions": {"question": "How bad?"}, "levels": ["Mild", ["Severe"]]}',
    )
    write(
        tmp_path / "triage/questions/team.json",
        '{"type": "choice", "instructions": "Who?", "options": {"billing": "Charges", "other": null}}',
    )
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge}) as engine:
        runs = [await engine.run("triage", "acme", "x") for _ in range(2)]
    expected: dict[str, Question] = {
        "severity": Score(instructions={"question": "How bad?"}, levels=["Mild", ["Severe"]]),
        "team": Choice(instructions="Who?", options={"billing": "Charges", "other": None}),
        "urgent": Check(instructions="Urgent?", yes={"what": "Down"}),
    }
    for run in runs:
        assert isinstance(run, Decision) and run.questions == expected
        assert list(run.questions) == ["severity", "team", "urgent"] and run.runtime == []
    assert [dict(asked) for _, asked in judge.calls] == [expected, expected]


def test_a_markdown_question_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    path = tmp_path / "triage/questions/urgent.md"
    write(path, "---\ntype: check\n---\nUrgent?\n")
    message = f"tree: question {path} is markdown; write it as urgent.json with type, instructions and its options, levels or yes and no"
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    assert str(raised.value) == message


@pytest.mark.parametrize("held", ["tasks/tone/polite.json", "tasks/tone.md"])
def test_a_decision_agent_with_tasks_is_refused(tmp_path: Path, held: str) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    write(tmp_path / "triage/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}')
    write(tmp_path / "triage" / held, '{"type": "check", "instructions": "Polite?"}')
    message = "tree: decision agent 'triage' has tasks/; a decision agent asks only questions/, so make another decision agent for another set of questions"
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    assert str(raised.value) == message


def test_a_question_script_without_its_json_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "matcher/AGENT.md", agent_md("matcher", kind="decision", model="judge"))
    path = tmp_path / "matcher/questions/order.py"
    write(path, ORDER)
    message = f"tree: {path} has no order.json beside it; a question is order.json, and order.py fills it from data"
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    assert str(raised.value) == message


def test_a_route_by_a_scripted_question_is_refused(tmp_path: Path) -> None:
    matcher(tmp_path, extra="route: {by: order, to: {acme-1: writer}}\n")
    write(tmp_path / "writer/AGENT.md", agent_md("writer"))
    message = "tree: route of 'matcher' is by 'order', whose options come from order.py; route by a question whose options are fixed"
    with pytest.raises(ConfigurationError) as raised:
        Engine(tmp_path, models={"judge": Answering(), "strong": Scripted()})
    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '{"type": "check", "instructions": "Urgent?", "option": "x"}',
            "has the unknown key 'option'; a question takes type, instructions, options, levels, yes and no",
        ),
        ('{"type": "check", "instructions": "Urgent?",}', "is not valid JSON: "),
        ('["check"]', "is not a JSON object; "),
        ('{"type": "choice", "instructions": "Who?"}', "options: field required"),
        (
            '{"type": "choice", "instructions": "Who?", "options": {"a": "A"}}',
            "a choice needs at least two options",
        ),
        ('{"type": "check"}', "instructions: field required"),
        (
            '{"type": "vote", "instructions": "Who?"}',
            "type: input should be 'choice', 'score' or 'check'",
        ),
    ],
)
def test_a_fixed_question_that_does_not_fit_is_refused_when_the_tree_loads(
    tmp_path: Path, text: str, message: str
) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    path = tmp_path / "triage/questions/urgent.json"
    write(path, text)
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    said = str(raised.value)
    assert said.startswith(f"tree: {path} ") or said.startswith(f"tree: invalid question {path}; ")
    assert message in said


async def test_a_script_that_raises_fails_the_run_before_the_decision_model_is_called(
    tmp_path: Path,
) -> None:
    matcher(
        tmp_path, "def run(input, call):\n    raise ValueError('no orders for ' + call['scope'])\n"
    )
    judge = Answering()
    recorded = Recorded()
    said = "engine: order.py of decision agent 'matcher' could not fill question 'order': ValueError: no orders for acme"
    async with Engine(tmp_path, models={"judge": judge}, refinery=recorded) as engine:
        with pytest.raises(WorkerError) as raised:
            await engine.run("matcher", "acme", "my desk")
    assert str(raised.value) == said
    assert judge.calls == []
    [failed] = recorded.runs
    assert isinstance(failed, Decision) and failed.answers == {}
    assert failed.error == f"WorkerError: {said}"


async def test_a_script_whose_worker_stops_names_the_question_it_was_filling(
    tmp_path: Path,
) -> None:
    matcher(tmp_path, "import os\ndef run(input, call):\n    os._exit(3)\n")
    async with Engine(tmp_path, models={"judge": Answering()}) as engine:
        with pytest.raises(WorkerError) as raised:
            await engine.run("matcher", "acme", "my desk")
    assert str(raised.value) == (
        "engine: order.py of decision agent 'matcher' could not fill question 'order': workers/pool: the worker for order stopped: exited with code 3"
    )


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            "def run(input, call):\n    return {'options': {'only': 'One'}}\n",
            "^engine: question 'order' of decision agent 'matcher' does not fit once order.py fills it: the question: models: a choice needs at least two options$",
        ),
        (
            "def run(input, call):\n    return {'options': {'a': 'A', 'b': 'B'}, 'hint': 'x'}\n",
            "^engine: question 'order' of decision agent 'matcher' does not fit once order.py fills it: hint: extra inputs are not permitted$",
        ),
        (
            "def run(input, call):\n    return {'options': {'a': 3, 'b': 'B'}}\n",
            "^engine: question 'order' of decision agent 'matcher' does not fit once order.py fills it: options.a.str: input should be a valid string; ",
        ),
        (
            "def run(input, call):\n    return [1, 2]\n",
            "^engine: order.py of decision agent 'matcher' returned a list, not a dict; it returns the fields it fills in order.json, such as",
        ),
        (
            "def run(input, call):\n    return 3\n",
            "^engine: order.py of decision agent 'matcher' returned an int, not a dict; ",
        ),
        (
            "def run(input, call):\n    return 'A-1'\n",
            "^engine: order.py of decision agent 'matcher' returned a str, not a dict; ",
        ),
        (
            "def run(input, call):\n    pass\n",
            "^engine: order.py of decision agent 'matcher' returned None, not a dict; ",
        ),
        (
            "def run(input, call):\n    return {'type': 'check'}\n",
            "^engine: order.py of decision agent 'matcher' returned 'type', which question 'order' sets in order.json; a script returns only what its question's JSON leaves out, such as",
        ),
        (
            "def run(input, call):\n    return {'instructions': 'Which one?', 'options': {'a': 'A', 'b': 'B'}}\n",
            "^engine: order.py of decision agent 'matcher' returned 'instructions', which question 'order' sets in order.json; ",
        ),
        (
            "def run(input, call):\n    return {'options': {'a': 'A', 'b': 'B'}, 'levels': ['low', 'high']}\n",
            "^engine: order.py of decision agent 'matcher' returned 'levels', which question 'order', a choice, does not take; a choice takes options, a score levels and a check yes and no$",
        ),
        (
            "def run(input, call):\n    return {'yes': 'Yes'}\n",
            "^engine: order.py of decision agent 'matcher' returned 'yes', which question 'order', a choice, does not take; ",
        ),
    ],
)
async def test_a_filled_question_that_does_not_fit_fails_the_run_before_the_model_is_called(
    tmp_path: Path, script: str, message: str
) -> None:
    matcher(tmp_path, script)
    judge = Answering()
    recorded = Recorded()
    async with Engine(tmp_path, models={"judge": judge}, refinery=recorded) as engine:
        with pytest.raises(ModelError, match=message):
            await engine.run("matcher", "acme", "my desk")
    assert judge.calls == []
    [failed] = recorded.runs
    assert isinstance(failed, Decision) and failed.error is not None
    assert failed.error.startswith("ModelError: engine: ")


async def test_a_script_past_the_tool_timeout_fails_the_run(tmp_path: Path) -> None:
    matcher(tmp_path, "import time\ndef run(input, call):\n    time.sleep(5)\n")
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge}, tool_timeout=0.3) as engine:
        with pytest.raises(TimeoutError, match="^workers/pool: order timed out after 0.3s$"):
            await engine.run("matcher", "acme", "my desk")
    assert judge.calls == []


async def test_a_script_reads_the_state_and_the_call(tmp_path: Path) -> None:
    matcher(
        tmp_path,
        "import json\n"
        "def run(input, call):\n"
        "    return {'options': {'input': json.dumps(input), 'call': json.dumps(call)}}\n",
    )
    judge = Seeing()
    async with Engine(tmp_path, models={"judge": judge}) as engine:
        given = await engine.run("matcher", "acme", {"message": "my desk", "plan": "team"})
        parts = await engine.run(
            "matcher",
            "acme",
            [Text(text="This one"), Image(media_type="image/png", data=b"\x89PNG")],
        )
        joined = await engine.run("matcher", "acme", [Text(text="a"), Text(text="b")])
    assert sent(given, "input") == {"message": "my desk", "plan": "team"}
    assert sent(given, "call") == {
        "scope": "acme",
        "run_id": given.run_id,
        "agent": "matcher",
        "question": "order",
    }
    assert sent(parts, "input") == [
        {"type": "text", "text": "This one"},
        {
            "type": "image",
            "media_type": "image/png",
            "data": base64.b64encode(b"\x89PNG").decode(),
            "url": None,
        },
    ]
    assert sent(joined, "input") == "a\nb"


async def test_several_scripted_questions_run_at_once(tmp_path: Path) -> None:
    write(tmp_path / "matcher/AGENT.md", agent_md("matcher", kind="decision", model="judge"))
    write(
        tmp_path / "matcher/questions/_meet.py", "import threading\nBOTH = threading.Barrier(2)\n"
    )
    for qid in ("order", "item"):
        write(
            tmp_path / f"matcher/questions/{qid}.json",
            '{"type": "choice", "instructions": "Which?"}',
        )
        write(
            tmp_path / f"matcher/questions/{qid}.py",
            "import _meet\n"
            "def run(input, call):\n"
            "    _meet.BOTH.wait(timeout=5)\n"
            "    return {'options': {call['question'] + '-a': 'A', call['question'] + '-b': 'B'}}\n",
        )
    async with Engine(tmp_path, models={"judge": Answering()}) as engine:
        run = await engine.run("matcher", "acme", "x")
    assert isinstance(run, Decision) and list(run.questions) == ["item", "order"]
    assert options(run.questions, "item") == ["item-a", "item-b"]


def test_a_question_script_without_a_run_at_its_top_level_is_refused(tmp_path: Path) -> None:
    matcher(tmp_path, "def fill(input, call):\n    return {}\n")
    write(tmp_path / "matcher/questions/_orders.py", "ORDERS = {}\n")
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    assert str(raised.value) == (
        f"tools/folder: {tmp_path / 'matcher/questions/order.py'} has no run at its top level; a question's script defines def run(input, call) or async def run(input, call)"
    )


@pytest.mark.parametrize(
    ("script", "awaits"), [(ORDER, False), (ORDER.replace("def run", "async def run"), True)]
)
def test_a_question_script_awaits_when_its_last_run_at_the_top_level_is_async(
    tmp_path: Path, script: str, awaits: bool
) -> None:
    matcher(tmp_path, script)
    _, held = load_tree(tmp_path)["matcher"].scripted["order"]
    assert held.awaits is awaits


def test_a_server_package_named_questions_is_the_server_s_code_to_a_question_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(tmp_path / "src")
    write(tmp_path / "pyproject.toml", "")
    write(tmp_path / "src/questions/__init__.py", "")
    matcher(tmp_path / "agents", "import questions\n" + ORDER)
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path / "agents")
    assert str(raised.value) == (
        f"tools/folder: {tmp_path / 'agents/matcher/questions/order.py'} imports 'questions' from {tmp_path / 'src/questions/__init__.py'}, the server's own code; a question's script imports the standard library, installed packages and helpers of its questions/ folder"
    )


def test_a_question_script_that_imports_the_engine_is_refused(tmp_path: Path) -> None:
    matcher(tmp_path, "import rfnry_engine\ndef run(input, call):\n    return {}\n")
    path = tmp_path / "matcher/questions/order.py"
    with pytest.raises(ConfigurationError, match="^tools/folder: ") as raised:
        load_tree(tmp_path)
    assert str(raised.value).startswith(f"tools/folder: {path} imports 'rfnry_engine'; ")


async def test_a_question_script_runs_in_its_agent_s_worker_group(tmp_path: Path) -> None:
    root = tmp_path / "agents"
    write(root / "desk/matcher/AGENT.md", agent_md("matcher", kind="decision", model="judge"))
    write(
        root / "desk/matcher/questions/order.json", '{"type": "choice", "instructions": "Which?"}'
    )
    pid = "import os\ndef run(input, call):\n    return {'options': {str(os.getpid()): 'Me', 'x': 'X'}}\n"
    write(root / "desk/matcher/questions/order.py", pid)
    write(root / "desk/support/AGENT.md", agent_md("support"))
    write(
        root / "desk/support/tools/pid.json",
        '{"description": "Pid", "input_schema": {"type": "object"}}',
    )
    write(
        root / "desk/support/tools/pid.py",
        "import os\ndef run(args, call):\n    return os.getpid()\n",
    )
    write(root / "solo/AGENT.md", agent_md("solo", kind="decision", model="judge"))
    write(root / "solo/questions/order.json", '{"type": "choice", "instructions": "Which?"}')
    write(root / "solo/questions/order.py", pid)
    async with Engine(root, models={"judge": Answering()}, workers={"desk": 1}) as engine:
        desk = await engine.run("matcher", "acme", "x")
        solo = await engine.run("solo", "acme", "x")
        tool = await EngineTools(engine, agent="support", scope="acme").dispatch("pid", {})
    assert isinstance(desk, Decision) and isinstance(solo, Decision)
    mine, theirs = (options(run.questions)[0] for run in (desk, solo))
    assert mine == str(tool) and theirs != mine


def test_a_question_script_is_part_of_the_agent_fingerprint(tmp_path: Path) -> None:
    matcher(tmp_path)
    before = load_tree(tmp_path)["matcher"].fingerprint
    write(tmp_path / "matcher/questions/order.py", ORDER.replace("First", "Earliest"))
    assert load_tree(tmp_path)["matcher"].fingerprint != before


async def test_a_decision_agent_takes_no_task(tmp_path: Path) -> None:
    matcher(tmp_path)
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge}) as engine:
        with pytest.raises(
            InputError,
            match="^engine: decision agent 'matcher' takes no task; it asks its questions/, so run another decision agent for another set of questions$",
        ):
            await engine.run("matcher", "acme", "x", task="tone")
    assert judge.calls == []


def test_a_route_entry_may_not_name_a_task_for_a_decision_agent(tmp_path: Path) -> None:
    write(
        tmp_path / "desk/AGENT.md",
        agent_md(
            "desk",
            kind="decision",
            model="judge",
            extra="route: {by: team, to: {a: {agent: triage, task: tone}}}\n",
        ),
    )
    write(
        tmp_path / "desk/questions/team.json",
        '{"type": "choice", "instructions": "?", "options": {"a": "A", "b": "B"}}',
    )
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    write(tmp_path / "triage/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}')
    message = "tree: route of 'desk' names task 'tone' for decision agent 'triage', which takes no task; drop task: from that entry"
    with pytest.raises(ConfigurationError) as raised:
        Engine(tmp_path, models={"judge": Answering()})
    assert str(raised.value) == message


async def test_a_route_entry_with_a_task_to_a_generation_agent_still_works(tmp_path: Path) -> None:
    write(
        tmp_path / "desk/AGENT.md",
        agent_md(
            "desk",
            kind="decision",
            model="judge",
            extra="route: {by: team, to: {b: {agent: writer, task: apology}}}\n",
        ),
    )
    write(
        tmp_path / "desk/questions/team.json",
        '{"type": "choice", "instructions": "?", "options": {"a": "A", "b": "B"}}',
    )
    write(tmp_path / "writer/AGENT.md", agent_md("writer"))
    write(tmp_path / "writer/tasks/apology.md", "Apologise first.\n")
    llm = Scripted(text_reply("Sorry."))
    run = await Engine(tmp_path, models={"judge": Answering(), "strong": llm}).run(
        "desk", "acme", "x"
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Generation)
    assert run.routed_to.task == "apology" and run.routed_to.reply == "Sorry."
    assert "<<<TASK>>>\nApologise first." in llm.calls[0][0]


async def test_questions_a_model_composes_reach_a_scripted_decision_agent(tmp_path: Path) -> None:
    matcher(tmp_path)
    write(tmp_path / "lead/AGENT.md", agent_md("lead", extra="agents: [matcher]\n"))
    angry = {"type": "check", "instructions": "Is the customer angry?"}
    llm = Scripted(
        delegating("matcher", "my desk", questions={"angry": angry}), text_reply("Done.")
    )
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge, "strong": llm}) as engine:
        run = await engine.run("lead", "acme", "Which order?")
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Decision) and child.runtime == ["angry"]
    assert list(child.questions) == ["order", "angry"]
    assert child.questions["angry"] == Check(instructions="Is the customer angry?")
    assert list(judge.calls[0][1]) == ["order", "angry"]


async def test_a_runtime_id_may_not_repeat_a_scripted_one(tmp_path: Path) -> None:
    matcher(tmp_path)
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge}) as engine:
        with pytest.raises(
            InputError,
            match="^engine: decision agent 'matcher' already asks 'order' from its folder$",
        ):
            await engine._start(
                "matcher", "acme", "x", questions={"order": Check(instructions="Which?")}
            )
    assert judge.calls == []


async def test_the_recorded_questions_hold_the_script_filled_options(
    tmp_path: Path, storage: Storage
) -> None:
    matcher(tmp_path)
    refinery = Refinery(storage)
    async with Engine(tmp_path, models={"judge": Answering()}, refinery=refinery) as engine:
        run = await engine.run("matcher", "acme", "my desk")
    read = await refinery.run("acme", run.run_id)
    assert isinstance(read, Decision) and read == run
    assert read.questions == {
        "order": Choice(
            instructions="Which order?", options={"acme-1": "First", "acme-2": "Second"}
        )
    }
    assert read.runtime == []


async def test_a_script_fills_its_question_per_tenant_and_never_changes_its_shape(
    tmp_path: Path,
) -> None:
    matcher(
        tmp_path,
        "def run(input, call):\n"
        "    if call['scope'] == 'other':\n"
        "        return {'type': 'check', 'yes': 'Something else entirely'}\n"
        "    return {'options': {call['scope'] + '-1': 'First', call['scope'] + '-2': 'Second'}}\n",
    )
    judge = Answering()
    async with Engine(tmp_path, models={"judge": judge}) as engine:
        acme = await engine.run("matcher", "acme", "my desk")
        again = await engine.run("matcher", "acme", "my desk")
        globex = await engine.run("matcher", "globex", "my desk")
        with pytest.raises(ModelError, match="returned 'type', which question 'order' sets"):
            await engine.run("matcher", "other", "my desk")
    for run, scope in ((acme, "acme"), (again, "acme"), (globex, "globex")):
        assert isinstance(run, Decision)
        assert run.questions["order"] == Choice(
            instructions="Which order?", options={f"{scope}-1": "First", f"{scope}-2": "Second"}
        )
    assert acme.fingerprint.questions == again.fingerprint.questions
    assert acme.fingerprint.questions != globex.fingerprint.questions
    assert acme.fingerprint.agent == globex.fingerprint.agent
    assert len(judge.calls) == 3


@pytest.mark.parametrize(
    ("text", "kind", "key", "script"),
    [
        (
            '{"type": "choice", "instructions": "Pick", "options": {"a": "A", "b": "B"}, "levels": ["low", "high"]}',
            "choice",
            "levels",
            False,
        ),
        (
            '{"type": "check", "instructions": "Is it?", "options": {"a": "A", "b": "B"}}',
            "check",
            "options",
            False,
        ),
        (
            '{"type": "score", "instructions": "How bad?", "levels": ["a", "b"], "no": "Fine"}',
            "score",
            "no",
            False,
        ),
        ('{"type": "choice", "instructions": "Which?", "yes": "Y"}', "choice", "yes", True),
    ],
)
def test_a_question_holding_a_field_of_another_type_is_refused_naming_its_file(
    tmp_path: Path, text: str, kind: str, key: str, script: bool
) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    path = tmp_path / "triage/questions/q.json"
    write(path, text)
    if script:
        write(tmp_path / "triage/questions/q.py", ORDER)
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    assert str(raised.value) == (
        f"tree: {path} is a {kind} question holding {key!r}; a choice takes options, a score levels and a check yes and no, so drop {key!r} or change its type"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '{"type": "choice", "instructions": "Which?", "option": "x"}',
            "has the unknown key 'option'; a question takes type, instructions, options, levels, yes and no",
        ),
        ('{"type": "choice", "instructions": "Which?",}', "is not valid JSON: "),
        ('{"type": "choice", "instructions": " "}', "instructions must say something"),
    ],
)
def test_a_scripted_question_s_json_that_does_not_fit_is_refused_beside_its_script(
    tmp_path: Path, text: str, message: str
) -> None:
    write(tmp_path / "triage/AGENT.md", agent_md("triage", kind="decision", model="judge"))
    path = tmp_path / "triage/questions/q.json"
    write(path, text)
    write(tmp_path / "triage/questions/q.py", ORDER)
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    said = str(raised.value)
    assert said.startswith(f"tree: {path} ") or said.startswith(f"tree: invalid question {path}; ")
    assert message in said


@pytest.mark.parametrize(
    "files",
    [{"x.md": "---\ntype: check\n---\nUrgent?\n"}, {"x.py": ORDER}, {"x.json": "{"}],
)
def test_a_generation_agent_with_questions_is_told_its_kind_before_anything_in_them(
    tmp_path: Path, files: dict[str, str]
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    for name, text in files.items():
        write(tmp_path / "a/questions" / name, text)
    with pytest.raises(ConfigurationError) as raised:
        load_tree(tmp_path)
    assert str(raised.value) == (
        "tree: generation agent 'a' cannot declare questions/; questions are for decision agents, so make it kind: decision or move them to one"
    )


def test_a_generation_agent_s_questions_folder_without_questions_loads(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a"))
    write(tmp_path / "a/questions/notes.txt", "Ideas for later.\n")
    write(tmp_path / "a/questions/_draft.json", "{")
    assert load_tree(tmp_path)["a"].questions == {}


def test_a_question_script_that_links_out_of_its_folder_is_refused(tmp_path: Path) -> None:
    matcher(tmp_path)
    (tmp_path / "matcher/questions/order.py").unlink()
    write(tmp_path / "elsewhere.py", ORDER)
    (tmp_path / "matcher/questions/order.py").symlink_to(tmp_path / "elsewhere.py")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tools/folder: {tmp_path / 'matcher/questions/order.py'} links to {tmp_path / 'elsewhere.py'}, outside the agent's folder; move the file into the agent's folder, or link the agent's whole folder"
    )


def test_a_question_json_that_links_out_of_its_folder_is_refused(tmp_path: Path) -> None:
    matcher(tmp_path)
    (tmp_path / "matcher/questions/order.json").replace(tmp_path / "elsewhere.json")
    (tmp_path / "matcher/questions/order.json").symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(
        ConfigurationError, match="^tree: .*order.json links to .*, outside the agent's folder;"
    ):
        load_tree(tmp_path)


@pytest.mark.parametrize("name", ["time", "math"])
def test_a_question_script_named_like_a_built_in_standard_module_beside_one_that_imports_it_loads(
    tmp_path: Path, name: str
) -> None:
    matcher(tmp_path, f"import {name}\n" + ORDER)
    write(
        tmp_path / f"matcher/questions/{name}.json", '{"type": "choice", "instructions": "Which?"}'
    )
    write(tmp_path / f"matcher/questions/{name}.py", f"import {name}\n" + ORDER)
    assert sorted(load_tree(tmp_path)["matcher"].scripted) == sorted(["order", name])


@pytest.mark.parametrize("suffix", [".json", ".py"])
def test_a_question_file_that_links_elsewhere_in_the_agent_s_folder_loads(
    tmp_path: Path, suffix: str
) -> None:
    matcher(tmp_path)
    (tmp_path / f"matcher/questions/order{suffix}").replace(tmp_path / f"matcher/kept{suffix}")
    (tmp_path / f"matcher/questions/order{suffix}").symlink_to(f"../kept{suffix}")
    assert list(load_tree(tmp_path)["matcher"].scripted) == ["order"]
