from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, JsonValue

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError, NotFoundError
from rfnry_engine.models import Check, Choice, ChoiceAnswer, Question, Score
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.refinery import Refinery

from .fakes import TOOLS, Answering, Scripted, delegate_triage, text_reply

AREA = Choice(
    instructions={"question": "Which area does `message` belong to?", "within": "orders"},
    options={
        "delivery": {"what": "Parcels in transit", "not_for": "Returns", "examples": ["Late"]},
        "returns": ["Sending it back", "A return label"],
        "other": None,
    },
)
HOW_BAD = Score(
    instructions={"question": "How bad is it?", "focus": "The effect, not the tone"},
    levels=[{"summary": "Mild", "signals": ["a question"]}, ["Severe"]],
)
ANGRY = Check(
    instructions="Is the customer angry?",
    yes={"what": "Insults, threats, all caps"},
    no=None,
)
RUNTIME: dict[str, Question] = {"area": AREA, "how_bad": HOW_BAD, "angry": ANGRY}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def grader(root: Path) -> Path:
    write(root / "grader/AGENT.md", "---\nname: grader\nkind: decision\n---\n")
    write(root / "grader/questions/urgent.md", "---\ntype: check\n---\nIs it urgent?\n")
    write(
        root / "grader/tasks/tone/polite.md",
        "---\ntype: check\nyes: Thanks, please\n---\nIs `reply` polite?\n",
    )
    write(
        root / "grader/tasks/tone/register.md",
        '---\ntype: choice\noptions:\n  formal:\n  casual: {what: "Contractions, slang"}\n---\n'
        "Which register?\n",
    )
    write(root / "grader/tasks/facts/grounded.md", "---\ntype: check\n---\nIs it grounded?\n")
    write(root / "blank/AGENT.md", "---\nname: blank\nkind: decision\n---\n")
    return root


async def test_runtime_questions_are_asked_with_the_folder_s_in_one_call(fixtures: Path) -> None:
    dm = Answering()
    run = await Engine(fixtures, decision=dm).run("triage", "acme", "It is down", questions=RUNTIME)
    [(state, asked)] = dm.calls
    assert state == "It is down" and isinstance(run, Decision)
    assert asked == {"urgent": run.questions["urgent"], "severity": run.questions["severity"]} | {
        qid: RUNTIME[qid] for qid in RUNTIME
    }
    assert run.questions == asked
    assert set(run.answers) == {"urgent", "severity", "area", "how_bad", "angry"}
    assert run.runtime == ["area", "how_bad", "angry"] and run.task is None
    assert run.answers["area"].confidence == pytest.approx(0.7)


async def test_a_folder_only_run_marks_nothing_as_runtime(fixtures: Path) -> None:
    run = await Engine(fixtures, decision=Answering()).run("triage", "acme", "x")
    assert (
        isinstance(run, Decision)
        and run.runtime == []
        and set(run.questions)
        == {
            "urgent",
            "severity",
        }
    )


async def test_a_runtime_id_that_is_already_in_the_folder_is_refused(fixtures: Path) -> None:
    dm = Answering()
    message = "^engine: decision agent 'triage' already asks 'urgent', 'severity' from its folder$"
    with pytest.raises(InputError, match=message):
        await Engine(fixtures, decision=dm).run(
            "triage", "acme", "x", questions={"urgent": ANGRY, "area": AREA, "severity": HOW_BAD}
        )
    assert dm.calls == []


async def test_a_decision_agent_with_no_questions_of_its_own_runs_on_runtime_ones(
    tmp_path: Path,
) -> None:
    root = grader(tmp_path)
    dm = Answering()
    engine = Engine(root, decision=dm)
    run = await engine.run("blank", "acme", {"message": "Where is it?"}, questions={"area": AREA})
    assert isinstance(run, Decision) and run.questions == {"area": AREA}
    assert run.runtime == ["area"] and run.answers["area"].type == "choice"
    nothing: list[dict[str, Question] | None] = [None, {}]
    for given in nothing:
        with pytest.raises(
            InputError,
            match="^engine: decision agent 'blank' has no questions to ask; pass questions=$",
        ):
            await engine.run("blank", "acme", "x", questions=given)
    assert len(dm.calls) == 1


async def test_a_generation_agent_takes_no_questions(fixtures: Path) -> None:
    llm = Scripted()
    engine = Engine(fixtures, generation=llm, decision=Answering(), tools=TOOLS)
    for given in (RUNTIME, {}):
        with pytest.raises(
            InputError, match="^engine: generation agent 'support' takes no questions$"
        ):
            await engine.run("support", "acme", "x", questions=given)
    assert llm.calls == []


@pytest.mark.parametrize(
    ("given", "message"),
    [
        (
            {"area": {"type": "choice", "instructions": "Which?", "options": {"a": None}}},
            "^engine: questions do not fit the contract: .*a choice needs at least two options",
        ),
        (
            {"area": {"type": "score", "instructions": "How?", "levels": ["one"]}},
            "^engine: questions do not fit the contract: .*a score needs at least two levels",
        ),
        (
            {"area": {"type": "check", "instructions": {"when": object()}}},
            "^engine: questions do not fit the contract",
        ),
        ({"area": "Which area?"}, "^engine: questions do not fit the contract"),
        ({"": ANGRY}, "^engine: a question id must be a non-empty string$"),
        (["area"], "^engine: questions must map question ids to questions$"),
    ],
)
async def test_a_runtime_question_that_does_not_fit_is_refused_before_the_model_is_called(
    fixtures: Path, given: Any, message: str
) -> None:
    dm = Answering()
    with pytest.raises(InputError, match=message):
        await Engine(fixtures, decision=dm).run("triage", "acme", "x", questions=given)
    assert dm.calls == []


class ForeignChoice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]


async def test_a_runtime_question_is_read_by_its_shape(fixtures: Path) -> None:
    foreign = ForeignChoice(instructions={"question": "Which?"}, options={"a": None, "b": ["B"]})
    plain: Any = {"type": "check", "instructions": "Is it?"}
    run = await Engine(fixtures, decision=Answering()).run(
        "triage", "acme", "x", questions={"pick": foreign, "sure": plain}
    )
    assert isinstance(run, Decision)
    assert run.questions["pick"] == Choice(
        instructions={"question": "Which?"}, options=foreign.options
    )
    assert run.questions["sure"] == Check(instructions="Is it?")


async def test_a_route_is_by_a_folder_question_and_runtime_ones_stay_with_the_router(
    fixtures: Path,
) -> None:
    llm = Scripted(text_reply("Support here."))
    dm = Answering()
    engine = Engine(fixtures, generation=llm, decision=dm, tools=TOOLS)
    with pytest.raises(
        InputError,
        match="^engine: decision agent 'frontdesk' already asks 'intent' from its folder$",
    ):
        await engine.run("frontdesk", "acme", "x", questions={"intent": AREA})
    run = await engine.run("frontdesk", "acme", "The app crashes", questions={"angry": ANGRY})
    assert isinstance(run, Decision) and run.runtime == ["angry"]
    intent = run.answers["intent"]
    assert isinstance(intent, ChoiceAnswer) and intent.choice == "other" and run.routed_to is None
    assert set(dm.calls[0][1]) == {"intent", "angry"}


async def test_a_routed_decision_run_asks_its_own_folder_only(tmp_path: Path) -> None:
    write(
        tmp_path / "desk/AGENT.md",
        "---\nname: desk\nkind: decision\nroute: {by: team, to: {b: triage}}\n---\n",
    )
    write(tmp_path / "desk/questions/team.md", "---\ntype: choice\noptions: {a: A, b: B}\n---\n?\n")
    write(tmp_path / "triage/AGENT.md", "---\nname: triage\nkind: decision\n---\n")
    write(tmp_path / "triage/questions/urgent.md", "---\ntype: check\n---\nUrgent?\n")
    dm = Answering()
    run = await Engine(tmp_path, decision=dm).run("desk", "acme", "x", questions={"angry": ANGRY})
    assert isinstance(run, Decision) and isinstance(run.routed_to, Decision)
    assert run.routed_to.questions == {"urgent": Check(instructions="Urgent?")}
    assert run.routed_to.runtime == [] and [set(q) for _, q in dm.calls] == [
        {"team", "angry"},
        {"urgent"},
    ]


async def test_a_delegated_decision_run_asks_its_folder(fixtures: Path) -> None:
    llm = Scripted(delegate_triage(), text_reply("Handled."))
    run = await Engine(fixtures, generation=llm, decision=Answering(), tools=TOOLS).run(
        "support", "acme", "It is down"
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Decision) and child.runtime == [] and child.task is None
    assert set(child.questions) == {"urgent", "severity"}


async def test_a_task_asks_its_folder_instead_of_questions(tmp_path: Path) -> None:
    root = grader(tmp_path)
    dm = Answering()
    engine = Engine(root, decision=dm)
    run = await engine.run("grader", "acme", {"reply": "Thanks!"}, task="tone")
    assert isinstance(run, Decision) and run.task == "tone" and run.runtime == []
    assert set(run.questions) == {"polite", "register"}
    assert run.questions["register"] == Choice(
        instructions="Which register?",
        options={"formal": None, "casual": {"what": "Contractions, slang"}},
    )
    plain = await engine.run("grader", "acme", "x")
    assert isinstance(plain, Decision) and set(plain.questions) == {"urgent"} and plain.task is None
    facts = await engine.run("grader", "acme", "x", task="facts", questions={"urgent": ANGRY})
    assert isinstance(facts, Decision) and set(facts.questions) == {"grounded", "urgent"}
    assert facts.runtime == ["urgent"] and facts.questions["urgent"] == ANGRY
    with pytest.raises(
        InputError, match="^engine: decision agent 'grader' already asks 'polite' from its folder$"
    ):
        await engine.run("grader", "acme", "x", task="tone", questions={"polite": ANGRY})
    assert [set(q) for _, q in dm.calls] == [
        {"polite", "register"},
        {"urgent"},
        {"grounded", "urgent"},
    ]


async def test_an_unknown_task_is_not_found_as_an_unknown_agent_is(tmp_path: Path) -> None:
    root = grader(tmp_path)
    dm = Answering()
    with pytest.raises(NotFoundError, match="^engine: agent 'grader' has no task 'ghost'$"):
        await Engine(root, decision=dm).run("grader", "acme", "x", task="ghost")
    with pytest.raises(NotFoundError, match="^engine: agent 'blank' has no task 'tone'$"):
        await Engine(root, decision=dm).run("blank", "acme", "x", task="tone", questions=RUNTIME)
    assert dm.calls == []


async def test_a_task_lives_in_the_folder_so_the_agent_fingerprint_covers_it(
    tmp_path: Path,
) -> None:
    root = grader(tmp_path)
    before = await Engine(root, decision=Answering()).run("grader", "acme", "x", task="tone")
    write(root / "grader/tasks/tone/polite.md", "---\ntype: check\n---\nIs `reply` kind?\n")
    edited = await Engine(root, decision=Answering()).run("grader", "acme", "x", task="tone")
    write(root / "grader/tasks/brevity/short.md", "---\ntype: check\n---\nIs it short?\n")
    added = await Engine(root, decision=Answering()).run("grader", "acme", "x")
    prints = [run.fingerprint for run in (before, edited, added)]
    assert all(p is not None for p in prints)
    assert len({p.agent for p in prints if p is not None}) == 3
    assert before.fingerprint is not None and edited.fingerprint is not None
    assert before.fingerprint.questions != edited.fingerprint.questions


async def test_runtime_questions_and_a_task_are_recorded_and_read_back(tmp_path: Path) -> None:
    root = grader(tmp_path)
    refinery = Refinery(InMemoryStorage())
    engine = Engine(root, decision=Answering(), refinery=refinery)
    run = await engine.run("grader", "acme", "x", task="tone", questions=RUNTIME)
    read = await refinery.run("acme", run.run_id)
    assert read == run and isinstance(read, Decision)
    assert read.task == "tone" and read.runtime == ["area", "how_bad", "angry"]
    assert read.questions["how_bad"] == HOW_BAD and read.questions["area"] == AREA
    [row] = await refinery.storage.runs("acme")
    assert row.task == "tone"
