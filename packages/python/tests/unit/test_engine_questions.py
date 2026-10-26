from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, JsonValue

from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import Check, Choice, ChoiceAnswer, Question, Score
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage

from .fakes import NO_QUESTIONS, Answering, Scripted, ask, delegate_triage, text_reply

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
    write(root / "grader/AGENT.md", "---\nname: grader\nkind: decision\nmodel: judge\n---\n")
    write(
        root / "grader/questions/urgent.json", '{"type": "check", "instructions": "Is it urgent?"}'
    )
    write(root / "blank/AGENT.md", "---\nname: blank\nkind: decision\nmodel: judge\n---\n")
    return root


async def test_runtime_questions_are_asked_with_the_folder_s_in_one_call(fixtures: Path) -> None:
    dm = Answering()
    run = await ask(
        Engine(fixtures, models={"judge": dm}), "triage", "acme", "It is down", questions=RUNTIME
    )
    [(state, asked)] = dm.calls
    assert state == "It is down" and isinstance(run, Decision)
    assert asked == {"urgent": run.questions["urgent"], "severity": run.questions["severity"]} | {
        qid: RUNTIME[qid] for qid in RUNTIME
    }
    assert run.questions == asked
    assert set(run.answers) == {"urgent", "severity", "area", "how_bad", "angry"}
    assert run.runtime == ["area", "how_bad", "angry"]
    assert run.answers["area"].confidence == pytest.approx(0.7)


async def test_a_folder_only_run_marks_nothing_as_runtime(fixtures: Path) -> None:
    run = await Engine(fixtures, models={"judge": Answering()}).run("triage", "acme", "x")
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
        await ask(
            Engine(fixtures, models={"judge": dm}),
            "triage",
            "acme",
            "x",
            questions={"urgent": ANGRY, "area": AREA, "severity": HOW_BAD},
        )
    assert dm.calls == []


async def test_a_decision_agent_with_no_questions_of_its_own_runs_on_runtime_ones(
    tmp_path: Path,
) -> None:
    root = grader(tmp_path)
    dm = Answering()
    engine = Engine(root, models={"judge": dm})
    run = await ask(engine, "blank", "acme", {"message": "Where is it?"}, questions={"area": AREA})
    assert isinstance(run, Decision) and run.questions == {"area": AREA}
    assert run.runtime == ["area"] and run.answers["area"].type == "choice"
    nothing: list[dict[str, Question] | None] = [None, {}]
    for given in nothing:
        with pytest.raises(
            InputError,
            match=f"^{NO_QUESTIONS}$",
        ):
            await ask(engine, "blank", "acme", "x", questions=given)
    assert len(dm.calls) == 1


async def test_a_generation_agent_takes_no_questions(fixtures: Path) -> None:
    llm = Scripted()
    engine = Engine(fixtures, models={"strong": llm, "judge": Answering()})
    for given in (RUNTIME, {}):
        with pytest.raises(
            InputError, match="^engine: generation agent 'support' takes no questions$"
        ):
            await ask(engine, "support", "acme", "x", questions=given)
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
        await ask(Engine(fixtures, models={"judge": dm}), "triage", "acme", "x", questions=given)
    assert dm.calls == []


class ForeignChoice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]


async def test_a_runtime_question_is_read_by_its_shape(fixtures: Path) -> None:
    foreign = ForeignChoice(instructions={"question": "Which?"}, options={"a": None, "b": ["B"]})
    plain: Any = {"type": "check", "instructions": "Is it?"}
    run = await ask(
        Engine(fixtures, models={"judge": Answering()}),
        "triage",
        "acme",
        "x",
        questions={"pick": foreign, "sure": plain},
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
    engine = Engine(fixtures, models={"strong": llm, "judge": dm})
    with pytest.raises(
        InputError,
        match="^engine: decision agent 'frontdesk' already asks 'intent' from its folder$",
    ):
        await ask(engine, "frontdesk", "acme", "x", questions={"intent": AREA})
    run = await ask(engine, "frontdesk", "acme", "The app crashes", questions={"angry": ANGRY})
    assert isinstance(run, Decision) and run.runtime == ["angry"]
    intent = run.answers["intent"]
    assert isinstance(intent, ChoiceAnswer) and intent.choice == "other" and run.routed_to is None
    assert set(dm.calls[0][1]) == {"intent", "angry"}


async def test_a_routed_decision_run_asks_its_own_folder_only(tmp_path: Path) -> None:
    write(
        tmp_path / "desk/AGENT.md",
        "---\nname: desk\nkind: decision\nmodel: judge\nroute: {by: team, to: {b: triage}}\n---\n",
    )
    write(
        tmp_path / "desk/questions/team.json",
        '{"type": "choice", "instructions": "?", "options": {"a": "A", "b": "B"}}',
    )
    write(tmp_path / "triage/AGENT.md", "---\nname: triage\nkind: decision\nmodel: judge\n---\n")
    write(tmp_path / "triage/questions/urgent.json", '{"type": "check", "instructions": "Urgent?"}')
    dm = Answering()
    run = await ask(
        Engine(tmp_path, models={"judge": dm}), "desk", "acme", "x", questions={"angry": ANGRY}
    )
    assert isinstance(run, Decision) and isinstance(run.routed_to, Decision)
    assert run.routed_to.questions == {"urgent": Check(instructions="Urgent?")}
    assert run.routed_to.runtime == [] and [set(q) for _, q in dm.calls] == [
        {"team", "angry"},
        {"urgent"},
    ]


async def test_a_delegated_decision_run_asks_its_folder(fixtures: Path) -> None:
    llm = Scripted(delegate_triage(), text_reply("Handled."))
    run = await Engine(fixtures, models={"strong": llm, "judge": Answering()}).run(
        "support", "acme", "It is down"
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Decision) and child.runtime == []
    assert set(child.questions) == {"urgent", "severity"}


async def test_a_task_is_refused_for_a_decision_agent_whatever_it_asks(tmp_path: Path) -> None:
    root = grader(tmp_path)
    dm = Answering()
    engine = Engine(root, models={"judge": dm})
    with pytest.raises(InputError, match="^engine: decision agent 'grader' takes no task; "):
        await engine.run("grader", "acme", "x", task="tone")
    with pytest.raises(InputError, match="^engine: decision agent 'blank' takes no task; "):
        await ask(engine, "blank", "acme", "x", task="tone", questions=RUNTIME)
    assert dm.calls == []


async def test_runtime_questions_are_recorded_and_read_back(
    tmp_path: Path, storage: Storage
) -> None:
    root = grader(tmp_path)
    refinery = Refinery(storage)
    engine = Engine(root, refinery=refinery, models={"judge": Answering()})
    run = await ask(engine, "grader", "acme", "x", questions=RUNTIME)
    read = await refinery.run("acme", run.run_id)
    assert read == run and isinstance(read, Decision)
    assert read.runtime == ["area", "how_bad", "angry"]
    assert list(read.questions) == ["urgent", "area", "how_bad", "angry"]
    assert read.questions["how_bad"] == HOW_BAD and read.questions["area"] == AREA
    [row] = await refinery.storage.runs("acme")
    assert "task" not in row.body
