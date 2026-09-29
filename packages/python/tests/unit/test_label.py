import base64
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, ModelError, NotFoundError
from rfnry_engine.models import (
    Answer,
    Audio,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Document,
    Image,
    Part,
    Question,
    Reply,
    Score,
    ScoreAnswer,
    Text,
    ToolCall,
)
from rfnry_engine.records import Generation
from rfnry_engine.refinery.inmemory import InMemoryStorage
from rfnry_engine.refinery.judge import TeacherAnswers, teacher_answers, teacher_prompt
from rfnry_engine.refinery.labels import Review
from rfnry_engine.refinery.refinery import Refinery, label_of
from rfnry_engine.refinery.storage import RunRow
from rfnry_engine.tools.tools import Tool, Tools

from .fakes import TOOLS, Fixed, MemoryBucket, Scripted, text_reply

JPEG = b"\xff\xd8\xff\xe0jpeg"
PDF = b"%PDF-1.7 policy"

JUDGEMENT: dict[str, Answer] = {
    "grounded": CheckAnswer(probability=0.9),
    "done": ScoreAnswer(score=2, probabilities=[0.0, 0.1, 0.9]),
}
TRIAGE: dict[str, Answer] = {
    "urgent": CheckAnswer(probability=0.6),
    "severity": ScoreAnswer(score=1, probabilities=[0.2, 0.5, 0.2, 0.1]),
}


def _teacher_reply() -> Reply:
    args = {"answers": {"urgent": True, "severity": 3}, "rationale": "An outage is urgent."}
    return Reply(tool_calls=[ToolCall(id="t1", name="Output", args=args)], stop_reason="tool_use")


async def test_judge_labels_generation_runs_once(fixtures: Path) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    dm = Fixed(JUDGEMENT)
    engine = Engine(
        fixtures,
        generation=Scripted(text_reply("Refunded.")),
        decision=dm,
        refinery=refinery,
        tools=TOOLS,
    )
    run = await engine.run("billing", "acme", "Charged twice.")
    labeled = await refinery.label("acme", engine, judge="judge")
    assert labeled.judged == 1 and labeled.errors == {}
    [row] = await storage.labels("acme", source="judge")
    label = label_of(row)
    assert label.run_id == run.run_id and label.answers is not None
    assert label.answers["grounded"].confidence == 0.8
    state = dm.calls[0][0]
    assert (
        isinstance(state, dict)
        and state["reply"] == "Refunded."
        and state["input"] == "Charged twice."
    )
    assert [r.agent for r in await storage.runs("acme")] == ["billing"]
    assert (await refinery.label("acme", engine, judge="judge")).judged == 0


async def test_teacher_answers_decision_runs(fixtures: Path) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    llm = Scripted(_teacher_reply())
    engine = Engine(
        fixtures, generation=llm, decision=Fixed(TRIAGE), refinery=refinery, tools=TOOLS
    )
    run = await engine.run("triage", "acme", {"message": "It is down"})
    labeled = await refinery.label("acme", engine, teacher="teacher")
    assert labeled.taught == 1 and labeled.errors == {}
    [row] = await storage.labels("acme", source="teacher")
    label = label_of(row)
    assert label.run_id == run.run_id and label.answers is not None
    assert label.answers["urgent"].model_dump()["probability"] == 1.0
    assert label.answers["severity"].model_dump()["score"] == 3.0
    prompt = llm.calls[0][1][-1].content
    assert "QUESTION severity (score)" in str(prompt) and "3: Active outage" in str(prompt)


async def test_a_failing_labeler_is_reported_per_run(fixtures: Path) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = Engine(
        fixtures,
        generation=Scripted(text_reply("x")),
        decision=Fixed({}),
        refinery=refinery,
        tools=TOOLS,
    )
    run = await engine.run("billing", "acme", "hi")
    labeled = await refinery.label("acme", engine, judge="judge")
    assert (
        labeled.judged == 0
        and run.run_id in labeled.errors
        and "ModelError" in labeled.errors[run.run_id]
    )


async def test_judge_and_teacher_never_label_their_own_runs(fixtures: Path) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    llm = Scripted(text_reply("taught"))
    engine = Engine(
        fixtures, generation=llm, decision=Fixed(JUDGEMENT), refinery=refinery, tools=TOOLS
    )
    await engine.run("teacher", "acme", "x")
    await engine.run("judge", "acme", "x")
    labeled = await refinery.label("acme", engine, judge="judge", teacher="teacher")
    assert (labeled.judged, labeled.taught, labeled.errors) == (0, 0, {})
    assert await storage.labels("acme") == []


async def test_a_judge_or_teacher_the_tree_cannot_use_is_refused_before_any_run(
    fixtures: Path,
) -> None:
    storage = InMemoryStorage()
    refinery = Refinery(storage)
    llm = Scripted(text_reply("Refunded."))
    dm = Fixed({**TRIAGE, **JUDGEMENT})
    engine = Engine(fixtures, generation=llm, decision=dm, refinery=refinery, tools=TOOLS)
    await engine.run("billing", "acme", "x")
    await engine.run("triage", "acme", "x")
    refused: list[tuple[str | None, str | None, type[Exception], str]] = [
        ("jduge", None, NotFoundError, "refinery: judge 'jduge' was not found"),
        (None, "teachr", NotFoundError, "refinery: teacher 'teachr' was not found"),
        ("support", None, ConfigurationError, "refinery: judge 'support' must be a decision agent"),
        (
            "judge",
            "judge",
            ConfigurationError,
            "refinery: teacher 'judge' must be a generation agent",
        ),
    ]
    for judge, teacher, error, message in refused:
        with pytest.raises(error, match=f"^{message}$"):
            await refinery.label("acme", engine, judge=judge, teacher=teacher)
    assert (len(llm.calls), len(dm.calls)) == (1, 1)
    assert await storage.labels("acme") == []


def test_teacher_answers_are_validated_against_the_questions() -> None:
    questions: dict[str, Question] = {
        "intent": Choice(instructions="Which?", options={"a": "A", "b": "B"}),
        "severity": Score(instructions="How bad?", levels=["low", "high"]),
        "urgent": Check(instructions="Urgent?"),
    }
    answers = teacher_answers(
        questions,
        TeacherAnswers(answers={"intent": "b", "severity": 1, "urgent": False}, rationale="r"),
    )
    assert answers["intent"] == ChoiceAnswer(
        choice="b", probabilities={"a": 0.0, "b": 1.0}, confidence=1.0
    )
    assert answers["severity"] == ScoreAnswer(score=1.0, probabilities=[0.0, 1.0], confidence=1.0)
    assert answers["urgent"] == CheckAnswer(probability=0.0, confidence=1.0)
    cases: list[tuple[dict[str, str | float | bool], str]] = [
        ({"intent": "a", "severity": 0}, "^judge: teacher gave no answer for 'urgent'$"),
        (
            {"intent": "c", "severity": 0, "urgent": True},
            "^judge: teacher chose 'c', not an option of 'intent'$",
        ),
        (
            {"intent": "a", "severity": 2, "urgent": True},
            "^judge: teacher gave level 2 for 'severity'$",
        ),
        (
            {"intent": "a", "severity": 0, "urgent": "false"},
            "^judge: teacher gave 'false', not true or false for 'urgent'$",
        ),
        (
            {"intent": "a", "severity": 0, "urgent": 1.0},
            "^judge: teacher gave 1.0, not true or false for 'urgent'$",
        ),
        (
            {"intent": "a", "severity": 0.7, "urgent": True},
            "^judge: teacher gave 0.7, not a level of 'severity'$",
        ),
        (
            {"intent": "a", "severity": True, "urgent": True},
            "^judge: teacher gave True, not a level of 'severity'$",
        ),
        (
            {"intent": "a", "severity": "1", "urgent": True},
            "^judge: teacher gave '1', not a level of 'severity'$",
        ),
    ]
    for given, message in cases:
        with pytest.raises(ModelError, match=message):
            teacher_answers(questions, TeacherAnswers(answers=given, rationale="r"))


def test_the_teacher_reads_what_yes_and_no_mean_for_a_check() -> None:
    questions = {
        "urgent": Check(instructions="Is it urgent?", yes="The service is down", no="A workaround"),
        "polite": Check(instructions="Is it polite?"),
    }
    now = datetime.now(UTC)
    row = RunRow(
        scope="acme",
        run_id="r",
        parent_run_id=None,
        agent="triage",
        kind="decision",
        task=None,
        at=now,
        recorded=now,
        duration_ms=0,
        model=None,
        body={
            "state": "It is down",
            "questions": {qid: q.model_dump(mode="json") for qid, q in questions.items()},
        },
    )
    assert teacher_prompt(row).endswith(
        "QUESTION urgent (check): Is it urgent?\n"
        "  yes: The service is down\n"
        "  no: A workaround\n"
        "QUESTION polite (check): Is it polite?"
    )


def test_the_teacher_reads_structure_as_compact_json_and_a_bare_option_as_its_key() -> None:
    questions: dict[str, Question] = {
        "team": Choice(
            instructions={"question": "Which team handles `ticket`?", "ticket": {"id": 7}},
            options={"billing": {"what": "Charges", "examples": ["charged twice"]}, "other": None},
        ),
        "scope": Score(instructions=["How focused?", "one change"], levels=["One", ["Several"]]),
        "creds": Check(instructions="Asks for a credential?", yes={"what": "A password"}),
    }
    now = datetime.now(UTC)
    row = RunRow(
        scope="acme",
        run_id="r",
        parent_run_id=None,
        agent="sorter",
        kind="decision",
        task=None,
        at=now,
        recorded=now,
        duration_ms=0,
        model=None,
        body={
            "state": "Hi",
            "questions": {qid: q.model_dump(mode="json") for qid, q in questions.items()},
        },
    )
    assert teacher_prompt(row).endswith(
        'QUESTION team (choice): {"question":"Which team handles `ticket`?","ticket":{"id":7}}\n'
        '  - billing: {"what":"Charges","examples":["charged twice"]}\n'
        "  - other\n"
        'QUESTION scope (score): ["How focused?","one change"]\n'
        "  0: One\n"
        '  1: ["Several"]\n'
        "QUESTION creds (check): Asks for a credential?\n"
        '  yes: {"what":"A password"}'
    )


async def test_a_teachers_answer_is_the_exported_truth_until_a_person_gives_one(
    fixtures: Path,
) -> None:
    refinery = Refinery(InMemoryStorage())
    engine = Engine(
        fixtures,
        generation=Scripted(_teacher_reply()),
        decision=Fixed(TRIAGE),
        refinery=refinery,
        tools=TOOLS,
    )
    run = await engine.run("triage", "acme", {"message": "It is down"})
    await refinery.label("acme", engine, teacher="teacher")

    async def truth() -> dict[str, tuple[object, object]]:
        bucket = MemoryBucket()
        await refinery.export("acme", bucket)
        rows = [
            json.loads(line)
            for data in bucket.objects.values()
            for line in data.decode().splitlines()
        ]
        return {row["question_id"]: (row["truth"], row["truth_source"]) for row in rows}

    assert await truth() == {"urgent": (True, "teacher"), "severity": (3.0, "teacher")}
    await refinery.review("acme", run.run_id, Review(correction={"urgent": False}))
    assert await truth() == {"urgent": (False, "human"), "severity": (3.0, "teacher")}


async def test_the_judge_sees_media_as_placeholders_never_base64(fixtures: Path) -> None:
    async def policy(args: dict[str, Any]) -> list[Part]:
        return [
            Text(text="the policy"),
            Document(media_type="application/pdf", data=PDF, name="policy.pdf"),
        ]

    storage = InMemoryStorage()
    refinery = Refinery(storage)
    dm = Fixed(JUDGEMENT)
    llm = Scripted(
        Reply(tool_calls=[ToolCall(id="p", name="policy", args={})], stop_reason="tool_use"),
        text_reply("Refunded."),
    )
    engine = Engine(fixtures, generation=llm, decision=dm, refinery=refinery, tools=TOOLS)
    parts: list[Part] = [
        Text(text="Is this refundable?"),
        Image(media_type="image/jpeg", data=JPEG),
        Audio(media_type="audio/wav", url="https://x/a.wav"),
    ]
    run = await engine.run("billing", "acme", parts, tools=Tools([Tool("policy", body=policy)]))
    labeled = await refinery.label("acme", engine, judge="judge")
    assert labeled.judged == 1 and labeled.errors == {}
    state = dm.calls[0][0]
    assert state == {
        "input": ["Is this refundable?", "[image: image/jpeg]", "[audio: audio/wav]"],
        "reply": "Refunded.",
        "steps": [
            {
                "tool": "policy",
                "args": {},
                "result": ["the policy", '[document: application/pdf, "policy.pdf"]'],
            }
        ],
    }
    for encoded in (base64.b64encode(JPEG).decode(), base64.b64encode(PDF).decode()):
        assert encoded not in json.dumps(state)
    assert isinstance(run, Generation) and run.input == parts
    [row] = await storage.runs("acme")
    assert row.body["input"][1]["data"] == base64.b64encode(JPEG).decode()
    assert row.body["steps"][0]["result"][1]["name"] == "policy.pdf"


async def test_the_judge_sees_media_loaded_from_the_folder_as_placeholders(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    png, pdf = b"\x89PNG\r\n\x1a\nbody", b"%PDF-1.7 rates"
    (root / "billing" / "knowledge").mkdir()
    (root / "billing" / "knowledge" / "diagram.png").write_bytes(png)
    (root / "billing" / "knowledge" / "rates.pdf").write_bytes(pdf)
    refinery = Refinery(InMemoryStorage())
    dm = Fixed(JUDGEMENT)
    reads = [
        ToolCall(id="i", name="Load", args={"path": "knowledge/diagram.png"}),
        ToolCall(id="d", name="Load", args={"path": "knowledge/rates.pdf"}),
    ]
    llm = Scripted(Reply(tool_calls=reads, stop_reason="tool_use"), text_reply("Express is $25."))
    engine = Engine(root, generation=llm, decision=dm, refinery=refinery, tools=TOOLS)
    await engine.run("billing", "acme", "What does express cost?")
    labeled = await refinery.label("acme", engine, judge="judge")
    assert labeled.judged == 1 and labeled.errors == {}
    state = dm.calls[0][0]
    assert isinstance(state, dict) and state["steps"] == [
        {
            "tool": "Load",
            "args": {"path": "knowledge/diagram.png"},
            "result": ["[image: image/png]"],
        },
        {
            "tool": "Load",
            "args": {"path": "knowledge/rates.pdf"},
            "result": ['[document: application/pdf, "rates.pdf"]'],
        },
    ]
    for encoded in (base64.b64encode(png).decode(), base64.b64encode(pdf).decode()):
        assert encoded not in json.dumps(state)
