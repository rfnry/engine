import base64
import json
import math
from datetime import date

import pytest
from pydantic import TypeAdapter, ValidationError

from rfnry_engine.errors import InputError, ModelError
from rfnry_engine.models import (
    Answer,
    Audio,
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Decided,
    Document,
    Image,
    Message,
    Question,
    Reply,
    Score,
    ScoreAnswer,
    Text,
    Usage,
    as_decided,
    as_reply,
    confidence,
    settle,
)


def test_confidence_is_peakedness() -> None:
    assert confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert confidence([0.8, 0.2]) == pytest.approx(0.6)


def test_settle_fills_confidence_and_keeps_answers() -> None:
    questions: dict[str, Question] = {
        "intent": Choice(instructions="Which?", options={"a": "A", "b": "B"}),
        "urgent": Check(instructions="Urgent?"),
        "severity": Score(instructions="How bad?", levels=["low", "mid", "high"]),
    }
    answers: dict[str, Answer] = {
        "intent": ChoiceAnswer(choice="a", probabilities={"a": 0.9, "b": 0.1}, confidence=0.123),
        "urgent": CheckAnswer(probability=0.75),
        "severity": ScoreAnswer(score=1.4, probabilities=[0.1, 0.4, 0.5]),
    }
    settled = settle(questions, answers)
    assert settled["intent"].confidence == pytest.approx(0.8)
    assert settled["urgent"].confidence == pytest.approx(0.5)
    assert settled["severity"].confidence == pytest.approx(0.25)
    severity = settled["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert severity.score == 1.4


def test_settle_rejects_contract_violations() -> None:
    q: dict[str, Question] = {"intent": Choice(instructions="Which?", options={"a": "A", "b": "B"})}
    with pytest.raises(ModelError):
        settle(q, {})
    with pytest.raises(ModelError):
        settle(q, {"intent": ChoiceAnswer(choice="c", probabilities={"a": 0.5, "b": 0.5})})
    with pytest.raises(ModelError):
        settle(q, {"intent": ChoiceAnswer(choice="a", probabilities={"a": 0.9, "b": 0.9})})
    with pytest.raises(ModelError):
        settle(q, {"intent": CheckAnswer(probability=0.5)})


def test_settle_refuses_probabilities_outside_zero_and_one() -> None:
    q: dict[str, Question] = {
        "pick": Choice(instructions="Which?", options={"a": "A", "b": "B"}),
        "level": Score(instructions="How bad?", levels=["low", "high"]),
    }
    level = ScoreAnswer(score=0, probabilities=[0.5, 0.5])
    pick = ChoiceAnswer(choice="a", probabilities={"a": 0.5, "b": 0.5})
    wrong: list[dict[str, Answer]] = [
        {"pick": ChoiceAnswer(choice="a", probabilities={"a": 1.5, "b": -0.5}), "level": level},
        {"pick": ChoiceAnswer(choice="a", probabilities={"a": math.nan, "b": 0.5}), "level": level},
        {"pick": pick, "level": ScoreAnswer(score=0, probabilities=[math.inf, -math.inf])},
        {"pick": pick, "level": ScoreAnswer(score=0, probabilities=[math.nan, math.nan])},
    ]
    for answers in wrong:
        with pytest.raises(ModelError, match="^models: probabilities for '(pick|level)' must lie"):
            settle(q, answers)


def test_settle_refuses_a_score_outside_its_levels() -> None:
    q: dict[str, Question] = {"level": Score(instructions="How bad?", levels=["low", "high"])}
    assert settle(q, {"level": ScoreAnswer(score=1.0, probabilities=[0.2, 0.8])})["level"]
    for score in (7.0, -0.5, math.nan, math.inf):
        with pytest.raises(ModelError, match="^models: score for 'level' lies outside its levels$"):
            settle(q, {"level": ScoreAnswer(score=score, probabilities=[0.2, 0.8])})


def test_questions_need_two_options_or_levels() -> None:
    with pytest.raises(ValidationError):
        Choice(instructions="?", options={"only": "one"})
    with pytest.raises(ValidationError):
        Score(instructions="?", levels=["one"])


def test_usage_forbids_unknown_keys_and_adds() -> None:
    with pytest.raises(ValidationError):
        Usage.model_validate({"input_tokens": 3})
    total = Usage(input=10, output=2) + Usage(input=5, cache_read=7)
    assert (total.input, total.output, total.cache_read) == (15, 2, 7)


def test_readers_take_results_by_their_fields_and_name_the_contract() -> None:
    assert as_reply({"text": "hi"}) == Reply(text="hi")
    assert as_decided({"urgent": {"type": "check", "probability": 0.9}}).answers[
        "urgent"
    ] == CheckAnswer(probability=0.9)
    assert as_decided(Decided(answers={}, model="d")).model == "d"
    with pytest.raises(
        ModelError, match="^models: the generation model's reply does not fit the contract"
    ):
        as_reply({"text": 1})
    with pytest.raises(
        ModelError, match="^models: the decision model's answers do not fit the contract"
    ):
        as_decided({"urgent": {"type": "check", "probability": 2}})


def test_parts_parse_by_their_type() -> None:
    message = Message.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "media_type": "image/png", "url": "https://x/y.png"},
            ],
        }
    )
    assert message.content == [
        Text(text="what is this?"),
        Image(media_type="image/png", url="https://x/y.png"),
    ]
    assert Message(role="user").content == "" and Message(role="assistant").tool_calls == []
    with pytest.raises(ValidationError):
        Message.model_validate(
            {"role": "user", "content": [{"type": "video", "media_type": "video/mp4", "url": "u"}]}
        )


@pytest.mark.parametrize("kind", [Image, Audio, Document])
def test_a_media_part_carries_data_or_url_not_both(kind: type[Image | Audio | Document]) -> None:
    kind(media_type="x/y", data=b"1")
    kind(media_type="x/y", url="https://x")
    name = kind.model_fields["type"].default
    with pytest.raises(InputError, match=f"^models: {name} carries exactly one of data or url$"):
        kind(media_type="x/y", data=b"1", url="https://x")
    with pytest.raises(InputError, match=f"^models: {name} carries exactly one of data or url$"):
        kind(media_type="x/y")


def test_part_bytes_travel_as_base64_in_json() -> None:
    raw = bytes(range(256))
    message = Message(
        role="tool",
        tool_call_id="c1",
        content=[Document(media_type="application/pdf", name="a.pdf", data=raw)],
    )
    dumped = message.model_dump(mode="json")
    assert dumped["content"][0]["data"] == base64.b64encode(raw).decode()
    assert Message.model_validate_json(message.model_dump_json()) == message
    assert Message.model_validate(dumped) == message
    assert (
        Audio.model_validate_json(Audio(media_type="audio/wav", data=raw).model_dump_json()).data
        == raw
    )


STRUCTURED: dict[str, Question] = {
    "team": Choice(
        instructions={"question": "Which team handles `ticket`?", "ticket": {"id": 7}},
        options={
            "billing": {"what": "Charges", "not_for": "Delivery", "examples": ["charged twice"]},
            "orders": ["Delivery", "Returns"],
            "other": None,
            "legal": "Contracts",
        },
    ),
    "scope": Score(
        instructions=["How focused is the change?", {"note": "count independent changes"}],
        levels=[{"summary": "One change", "signals": ["a single fix"]}, ["Several"], "Many"],
    ),
    "credentials": Check(
        instructions={"question": "Does `message` ask for a credential?", "inspect": "message"},
        yes={"what": "Asks for a password or code", "examples": ["Reply with your PIN"]},
        no=["No credential is requested", {"examples": ["Reset it in settings"]}],
    ),
}


def test_questions_take_structure_and_round_trip_through_json() -> None:
    adapter = TypeAdapter(dict[str, Question])
    dumped = adapter.dump_json(STRUCTURED)
    assert adapter.validate_json(dumped) == STRUCTURED
    assert json.loads(dumped)["team"]["options"]["other"] is None
    assert json.loads(dumped)["credentials"]["no"][1] == {"examples": ["Reset it in settings"]}
    plain = Choice(instructions="Which?", options={"a": "A", "b": "B"})
    assert plain.instructions == "Which?" and plain.options == {"a": "A", "b": "B"}
    assert Check(instructions="Is it?").yes is None


def test_structure_is_json_and_still_needs_two_options_or_levels() -> None:
    with pytest.raises(ValidationError):
        Choice(instructions="Which?", options={"only": {"what": "one"}})
    with pytest.raises(ValidationError):
        Score(instructions="How?", levels=[{"summary": "one"}])
    for fields in [
        {"type": "score", "instructions": "How?", "levels": ["low", None]},
        {"type": "check", "instructions": {"since": date(2026, 1, 1)}},
        {"type": "choice", "instructions": "Which?", "options": {"a": 1, "b": "B"}},
    ]:
        with pytest.raises(ValidationError):
            TypeAdapter(Question).validate_python(fields)
