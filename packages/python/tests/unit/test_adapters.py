import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from rfnry_engine.models import (
    Check,
    CheckAnswer,
    Choice,
    ChoiceAnswer,
    Question,
    Score,
    ScoreAnswer,
)

EXAMPLES = Path(__file__).parents[4] / "examples"

QUESTIONS: dict[str, Question] = {
    "team": Choice(
        instructions={"question": "Which team handles `message`?", "focus": "The main request"},
        options={
            "security": {"what": "Phishing", "not_for": "Payroll", "examples": ["my password"]},
            "payroll": ["Bonuses", "Salaries"],
            "other": None,
        },
    ),
    "risk": Score(
        instructions=["How risky is `message`?", {"inspect": "message"}],
        levels=[{"summary": "Harmless"}, ["Suspicious"], "Dangerous"],
    ),
    "credentials": Check(
        instructions={"question": "Does `message` ask for a credential?"},
        yes={"what": "A password or code"},
        no=["Nothing secret"],
    ),
    "bank": Check(instructions="Is `sender` a bank?", yes="A licensed bank"),
    "english": Check(instructions="Is it written in English?"),
}


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    anthropic, types = ModuleType("anthropic"), ModuleType("anthropic.types")
    vars(anthropic).update(AsyncAnthropic=object, types=types)
    vars(types).update(Message=object)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    monkeypatch.setitem(sys.modules, "anthropic.types", types)
    spec = importlib.util.spec_from_file_location(
        "healthcare_providers", EXAMPLES / "healthcare_assistant" / "src" / "providers.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replying(
    sent: list[dict[str, Any]], answer: Callable[[dict[str, Any]], dict[str, Any]]
) -> httpx.AsyncClient:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json=answer(body))

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def systemone(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "security",
                "confidence": 1.0,
                "probabilities": {"payroll": 0.0, "other": 0.0, "security": 1.0},
            },
            "risk": {
                "type": "score",
                "score": 2.0,
                "confidence": 1.0,
                "legend": dict(enumerate(body["questions"]["risk"]["criteria"])),
                "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
            },
            "credentials": {"type": "noul", "noul": 0.99},
            "bank": {"type": "noul", "noul": 0.01},
            "english": {"type": "noul", "noul": 0.99},
        },
        "usage": {"input_tokens": 692, "output_tokens": 102},
    }


async def test_the_decision_provider_sends_the_structure_as_it_is(
    providers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DECISION_API_KEY", "offline")
    provider = providers.TypeSafeProvider()
    sent: list[dict[str, Any]] = []
    provider.client = replying(sent, systemone)
    decided = await provider.decide({"message": "Reply with your password"}, QUESTIONS)
    [body] = sent
    assert body["state"] == {"message": "Reply with your password"}
    assert body["questions"] == {
        "team": {
            "type": "choice",
            "instructions": {
                "question": "Which team handles `message`?",
                "focus": "The main request",
            },
            "criteria": {
                "security": {"what": "Phishing", "not_for": "Payroll", "examples": ["my password"]},
                "payroll": ["Bonuses", "Salaries"],
                "other": None,
            },
        },
        "risk": {
            "type": "score",
            "instructions": ["How risky is `message`?", {"inspect": "message"}],
            "criteria": [{"summary": "Harmless"}, ["Suspicious"], "Dangerous"],
        },
        "credentials": {
            "type": "noul",
            "instructions": {"question": "Does `message` ask for a credential?"},
            "criteria": {"true": {"what": "A password or code"}, "false": ["Nothing secret"]},
        },
        "bank": {
            "type": "noul",
            "instructions": "Is `sender` a bank?",
            "criteria": {"true": "A licensed bank", "false": None},
        },
        "english": {"type": "noul", "instructions": "Is it written in English?"},
    }
    assert decided.answers == {
        "team": ChoiceAnswer(
            choice="security", probabilities={"security": 1.0, "payroll": 0.0, "other": 0.0}
        ),
        "risk": ScoreAnswer(score=2.0, probabilities=[0.0, 0.0, 1.0]),
        "credentials": CheckAnswer(probability=0.99),
        "bank": CheckAnswer(probability=0.01),
        "english": CheckAnswer(probability=0.99),
    }
    assert decided.model == "jev-1.13.0" and decided.usage.input == 692


async def test_the_decision_provider_refuses_what_its_vendor_cannot_take_before_sending(
    providers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DECISION_API_KEY", "offline")
    provider = providers.TypeSafeProvider()
    sent: list[dict[str, Any]] = []
    provider.client = replying(sent, systemone)
    many = Choice(instructions="Which?", options={f"o{i}": None for i in range(256)})
    with pytest.raises(ValueError, match="^typesafe: a choice takes 255 options at most$"):
        await provider.decide("x", {"q": many})
    deep = Score(instructions="How?", levels=[f"level {i}" for i in range(11)])
    with pytest.raises(ValueError, match="^typesafe: a score takes 10 levels at most$"):
        await provider.decide("x", {"q": deep})
    assert sent == []


def lettered(body: dict[str, Any]) -> dict[str, Any]:
    letters = [line[0] for line in body["messages"][1]["content"].splitlines() if line[1:3] == ") "]
    top = [
        {"token": letter, "logprob": -0.1 if i == 0 else -5.0} for i, letter in enumerate(letters)
    ]
    return {
        "model": "local",
        "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }


async def test_the_local_provider_writes_structure_as_compact_json_in_its_prompt(
    providers: ModuleType,
) -> None:
    provider = providers.LocalDecisionProvider()
    sent: list[dict[str, Any]] = []
    provider.client = replying(sent, lettered)
    decided = await provider.decide("Reply with your password", QUESTIONS)
    prompts = {
        qid: body["messages"][1]["content"] for qid, body in zip(QUESTIONS, sent, strict=True)
    }
    assert prompts["team"] == (
        "Reply with your password\n\n"
        'Question: {"question":"Which team handles `message`?","focus":"The main request"}\n\n'
        'A) security: {"what":"Phishing","not_for":"Payroll","examples":["my password"]}\n'
        'B) payroll: ["Bonuses","Salaries"]\n'
        "C) other\n\n"
        "Answer with one letter: A, B or C."
    )
    assert prompts["risk"] == (
        "Reply with your password\n\n"
        'Question: ["How risky is `message`?",{"inspect":"message"}]\n\n'
        'A) {"summary":"Harmless"}\nB) ["Suspicious"]\nC) Dangerous\n\n'
        "Answer with one letter: A, B or C."
    )
    assert prompts["credentials"] == (
        "Reply with your password\n\n"
        'Question: {"question":"Does `message` ask for a credential?"}\n\n'
        'A) yes: {"what":"A password or code"}\nB) no: ["Nothing secret"]\n\n'
        "Answer with one letter: A or B."
    )
    assert prompts["bank"] == (
        "Reply with your password\n\nQuestion: Is `sender` a bank?\n\n"
        "A) yes: A licensed bank\nB) no\n\nAnswer with one letter: A or B."
    )
    assert decided.answers["team"].choice == "security"


def test_the_example_picks_the_local_decision_model_when_one_is_named(
    providers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DECISION_API_KEY", "offline")
    monkeypatch.delenv("LOCAL_DECISION_MODEL", raising=False)
    assert isinstance(providers.decision(), providers.TypeSafeProvider)
    monkeypatch.setenv("LOCAL_DECISION_MODEL", "qwen3:8b")
    local = providers.decision()
    assert isinstance(local, providers.LocalDecisionProvider) and local.model == "qwen3:8b"
