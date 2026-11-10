import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Check, Choice, Score

from .trees import add_agent, deciding, write_folder

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"
CHOICE = '{"type": "choice", "instructions": "Which?", "options": {"x": "X", "y": "Y"}}'
CHECK = '{"type": "check", "instructions": "Is it?"}'


def _generation(root: Path, name: str = "a", **entry: Any) -> Path:
    return add_agent(
        root,
        name,
        {"kind": "generation", "model": "strong", "instructions": f"{name}.md", **entry},
        files={f"{name}.md": "Work.\n"},
    )


def test_load_folder_finds_every_agent_with_its_kind() -> None:
    agents = load_folder(FIXTURES)
    assert {n: a.kind for n, a in agents.items()} == {
        "support": "generation",
        "billing": "generation",
        "triage": "decision",
        "frontdesk": "decision",
        "judge": "decision",
        "teacher": "generation",
    }


def test_generation_agent_carries_body_agents_tools_skills() -> None:
    support = load_folder(FIXTURES)["support"]
    assert support.body.startswith("You are the support desk")
    assert support.agents == ["triage"]
    assert list(support.skills) == ["escalate"]
    assert list(support.folder_tools) == ["lookup_order", "now"]


def test_decision_agent_carries_typed_questions_and_route() -> None:
    agents = load_folder(FIXTURES)
    severity = agents["triage"].questions["severity"]
    assert isinstance(severity, Score) and len(severity.levels) == 4
    intent = agents["frontdesk"].questions["intent"]
    assert isinstance(intent, Choice) and set(intent.options) == {"billing", "technical", "other"}
    route = agents["frontdesk"].route
    assert route is not None and route.by == "intent" and route.min_confidence == 0.6


def test_question_with_one_option_is_refused(tmp_path: Path) -> None:
    deciding(tmp_path, "d", {"q": '{"type": "choice", "instructions": "?", "options": {"x": "X"}}'})
    with pytest.raises(ConfigurationError, match="a choice needs at least two options"):
        load_folder(tmp_path)


def test_an_agent_carries_the_map_of_its_folder() -> None:
    support = load_folder(FIXTURES)["support"]
    assert support.listing == (
        "support/\n  knowledge/\n    plans.md — The three plans\n"
        "  rules/\n    refunds.md — When a refund is owed"
    )


def test_an_unknown_key_in_a_question_is_refused_by_name(tmp_path: Path) -> None:
    typo = '{"type": "choice", "instructions": "Which?", "optoins": {"x": "X", "y": "Y"}}'
    deciding(tmp_path, "d", {"q": typo})
    with pytest.raises(ConfigurationError) as refused:
        load_folder(tmp_path)
    path = tmp_path / "d" / "q.json"
    assert str(refused.value) == (
        f"tree: {path} has the unknown key 'optoins'; a question takes type, instructions, options, levels, yes and no"
    )


def test_a_question_takes_structure_as_json(tmp_path: Path) -> None:
    choice = {
        "type": "choice",
        "instructions": {
            "question": "Which team handles `ticket`?",
            "ticket": {"id": 7, "vip": True},
        },
        "options": {
            "billing": {"what": "Charges", "examples": ["charged twice", "yes", "on"]},
            "orders": ["Delivery", "Returns"],
            "other": None,
            "legal": "Contracts",
        },
    }
    score = {
        "type": "score",
        "instructions": "How focused is the change?",
        "levels": [{"summary": "One change", "signals": ["a single fix"]}, ["Several"], "Many"],
    }
    check = {
        "type": "check",
        "instructions": ["Does `message` ask for a credential?", "message"],
        "yes": {"what": "Asks for a password", "examples": ["Reply with your PIN"]},
        "no": ["No credential is requested"],
    }
    deciding(
        tmp_path,
        "d",
        {"team": json.dumps(choice), "scope": json.dumps(score), "creds": json.dumps(check)},
    )
    assert load_folder(tmp_path)["d"].questions == {
        "creds": Check(
            instructions=["Does `message` ask for a credential?", "message"],
            yes={"what": "Asks for a password", "examples": ["Reply with your PIN"]},
            no=["No credential is requested"],
        ),
        "scope": Score(
            instructions="How focused is the change?",
            levels=[{"summary": "One change", "signals": ["a single fix"]}, ["Several"], "Many"],
        ),
        "team": Choice(
            instructions={
                "question": "Which team handles `ticket`?",
                "ticket": {"id": 7, "vip": True},
            },
            options={
                "billing": {"what": "Charges", "examples": ["charged twice", "yes", "on"]},
                "orders": ["Delivery", "Returns"],
                "other": None,
                "legal": "Contracts",
            },
        ),
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"type": "check"}', "instructions: field required"),
        (
            '{"type": "check", "instructions": "  "}',
            "instructions: instructions must say something",
        ),
        ('{"type": "check", "instructions": {}}', "instructions: instructions must say something"),
        ('{"type": "check", "instructions": []}', "instructions: instructions must say something"),
    ],
)
def test_a_question_has_instructions_that_say_something(
    tmp_path: Path, text: str, message: str
) -> None:
    deciding(tmp_path, "d", {"q": text})
    path = tmp_path / "d" / "q.json"
    with pytest.raises(ConfigurationError) as refused:
        load_folder(tmp_path)
    assert str(refused.value).startswith(f"tree: invalid question {path}; {message}")


@pytest.mark.parametrize(
    "text",
    [
        '{"type": "score", "instructions": "How?", "levels": ["low", null]}',
        '{"type": "score", "instructions": "How?", "levels": ["low", 2]}',
        '{"type": "choice", "instructions": "Which?", "options": {"a": "A", "b": 2026}}',
        '{"type": "check", "instructions": "Is it?", "yes": true}',
    ],
)
def test_a_question_s_text_is_a_string_an_object_or_a_list(tmp_path: Path, text: str) -> None:
    deciding(tmp_path, "d", {"q": text})
    with pytest.raises(ConfigurationError, match="^tree: invalid question "):
        load_folder(tmp_path)


def test_a_decision_agent_may_have_no_questions_of_its_own(tmp_path: Path) -> None:
    deciding(tmp_path, "a", {})
    spec = load_folder(tmp_path)["a"]
    assert spec.questions == {} and spec.scripted == {}


def test_a_delegate_target_needs_no_questions_of_its_own(tmp_path: Path) -> None:
    _generation(tmp_path, orchestration={"delegate_to": ["b", "c"]})
    deciding(tmp_path, "b", {})
    deciding(tmp_path, "c", {})
    assert sorted(load_folder(tmp_path)) == ["a", "b", "c"]


def test_a_route_entry_that_is_not_a_name_fails_as_any_bad_value(tmp_path: Path) -> None:
    deciding(
        tmp_path,
        "a",
        {"team": CHOICE},
        route={"by": "team", "to": {"x": {"agent": "support", "task": "triage"}}},
    )
    with pytest.raises(ConfigurationError, match=r"route\.to\.x: input should be a valid string"):
        load_folder(tmp_path)


def _shared(root: Path) -> Path:
    for name in ("a", "b"):
        _generation(root, name, files={"include": ["kb"]})
    return root


def test_an_unreadable_folder_is_warned_about_once_per_load_and_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root = _shared(tmp_path)
    for name in ("kb/locked/x.md", "kb/z.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x\n")
    locked = (root / "kb/locked").resolve()
    real = os.listdir

    def listdir(path: Any = ".") -> Any:
        if Path(str(path)).resolve() == locked:
            raise PermissionError(13, "denied", str(path))
        return real(path)

    monkeypatch.setattr(os, "listdir", listdir)
    warned = [f"folders: cannot read {locked}: [Errno 13] denied: '{locked}'; it is left out"]
    for _ in range(2):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            agents = load_folder(root)
        assert [m for m in caplog.messages if "cannot read" in m] == warned
    assert "z.md" in agents["a"].listing and "x.md" not in agents["b"].listing


def test_an_unreadable_file_is_warned_about_once_per_load_and_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    roots = (tmp_path / "with", tmp_path / "without")
    for root in roots:
        root.mkdir()
        _shared(root)
        (root / "kb").mkdir()
        (root / "kb/open.md").write_text("---\ndescription: Open\n---\nx\n")
    locked = (tmp_path / "with/kb/notes.md").resolve()
    locked.write_text("---\ndescription: Locked\n---\ny\n")
    locked.chmod(0)
    try:
        warned = [
            f"folders: cannot read {locked}: [Errno 13] Permission denied: '{locked}'; it is left out"
        ]
        for _ in range(2):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                agents = load_folder(tmp_path / "with")
            assert [m for m in caplog.messages if "cannot read" in m] == warned
        alike = load_folder(tmp_path / "without")
        assert {name: spec.fingerprint for name, spec in agents.items()} == {
            name: spec.fingerprint for name, spec in alike.items()
        }
        assert "notes.md" not in agents["a"].listing + agents["b"].listing
        assert "open.md — Open" in agents["a"].listing
    finally:
        locked.chmod(0o644)


DECLARED_FILES = {
    "a.md": "body\n",
    "t/x.json": '{"description": "X", "input_schema": {"type": "object"}}',
    "t/x.py": "import shared\ndef run(args, call):\n    return 'x'\n",
    "t/shared.py": "VALUE = 1\n",
    "d.md": "",
    "q/ok.json": '{"type": "check", "instructions": "ok?"}',
    "q/filled.json": '{"type": "check"}',
    "q/filled.py": "def run(input, call):\n    return {'instructions': 'q?'}\n",
}


def _declared(root: Path) -> Path:
    return write_folder(
        root,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "tools": {"include": ["x"]},
            },
            "d": {
                "kind": "decision",
                "model": "judge",
                "instructions": "d.md",
                "questions": ["q/*.json"],
            },
        },
        tools={"x": {"module": "t/x.py", "schema": "t/x.json"}},
        files=DECLARED_FILES,
    )


@pytest.mark.parametrize(
    ("locked", "module"),
    [
        ("q/ok.json", "tree"),
        ("t/x.json", "tools/folder"),
        ("t/x.py", "tools/folder"),
        ("t/shared.py", "tools/folder"),
        ("q/filled.py", "tools/folder"),
    ],
)
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_declared_file_that_cannot_be_read_is_refused_naming_it(
    tmp_path: Path, locked: str, module: str
) -> None:
    root = _declared(tmp_path)
    assert sorted(load_folder(root)) == ["a", "d"]
    path = tmp_path / locked
    path.chmod(0)
    try:
        with pytest.raises(ConfigurationError) as caught:
            load_folder(root)
    finally:
        path.chmod(0o644)
    assert (
        str(caught.value)
        == f"{module}: cannot read {path.resolve()}: Permission denied; make it readable to the engine"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_an_unreadable_skill_is_refused_at_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_folder(
        tmp_path,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "skills": {"include": ["skills/*"]},
            }
        },
        files={
            "a.md": "body\n",
            "skills/deep/SKILL.md": "---\ndescription: Deep\n---\nGo deep.\n",
            "skills/deep/notes.md": "Notes.\n",
        },
    )
    notes = (tmp_path / "skills/deep/notes.md").resolve()
    path = (tmp_path / "skills/deep/SKILL.md").resolve()
    notes.chmod(0)
    try:
        with caplog.at_level(logging.WARNING):
            assert sorted(load_folder(tmp_path)["a"].skills) == ["deep"]
        assert [m for m in caplog.messages if "cannot read" in m] == [
            f"folders: cannot read {notes}: [Errno 13] Permission denied: '{notes}'; it is left out"
        ]
        path.chmod(0)
        try:
            with pytest.raises(ConfigurationError) as caught:
                load_folder(tmp_path)
        finally:
            path.chmod(0o644)
    finally:
        notes.chmod(0o644)
    assert str(caught.value) == (
        "compiled: agent 'a' lists 'skills/*' in skills.include, which reaches no skill; a skill is a folder holding SKILL.md, so write such a folder or a glob from the folder root that reaches one, or remove it"
    )


class _Vanishing:
    def __init__(self, folder: str, name: str) -> None:
        self.entries = SCANDIR(folder)
        self.name = name

    def __enter__(self) -> _Vanishing:
        self.entries.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.entries.__exit__(None, None, None)

    def __iter__(self) -> Iterator[os.DirEntry[str]]:
        for entry in self.entries:
            if entry.name == self.name:
                os.unlink(entry.path)
            yield entry


SCANDIR = os.scandir


def test_a_file_removed_while_its_question_folder_is_listed_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _declared(tmp_path)
    (root / "q/_old.json").write_text("a temporary file\n")
    place = str(root / "q")

    def racing(path: Any = ".") -> Any:
        listed = str(path) in (place, os.path.realpath(place))
        by_check = sys._getframe(1).f_code.co_name == "_declared"
        return _Vanishing(path, "_old.json") if listed and by_check else SCANDIR(path)

    monkeypatch.setattr(os, "scandir", racing)
    agents = load_folder(root)
    monkeypatch.undo()
    assert not (root / "q/_old.json").exists()
    assert sorted([*agents["d"].questions, *agents["d"].scripted]) == ["filled", "ok"]
