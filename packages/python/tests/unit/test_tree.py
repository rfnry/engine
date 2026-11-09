import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Check, Choice, Score
from rfnry_engine.tree import load_agent, load_tree, validate_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"


def test_load_tree_finds_every_agent_with_its_kind() -> None:
    agents = load_tree(FIXTURES)
    assert {n: a.kind for n, a in agents.items()} == {
        "support": "generation",
        "billing": "generation",
        "triage": "decision",
        "frontdesk": "decision",
        "judge": "decision",
        "teacher": "generation",
    }


def test_generation_agent_carries_body_agents_tools_skills() -> None:
    support = load_tree(FIXTURES)["support"]
    assert support.body.startswith("You are the support desk")
    assert support.agents == ["triage"]
    assert list(support.skills) == ["escalate"]
    assert list(support.folder_tools) == ["lookup_order", "now"] and support.keep is None


def test_decision_agent_carries_typed_questions_and_route() -> None:
    agents = load_tree(FIXTURES)
    severity = agents["triage"].questions["severity"]
    assert isinstance(severity, Score) and len(severity.levels) == 4
    intent = agents["frontdesk"].questions["intent"]
    assert isinstance(intent, Choice) and set(intent.options) == {"billing", "technical", "other"}
    route = agents["frontdesk"].route
    assert route is not None and route.by == "intent" and route.min_confidence == 0.6


def test_validate_tree_accepts_the_fixture() -> None:
    validate_tree(load_tree(FIXTURES))


def test_single_agent_root(tmp_path: Path) -> None:
    (tmp_path / "solo").mkdir()
    (tmp_path / "solo" / "AGENT.md").write_text(
        "---\nname: solo\nkind: generation\nmodel: strong\n---\nHi\n"
    )
    assert list(load_tree(tmp_path / "solo")) == ["solo"]


def test_route_must_point_at_existing_agents(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "questions").mkdir()
    (tmp_path / "d" / "AGENT.md").write_text(
        "---\nname: a\nkind: decision\nmodel: judge\nroute: {by: q, to: {x: ghost}}\n---\n"
    )
    (tmp_path / "d" / "questions" / "q.json").write_text(CHOICE)
    with pytest.raises(ConfigurationError, match="ghost"):
        validate_tree(load_tree(tmp_path))


def test_question_with_one_option_is_refused(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "questions").mkdir()
    (tmp_path / "d" / "AGENT.md").write_text("---\nname: d\nkind: decision\nmodel: judge\n---\n")
    (tmp_path / "d" / "questions" / "q.json").write_text(
        '{"type": "choice", "instructions": "?", "options": {"x": "X"}}'
    )
    with pytest.raises(ConfigurationError, match="a choice needs at least two options"):
        load_agent(tmp_path / "d")


def test_an_agent_carries_the_map_of_its_folder() -> None:
    support = load_tree(FIXTURES)["support"]
    assert support.listing == (
        "knowledge/\n  plans.md — The three plans\nrules/\n  refunds.md — When a refund is owed"
    )


def test_a_bad_description_stops_the_tree_naming_the_file(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\nmodel: strong\n")
    (tmp_path / "a" / "notes.md").write_text("---\ndescription: [x\n---\n")
    with pytest.raises(ConfigurationError, match=r"^frontmatter: invalid YAML in .*notes\.md"):
        load_tree(tmp_path)


def _agent(
    root: Path,
    name: str,
    front: str,
    questions: dict[str, str] | None = None,
    tools: tuple[str, ...] = (),
) -> None:
    (root / name).mkdir(parents=True)
    listed = f"tools: [{', '.join(tools)}]\n" if tools else ""
    (root / name / "AGENT.md").write_text(f"---\nname: {name}\n{front}{listed}---\n")
    for qid, text in (questions or {}).items():
        (root / name / "questions").mkdir(exist_ok=True)
        (root / name / "questions" / f"{qid}.json").write_text(text)


CHOICE = '{"type": "choice", "instructions": "Which?", "options": {"x": "X", "y": "Y"}}'
CHECK = '{"type": "check", "instructions": "Is it?"}'


@pytest.mark.parametrize(
    ("front", "questions", "tools", "message"),
    [
        (
            "kind: decision\nmodel: judge\n",
            {"q": CHECK},
            ("t",),
            "decision agent 'a' cannot list tools:",
        ),
        (
            "kind: decision\nmodel: judge\nworkspace: true\n",
            {"q": CHECK},
            (),
            "decision agent 'a' cannot declare workspace: true",
        ),
        (
            "kind: generation\nmodel: strong\n",
            {"q": CHECK},
            (),
            "generation agent 'a' cannot declare questions/",
        ),
        (
            "kind: generation\nmodel: strong\nroute: {by: q, to: {x: a}}\n",
            {},
            (),
            "generation agent 'a' cannot declare route:",
        ),
    ],
)
def test_each_kind_holds_only_what_it_can_use(
    tmp_path: Path, front: str, questions: dict[str, str], tools: tuple[str, ...], message: str
) -> None:
    _agent(tmp_path, "a", front, questions, tools)
    with pytest.raises(ConfigurationError, match=f"^tree: {message}(;|$)"):
        load_agent(tmp_path / "a")


@pytest.mark.parametrize(
    ("front", "message"),
    [
        (
            "kind: decision\nmodel: judge\nroute: {by: urgent, to: {x: b}}\n",
            "route of 'a' must name a choice question",
        ),
        (
            "kind: decision\nmodel: judge\nroute: {by: q, to: {z: b}}\n",
            "route of 'a' maps unknown option 'z'",
        ),
        (
            "kind: decision\nmodel: judge\nagents: [ghost]\n",
            "agent 'a' names unknown agent 'ghost'",
        ),
    ],
)
def test_routes_and_agents_must_resolve(tmp_path: Path, front: str, message: str) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE, "urgent": CHECK})
    _agent(tmp_path, "b", "kind: generation\nmodel: strong\n")
    with pytest.raises(ConfigurationError, match=f"^tree: {message}(;|$)"):
        validate_tree(load_tree(tmp_path))


@pytest.mark.parametrize(
    ("front", "key"),
    [
        ("kind: generation\nmodel: strong\nagent: [b]\n", "agent"),
        ("kind: generation\nmodel: strong\nworkpsace: true\n", "workpsace"),
        (
            "kind: decision\nmodel: judge\nroute: {by: q, to: {x: b}, min_confidance: 0.9}\n",
            "route.min_confidance",
        ),
        ("kind: generation\nmodel: strong\ntools: [a]\ntool: [b]\n", "tool"),
    ],
)
def test_an_unknown_frontmatter_key_is_refused_by_name(
    tmp_path: Path, front: str, key: str
) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE} if "decision" in front else {})
    path = tmp_path / "a" / "AGENT.md"
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "a")
    said = str(refused.value)
    assert said.startswith(f"frontmatter: unknown key {key!r} in {path}")
    assert said.split(";")[0].endswith("AGENT.md")
    assert ("; check its spelling" if "." in key else "; it takes ") in said


def test_only_true_and_false_are_booleans_in_frontmatter(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\nmodel: strong\nworkspace: true\n")
    assert load_agent(tmp_path / "a").workspace is True
    _agent(tmp_path, "b", "kind: generation\nmodel: strong\nreads: [yes, on]\n")
    (tmp_path / "yes").mkdir()
    (tmp_path / "on").mkdir()
    assert list(load_tree(tmp_path)["b"].reads) == ["yes", "on"]


def test_an_unknown_key_in_a_question_is_refused_by_name(tmp_path: Path) -> None:
    typo = '{"type": "choice", "instructions": "Which?", "optoins": {"x": "X", "y": "Y"}}'
    _agent(tmp_path, "d", "kind: decision\nmodel: judge\n", {"q": typo})
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "d")
    path = tmp_path / "d" / "questions" / "q.json"
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
    _agent(
        tmp_path,
        "d",
        "kind: decision\nmodel: judge\n",
        {"team": json.dumps(choice), "scope": json.dumps(score), "creds": json.dumps(check)},
    )
    assert load_agent(tmp_path / "d").questions == {
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
    _agent(tmp_path, "d", "kind: decision\nmodel: judge\n", {"q": text})
    path = tmp_path / "d" / "questions" / "q.json"
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "d")
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
    _agent(tmp_path, "d", "kind: decision\nmodel: judge\n", {"q": text})
    with pytest.raises(ConfigurationError, match="^tree: invalid question "):
        load_agent(tmp_path / "d")


def test_a_decision_agent_may_have_no_questions_of_its_own(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\nmodel: judge\n")
    spec = load_agent(tmp_path / "a")
    assert spec.questions == {} and spec.scripted == {}


def test_a_delegate_target_needs_no_questions_of_its_own(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\nmodel: strong\nagents: [b, c]\n")
    _agent(tmp_path, "b", "kind: decision\nmodel: judge\n")
    _agent(tmp_path, "c", "kind: decision\nmodel: judge\n")
    validate_tree(load_tree(tmp_path))


def _router(root: Path, to: str) -> None:
    _agent(
        root,
        "a",
        f"kind: decision\nmodel: judge\nroute: {{by: q, to: {to}}}\n",
        {"q": CHOICE},
    )
    _agent(root, "b", "kind: decision\nmodel: judge\n")
    _agent(root, "g", "kind: generation\nmodel: strong\n")


def test_a_route_to_a_decision_agent_that_could_never_ask_anything_is_refused(
    tmp_path: Path,
) -> None:
    _router(tmp_path, "{x: b}")
    message = (
        "^tree: route of 'a' gives decision agent 'b' nothing to ask: it has no questions/; "
        "give it questions/, since a route hands it only the input$"
    )
    with pytest.raises(ConfigurationError, match=message):
        validate_tree(load_tree(tmp_path))


def test_a_route_entry_is_an_agent_s_name(tmp_path: Path) -> None:
    team = (
        '{"type": "choice", "instructions": "Which?", "options": {"technical": "T", "other": "O"}}'
    )
    _agent(
        tmp_path,
        "a",
        "kind: decision\nmodel: judge\nroute: {by: team, to: {technical: support}}\n",
        {"team": team},
    )
    _agent(tmp_path, "support", "kind: generation\nmodel: strong\n")
    agents = load_tree(tmp_path)
    validate_tree(agents)
    route = agents["a"].route
    assert route is not None and route.to == {"technical": "support"}
    (tmp_path / "a" / "AGENT.md").write_text(
        "---\nname: a\nkind: decision\nmodel: judge\n"
        "route: {by: team, to: {technical: {agent: support}}}\n---\n"
    )
    with pytest.raises(
        ConfigurationError, match="route.to.technical: input should be a valid string"
    ):
        load_tree(tmp_path)


@pytest.mark.parametrize(
    "kind", ["kind: generation\nmodel: strong\n", "kind: decision\nmodel: judge\n"]
)
def test_a_tasks_folder_in_an_agent_s_folder_is_refused_naming_its_first_file(
    tmp_path: Path, kind: str
) -> None:
    _agent(tmp_path, "support", kind)
    tasks = tmp_path / "support" / "tasks"
    tasks.mkdir()
    for name in ("b.md", "a.md", ".#a.md"):
        (tasks / name).write_text("Do it.\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: tasks are gone ({tmp_path}/support/tasks/a.md); make it a skill, or say it in the input"
    )
    for name in ("b.md", "a.md"):
        (tasks / name).unlink()
    (tasks / "a.md").mkdir()
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: tasks are gone ({tmp_path}/support/tasks); make it a skill, or say it in the input"
    )


def test_a_tasks_folder_holding_an_agent_md_is_refused_naming_it(tmp_path: Path) -> None:
    _agent(tmp_path, "support", "kind: generation\nmodel: strong\n")
    (tmp_path / "support" / "tasks").mkdir()
    (tmp_path / "support" / "tasks" / "AGENT.md").write_text("Not an agent: a task.\n")
    (tmp_path / "support" / "tasks" / "close.md").write_text("Close the ticket.\n")
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: tasks are gone ({tmp_path}/support/tasks/AGENT.md); make it a skill, or say it in the input"
    )


def test_a_tasks_folder_that_cannot_be_read_is_refused_naming_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _agent(tmp_path, "support", "kind: generation\nmodel: strong\n")
    tasks = tmp_path / "support" / "tasks"
    tasks.mkdir()
    (tasks / "triage.md").write_text("Triage it.\n")
    tasks.chmod(0)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ConfigurationError) as caught:
            load_tree(tmp_path)
    finally:
        tasks.chmod(0o755)
    assert str(caught.value) == (
        f"tree: tasks are gone ({tasks}); make it a skill, or say it in the input"
    )
    assert [m for m in caplog.messages if "cannot read" in m] == []


def test_a_tasks_folder_outside_any_agent_is_content(tmp_path: Path) -> None:
    _agent(tmp_path, "support", "kind: generation\nmodel: strong\nreads: [shared]\n")
    (tmp_path / "shared" / "tasks").mkdir(parents=True)
    (tmp_path / "shared" / "tasks" / "x.md").write_text("Do it.\n")
    (tmp_path / "tools" / "kit" / "tasks").mkdir(parents=True)
    (tmp_path / "tools" / "kit" / "AGENT.md").write_text("---\nname: kit\n---\n")
    (tmp_path / "tools" / "kit" / "tasks" / "y.md").write_text("Do it.\n")
    agents = load_tree(tmp_path)
    assert list(agents) == ["support"]
    listing = agents["support"].listing.splitlines()
    assert listing[listing.index("../shared/") :] == ["../shared/", "  tasks/", "    x.md"]


def _shelf(root: Path, coding: str = "reads: [books, writing]\n") -> Path:
    _agent(root, "coding", f"kind: generation\nmodel: strong\n{coding}")
    _agent(
        root,
        "writing",
        "kind: generation\nmodel: strong\ndescription: Writes\n",
    )
    (root / "writing" / "voice.md").write_text("---\ndescription: The voice\n---\nWarm.\n")
    (root / "books" / "python").mkdir(parents=True)
    (root / "books" / "python" / "async.md").write_text("---\ndescription: Async\n---\nAwait.\n")
    return root


def test_an_agent_reads_the_folders_it_names(tmp_path: Path) -> None:
    agents = load_tree(_shelf(tmp_path))
    coding = agents["coding"]
    assert coding.reads == {
        "books": (tmp_path / "books").resolve(),
        "writing": (tmp_path / "writing").resolve(),
    }
    assert coding.listing == (
        "../books/\n  python/\n    async.md — Async\n../writing/\n  voice.md — The voice"
    )
    assert agents["writing"].reads == {} and "books" not in agents


@pytest.mark.parametrize(
    ("front", "message"),
    [
        (
            "kind: generation\nmodel: strong\nreads: [ghost]\n",
            "agent 'coding' reads unknown folder 'ghost'",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [coding]\n",
            "agent 'coding' reads 'coding', its own folder or one that holds it",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [.hidden]\n",
            "agent 'coding' reads '.hidden', which is not a path from the root",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [outside]\n",
            "agent 'coding' reads 'outside', which resolves outside the root to .*/elsewhere",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [alias]\n",
            "agent 'coding' reads unknown folder 'alias'",
        ),
        (
            "kind: decision\nmodel: judge\nreads: [books]\n",
            "decision agent 'coding' cannot declare reads:",
        ),
    ],
)
def test_reads_names_a_readable_folder_of_the_root(
    tmp_path: Path, front: str, message: str
) -> None:
    root = tmp_path / "root"
    _agent(root, "coding", front, {"q": CHECK} if "decision" in front else None)
    (root / "books").mkdir()
    (root / ".hidden").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (root / "outside").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    (root / "alias").symlink_to(root / "books", target_is_directory=True)
    with pytest.raises(ConfigurationError, match=f"^tree: {message}(;|$)"):
        load_tree(root)


@pytest.mark.parametrize(
    ("front", "message"),
    [
        (
            "kind: decision\nmodel: judge\nreads: [ghost]\n",
            "decision agent 'coding' cannot declare reads:",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [coding, ghost]\n",
            "agent 'coding' reads unknown folder 'ghost'",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [outside, coding]\n",
            "agent 'coding' reads 'outside', which resolves outside the root to .*/elsewhere",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [books, books/python, coding]\n",
            "agent 'coding' reads 'coding', its own folder or one that holds it",
        ),
    ],
)
def test_reads_that_breaks_two_rules_is_refused_for_the_rule_checked_first(
    tmp_path: Path, front: str, message: str
) -> None:
    root = tmp_path / "root"
    _agent(root, "coding", front, {"q": CHECK} if "decision" in front else None)
    (root / "books" / "python").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (root / "outside").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ConfigurationError, match=f"^tree: {message}(;|$)"):
        load_tree(root)


def test_a_single_agent_root_reads_nothing(tmp_path: Path) -> None:
    _agent(tmp_path, "solo", "kind: generation\nmodel: strong\nreads: [books]\n")
    (tmp_path / "solo" / "books").mkdir()
    with pytest.raises(
        ConfigurationError, match="agent 'solo' reads 'books', inside its own folder"
    ):
        load_tree(tmp_path / "solo")


def test_reads_needs_the_agent_s_folder_inside_the_root_not_a_link_to_elsewhere(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "books").mkdir(parents=True)
    _agent(
        tmp_path / "elsewhere",
        "coding",
        "kind: generation\nmodel: strong\nreads: [books]\n",
    )
    (root / "coding").symlink_to(tmp_path / "elsewhere" / "coding", target_is_directory=True)
    message = r"^tree: agent 'coding' declares reads: but its folder is a link to "
    with pytest.raises(ConfigurationError, match=message):
        load_tree(root)
    (tmp_path / "elsewhere" / "coding" / "AGENT.md").write_text(
        "---\nname: coding\nkind: generation\nmodel: strong\n---\n"
    )
    assert load_tree(root)["coding"].reads == {}


def test_an_unreadable_folder_is_warned_about_once_per_loaded_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _agent(tmp_path, "a", "kind: generation\nmodel: strong\npreload: true\n")
    _agent(tmp_path, "b", "kind: generation\nmodel: strong\nreads: [a/kb]\n")
    for name in ("a/kb/locked/x.md", "a/notes/locked/y.md", "a/kb/z.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x\n")
    kb, notes = (tmp_path / "a/kb/locked").resolve(), (tmp_path / "a/notes/locked").resolve()
    real = os.scandir

    def scandir(path: object = ".") -> object:
        if Path(str(path)).resolve() in (kb, notes):
            raise PermissionError(13, "denied", str(path))
        return real(path)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "scandir", scandir)
    warned = [
        f"tree: cannot read {kb}: [Errno 13] denied: '{kb}'; it is left out",
        f"tree: cannot read {notes}: [Errno 13] denied: '{notes}'; it is left out",
    ]
    with caplog.at_level(logging.WARNING):
        assert list(load_tree(tmp_path)) == ["a", "b"]
    assert [m for m in caplog.messages if "cannot read" in m] == warned
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        load_tree(tmp_path)
    assert [m for m in caplog.messages if "cannot read" in m] == warned


@pytest.mark.parametrize(
    ("agent", "link"),
    [(".", ".lib"), ("a", "lib")],
)
def test_an_agent_s_own_tools_linked_to_a_hidden_folder_inside_it_is_refused_at_load(
    tmp_path: Path, agent: str, link: str
) -> None:
    folder = tmp_path / agent
    if agent == ".":
        (folder / "AGENT.md").write_text("---\nname: root\nkind: generation\nmodel: strong\n---\n")
    else:
        _agent(tmp_path, agent, "kind: generation\nmodel: strong\n")
    (folder / ".lib").mkdir()
    (folder / ".lib/notify.json").write_text(
        '{"description": "Do it", "input_schema": {"type": "object"}}'
    )
    (folder / ".lib/notify.py").write_text("def run(args, call):\n    return {}\n")
    if link == "lib":
        (folder / "lib").symlink_to(".lib", target_is_directory=True)
    (folder / "tools").symlink_to(link, target_is_directory=True)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == (
        f"tree: {folder / 'tools'} links to {folder.resolve() / '.lib'}, a hidden folder the agent's fingerprint leaves out; move the folder to a visible name"
    )


def test_an_unreadable_file_is_warned_about_once_per_loaded_tree_and_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for root in (tmp_path / "with", tmp_path / "without"):
        _agent(root, "a", "kind: generation\nmodel: strong\npreload: true\n")
        _agent(root, "b", "kind: generation\nmodel: strong\nreads: [a/kb]\n")
        (root / "a/kb").mkdir()
        (root / "a/kb/open.md").write_text("---\ndescription: Open\n---\nx\n")
    locked = (tmp_path / "with/a/kb/notes.md").resolve()
    locked.write_text("---\ndescription: Locked\n---\ny\n")
    locked.chmod(0)
    try:
        warned = [
            f"folders: cannot read {locked}: [Errno 13] Permission denied: '{locked}'; it is left out"
        ]
        for _ in range(2):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                agents = load_tree(tmp_path / "with")
            assert [m for m in caplog.messages if "cannot read" in m] == warned
        alike = load_tree(tmp_path / "without")
        assert {name: spec.fingerprint for name, spec in agents.items()} == {
            name: spec.fingerprint for name, spec in alike.items()
        }
        assert "notes.md" not in agents["a"].listing + agents["b"].listing
        assert "open.md — Open" in agents["a"].listing
    finally:
        locked.chmod(0o644)


@pytest.mark.parametrize("kind", ["file", "folder"])
def test_a_name_that_is_not_utf8_is_refused_naming_its_path(tmp_path: Path, kind: str) -> None:
    _agent(tmp_path, "a", "kind: generation\nmodel: strong\n")
    named = os.path.join(os.fsencode(tmp_path / "a"), b"caf\xe9")
    if kind == "file":
        with open(named, "wb") as file:
            file.write(b"menu")
    else:
        os.mkdir(named)
    shown = repr(os.fsdecode(named))
    with pytest.raises(ConfigurationError) as caught:
        load_tree(tmp_path)
    assert str(caught.value) == f"tree: the name of {shown} is not UTF-8; rename it"


DECLARED_FILES = {
    "a/AGENT.md": "---\nname: a\nkind: generation\nmodel: strong\n---\nbody\n",
    "a/tools/x.json": '{"description": "X", "input_schema": {"type": "object"}}',
    "a/tools/x.py": "def run(args, call):\n    return 'x'\n",
    "a/tools/_shared.py": "VALUE = 1\n",
    "d/AGENT.md": "---\nname: d\nkind: decision\nmodel: judge\n---\n",
    "d/questions/ok.json": '{"type": "check", "instructions": "ok?"}',
    "d/questions/filled.json": '{"type": "check"}',
    "d/questions/filled.py": "def run(input, call):\n    return {'instructions': 'q?'}\n",
}


@pytest.mark.parametrize(
    ("locked", "module"),
    [
        ("a/AGENT.md", "tree"),
        ("d/questions/ok.json", "tree"),
        ("a/tools/x.json", "tools/folder"),
        ("a/tools/x.py", "tools/folder"),
        ("a/tools/_shared.py", "tools/folder"),
        ("d/questions/filled.py", "tools/folder"),
    ],
)
def test_a_declared_file_that_cannot_be_read_is_refused_naming_it(
    tmp_path: Path, locked: str, module: str
) -> None:
    for rel, text in DECLARED_FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert sorted(load_tree(tmp_path)) == ["a", "d"]
    path = tmp_path / locked
    path.chmod(0)
    try:
        with pytest.raises(ConfigurationError) as caught:
            load_tree(tmp_path)
    finally:
        path.chmod(0o644)
    assert (
        str(caught.value)
        == f"{module}: cannot read {path.resolve()}: Permission denied; make it readable to the engine"
    )


@pytest.mark.parametrize("locked", ["a/skills/brief.md", "a/skills/deep/SKILL.md"])
def test_an_unreadable_skill_is_refused_at_load(
    tmp_path: Path, locked: str, caplog: pytest.LogCaptureFixture
) -> None:
    for rel, text in {
        "a/AGENT.md": "---\nname: a\nkind: generation\nmodel: strong\n---\nbody\n",
        "a/skills/brief.md": "---\ndescription: Brief\n---\nBe brief.\n",
        "a/skills/deep/SKILL.md": "---\ndescription: Deep\n---\nGo deep.\n",
        "a/skills/deep/notes.md": "Notes.\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    notes = (tmp_path / "a/skills/deep/notes.md").resolve()
    path = (tmp_path / locked).resolve()
    notes.chmod(0)
    try:
        with caplog.at_level(logging.WARNING):
            assert sorted(load_tree(tmp_path)["a"].skills) == ["brief", "deep"]
        assert [m for m in caplog.messages if "cannot read" in m] == [
            f"folders: cannot read {notes}: [Errno 13] Permission denied: '{notes}'; it is left out"
        ]
        path.chmod(0)
        try:
            with pytest.raises(ConfigurationError) as caught:
                load_tree(tmp_path)
        finally:
            path.chmod(0o644)
    finally:
        notes.chmod(0o644)
    assert str(caught.value) == (
        f"tree: cannot read {path}: Permission denied; make it readable to the engine"
    )


@pytest.mark.parametrize("mode", [0, 0o600], ids=["unreadable", "unsearchable"])
@pytest.mark.parametrize("folder", ["a/tools", "d/questions", "a/skills", "a/skills/deep"])
def test_a_declared_folder_that_cannot_be_read_or_searched_is_refused_naming_it(
    tmp_path: Path, folder: str, mode: int, caplog: pytest.LogCaptureFixture
) -> None:
    for rel, text in {
        **DECLARED_FILES,
        "a/skills/brief.md": "---\ndescription: Brief\n---\nBe brief.\n",
        "a/skills/deep/SKILL.md": "---\ndescription: Deep\n---\nGo deep.\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert sorted(load_tree(tmp_path)["a"].skills) == ["brief", "deep"]
    place = tmp_path / folder
    place.chmod(mode)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ConfigurationError) as caught:
            load_tree(tmp_path)
    finally:
        place.chmod(0o755)
    assert str(caught.value) == (
        f"tree: cannot read {place}: Permission denied; make it readable to the engine"
    )
    assert [m for m in caplog.messages if "cannot read" in m] == []


@pytest.mark.parametrize("folder", ["a/skills", "a/skills/deep"])
def test_an_unreadable_skills_folder_is_refused_before_an_agent_above_it_walks_it(
    tmp_path: Path, folder: str, caplog: pytest.LogCaptureFixture
) -> None:
    for rel, text in {
        "AGENT.md": "---\nname: top\nkind: generation\nmodel: strong\n---\nbody\n",
        "a/AGENT.md": "---\nname: a\nkind: generation\nmodel: strong\n---\nbody\n",
        "a/skills/deep/SKILL.md": "---\ndescription: Deep\n---\nGo deep.\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    place = tmp_path / folder
    place.chmod(0)
    try:
        with caplog.at_level(logging.WARNING), pytest.raises(ConfigurationError) as caught:
            load_tree(tmp_path)
    finally:
        place.chmod(0o755)
    assert str(caught.value) == (
        f"tree: cannot read {place.resolve()}: Permission denied; make it readable to the engine"
    )
    assert [m for m in caplog.messages if "cannot read" in m] == []


@pytest.mark.parametrize("folder", ["d/skills", "d/skills/deep"])
def test_an_unreadable_skills_folder_of_a_decision_agent_is_left_out_with_a_warning(
    tmp_path: Path, folder: str, caplog: pytest.LogCaptureFixture
) -> None:
    for rel, text in {
        **DECLARED_FILES,
        "d/skills/deep/SKILL.md": "---\ndescription: Deep\n---\nGo deep.\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    place = (tmp_path / folder).resolve()
    place.chmod(0)
    try:
        with caplog.at_level(logging.WARNING):
            assert sorted(load_tree(tmp_path)) == ["a", "d"]
    finally:
        place.chmod(0o755)
    assert [m for m in caplog.messages if "cannot read" in m] == [
        f"folders: cannot read {place}: [Errno 13] Permission denied: '{place}'; it is left out"
    ]


@pytest.mark.parametrize(
    "linked", ["a/skills/outside", "a/skills/outside.md", "a/skills/deep/SKILL.md"]
)
def test_a_link_under_skills_that_no_skill_is_loaded_from_is_never_read(
    tmp_path: Path, linked: str
) -> None:
    root, out = tmp_path / "root", tmp_path / "out"
    for rel, text in {
        "a/AGENT.md": "---\nname: a\nkind: generation\nmodel: strong\n---\nbody\n",
        "a/skills/brief.md": "---\ndescription: Brief\n---\nBe brief.\n",
        "a/skills/deep/notes.md": "Notes.\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (out / "skill").mkdir(parents=True)
    (out / "skill" / "SKILL.md").write_text("---\ndescription: Out\n---\nout\n")
    target = out / "skill" if linked.endswith("outside") else out / "skill" / "SKILL.md"
    (root / linked).symlink_to(target)
    assert sorted(load_tree(root)["a"].skills) == ["brief"]
    target.chmod(0)
    try:
        assert sorted(load_tree(root)["a"].skills) == ["brief"]
    finally:
        target.chmod(0o755)


def test_a_skills_folder_linked_out_of_the_agent_s_folder_is_refused(tmp_path: Path) -> None:
    root, out = tmp_path / "root", tmp_path / "out"
    a, target = root / "a", out.resolve()
    a.mkdir(parents=True)
    (a / "AGENT.md").write_text("---\nname: a\nkind: generation\nmodel: strong\n---\nbody\n")
    (out / "deep").mkdir(parents=True)
    (out / "deep" / "SKILL.md").write_text("---\ndescription: Out\n---\nout\n")
    (a / "skills").symlink_to(out)
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    assert str(caught.value) == (
        f"tree: {a}/skills links to {target}, outside the agent's folder; move the folder into the agent's folder, or link the agent's whole folder"
    )


SKILLED = "---\nname: a\nkind: generation\nmodel: strong\n---\nbody\n"
SKILL = "---\ndescription: S\n---\nskill\n"


@pytest.mark.parametrize(
    ("link", "target", "skill"),
    [
        ("skills/x.md", ".hidden/x.md", "x"),
        ("skills/deep/SKILL.md", ".hidden/SKILL.md", "deep"),
        ("skills/x.md", "__pycache__/x.md", "x"),
    ],
)
def test_a_skill_file_linked_inside_the_agent_s_folder_is_loaded_and_so_must_be_readable(
    tmp_path: Path, link: str, target: str, skill: str
) -> None:
    agent = tmp_path / "a"
    (agent / target).parent.mkdir(parents=True)
    (agent / "AGENT.md").write_text(SKILLED)
    (agent / target).write_text(SKILL)
    (agent / link).parent.mkdir(parents=True, exist_ok=True)
    (agent / link).symlink_to(agent / target)
    assert sorted(load_tree(tmp_path)["a"].skills) == [skill]
    (agent / target).chmod(0)
    try:
        with pytest.raises(ConfigurationError) as caught:
            load_tree(tmp_path)
    finally:
        (agent / target).chmod(0o644)
    shown = tmp_path.resolve() / "a" / link
    assert str(caught.value) == (
        f"tree: cannot read {shown}: Permission denied; make it readable to the engine"
    )


@pytest.mark.parametrize(
    ("link", "target", "made", "locked"),
    [
        ("skills/deep", "lib/deep", "lib/deep/SKILL.md", "lib/deep/SKILL.md"),
        ("skills/deep", "lib/deep", "lib/deep/SKILL.md", "lib/deep"),
        ("skills", "lib/sk", "lib/sk/x.md", "lib/sk/x.md"),
    ],
    ids=["folder-link-skill-locked", "folder-link-locked", "skills-link-skill-locked"],
)
def test_a_folder_link_under_skills_is_never_entered_so_no_skill_is_read_through_it(
    tmp_path: Path, link: str, target: str, made: str, locked: str
) -> None:
    agent = tmp_path / "a"
    (agent / made).parent.mkdir(parents=True)
    (agent / "AGENT.md").write_text(SKILLED)
    (agent / made).write_text(SKILL)
    (agent / link).parent.mkdir(parents=True, exist_ok=True)
    (agent / link).symlink_to(agent / target)
    assert load_tree(tmp_path)["a"].skills == {}
    (agent / locked).chmod(0)
    try:
        assert load_tree(tmp_path)["a"].skills == {}
    finally:
        (agent / locked).chmod(0o755)


def test_an_agent_md_inside_skills_is_never_an_agent(tmp_path: Path) -> None:
    for rel, text in {
        "a/AGENT.md": SKILLED,
        "a/skills/brief.md": SKILL,
        "a/skills/sub/AGENT.md": "---\nname: sub\nkind: generation\nmodel: strong\n---\nbody\n",
        "a/skills/sub/SKILL.md": SKILL,
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert sorted(load_tree(tmp_path)["a"].skills) == ["brief"]
    locked = tmp_path / "a/skills/sub/SKILL.md"
    locked.chmod(0)
    try:
        agents = load_tree(tmp_path)
    finally:
        locked.chmod(0o644)
    assert sorted(agents) == ["a"] and sorted(agents["a"].skills) == ["brief"]


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


@pytest.mark.parametrize(
    ("folder", "name"),
    [
        ("a/tools", ".x.py.swp"),
        ("d/questions", "_old.json"),
        ("a/skills", "~brief.md"),
        ("a/skills/deep", ".SKILL.md.swp"),
    ],
)
def test_a_file_removed_while_its_declared_folder_is_listed_is_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str, name: str
) -> None:
    for rel, text in {
        **DECLARED_FILES,
        "a/skills/brief.md": "---\ndescription: Brief\n---\nBe brief.\n",
        "a/skills/deep/SKILL.md": "---\ndescription: Deep\n---\nGo deep.\n",
        f"{folder}/{name}": "a temporary file\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    place = str(tmp_path / folder)

    def racing(path: Any = ".") -> Any:
        listed = str(path) in (place, os.path.realpath(place))
        by_check = sys._getframe(1).f_code.co_name == "_declared"
        return _Vanishing(path, name) if listed and by_check else SCANDIR(path)

    monkeypatch.setattr(os, "scandir", racing)
    agents = load_tree(tmp_path)
    monkeypatch.undo()
    assert not (tmp_path / folder / name).exists()
    assert sorted(agents["a"].skills) == ["brief", "deep"]
    assert sorted(agents["a"].folder_tools) == ["x"]
    assert sorted([*agents["d"].questions, *agents["d"].scripted]) == ["filled", "ok"]
