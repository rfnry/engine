import json
from pathlib import Path

import pytest

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.models import Check, Choice, Score
from rfnry_engine.tree import load_agent, load_tree, validate_tree

FIXTURES = Path(__file__).parents[1] / "fixtures" / "agents"
EXAMPLES = Path(__file__).parents[4] / "examples"


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


def test_generation_agent_carries_body_agents_tools_tasks() -> None:
    support = load_tree(FIXTURES)["support"]
    assert support.body.startswith("You are the support desk")
    assert support.agents == ["triage"]
    assert support.tasks == {"triage": ""}
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
        "knowledge/\n  plans.md — The three plans\nrules/\n  refunds.md — When a refund is owed\n"
        "skills/\n  escalate.md — Hand a conversation to a person"
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
    with pytest.raises(ConfigurationError, match=f"^tree: {message}"):
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
    with pytest.raises(ConfigurationError, match=f"^tree: {message}"):
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


def test_every_example_tree_loads_with_its_checks_criteria() -> None:
    roots = sorted(EXAMPLES.glob("*/agents"))
    assert roots
    for root in roots:
        agents = load_tree(root)
        validate_tree(agents)
        for spec in agents.values():
            for qid, question in spec.questions.items():
                if not isinstance(question, Check):
                    continue
                written = json.loads((spec.root / "questions" / f"{qid}.json").read_text())
                assert (question.yes is not None) == ("yes" in written), (root, qid)
                assert (question.no is not None) == ("no" in written), (root, qid)


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
    assert spec.questions == {} and spec.tasks == {} and spec.scripted == {}


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
    (root / "g" / "tasks").mkdir()
    (root / "g" / "tasks" / "billing.md").write_text("Answer the billing question.\n")


@pytest.mark.parametrize("to", ["{x: b}", "{x: {agent: b}}"])
def test_a_route_to_a_decision_agent_that_could_never_ask_anything_is_refused(
    tmp_path: Path, to: str
) -> None:
    _router(tmp_path, to)
    message = (
        "^tree: route of 'a' gives decision agent 'b' nothing to ask: it has no questions/; "
        "give it questions/, since a route hands it only the input$"
    )
    with pytest.raises(ConfigurationError, match=message):
        validate_tree(load_tree(tmp_path))


def test_a_route_names_only_a_task_its_agent_has(tmp_path: Path) -> None:
    _router(tmp_path, "{x: {agent: g, task: tone}}")
    message = (
        "^tree: route of 'a' points at agent 'g', which has no task 'tone'; its tasks are billing$"
    )
    with pytest.raises(ConfigurationError, match=message):
        validate_tree(load_tree(tmp_path))


def test_a_route_entry_is_an_agent_or_a_generation_agent_with_one_of_its_tasks(
    tmp_path: Path,
) -> None:
    _router(tmp_path, "{x: g, y: {agent: g, task: billing}}")
    agents = load_tree(tmp_path)
    validate_tree(agents)
    route = agents["a"].route
    assert route is not None
    assert {option: (t.agent, t.task) for option, t in route.to.items()} == {
        "x": ("g", None),
        "y": ("g", "billing"),
    }
    named = load_tree(FIXTURES)["frontdesk"].route
    assert named is not None
    assert {option: (t.agent, t.task) for option, t in named.to.items()} == {
        "billing": ("billing", None),
        "technical": ("support", None),
    }


def test_a_route_entry_with_an_unknown_key_names_it(tmp_path: Path) -> None:
    _router(tmp_path, "{x: {agent: b, tsak: grade}}")
    message = "^frontmatter: unknown key 'route.to.x.tsak' in "
    with pytest.raises(ConfigurationError, match=message):
        load_agent(tmp_path / "a")


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
            "agent 'coding' reads 'coding', its own folder",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [.hidden]\n",
            "agent 'coding' reads '.hidden', which is not a path from the root",
        ),
        (
            "kind: generation\nmodel: strong\nreads: [outside]\n",
            "agent 'coding' reads 'outside', which resolves outside the root",
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
    with pytest.raises(ConfigurationError, match=f"^tree: {message}"):
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
