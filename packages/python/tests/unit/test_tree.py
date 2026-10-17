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
    assert support.tasks == ["triage"]
    contract = support.tools["lookup_order"]
    assert contract.input is not None and contract.input["order_id"].required is True
    assert contract.description is not None and "Returns the order as JSON." in contract.description
    adoption = support.tools["now"]
    assert adoption.description is None and adoption.input is None


def test_a_tool_file_is_a_contract_or_a_name(tmp_path: Path) -> None:
    (tmp_path / "a" / "tools").mkdir(parents=True)
    (tmp_path / "a" / "AGENT.md").write_text("---\nkind: generation\n---\n")
    (tmp_path / "a" / "tools" / "x.md").write_text("---\ninput: {q: {type: string}}\n---\n")
    with pytest.raises(ConfigurationError, match="description"):
        load_agent(tmp_path / "a")


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
    (tmp_path / "solo" / "AGENT.md").write_text("---\nkind: generation\n---\nHi\n")
    assert list(load_tree(tmp_path / "solo")) == ["solo"]


def test_the_folder_names_the_agent(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "AGENT.md").write_text("---\nkind: generation\n---\n")
    assert load_agent(tmp_path / "a").name == "a"
    (tmp_path / "a" / "AGENT.md").write_text("---\nname: a\nkind: generation\n---\n")
    with pytest.raises(
        ConfigurationError, match="unknown key 'name'.*; its folder or file names it"
    ):
        load_agent(tmp_path / "a")


def test_route_must_point_at_existing_agents(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "questions").mkdir()
    (tmp_path / "d" / "AGENT.md").write_text(
        "---\nkind: decision\nroute: {by: q, to: {x: ghost}}\n---\n"
    )
    (tmp_path / "d" / "questions" / "q.md").write_text(
        "---\ntype: choice\noptions: {x: X, y: Y}\n---\nWhich?\n"
    )
    with pytest.raises(ConfigurationError, match="ghost"):
        validate_tree(load_tree(tmp_path))


def test_question_with_one_option_is_refused(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "questions").mkdir()
    (tmp_path / "d" / "AGENT.md").write_text("---\nkind: decision\n---\n")
    (tmp_path / "d" / "questions" / "q.md").write_text(
        "---\ntype: choice\noptions: {x: X}\n---\n?\n"
    )
    with pytest.raises(ConfigurationError):
        load_agent(tmp_path / "d")


def test_an_agent_carries_the_map_of_its_folder() -> None:
    support = load_tree(FIXTURES)["support"]
    assert support.listing == (
        "knowledge/\n  plans.md — The three plans\nrules/\n  refunds.md — When a refund is owed\n"
        "skills/\n  escalate.md — Hand a conversation to a person"
    )


def test_a_bad_description_stops_the_tree_naming_the_file(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\n")
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
    (root / name / "AGENT.md").write_text(f"---\n{front}---\n")
    for qid, text in (questions or {}).items():
        (root / name / "questions").mkdir(exist_ok=True)
        (root / name / "questions" / f"{qid}.md").write_text(text)
    if tools:
        (root / name / "TOOLS.md").write_text("---\n" + "".join(f"{t}:\n" for t in tools) + "---\n")


CHOICE = "---\ntype: choice\noptions: {x: X, y: Y}\n---\nWhich?\n"
CHECK = "---\ntype: check\n---\nIs it?\n"


@pytest.mark.parametrize(
    ("front", "questions", "tools", "message"),
    [
        ("kind: decision\n", {"q": CHECK}, ("t",), "decision agent 'a' cannot declare tools"),
        (
            "kind: decision\nworkspace: true\n",
            {"q": CHECK},
            (),
            "decision agent 'a' cannot declare workspace: true",
        ),
        ("kind: generation\n", {"q": CHECK}, (), "generation agent 'a' cannot declare questions/"),
        (
            "kind: generation\nroute: {by: q, to: {x: a}}\n",
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
            "kind: decision\nroute: {by: urgent, to: {x: b}}\n",
            "route of 'a' must name a choice question",
        ),
        ("kind: decision\nroute: {by: q, to: {z: b}}\n", "route of 'a' maps unknown option 'z'"),
        ("kind: decision\nagents: [ghost]\n", "agent 'a' names unknown agent 'ghost'"),
    ],
)
def test_routes_and_agents_must_resolve(tmp_path: Path, front: str, message: str) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE, "urgent": CHECK})
    _agent(tmp_path, "b", "kind: generation\n")
    with pytest.raises(ConfigurationError, match=f"^tree: {message}(;|$)"):
        validate_tree(load_tree(tmp_path))


@pytest.mark.parametrize(
    ("front", "tool", "key", "where"),
    [
        ("kind: generation\nagent: [b]\n", None, "agent", "AGENT.md"),
        ("kind: generation\nworkpsace: true\n", None, "workpsace", "AGENT.md"),
        (
            "kind: decision\nroute: {by: q, to: {x: b}, min_confidance: 0.9}\n",
            None,
            "route.min_confidance",
            "AGENT.md",
        ),
        (
            "kind: generation\n",
            "---\ndescription: T\nsummary: S\n---\n",
            "summary",
            "t.md",
        ),
        (
            "kind: generation\n",
            "---\ndescription: T\ninput:\n  q: {type: string, requried: true}\n---\n",
            "input.q.requried",
            "t.md",
        ),
    ],
)
def test_an_unknown_frontmatter_key_is_refused_by_name(
    tmp_path: Path, front: str, tool: str | None, key: str, where: str
) -> None:
    _agent(tmp_path, "a", front, {"q": CHOICE} if "decision" in front else {})
    if tool is not None:
        (tmp_path / "a" / "tools").mkdir()
        (tmp_path / "a" / "tools" / "t.md").write_text(tool)
    path = tmp_path / "a" / ("tools/t.md" if tool is not None else "AGENT.md")
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "a")
    said = str(refused.value)
    assert said.startswith(f"frontmatter: unknown key {key!r} in {path}")
    assert said.split(";")[0].endswith(where)
    assert ("; check its spelling" if "." in key else "; it takes ") in said


def test_only_true_and_false_are_booleans_in_frontmatter(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\nworkspace: true\n")
    (tmp_path / "a" / "tools").mkdir()
    (tmp_path / "a" / "tools" / "t.md").write_text(
        "---\ndescription: T\ninput:\n"
        "  q: {type: string, required: true}\n"
        "  r: {type: string, required: yes}\n"
        "  s: {type: string, required: False}\n"
        "---\n"
    )
    spec = load_agent(tmp_path / "a")
    assert spec.workspace is True and spec.tools["t"].input is not None
    assert {k: f.required for k, f in spec.tools["t"].input.items()} == {
        "q": True,
        "r": True,
        "s": False,
    }
    choice = "---\ntype: choice\noptions: {yes: Agrees, no: Disagrees, off: Unrelated}\n---\n?\n"
    _agent(tmp_path, "d", "kind: decision\n", {"q": choice})
    question = load_agent(tmp_path / "d").questions["q"]
    assert isinstance(question, Choice)
    assert question.options == {"yes": "Agrees", "no": "Disagrees", "off": "Unrelated"}


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
                text = (spec.root / "questions" / f"{qid}.md").read_text(encoding="utf-8")
                assert (question.yes is not None) == ("\nyes:" in text), (root, qid)
                assert (question.no is not None) == ("\nno:" in text), (root, qid)


def test_an_unknown_key_in_a_question_is_refused_by_name(tmp_path: Path) -> None:
    typo = "---\ntype: choice\noptoins: {x: X, y: Y}\n---\nWhich?\n"
    _agent(tmp_path, "d", "kind: decision\n", {"q": typo})
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "d")
    path = tmp_path / "d" / "questions" / "q.md"
    assert str(refused.value) == (
        f"frontmatter: unknown key 'optoins' in {path}; it takes type, instructions, options, levels, yes, no"
    )


def test_a_question_takes_structure_through_its_frontmatter(tmp_path: Path) -> None:
    choice = (
        "---\ntype: choice\n"
        "instructions:\n  question: Which team handles `ticket`?\n  ticket: {id: 7, vip: true}\n"
        "options:\n"
        "  billing:\n    what: Charges\n    examples: [charged twice, yes, on]\n"
        "  orders: [Delivery, Returns]\n"
        "  other:\n"
        "  legal: Contracts\n"
        "---\n"
    )
    score = (
        "---\ntype: score\nlevels:\n  - {summary: One change, signals: [a single fix]}\n"
        "  - [Several]\n  - Many\n---\nHow focused is the change?\n"
    )
    check = (
        "---\ntype: check\ninstructions:\n  - Does `message` ask for a credential?\n  - message\n"
        "yes: {what: Asks for a password, examples: [Reply with your PIN]}\n"
        "no: [No credential is requested]\n---\n"
    )
    _agent(tmp_path, "d", "kind: decision\n", {"team": choice, "scope": score, "creds": check})
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
        (
            "---\ntype: check\ninstructions:\n  question: Is it?\n---\nIs it?\n",
            "gives instructions both in its frontmatter and in its body",
        ),
        ("---\ntype: check\n---\n\n", "gives no instructions, in its body or its frontmatter"),
        (
            "---\ntype: check\ninstructions: {}\n---\n",
            "gives no instructions, in its body or its frontmatter",
        ),
    ],
)
def test_a_question_has_its_instructions_in_its_frontmatter_or_its_body(
    tmp_path: Path, text: str, message: str
) -> None:
    _agent(tmp_path, "d", "kind: decision\n", {"q": text})
    path = tmp_path / "d" / "questions" / "q.md"
    with pytest.raises(ConfigurationError) as refused:
        load_agent(tmp_path / "d")
    assert str(refused.value).startswith(f"tree: question {path} {message}; ")


@pytest.mark.parametrize(
    "text",
    [
        "---\ntype: check\ninstructions: Is it?\n---\n",
        "---\ntype: score\nlevels: [low, ~]\n---\nHow?\n",
        "---\ntype: score\nlevels: [low, 2]\n---\nHow?\n",
        "---\ntype: choice\noptions: {a: A, b: 2026-01-01}\n---\nWhich?\n",
    ],
)
def test_structure_in_frontmatter_is_an_object_or_a_list_of_json(tmp_path: Path, text: str) -> None:
    _agent(tmp_path, "d", "kind: decision\n", {"q": text})
    with pytest.raises(ConfigurationError, match="^frontmatter: invalid question "):
        load_agent(tmp_path / "d")


def test_a_decision_agent_may_have_no_questions_of_its_own(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n")
    spec = load_agent(tmp_path / "a")
    assert spec.questions == {} and spec.tasks == [] and spec.task_questions == {}


def test_a_decision_agent_s_tasks_are_folders_of_questions(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n", {"urgent": CHECK})
    for task, qid, text in (
        ("grade", "tone", CHECK),
        ("grade", "pick", CHOICE),
        ("sort", "q", CHECK),
    ):
        (tmp_path / "a" / "tasks" / task).mkdir(parents=True, exist_ok=True)
        (tmp_path / "a" / "tasks" / task / f"{qid}.md").write_text(text)
    (tmp_path / "a" / "tasks" / "_drafts").mkdir()
    (tmp_path / "a" / "tasks" / "_drafts" / "x.md").write_text(CHECK)
    (tmp_path / "a" / "tasks" / "_index.md").write_text("grade, sort\n")
    spec = load_agent(tmp_path / "a")
    assert spec.tasks == ["grade", "sort"] and set(spec.questions) == {"urgent"}
    assert {task: sorted(qs) for task, qs in spec.task_questions.items()} == {
        "grade": ["pick", "tone"],
        "sort": ["q"],
    }
    assert isinstance(spec.task_questions["grade"]["pick"], Choice)


def test_a_bad_question_in_a_task_is_refused_like_any_question_file(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n")
    (tmp_path / "a" / "tasks" / "grade").mkdir(parents=True)
    (tmp_path / "a" / "tasks" / "grade" / "q.md").write_text(
        "---\ntype: choice\noptions: {x: X}\n---\n?\n"
    )
    with pytest.raises(ConfigurationError, match="^tree: invalid question .*q.md"):
        load_agent(tmp_path / "a")


def test_a_decision_agent_s_task_is_a_folder_not_a_file(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n", {"q": CHECK})
    (tmp_path / "a" / "tasks").mkdir()
    (tmp_path / "a" / "tasks" / "grade.md").write_text("Grade it.\n")
    message = "^tree: decision agent 'a' has tasks/grade.md, but keeps each task's questions in a folder; move it to tasks/grade/$"
    with pytest.raises(ConfigurationError, match=message):
        load_agent(tmp_path / "a")


def _task(root: Path, agent: str, task: str, qid: str, text: str = CHECK) -> None:
    (root / agent / "tasks" / task).mkdir(parents=True, exist_ok=True)
    (root / agent / "tasks" / task / f"{qid}.md").write_text(text)


def test_a_task_adds_questions_so_it_repeats_no_id_of_questions(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n", {"urgent": CHECK, "tone": CHECK})
    _task(tmp_path, "a", "grade", "pick", CHOICE)
    _task(tmp_path, "a", "grade", "tone")
    _task(tmp_path, "a", "grade", "urgent")
    message = "^tree: decision agent 'a' asks 'tone', 'urgent' in both questions/ and tasks/grade/; rename one$"
    with pytest.raises(ConfigurationError, match=message):
        load_agent(tmp_path / "a")


def test_two_tasks_may_share_an_id_since_they_are_never_asked_together(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n", {"urgent": CHECK})
    _task(tmp_path, "a", "grade", "tone")
    _task(tmp_path, "a", "sort", "tone", CHOICE)
    spec = load_agent(tmp_path / "a")
    assert isinstance(spec.task_questions["grade"]["tone"], Check)
    assert isinstance(spec.task_questions["sort"]["tone"], Choice)


def test_a_decision_agent_s_questions_may_all_live_in_tasks(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\n")
    _task(tmp_path, "a", "grade", "tone")
    spec = load_agent(tmp_path / "a")
    assert spec.questions == {} and spec.tasks == ["grade"]


def test_a_routing_agent_may_declare_tasks(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: decision\nroute: {by: q, to: {x: b}}\n", {"q": CHOICE})
    _agent(tmp_path, "b", "kind: generation\n")
    _task(tmp_path, "a", "grade", "tone")
    agents = load_tree(tmp_path)
    validate_tree(agents)
    assert agents["a"].tasks == ["grade"] and set(agents["a"].questions) == {"q"}


def test_a_delegate_target_needs_no_questions_of_its_own(tmp_path: Path) -> None:
    _agent(tmp_path, "a", "kind: generation\nagents: [b, c]\n")
    _agent(tmp_path, "b", "kind: decision\n")
    _agent(tmp_path, "c", "kind: decision\n")
    _task(tmp_path, "c", "grade", "tone")
    validate_tree(load_tree(tmp_path))


def _router(root: Path, to: str) -> None:
    _agent(root, "a", f"kind: decision\nroute: {{by: q, to: {to}}}\n", {"q": CHOICE})
    _agent(root, "b", "kind: decision\n")
    _task(root, "b", "grade", "tone")
    (root / "b" / "tasks" / "empty").mkdir()
    _agent(root, "g", "kind: generation\n")
    (root / "g" / "tasks").mkdir()
    (root / "g" / "tasks" / "billing.md").write_text("Answer the billing question.\n")


@pytest.mark.parametrize(
    "to",
    [
        "{x: b}",
        "{x: {agent: b}}",
        "{x: {agent: b, task: empty}}",
    ],
)
def test_a_route_to_a_decision_agent_that_could_never_ask_anything_is_refused(
    tmp_path: Path, to: str
) -> None:
    _router(tmp_path, to)
    message = (
        "^tree: route of 'a' gives decision agent 'b' nothing to ask: it has no questions/ "
        "and the route names no task with questions; give it questions/"
    )
    with pytest.raises(ConfigurationError, match=message):
        validate_tree(load_tree(tmp_path))


@pytest.mark.parametrize(
    ("to", "agent", "task"),
    [("{x: {agent: b, task: ghost}}", "b", "ghost"), ("{x: {agent: g, task: tone}}", "g", "tone")],
)
def test_a_route_names_only_a_task_its_agent_has(
    tmp_path: Path, to: str, agent: str, task: str
) -> None:
    _router(tmp_path, to)
    message = f"^tree: route of 'a' points at agent '{agent}', which has no task '{task}';"
    with pytest.raises(ConfigurationError, match=message):
        validate_tree(load_tree(tmp_path))


def test_a_route_entry_is_an_agent_or_an_agent_with_one_of_its_tasks(tmp_path: Path) -> None:
    _router(tmp_path, "{x: {agent: b, task: grade}, y: {agent: g, task: billing}}")
    agents = load_tree(tmp_path)
    validate_tree(agents)
    route = agents["a"].route
    assert route is not None
    assert {option: (t.agent, t.task) for option, t in route.to.items()} == {
        "x": ("b", "grade"),
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
    _agent(root, "coding", f"kind: generation\n{coding}")
    _agent(root, "writing", "kind: generation\ndescription: Writes\n", tools=("draft",))
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
        ("kind: generation\nreads: [ghost]\n", "agent 'coding' reads unknown folder 'ghost'"),
        ("kind: generation\nreads: [coding]\n", "agent 'coding' reads unknown folder 'coding'"),
        ("kind: generation\nreads: [.hidden]\n", "agent 'coding' reads unknown folder '.hidden'"),
        ("kind: generation\nreads: [outside]\n", "agent 'coding' reads unknown folder 'outside'"),
        ("kind: generation\nreads: [loop]\n", "agent 'coding' reads unknown folder 'loop'"),
        (
            "kind: decision\nreads: [books]\n",
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
    (root / "loop").symlink_to(root, target_is_directory=True)
    with pytest.raises(ConfigurationError, match=f"^tree: {message}(;|$)"):
        load_tree(root)


def test_a_single_agent_root_reads_nothing(tmp_path: Path) -> None:
    _agent(tmp_path, "solo", "kind: generation\nreads: [books]\n")
    with pytest.raises(ConfigurationError, match="root agent 'solo' cannot declare reads:"):
        load_tree(tmp_path / "solo")


def test_reads_needs_the_agent_s_folder_inside_the_root_not_a_link_to_elsewhere(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "books").mkdir(parents=True)
    _agent(tmp_path / "elsewhere", "coding", "kind: generation\nreads: [books]\n")
    (root / "coding").symlink_to(tmp_path / "elsewhere" / "coding", target_is_directory=True)
    message = r"^tree: agent 'coding' declares reads: but its folder is a link to "
    with pytest.raises(ConfigurationError, match=message):
        load_tree(root)
    (tmp_path / "elsewhere" / "coding" / "AGENT.md").write_text("---\nkind: generation\n---\n")
    assert load_tree(root)["coding"].reads == {}
