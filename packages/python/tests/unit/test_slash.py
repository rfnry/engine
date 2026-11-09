from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.models import CheckAnswer, ChoiceAnswer, Content, Image, State, Text
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.tools.engine_tools import EngineTools
from rfnry_engine.tree import load_tree

from .fakes import Fixed, Scripted, delegating, text_reply

ESCALATE = "---\ndescription: Hand it to a person\n---\nSay so, then stop.\n"
CLOSE = "---\ndescription: Close the ticket\n---\nClose it.\n"
PUBLIC = "public_skills: [escalate, close]\n"
SKILL = Text(text="<<<SKILL skills/escalate.md>>>\nSay so, then stop.")
CLOSING = Text(text="<<<SKILL skills/close.md>>>\nClose it.")
IMAGE = Image(media_type="image/png", url="https://x/y.png")
BOTH = (
    "engine: a run invokes one skill, by skill= or by an input that starts with /<name>; "
    "got skill='escalate' and /close"
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agents(root: Path, line: str = PUBLIC) -> Path:
    write(
        root / "reviewer/AGENT.md",
        f"---\nname: reviewer\nkind: generation\nmodel: strong\ndescription: Reviews\n{line}---\nReview.\n",
    )
    write(root / "reviewer/skills/escalate.md", ESCALATE)
    write(root / "reviewer/skills/close.md", CLOSE)
    return root


async def ran(root: Path, input: State, **given: Any) -> tuple[Generation, Scripted]:
    llm = Scripted(text_reply("Done."))
    run = await Engine(root, models={"strong": llm}).run("reviewer", "acme", input, **given)
    assert isinstance(run, Generation)
    return run, llm


def sent(llm: Scripted) -> Content:
    return llm.calls[0][1][-1].content


@pytest.mark.parametrize("line", ["", "public_skills: none\n"], ids=["absent", "none"])
async def test_no_skill_is_public_by_default(tmp_path: Path, line: str) -> None:
    root = agents(tmp_path, line)
    run, llm = await ran(root, "/escalate x")
    assert run.skill is None and run.input == "/escalate x" and sent(llm) == "/escalate x"
    assert Engine(root, models={}).agents["reviewer"].public_skills == {}
    invoked, llm = await ran(root, "x", skill="escalate")
    assert invoked.skill == "escalate" and sent(llm) == [SKILL, Text(text="x")]


async def test_a_skill_that_is_not_public_is_no_command(tmp_path: Path) -> None:
    root = agents(tmp_path, "public_skills: [close]\n")
    run, llm = await ran(root, "/escalate x")
    assert run.skill is None and run.input == "/escalate x" and sent(llm) == "/escalate x"
    closed, llm = await ran(root, "/close x")
    assert closed.skill == "close" and closed.input == "x"
    assert sent(llm) == [CLOSING, Text(text="x")]
    both, llm = await ran(root, "/escalate x", skill="close")
    assert both.skill == "close" and both.input == "/escalate x"
    assert sent(llm) == [CLOSING, Text(text="/escalate x")]


def test_public_skills_lists_them_with_their_descriptions(tmp_path: Path) -> None:
    engine = Engine(agents(tmp_path), models={})
    assert engine.agents["reviewer"].public_skills == {
        "close": "Close the ticket",
        "escalate": "Hand it to a person",
    }
    assert list(engine.agents["reviewer"].public_skills) == ["close", "escalate"]


@pytest.mark.parametrize(
    ("line", "named", "have"),
    [
        ("public_skills: [ghost]\n", "ghost", "its skills are close, escalate"),
        ("skills: [close]\npublic_skills: [escalate]\n", "escalate", "its skills are close"),
        ("skills: none\npublic_skills: [escalate]\n", "escalate", "it has no skills"),
    ],
    ids=["unknown", "left-out-by-skills", "no-skills"],
)
def test_public_skills_names_only_skills_the_agent_has(
    tmp_path: Path, line: str, named: str, have: str
) -> None:
    agents(tmp_path, line)
    with pytest.raises(ConfigurationError) as refused:
        load_tree(tmp_path)
    assert str(refused.value) == (
        f"tree: {tmp_path / 'reviewer'}/AGENT.md names {named!r} in public_skills:, a skill agent "
        f"'reviewer' does not have; {have}"
    )


@pytest.mark.parametrize(
    ("line", "said"),
    [
        (
            "public_skills: [' x']\n",
            "public_skills: an entry must not be blank or padded, got ' x'; write the name of a skill people may call, such as public_skills: [escalate]",
        ),
        (
            "public_skills: ['']\n",
            "public_skills: an entry must not be blank or padded, got ''; write the name of a skill people may call, such as public_skills: [escalate]",
        ),
        (
            "public_skills: [close, close]\n",
            "public_skills: 'close' is listed twice; list each skill once",
        ),
        (
            "public_skills: ['my skill']\n",
            "public_skills: 'my skill' holds whitespace, so no one can type /my skill; rename the skill to one word, such as my-skill",
        ),
        (
            'public_skills: ["my\\tskill"]\n',
            "public_skills: 'my\\tskill' holds whitespace, so no one can type /my\tskill; rename the skill to one word, such as my-skill",
        ),
    ],
    ids=["padded", "blank", "twice", "a-space", "a-tab"],
)
def test_public_skills_entries_are_plain_names_each_once(
    tmp_path: Path, line: str, said: str
) -> None:
    agents(tmp_path, line)
    with pytest.raises(ConfigurationError) as refused:
        load_tree(tmp_path)
    assert (
        str(refused.value) == f"frontmatter: invalid agent {tmp_path / 'reviewer/AGENT.md'}; {said}"
    )


async def test_a_name_that_holds_whitespace_stays_a_skill_the_code_may_invoke(
    tmp_path: Path,
) -> None:
    root = agents(tmp_path, "skills: ['my skill']\n")
    write(root / "reviewer/skills/my skill.md", "Do it.\n")
    run, llm = await ran(root, "x", skill="my skill")
    assert run.skill == "my skill"
    assert sent(llm) == [Text(text="<<<SKILL skills/my skill.md>>>\nDo it."), Text(text="x")]


@pytest.mark.parametrize(
    "line", ["public_skills: [escalate]\n", "public_skills: all\n", "public_skills: none\n"]
)
def test_public_skills_on_a_decision_agent_is_refused(tmp_path: Path, line: str) -> None:
    write(
        tmp_path / "grader/AGENT.md",
        f"---\nname: grader\nkind: decision\nmodel: judge\n{line}---\n",
    )
    write(tmp_path / "grader/questions/ok.json", '{"type": "check", "instructions": "Ok?"}')
    with pytest.raises(ConfigurationError) as refused:
        load_tree(tmp_path)
    assert str(refused.value) == (
        "tree: decision agent 'grader' cannot declare public_skills:; skills are for generation agents"
    )


@pytest.mark.parametrize(
    ("given", "rest"),
    [
        ("/escalate  A third complaint.", "A third complaint."),
        ("/escalate\tx", "x"),
        ("/escalate\nx", "x"),
    ],
    ids=["spaces", "tab", "newline"],
)
async def test_an_input_starting_with_slash_name_invokes_that_skill(
    tmp_path: Path, given: str, rest: str
) -> None:
    run, llm = await ran(agents(tmp_path), given)
    assert run.skill == "escalate" and run.input == rest
    assert sent(llm) == [SKILL, Text(text=rest)]


@pytest.mark.parametrize("given", ["/escalate", [Text(text="/escalate")]], ids=["str", "list"])
async def test_slash_name_alone_runs_on_an_empty_input(tmp_path: Path, given: State) -> None:
    run, llm = await ran(agents(tmp_path), given)
    assert run.skill == "escalate" and run.input == "" and sent(llm) == [SKILL]


async def test_a_list_input_starting_with_slash_name_invokes_it(tmp_path: Path) -> None:
    root = agents(tmp_path)
    run, llm = await ran(root, [Text(text="/escalate look"), IMAGE])
    assert run.skill == "escalate" and run.input == [Text(text="look"), IMAGE]
    assert sent(llm) == [SKILL, Text(text="look"), IMAGE]
    bare, llm = await ran(root, [Text(text="/escalate"), IMAGE])
    assert bare.skill == "escalate" and bare.input == [IMAGE]
    assert sent(llm) == [SKILL, IMAGE]


@pytest.mark.parametrize(
    "given",
    [
        "/refund x",
        "/ESCALATE x",
        "/etc/passwd please",
        "/escalate/x",
        "/escalatex",
        " /escalate",
        {"text": "/escalate"},
        ["/escalate x"],
        [IMAGE, Text(text="/escalate")],
    ],
    ids=[
        "unknown-name",
        "another-case",
        "path",
        "name-then-slash",
        "name-then-more",
        "leading-space",
        "json-object",
        "json-list",
        "text-after-a-part",
    ],
)
async def test_an_input_that_only_looks_like_a_command_is_ordinary_input(
    tmp_path: Path, given: State
) -> None:
    run, _ = await ran(agents(tmp_path), given)
    assert run.skill is None and run.input == given


async def test_a_skill_skills_leaves_out_is_no_command(tmp_path: Path) -> None:
    root = agents(tmp_path, "skills: [close]\npublic_skills: [close]\n")
    run, llm = await ran(root, "/escalate x")
    assert run.skill is None and run.input == "/escalate x" and sent(llm) == "/escalate x"


async def test_skill_and_slash_name_at_once_is_refused_before_any_model_call(
    tmp_path: Path,
) -> None:
    llm = Scripted(text_reply("Done."))
    engine = Engine(agents(tmp_path), models={"strong": llm})
    with pytest.raises(InputError) as refused:
        await engine.run("reviewer", "acme", "/close x", skill="escalate")
    assert str(refused.value) == BOTH and llm.calls == []
    run = await engine.run("reviewer", "acme", "/ghost x", skill="escalate")
    assert isinstance(run, Generation)
    assert run.skill == "escalate" and run.input == "/ghost x"


async def test_a_decision_agent_reads_slash_as_input(tmp_path: Path) -> None:
    write(tmp_path / "grader/AGENT.md", "---\nname: grader\nkind: decision\nmodel: judge\n---\n")
    write(tmp_path / "grader/questions/ok.json", '{"type": "check", "instructions": "Ok?"}')
    model = Fixed({"ok": CheckAnswer(probability=0.9)})
    await Engine(tmp_path, models={"judge": model}).run("grader", "acme", "/escalate x")
    assert model.calls[0][0] == "/escalate x"


class Recorded:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


def untouched(run: Run) -> bool:
    return isinstance(run, Generation) and run.skill is None and run.input == "/escalate x"


async def test_engine_tool_and_delegate_inputs_never_invoke_a_skill(tmp_path: Path) -> None:
    root = agents(tmp_path)
    write(
        root / "lead/AGENT.md",
        "---\nname: lead\nkind: generation\nmodel: strong\nagents: [reviewer]\n---\nLead.\n",
    )
    write(
        root / "frontdesk/AGENT.md",
        "---\nname: frontdesk\nkind: decision\nmodel: judge\nroute:\n  by: intent\n  to:\n    review: reviewer\n---\n",
    )
    write(
        root / "frontdesk/questions/intent.json",
        '{"type": "choice", "instructions": "Which?", "options": {"review": null, "other": null}}',
    )
    recorded = Recorded()
    llm = Scripted(*[text_reply("Done.")] * 2)
    judge = Fixed(
        {"intent": ChoiceAnswer(choice="review", probabilities={"review": 0.9, "other": 0.1})}
    )
    engine = Engine(root, refinery=recorded, models={"strong": llm, "judge": judge})
    consult = engine.tool("reviewer", scope="acme")
    assert await consult.dispatch("reviewer", {"input": "/escalate x"}) == "Done."
    assert untouched(recorded.runs[-1]) and sent(llm) == "/escalate x"
    await EngineTools(engine, agent="lead", scope="acme").dispatch(
        "Delegate", {"agent": "reviewer", "input": "/escalate x"}
    )
    assert untouched(recorded.runs[-1])
    llm = Scripted(delegating("reviewer", "/escalate x"), text_reply("Done."), text_reply("Done."))
    run = await Engine(root, models={"strong": llm, "judge": judge}).run("lead", "acme", "Go.")
    assert isinstance(run, Generation)
    [child] = run.children
    assert untouched(child)
    llm = Scripted(text_reply("Done."))
    decided = await Engine(root, models={"strong": llm, "judge": judge}).run(
        "frontdesk", "acme", "/escalate x"
    )
    assert isinstance(decided, Decision) and decided.routed_to is not None
    assert untouched(decided.routed_to)
