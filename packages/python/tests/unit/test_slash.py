from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.models import CheckAnswer, ChoiceAnswer, Content, Image, State, Text
from rfnry_engine.records import Decision, Generation, Run
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Fixed, Scripted, delegating, text_reply
from .trees import deciding, provided, write_folder

ESCALATE = "---\ndescription: Hand it to a person\n---\nSay so, then stop.\n"
CLOSE = "---\ndescription: Close the ticket\n---\nClose it.\n"
BOTH_PUBLIC = ["escalate", "close"]
SKILL = Text(text="<<<SKILL skills/escalate/SKILL.md>>>\nSay so, then stop.")
CLOSING = Text(text="<<<SKILL skills/close/SKILL.md>>>\nClose it.")
IMAGE = Image(media_type="image/png", url="https://x/y.png")
BOTH = (
    "engine: a run invokes one skill, by skill= or by an input that starts with /<name>; "
    "got skill='escalate' and /close"
)
REVIEWER_BODY = {"reviewer.md": "Review.\n"}
SKILLS = {"skills/escalate/SKILL.md": ESCALATE, "skills/close/SKILL.md": CLOSE}


def agents(root: Path, public: list[str] | str | None = BOTH_PUBLIC) -> Path:
    skills: dict[str, Any] = {"allow": ["skills/*"]}
    if public is not None:
        skills["public"] = public
    return write_folder(
        root,
        {
            "reviewer": {
                "kind": "generation",
                "model": "strong",
                "description": "Reviews",
                "instructions": "reviewer.md",
                "skills": skills,
            }
        },
        files={**REVIEWER_BODY, **SKILLS},
    )


async def ran(root: Path, input: State, **given: Any) -> tuple[Generation, Scripted]:
    llm = Scripted(text_reply("Done."))
    run = await Engine(root, providers=provided({"strong": llm})).run(
        "reviewer", "acme", input, **given
    )
    assert isinstance(run, Generation)
    return run, llm


def sent(llm: Scripted) -> Content:
    return llm.calls[0][1][-1].content


@pytest.mark.parametrize("public", [None], ids=["absent"])
async def test_no_skill_is_public_by_default(tmp_path: Path, public: None) -> None:
    root = agents(tmp_path, public)
    run, llm = await ran(root, "/escalate x")
    assert run.skill is None and run.input == "/escalate x" and sent(llm) == "/escalate x"
    assert Engine(root, providers=provided({"strong": None})).agents["reviewer"].public == {}
    invoked, llm = await ran(root, "x", skill="escalate")
    assert invoked.skill == "escalate" and sent(llm) == [SKILL, Text(text="x")]


async def test_a_skill_that_is_not_public_is_no_command(tmp_path: Path) -> None:
    root = agents(tmp_path, ["close"])
    run, llm = await ran(root, "/escalate x")
    assert run.skill is None and run.input == "/escalate x" and sent(llm) == "/escalate x"
    closed, llm = await ran(root, "/close x")
    assert closed.skill == "close" and closed.input == "x"
    assert sent(llm) == [CLOSING, Text(text="x")]
    both, llm = await ran(root, "/escalate x", skill="close")
    assert both.skill == "close" and both.input == "/escalate x"
    assert sent(llm) == [CLOSING, Text(text="/escalate x")]


def test_the_public_skills_list_them_with_their_descriptions(tmp_path: Path) -> None:
    engine = Engine(agents(tmp_path), providers=provided({"strong": None}))
    assert engine.agents["reviewer"].public == {
        "close": "Close the ticket",
        "escalate": "Hand it to a person",
    }
    assert list(engine.agents["reviewer"].public) == ["close", "escalate"]


async def test_a_name_that_holds_whitespace_stays_a_skill_the_code_may_invoke(
    tmp_path: Path,
) -> None:
    root = agents(tmp_path, None)
    (root / "skills/my skill").mkdir(parents=True)
    (root / "skills/my skill/SKILL.md").write_text("Do it.\n", encoding="utf-8")
    run, llm = await ran(root, "x", skill="my skill")
    assert run.skill == "my skill"
    assert sent(llm) == [Text(text="<<<SKILL skills/my skill/SKILL.md>>>\nDo it."), Text(text="x")]


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


async def test_skill_and_slash_name_at_once_is_refused_before_any_model_call(
    tmp_path: Path,
) -> None:
    llm = Scripted(text_reply("Done."))
    engine = Engine(agents(tmp_path), providers=provided({"strong": llm}))
    with pytest.raises(InputError) as refused:
        await engine.run("reviewer", "acme", "/close x", skill="escalate")
    assert str(refused.value) == BOTH and llm.calls == []
    run = await engine.run("reviewer", "acme", "/ghost x", skill="escalate")
    assert isinstance(run, Generation)
    assert run.skill == "escalate" and run.input == "/ghost x"


async def test_a_decision_agent_reads_slash_as_input(tmp_path: Path) -> None:
    deciding(tmp_path, "grader", {"ok": '{"type": "check", "instructions": "Ok?"}'})
    model = Fixed({"ok": CheckAnswer(probability=0.9)})
    await Engine(tmp_path, providers=provided({"judge": model})).run(
        "grader", "acme", "/escalate x"
    )
    assert model.calls[0][0] == "/escalate x"


class Recorded:
    def __init__(self) -> None:
        self.runs: list[Run] = []

    async def record(self, run: Run) -> None:
        self.runs.append(run)


def untouched(run: Run) -> bool:
    return isinstance(run, Generation) and run.skill is None and run.input == "/escalate x"


async def test_engine_tool_and_delegate_inputs_never_invoke_a_skill(tmp_path: Path) -> None:
    root = write_folder(
        tmp_path,
        {
            "reviewer": {
                "kind": "generation",
                "model": "strong",
                "description": "Reviews",
                "instructions": "reviewer.md",
                "skills": {"allow": ["skills/*"], "public": BOTH_PUBLIC},
            },
            "lead": {
                "kind": "generation",
                "model": "strong",
                "instructions": "lead.md",
                "orchestration": {"delegate_to": ["reviewer"]},
            },
            "frontdesk": {
                "kind": "decision",
                "model": "judge",
                "instructions": "frontdesk.md",
                "questions": ["q/*.json"],
                "orchestration": {"route": {"by": "intent", "to": {"review": "reviewer"}}},
            },
        },
        files={
            **REVIEWER_BODY,
            **SKILLS,
            "lead.md": "Lead.\n",
            "frontdesk.md": "",
            "q/intent.json": '{"type": "choice", "instructions": "Which?", "options": {"review": null, "other": null}}',
        },
    )
    recorded = Recorded()
    llm = Scripted(*[text_reply("Done.")] * 2)
    judge = Fixed(
        {"intent": ChoiceAnswer(choice="review", probabilities={"review": 0.9, "other": 0.1})}
    )
    engine = Engine(root, refinery=recorded, providers=provided({"strong": llm, "judge": judge}))
    consult = engine.tool("reviewer", scope="acme")
    assert await consult.dispatch("reviewer", {"input": "/escalate x"}) == "Done."
    assert untouched(recorded.runs[-1]) and sent(llm) == "/escalate x"
    await EngineTools(engine, agent="lead", scope="acme").dispatch(
        "Delegate", {"agent": "reviewer", "input": "/escalate x"}
    )
    assert untouched(recorded.runs[-1])
    llm = Scripted(delegating("reviewer", "/escalate x"), text_reply("Done."), text_reply("Done."))
    run = await Engine(root, providers=provided({"strong": llm, "judge": judge})).run(
        "lead", "acme", "Go."
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert untouched(child)
    llm = Scripted(text_reply("Done."))
    decided = await Engine(root, providers=provided({"strong": llm, "judge": judge})).run(
        "frontdesk", "acme", "/escalate x"
    )
    assert isinstance(decided, Decision) and decided.routed_to is not None
    assert untouched(decided.routed_to)
