import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import (
    Engine,
    Image,
    InputError,
    Message,
    NotFoundError,
    Part,
    Refinery,
    Reply,
    Storage,
    Text,
    ToolCall,
)
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Fixed, MemoryBucket, Scripted, text_reply
from .trees import provided, write_folder

REVIEWER: dict[str, Any] = {
    "kind": "generation",
    "model": "strong",
    "description": "Reviews contracts",
    "instructions": "reviewer/instructions.md",
    "skills": {"allow": ["reviewer/skills/*"]},
}


def tree(root: Path, agents: dict[str, dict[str, Any]], files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return write_folder(root, agents, files=files)


def reviewer(tmp_path: Path, **files: str) -> Path:
    return tree(
        tmp_path / "agents",
        {"reviewer": REVIEWER},
        {
            "reviewer/instructions.md": "You review contracts.\n",
            **{f"reviewer/{k}": v for k, v in files.items()},
        },
    )


REDLINE = (
    "---\nname: redline\ndescription: How to write a redline\nallowed-tools: Read\n---\n"
    "Write the redline.\nSee examples.md.\n"
)
SKILLED: dict[str, Any] = {
    "reviewer": {
        **REVIEWER,
        "description": "Reviews",
        "orchestration": {"delegate_to": ["chat"]},
        "files": {"allow": ["reviewer/rules"]},
    },
    "chat": {
        "kind": "generation",
        "model": "strong",
        "description": "Talks",
        "instructions": "chat.md",
    },
    "grader": {
        "kind": "decision",
        "model": "judge",
        "description": "Grades",
        "instructions": "grader.md",
        "questions": ["grader/questions/*.json"],
    },
}


def skills(tmp_path: Path, **more: str) -> Path:
    return tree(
        tmp_path / "agents",
        SKILLED,
        {
            "reviewer/instructions.md": "You review contracts.\n",
            "reviewer/skills/redline/SKILL.md": REDLINE,
            "reviewer/skills/redline/examples.md": "An example.\n",
            "reviewer/skills/risk/SKILL.md": "---\ndescription: How to rate a risk\n---\nRate it.\n",
            "reviewer/skills/plain/SKILL.md": "No frontmatter here.\n",
            "reviewer/skills/.hidden/SKILL.md": "Hidden.\n",
            "reviewer/skills/deep/inner/SKILL.md": "Too deep.\n",
            "reviewer/rules/playbook.md": "The rules.\n",
            "chat.md": "You talk.\n",
            "grader.md": "",
            "grader/questions/ok.json": '{"type": "check", "instructions": "Ok?"}',
            **more,
        },
    )


SKILL = Text(
    text="<<<SKILL reviewer/skills/redline/SKILL.md>>>\nWrite the redline.\nSee examples.md."
)


def test_a_skill_is_a_folder_of_skills_listed_with_its_description(
    tmp_path: Path,
) -> None:
    agents = Engine(skills(tmp_path), providers=provided({"strong": None, "judge": None})).agents
    assert agents["reviewer"].skills == {
        "plain": "",
        "redline": "How to write a redline",
        "risk": "How to rate a risk",
    }
    assert agents["chat"].skills == {} and agents["grader"].skills == {}


async def test_skills_are_listed_in_their_own_section_after_the_files(tmp_path: Path) -> None:
    run = await Engine(
        skills(tmp_path),
        providers=provided({"strong": Scripted(text_reply("Done.")), "judge": None}),
    ).run("reviewer", "acme", "clause 7")
    assert isinstance(run, Generation)
    assert (
        "<<<SKILLS>>>\nreviewer/skills/plain/SKILL.md\nreviewer/skills/redline/SKILL.md — How to write a redline\n"
        "reviewer/skills/risk/SKILL.md — How to rate a risk\n\n<<<TOOLS>>>"
    ) in run.system
    assert run.system.index("<<<FILES>>>") < run.system.index("<<<SKILLS>>>")


async def test_bad_frontmatter_in_a_skill_s_other_file_never_stops_the_tree(tmp_path: Path) -> None:
    examples = "---\ndescription: [bad\n---\nAn example.\n"
    root = reviewer(
        tmp_path,
        **{
            "skills/x/SKILL.md": "---\ndescription: Do x\n---\nDo it.\n",
            "skills/x/examples.md": examples,
        },
    )
    engine = Engine(root, providers=provided({"strong": Scripted()}))
    assert set(engine.agents["reviewer"].skills) == {"x"}
    tools = EngineTools(engine, agent="reviewer", scope="acme")
    assert await tools.dispatch("Read", {"path": "reviewer/skills/x/examples.md"}) == examples


async def test_an_invoked_skill_rides_in_front_of_the_input(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Redlined."))
    run = await Engine(skills(tmp_path), providers=provided({"strong": llm, "judge": None})).run(
        "reviewer", "acme", "clause 7", skill="redline"
    )
    sent = Message(role="user", content=[SKILL, Text(text="clause 7")])
    assert llm.calls[0][1] == [sent]
    assert isinstance(run, Generation) and run.skill == "redline" and run.input == "clause 7"
    assert run.messages == [sent] and run.conversation[0] == sent
    assert "<<<SKILL " not in llm.calls[0][0]


@pytest.mark.parametrize(
    ("given", "rest"),
    [
        ("", []),
        ([Text(text="look"), Image(media_type="image/png", url="https://x/y.png")], None),
        ({"clause": 7}, [Text(text='{"clause": 7}')]),
    ],
)
async def test_an_invoked_skill_keeps_the_input_as_it_came(
    tmp_path: Path, given: Any, rest: list[Part] | None
) -> None:
    llm = Scripted(text_reply("Done."))
    await Engine(skills(tmp_path), providers=provided({"strong": llm, "judge": None})).run(
        "reviewer", "acme", given, skill="redline"
    )
    assert llm.calls[0][1][-1].content == [SKILL, *(given if rest is None else rest)]


async def test_a_run_that_continues_still_has_the_skill(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Redlined."), text_reply("Shorter."))
    engine = Engine(skills(tmp_path), providers=provided({"strong": llm, "judge": None}))
    first = await engine.run("reviewer", "acme", "clause 7", skill="redline")
    assert isinstance(first, Generation)
    second = await engine.run("reviewer", "acme", "shorter", history=first.conversation)
    assert isinstance(second, Generation) and second.skill is None
    assert llm.calls[1][1][0].content == [SKILL, Text(text="clause 7")]


@pytest.mark.parametrize(
    ("root", "agent", "error", "message"),
    [
        (
            skills,
            "reviewer",
            NotFoundError,
            "engine: agent 'reviewer' has no skill 'ghost'; its skills are plain, redline, risk",
        ),
        (
            skills,
            "chat",
            NotFoundError,
            "engine: agent 'chat' has no skill 'ghost'; it has no skills; name a folder "
            "holding SKILL.md in skills.allow or skills.preload",
        ),
        (
            skills,
            "grader",
            InputError,
            "engine: decision agent 'grader' takes no skill; skills are for generation agents",
        ),
    ],
)
async def test_a_skill_the_agent_cannot_take_is_refused_before_any_model_call(
    tmp_path: Path, root: Any, agent: str, error: type[Exception], message: str
) -> None:
    llm = Scripted()
    engine = Engine(root(tmp_path), providers=provided({"strong": llm, "judge": Fixed({})}))
    with pytest.raises(error) as refused:
        await engine.run(agent, "acme", "x", skill="ghost")
    assert str(refused.value) == message and llm.calls == []


async def test_a_sub_agent_never_takes_the_callers_skill(tmp_path: Path) -> None:
    ask = ToolCall(id="d1", name="Delegate", args={"agent": "chat", "input": "hi"})
    llm = Scripted(
        Reply(tool_calls=[ask], stop_reason="tool_use"), text_reply("Hi."), text_reply("Done.")
    )
    run = await Engine(skills(tmp_path), providers=provided({"strong": llm, "judge": None})).run(
        "reviewer", "acme", "clause 7", skill="redline"
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Generation) and child.skill is None
    assert llm.calls[1][1] == [Message(role="user", content="hi")]


async def test_the_skill_is_recorded_replayed_and_exported(
    tmp_path: Path, storage: Storage
) -> None:
    refinery = Refinery(storage)
    recorder = Engine(
        skills(tmp_path / "a"),
        refinery=refinery,
        providers=provided({"strong": Scripted(text_reply("Redlined.")), "judge": None}),
    )
    run = await recorder.run("reviewer", "acme", "clause 7", skill="redline")
    back = await refinery.run("acme", run.run_id)
    assert isinstance(back, Generation) and back.skill == "redline"
    newer = skills(tmp_path / "b")
    (newer / "reviewer" / "skills" / "redline" / "SKILL.md").write_text("Redline it the new way.\n")
    llm = Scripted(text_reply("Again."))
    [pair] = [
        pair
        async for pair in refinery.replay(
            "acme",
            Engine(newer, providers=provided({"strong": llm, "judge": None})),
            agent="reviewer",
        )
    ]
    assert isinstance(pair.candidate, Generation) and pair.candidate.skill == "redline"
    assert llm.calls[0][1] == [
        Message(
            role="user",
            content=[
                Text(text="<<<SKILL reviewer/skills/redline/SKILL.md>>>\nRedline it the new way."),
                Text(text="clause 7"),
            ],
        )
    ]
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [rows] = [data for key, data in bucket.objects.items() if "/generation/" in key]
    [row] = [json.loads(line) for line in rows.decode().splitlines()]
    assert row["skill"] == "redline" and "task" not in row


async def test_a_byte_order_mark_never_reaches_the_model(tmp_path: Path) -> None:
    root = reviewer(tmp_path, **{"skills/review/SKILL.md": "\ufeffReview it.\n"})
    llm = Scripted(text_reply("Done."))
    await Engine(root, providers=provided({"strong": llm})).run(
        "reviewer", "acme", "x", skill="review"
    )
    system, messages, _ = llm.calls[0]
    sent = json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False)
    assert "\ufeff" not in system and "\ufeff" not in sent


def test_a_block_after_a_blank_line_is_body_in_skills_and_in_the_prompt(tmp_path: Path) -> None:
    from rfnry_engine.frontmatter import body_of

    root = reviewer(
        tmp_path, **{"skills/risk/SKILL.md": "\n---\ndescription: Rate it\n---\nRate.\n"}
    )
    assert Engine(root, providers=provided({"strong": None})).agents["reviewer"].skills == {
        "risk": ""
    }
    assert body_of((root / "reviewer/skills/risk/SKILL.md").read_text(encoding="utf-8")).startswith(
        "---\ndescription"
    )


async def test_an_invoked_skill_edited_after_the_tree_loads_is_the_loaded_one(
    tmp_path: Path,
) -> None:
    root = reviewer(
        tmp_path, **{"skills/risk/SKILL.md": "---\ndescription: Rate it\n---\nRate the risk.\n"}
    )
    llm = Scripted(text_reply("Done."))
    engine = Engine(root, providers=provided({"strong": llm}))
    (root / "reviewer/skills/risk/SKILL.md").write_text("Changed.\n", encoding="utf-8")
    await engine.run("reviewer", "acme", "Look.", skill="risk")
    sent = llm.calls[0][1][-1].content[0]
    assert isinstance(sent, Text) and "Rate the risk." in sent.text and "Changed." not in sent.text


async def test_a_skill_the_model_reads_with_read_is_the_file_on_disk(tmp_path: Path) -> None:
    root = reviewer(
        tmp_path, **{"skills/risk/SKILL.md": "---\ndescription: Rate it\n---\nRate the risk.\n"}
    )
    llm = Scripted(
        Reply(
            tool_calls=[
                ToolCall(id="r1", name="Read", args={"path": "reviewer/skills/risk/SKILL.md"})
            ],
            stop_reason="tool_use",
        ),
        text_reply("Done."),
    )
    engine = Engine(root, providers=provided({"strong": llm}))
    (root / "reviewer/skills/risk/SKILL.md").write_text("Changed.\n", encoding="utf-8")
    run = await engine.run("reviewer", "acme", "Look.")
    assert isinstance(run, Generation) and run.steps[0].result == "Changed.\n"
