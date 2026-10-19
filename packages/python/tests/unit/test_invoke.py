import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import (
    ConfigurationError,
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

from .fakes import Fixed, MemoryBucket, Scripted, text_reply

REVIEWER = "---\nkind: generation\ndescription: Reviews contracts\n---\nYou review contracts.\n"


def tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "agents"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def reviewer(tmp_path: Path, **files: str) -> Path:
    return tree(
        tmp_path,
        {"reviewer/AGENT.md": REVIEWER, **{f"reviewer/{k}": v for k, v in files.items()}},
    )


def tasks(tmp_path: Path) -> Path:
    return tree(
        tmp_path,
        {
            "reviewer/AGENT.md": REVIEWER,
            "reviewer/tasks/review.md": (
                "---\ndescription: Review  a whole\n  contract\n---\nReview every clause.\n"
            ),
            "reviewer/tasks/triage.md": "Sort the request.\n",
            "chat/AGENT.md": "---\nkind: generation\ndescription: Talks\n---\nYou talk.\n",
            "grader/AGENT.md": "---\nkind: decision\ndescription: Grades\n---\n",
            "grader/tasks/tone/polite.md": "---\ntype: check\n---\nPolite?\n",
        },
    )


def test_a_task_is_listed_with_its_description(tmp_path: Path) -> None:
    agents = Engine(tasks(tmp_path)).agents
    assert agents["reviewer"].tasks == {"review": "Review a whole contract", "triage": ""}
    assert agents["chat"].tasks == {} and agents["grader"].tasks == {"tone": ""}


async def test_only_a_task_s_body_reaches_the_model(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Done."), text_reply("Sorted."))
    engine = Engine(tasks(tmp_path), generation=llm)
    await engine.run("reviewer", "acme", "Clause 7", task="review")
    await engine.run("reviewer", "acme", "Clause 7", task="triage")
    first, second = (system for system, _, _ in llm.calls)
    assert "<<<TASK>>>\nReview every clause.\n\n<<<TOOLS>>>" in first
    assert "description" not in first.split("<<<TOOLS>>>")[0]
    assert "<<<TASK>>>\nSort the request.\n\n" in second


@pytest.mark.parametrize(
    ("front", "said"),
    [
        ("name: review", "unknown key 'name'"),
        ("when: always", "unknown key 'when' in .*; it takes description$"),
        ("description: [a, b]", "invalid task"),
    ],
)
def test_a_task_takes_only_a_description(tmp_path: Path, front: str, said: str) -> None:
    root = reviewer(tmp_path, **{"tasks/review.md": f"---\n{front}\n---\nReview.\n"})
    with pytest.raises(ConfigurationError, match=said):
        Engine(root)


@pytest.mark.parametrize(
    ("agent", "fix"),
    [("reviewer", "its tasks are review, triage"), ("chat", "it has no tasks")],
)
async def test_a_task_the_agent_lacks_is_refused_with_the_ones_it_has(
    tmp_path: Path, agent: str, fix: str
) -> None:
    llm = Scripted()
    engine = Engine(tasks(tmp_path), generation=llm)
    with pytest.raises(NotFoundError) as refused:
        await engine.run(agent, "acme", "x", task="ghost")
    assert str(refused.value) == f"engine: agent '{agent}' has no task 'ghost'; {fix}"
    assert llm.calls == []


async def test_a_run_with_a_task_records_it(tmp_path: Path) -> None:
    run = await Engine(tasks(tmp_path), generation=Scripted(text_reply("Done."))).run(
        "reviewer", "acme", "x", task="review"
    )
    assert isinstance(run, Generation) and run.task == "review"


REDLINE = (
    "---\nname: redline\ndescription: How to write a redline\nallowed-tools: Read\n---\n"
    "Write the redline.\nSee examples.md.\n"
)


def skills(tmp_path: Path, **more: str) -> Path:
    return tree(
        tmp_path,
        {
            "reviewer/AGENT.md": (
                "---\nkind: generation\ndescription: Reviews\nagents: [chat]\nreads: [books]\n"
                "---\nYou review contracts.\n"
            ),
            "reviewer/skills/redline/SKILL.md": REDLINE,
            "reviewer/skills/redline/examples.md": "An example.\n",
            "reviewer/skills/risk.md": "---\ndescription: How to rate a risk\n---\nRate it.\n",
            "reviewer/skills/plain.md": "No frontmatter here.\n",
            "reviewer/skills/_draft.md": "A draft.\n",
            "reviewer/skills/_old/SKILL.md": "Old.\n",
            "reviewer/skills/.hidden/SKILL.md": "Hidden.\n",
            "reviewer/skills/deep/inner/SKILL.md": "Too deep.\n",
            "reviewer/rules/playbook.md": "The rules.\n",
            "books/skills/borrowed.md": "Not the reviewer's.\n",
            "chat/AGENT.md": "---\nkind: generation\ndescription: Talks\n---\nYou talk.\n",
            "grader/AGENT.md": "---\nkind: decision\ndescription: Grades\n---\n",
            "grader/questions/ok.md": "---\ntype: check\n---\nOk?\n",
            "grader/skills/x.md": "A file.\n",
            **more,
        },
    )


SKILL = Text(text="<<<SKILL skills/redline/SKILL.md>>>\nWrite the redline.\nSee examples.md.")


def test_a_skill_is_a_file_or_a_folder_in_skills_listed_with_its_description(
    tmp_path: Path,
) -> None:
    agents = Engine(skills(tmp_path)).agents
    assert agents["reviewer"].skills == {
        "plain": "",
        "redline": "How to write a redline",
        "risk": "How to rate a risk",
    }
    assert agents["chat"].skills == {} and agents["grader"].skills == {}


def test_a_skill_named_twice_is_refused(tmp_path: Path) -> None:
    root = skills(tmp_path, **{"reviewer/skills/risk/SKILL.md": "Again.\n"})
    message = (
        "^tree: agent 'reviewer' has the skill 'risk' as both skills/risk.md and "
        "skills/risk/SKILL.md; keep one$"
    )
    with pytest.raises(ConfigurationError, match=message):
        Engine(root)


def test_a_plain_folder_has_skills_too(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    (root / "skills").mkdir(parents=True)
    (root / "skills" / "tone.md").write_text("---\ndescription: The house tone\n---\nBe brief.\n")
    assert Engine(root).agents["notes"].skills == {"tone": "The house tone"}


async def test_an_invoked_skill_rides_in_front_of_the_input(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Redlined."))
    run = await Engine(skills(tmp_path), generation=llm).run(
        "reviewer", "acme", "clause 7", skill="redline"
    )
    sent = Message(role="user", content=[SKILL, Text(text="clause 7")])
    assert llm.calls[0][1] == [sent]
    assert isinstance(run, Generation) and run.skill == "redline" and run.input == "clause 7"
    assert run.messages == [sent] and run.conversation[0] == sent
    assert "<<<SKILL" not in llm.calls[0][0]


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
    await Engine(skills(tmp_path), generation=llm).run("reviewer", "acme", given, skill="redline")
    assert llm.calls[0][1][-1].content == [SKILL, *(given if rest is None else rest)]


async def test_a_run_that_continues_still_has_the_skill(tmp_path: Path) -> None:
    llm = Scripted(text_reply("Redlined."), text_reply("Shorter."))
    engine = Engine(skills(tmp_path), generation=llm)
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
            "engine: agent 'chat' has no skill 'ghost'; it has no skills; a skill is "
            "skills/<name>.md or skills/<name>/SKILL.md",
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
    engine = Engine(root(tmp_path), generation=llm, decision=Fixed({}))
    with pytest.raises(error) as refused:
        await engine.run(agent, "acme", "x", skill="ghost")
    assert str(refused.value) == message and llm.calls == []


async def test_a_sub_agent_never_takes_the_callers_skill(tmp_path: Path) -> None:
    ask = ToolCall(id="d1", name="Delegate", args={"agent": "chat", "input": "hi"})
    llm = Scripted(
        Reply(tool_calls=[ask], stop_reason="tool_use"), text_reply("Hi."), text_reply("Done.")
    )
    run = await Engine(skills(tmp_path), generation=llm).run(
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
        skills(tmp_path / "a"), generation=Scripted(text_reply("Redlined.")), refinery=refinery
    )
    run = await recorder.run("reviewer", "acme", "clause 7", skill="redline")
    back = await refinery.run("acme", run.run_id)
    assert isinstance(back, Generation) and back.skill == "redline"
    newer = skills(tmp_path / "b")
    (newer / "reviewer" / "skills" / "redline" / "SKILL.md").write_text("Redline it the new way.\n")
    llm = Scripted(text_reply("Again."))
    [pair] = [
        pair
        async for pair in refinery.replay("acme", Engine(newer, generation=llm), agent="reviewer")
    ]
    assert isinstance(pair.candidate, Generation) and pair.candidate.skill == "redline"
    assert llm.calls[0][1] == [
        Message(
            role="user",
            content=[
                Text(text="<<<SKILL skills/redline/SKILL.md>>>\nRedline it the new way."),
                Text(text="clause 7"),
            ],
        )
    ]
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    [rows] = [data for key, data in bucket.objects.items() if "/generation/" in key]
    [row] = [json.loads(line) for line in rows.decode().splitlines()]
    assert row["skill"] == "redline" and row["task"] is None


def test_a_task_that_is_not_utf8_is_refused_with_its_path(tmp_path: Path) -> None:
    root = reviewer(tmp_path)
    (root / "reviewer" / "tasks").mkdir()
    (root / "reviewer" / "tasks" / "review.md").write_bytes("Revisão\n".encode("latin-1"))
    with pytest.raises(ConfigurationError, match=r"^tree: .*review\.md is not UTF-8 text; "):
        Engine(root)


def test_a_skill_that_is_not_utf8_is_refused_when_the_tree_loads(tmp_path: Path) -> None:
    root = reviewer(tmp_path)
    (root / "reviewer" / "skills").mkdir()
    (root / "reviewer" / "skills" / "risk.md").write_bytes("Risco: médio\n".encode("latin-1"))
    with pytest.raises(ConfigurationError, match=r"^tree: .*risk\.md is not UTF-8 text; "):
        Engine(root)


@pytest.mark.parametrize("name", ["tasks/review.md", "skills/review.md"])
async def test_a_byte_order_mark_never_reaches_the_model(tmp_path: Path, name: str) -> None:
    root = reviewer(tmp_path, **{name: "\ufeffReview it.\n"})
    llm = Scripted(text_reply("Done."))
    tasked = name.startswith("tasks")
    await Engine(root, generation=llm).run(
        "reviewer",
        "acme",
        "x",
        task="review" if tasked else None,
        skill=None if tasked else "review",
    )
    system, messages, _ = llm.calls[0]
    sent = json.dumps([m.model_dump(mode="json") for m in messages], ensure_ascii=False)
    assert "\ufeff" not in system and "\ufeff" not in sent


def test_a_block_between_dashes_that_holds_no_keys_says_so(tmp_path: Path) -> None:
    root = reviewer(tmp_path, **{"tasks/review.md": "---\nJust a rule above.\n---\nReview.\n"})
    with pytest.raises(ConfigurationError) as refused:
        Engine(root)
    said = str(refused.value)
    assert said.startswith("frontmatter: ") and "TaskFront" not in said
    assert said.endswith(
        "write key: value lines between the two --- lines, or start the file without ---"
    )
