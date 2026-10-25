import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.fingerprint import tools_fingerprint
from rfnry_engine.jail import Jail
from rfnry_engine.models import (
    Document,
    GenerationModel,
    Image,
    Message,
    Part,
    Reply,
    Text,
    ToolCall,
    ToolSpec,
)
from rfnry_engine.records import Generation
from rfnry_engine.refinery.labels import Review
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import (
    Answering,
    MemoryBucket,
    Scripted,
    by_hand,
    delegating,
    lead,
    said,
    text_reply,
)

PNG = b"\x89PNG\r\n\x1a\ndiagram"
PDF = b"%PDF-1.7 \xe2\x28\xa1 rates"
RULE = "A duplicate charge is refunded in full without approval.\n"


def tree(fixtures: Path, tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    return root


def preload_all(folder: Path) -> None:
    agent = folder / "AGENT.md"
    text = agent.read_text(encoding="utf-8")
    agent.write_text(text.replace("\n---\n", "\npreload: true\n---\n", 1), encoding="utf-8")


def furnish(support: Path) -> None:
    (support / "knowledge" / "diagram.png").write_bytes(PNG)
    (support / "knowledge" / "rates.pdf").write_bytes(PDF)
    (support / "rules" / "chargebacks").mkdir()
    (support / "rules" / "chargebacks" / "disputed.md").write_text(
        "A disputed charge waits for the bank.\n", encoding="utf-8"
    )
    (support / "rules" / ".draft.md").write_text("Not yet a rule.\n", encoding="utf-8")


def engine(root: Path, llm: GenerationModel) -> Engine:
    return Engine(root, models={"strong": llm, "judge": Answering()})


def preloaded(support: Path) -> list[Part]:
    def text(name: str) -> Text:
        return Text(text=f"<<<{name}>>>\n" + (support / name).read_text(encoding="utf-8"))

    return [
        Text(text="<<<knowledge/diagram.png>>>"),
        Image(media_type="image/png", data=PNG),
        text("knowledge/plans.md"),
        Text(text="<<<knowledge/rates.pdf>>>"),
        Document(media_type="application/pdf", name="rates.pdf", data=PDF),
        text("rules/chargebacks/disputed.md"),
        text("rules/refunds.md"),
        text("skills/escalate.md"),
    ]


async def test_preload_all_puts_every_rule_skill_and_knowledge_file_before_the_model_in_path_order(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    plain = Scripted(text_reply("ok"))
    before = await engine(root, plain).run("support", "acme", "Charged twice.")
    preload_all(support)
    llm = Scripted(
        Reply(tool_calls=[ToolCall(id="c", name="now", args={})], stop_reason="tool_use"),
        text_reply("Refunded."),
    )
    async with engine(root, llm) as running:
        run = await running.run("support", "acme", "Charged twice.")
    assert isinstance(run, Generation) and isinstance(before, Generation)
    parts = preloaded(support)
    assert run.preload == parts and before.preload == []
    first, second = llm.calls
    assert first[0] == run.system == before.system
    assert first[1] == [Message(role="user", content=parts), *run.messages]
    assert second[1] == [Message(role="user", content=parts), *said(run.conversation[:-1])]
    assert run.messages == [Message(role="user", content="Charged twice.")]
    assert all(message.content != parts for message in run.conversation)
    assert [t.name for t in first[2]] == [t.name for t in plain.calls[0][2]]
    assert run.fingerprint.agent != before.fingerprint.agent


async def test_a_binary_knowledge_file_goes_in_as_the_part_load_hands_over(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    preload_all(support)
    llm = Scripted(text_reply("ok"))
    built = engine(root, llm)
    run = await built.run("support", "acme", "What does express cost?")
    assert isinstance(run, Generation)
    surface = EngineTools(built, agent="support", scope="acme")
    for name in ("knowledge/rates.pdf", "knowledge/diagram.png"):
        loaded = await surface.dispatch("Load", {"path": name})
        assert isinstance(loaded, list) and len(loaded) == 1
        at = run.preload.index(Text(text=f"<<<{name}>>>"))
        assert run.preload[at + 1 : at + 2] == loaded


async def test_a_preloaded_link_goes_in_as_the_part_load_hands_over(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    (support / "knowledge" / "notes.md").write_bytes(b"# Notes\n\x00 kept as bytes\n")
    (support / "knowledge" / "notes").symlink_to("notes.md")
    preload_all(support)
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "What does express cost?")
    assert isinstance(run, Generation)
    loaded = await EngineTools(built, agent="support", scope="acme").dispatch(
        "Load", {"path": "knowledge/notes"}
    )
    assert loaded == [
        Document(media_type="text/markdown", name="notes.md", data=b"# Notes\n\x00 kept as bytes\n")
    ]
    at = run.preload.index(Text(text="<<<knowledge/notes>>>"))
    assert run.preload[at + 1 : at + 2] == loaded


class RuleFollower:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        self.calls.append(list(messages))
        seen = [
            part.text
            for message in messages
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, Text)
        ] + [m.content for m in messages if isinstance(m.content, str)]
        if any("refunded in full without approval" in text for text in seen):
            return text_reply("Refunded in full; no approval needed.")
        read = ToolCall(id=f"r{len(self.calls)}", name="Read", args={"path": "rules/refunds.md"})
        return Reply(tool_calls=[read], stop_reason="tool_use")


async def test_a_run_applies_a_rule_held_only_in_its_file_without_reading_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    assert (support / "rules" / "refunds.md").read_text(encoding="utf-8").endswith(RULE)
    assert (
        "without approval"
        not in Engine(root, models={"strong": Scripted()})._spec("support").listing
    )
    reading = RuleFollower()
    on_demand = await engine(root, reading).run("support", "acme", "Charged twice.")
    assert isinstance(on_demand, Generation)
    assert [step.tool for step in on_demand.steps] == ["Read"]
    preload_all(support)
    preloading = RuleFollower()
    run = await engine(root, preloading).run("support", "acme", "Charged twice.")
    assert isinstance(run, Generation)
    assert run.steps == [] and run.calls == 1
    assert run.reply == "Refunded in full; no approval needed."


async def test_without_preload_the_model_is_sent_exactly_what_it_was_sent_before(
    fixtures: Path,
) -> None:
    llm = Scripted(text_reply("ok"))
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    await Engine(fixtures, models={"strong": llm}).run(
        "support",
        "acme",
        "Charged twice.",
        task="triage",
        context="Acme is on the enterprise plan.",
        history=history,
    )
    [(system, messages, tools)] = llm.calls
    assert system == (
        "<<<AGENT>>>\nYou are the support desk. Be brief, cite the rule you apply.\n\n"
        "<<<TASK>>>\nDecide whether this message needs a person.\n\n"
        "<<<FILES>>>\nknowledge/\n  plans.md — The three plans\nrules/\n"
        "  refunds.md — When a refund is owed\nskills/\n"
        "  escalate.md — Hand a conversation to a person\n\n"
        "<<<TOOLS>>>\n"
        "- Read: Read a text file by its path in the FILES map. Optional offset/limit select lines. For an image, a PDF, audio or any other file, use Load.\n"
        "- Grep: Search file contents with a regex. mode=files lists matching files; mode=content lists matching lines. path is a file or folder of the FILES map; without it, your own folder is searched.\n"
        "- Glob: Find files by glob pattern, e.g. '**/*.md'. Paths are those of the FILES map.\n"
        "- LS: List a folder by its path in the FILES map; without a path, your own folder.\n"
        "- Load: Load a file of the FILES map as it is, such as an image, a PDF or audio, so you see or hear it yourself.\n"
        "- lookup_order: Fetch an order by id. Returns the order as JSON.\n"
        "- now: The current UTC time\n"
        "- Delegate: Hand an input to another agent and get its reply.\n\n"
        "<<<AGENTS>>>\n- triage: Judges urgency and severity of a message\n\n"
        "<<<CONTEXT>>>\nAcme is on the enterprise plan."
    )
    assert messages == [*history, Message(role="user", content="Charged twice.")]
    assert tools_fingerprint(tools) == "eaff21b12c6a840e"
    assert [t.name for t in tools] == [
        "Read",
        "Grep",
        "Glob",
        "LS",
        "Load",
        "lookup_order",
        "now",
        "Delegate",
    ]


@pytest.mark.parametrize(
    ("front", "message"),
    [
        (
            "name: a\nkind: decision\nmodel: judge\npreload: true\n",
            "^tree: decision agent 'a' cannot declare preload:; a decision agent reads no files",
        ),
        (
            "name: a\nkind: generation\nmodel: strong\npreload: some\n",
            "^frontmatter: invalid agent .*AGENT.md",
        ),
    ],
)
def test_preload_is_true_and_a_decision_agent_cannot_declare_it(
    tmp_path: Path, front: str, message: str
) -> None:
    (tmp_path / "a" / "questions").mkdir(parents=True)
    (tmp_path / "a" / "AGENT.md").write_text(f"---\n{front}---\n", encoding="utf-8")
    (tmp_path / "a" / "questions" / "q.json").write_text(
        '{"type": "check", "instructions": "Is it?"}'
    )
    with pytest.raises(ConfigurationError, match=message):
        Engine(tmp_path, models={"strong": Scripted(), "judge": Answering()})


def test_a_preloaded_file_that_is_not_the_type_its_name_says_is_refused_when_the_tree_loads(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    (root / "support" / "knowledge" / "diagram.png").write_bytes(b"\0not a png")
    preload_all(root / "support")
    with pytest.raises(
        ConfigurationError,
        match=r"^tree: agent 'support' cannot preload its folder: builtins: diagram.png is not the image/png its name says; move that file out of its folders, or drop preload: true$",
    ):
        engine(root, Scripted())


async def test_the_preload_is_read_when_the_tree_loads_as_the_fingerprint_is_taken(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    preload_all(support)
    llm = Scripted(text_reply("a"), text_reply("b"))
    loaded = engine(root, llm)
    (support / "rules" / "refunds.md").write_text("Refunds need approval.\n", encoding="utf-8")
    old = await loaded.run("support", "acme", "x")
    new = await engine(root, Scripted(text_reply("c"))).run("support", "acme", "x")
    assert isinstance(old, Generation) and isinstance(new, Generation)
    held = f"<<<rules/refunds.md>>>\n---\ndescription: When a refund is owed\n---\n{RULE}"
    assert Text(text=held) in old.preload
    assert Text(text="<<<rules/refunds.md>>>\nRefunds need approval.\n") in new.preload
    assert old.fingerprint != new.fingerprint


async def test_a_conversation_passed_back_carries_no_second_copy_of_the_folder(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    preload_all(root / "support")
    llm = Scripted(text_reply("Refunded."), text_reply("Done."))
    built = engine(root, llm)
    first = await built.run("support", "acme", "Charged twice.")
    assert isinstance(first, Generation)
    await built.run("support", "acme", "And the second?", history=first.conversation)
    preload = Message(role="user", content=first.preload)
    assert llm.calls[1][1] == [
        preload,
        *said(first.conversation),
        Message(role="user", content="And the second?"),
    ]


async def test_a_delegated_agent_preloads_its_own_folder_and_only_its_own(tmp_path: Path) -> None:
    root = lead(tmp_path)
    (root / "lead" / "rules").mkdir()
    (root / "lead" / "rules" / "tone.md").write_text("Be kind.\n", encoding="utf-8")
    (root / "writer" / "knowledge").mkdir()
    (root / "writer" / "knowledge" / "style.md").write_text("Short words.\n", encoding="utf-8")
    preload_all(root / "lead")
    preload_all(root / "writer")
    llm = Scripted(delegating("writer", "Draft it."), text_reply("Drafted."), text_reply("Sent."))
    run = await Engine(root, models={"strong": llm, "judge": Answering()}).run(
        "lead", "acme", "Go."
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Generation)
    assert run.preload == [Text(text="<<<rules/tone.md>>>\nBe kind.\n")]
    assert child.preload == [Text(text="<<<knowledge/style.md>>>\nShort words.\n")]
    assert llm.calls[1][1] == [
        Message(role="user", content=child.preload),
        Message(role="user", content="Draft it."),
    ]


async def test_the_preload_is_recorded_read_back_exported_and_left_to_the_candidate_on_replay(
    fixtures: Path, tmp_path: Path, storage: Storage
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    preload_all(support)
    refinery = Refinery(storage)
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    built = Engine(root, refinery=refinery, models={"strong": Scripted(text_reply("Sorry."))})
    run = await built.run("support", "acme", "Charged twice.", history=history)
    assert isinstance(run, Generation)
    assert await refinery.run("acme", run.run_id) == run
    await refinery.review("acme", run.run_id, Review(correction="Refunded."))
    bucket = MemoryBucket()
    await refinery.export("acme", bucket)
    rows = {
        key.split("/")[2]: json.loads(data.decode().splitlines()[0])
        for key, data in bucket.objects.items()
    }
    sent = [part.model_dump(mode="json") for part in run.preload]
    assert rows["generation"]["preload"] == sent
    assert rows["preference"]["prompt"]["preload"] == sent
    candidate_model = Scripted(text_reply("Refunded."))
    candidate = Engine(root, models={"strong": candidate_model})
    async for pair in refinery.replay("acme", candidate, agent="support"):
        assert pair.candidate is not None
    [(_, messages, _)] = candidate_model.calls
    assert messages == [
        Message(role="user", content=run.preload),
        *history,
        Message(role="user", content="Charged twice."),
    ]


class PicturesAndDocuments(Scripted):
    media_types = {"image/*", "application/pdf"}


async def refusal(root: Path, llm: GenerationModel, name: str, tool: str = "Load") -> str:
    surface = EngineTools(engine(root, llm), agent="support", scope="acme")
    with pytest.raises(InputError) as refused:
        await surface.dispatch(tool, {"path": name})
    return str(refused.value)


async def test_a_file_over_load_s_cap_is_refused_when_the_tree_loads_with_load_s_message(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    with (root / "support" / "knowledge" / "archive.bin").open("wb") as file:
        file.truncate(20 * 1024 * 1024 + 1)
    load = await refusal(root, Scripted(), "knowledge/archive.bin")
    assert load == "builtins: file exceeds 20971520 bytes: knowledge/archive.bin"
    preload_all(root / "support")
    with pytest.raises(ConfigurationError) as refused:
        engine(root, Scripted())
    assert str(refused.value) == (
        f"tree: agent 'support' cannot preload its folder: {load}; "
        "move that file out of its folders, or drop preload: true"
    )


async def test_a_text_file_over_read_s_cap_goes_in_as_load_hands_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    name = "knowledge/notes.txt"
    (root / "support" / name).write_text("a" * (5 * 1024 * 1024 + 1), encoding="utf-8")
    read = await refusal(root, Scripted(), name, tool="Read")
    assert read == f"builtins: file exceeds 5242880 bytes: {name}"
    preload_all(root / "support")
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "x")
    assert isinstance(run, Generation)
    loaded = await EngineTools(built, agent="support", scope="acme").dispatch(
        "Load", {"path": name}
    )
    assert loaded == [
        Document(
            media_type="text/plain", name="notes.txt", data=(root / "support" / name).read_bytes()
        )
    ]
    at = run.preload.index(Text(text=f"<<<{name}>>>"))
    assert run.preload[at + 1 : at + 2] == loaded


async def test_a_file_the_model_does_not_take_is_refused_when_the_tree_loads_with_load_s_message(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    name = "knowledge/greeting.wav"
    (root / "support" / name).write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00")
    load = await refusal(root, PicturesAndDocuments(), name)
    kind = mimetypes.guess_type(name)[0]
    assert load == f"builtins: the model does not take {kind} files: {name}"
    preload_all(root / "support")
    with pytest.raises(ConfigurationError) as refused:
        engine(root, PicturesAndDocuments())
    assert str(refused.value) == (
        f"tree: agent 'support' cannot preload its folder: {load}; "
        "move that file out of its folders, or drop preload: true"
    )
    assert engine(root, Scripted()).agents["support"].kind == "generation"


def test_the_preload_and_the_fingerprint_come_from_one_read_of_each_file(
    fixtures: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tree(fixtures, tmp_path)
    support = (root / "support").resolve()
    furnish(support)
    preload_all(support)
    opened: list[Path] = []
    original = Path.open

    jailed = Jail.open

    def counting(self: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(self)
        return original(self, *args, **kwargs)

    def walking(self: Jail, path: str | Path, *args: Any) -> int:
        opened.append(self.root / path)
        return jailed(self, path, *args)

    monkeypatch.setattr(Path, "open", counting)
    monkeypatch.setattr(Jail, "open", walking)
    spec = engine(root, Scripted())._agents["support"]
    monkeypatch.undo()
    preloaded = ["knowledge/diagram.png", "knowledge/plans.md", "knowledge/rates.pdf"]
    preloaded += ["rules/chargebacks/disputed.md", "rules/refunds.md", "skills/escalate.md"]
    opens = {name: 1 + name.endswith(".md") + name.startswith("skills/") for name in preloaded}
    assert {name: opened.count(support / name) for name in preloaded} == opens
    assert spec.fingerprint == by_hand(support)


async def test_preload_all_sends_every_file_but_agent_md_and_the_declarations(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    (support / "notes.txt").write_text("Open on weekdays.\n", encoding="utf-8")
    (support / "docs" / "tools").mkdir(parents=True)
    (support / "docs" / "tools" / "hammer.md").write_text("Content.\n", encoding="utf-8")
    preload_all(support)
    run = await engine(root, Scripted(text_reply("ok"))).run("support", "acme", "Hi")
    assert isinstance(run, Generation)
    names = [p.text.split("\n", 1)[0] for p in run.preload if isinstance(p, Text)]
    assert names == [
        "<<<docs/tools/hammer.md>>>",
        "<<<knowledge/plans.md>>>",
        "<<<notes.txt>>>",
        "<<<rules/refunds.md>>>",
        "<<<skills/escalate.md>>>",
    ]


def test_preload_all_sends_the_read_folders_too_in_path_order(tmp_path: Path) -> None:
    (tmp_path / "coding").mkdir()
    (tmp_path / "coding" / "AGENT.md").write_text(
        "---\nname: coding\nkind: generation\nmodel: strong\nreads: [books]\npreload: true\n---\nHi\n"
    )
    (tmp_path / "coding" / "notes.md").write_text("Own.\n")
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "a.md").write_text("Shared.\n")
    spec = Engine(tmp_path, models={"strong": Scripted()})._spec("coding")
    assert spec.preload == [
        Text(text="<<<../books/a.md>>>\nShared.\n"),
        Text(text="<<<notes.md>>>\nOwn.\n"),
    ]
