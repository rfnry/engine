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
    Audio,
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
from rfnry_engine.tree import AgentSpec, load_tree

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
    agent.write_text(text.replace("\n---\n", "\npreload: ['.']\n---\n", 1), encoding="utf-8")


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
        Text(text="<<<knowledge/diagram.png>>>\nimage/png, 15 bytes; Load opens it"),
        text("knowledge/plans.md"),
        Text(text="<<<knowledge/rates.pdf>>>\napplication/pdf, 18 bytes; Load opens it"),
        text("rules/chargebacks/disputed.md"),
        text("rules/refunds.md"),
    ]


async def test_preload_all_puts_every_rule_and_knowledge_file_before_the_model_in_path_order(
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


async def test_a_file_that_is_not_text_goes_in_as_a_reference_load_opens(
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
    assert not any(isinstance(part, Image | Document) for part in run.preload)
    for line in (
        "<<<knowledge/diagram.png>>>\nimage/png, 15 bytes; Load opens it",
        "<<<knowledge/rates.pdf>>>\napplication/pdf, 18 bytes; Load opens it",
    ):
        assert Text(text=line) in run.preload
    for name, part in {
        "knowledge/rates.pdf": Document(media_type="application/pdf", name="rates.pdf", data=PDF),
        "knowledge/diagram.png": Image(media_type="image/png", data=PNG),
    }.items():
        assert await surface.dispatch("Load", {"path": name}) == [part]


async def test_a_preloaded_link_goes_in_as_a_reference(fixtures: Path, tmp_path: Path) -> None:
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
    assert Text(text="<<<knowledge/notes>>>\ntext/markdown, 24 bytes; Load opens it") in run.preload


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
        context="Acme is on the enterprise plan.",
        history=history,
    )
    [(system, messages, tools)] = llm.calls
    assert system == (
        "<<<AGENT>>>\nYou are the support desk. Be brief, cite the rule you apply.\n\n"
        "<<<FILES>>>\nknowledge/\n  plans.md — The three plans\nrules/\n"
        "  refunds.md — When a refund is owed\n\n"
        "<<<SKILLS>>>\nskills/escalate.md — Hand a conversation to a person\n\n"
        "<<<TOOLS>>>\n"
        "- Read: Read a text file by its path in the FILES map or SKILLS. Optional offset/limit select lines. For an image, a PDF, audio or any other file, use Load.\n"
        "- Grep: Search file contents with a regex. mode=files lists matching files; mode=content lists matching lines. path is a file or folder from your own folder, as the map shows it; without it, your own folder is searched.\n"
        "- Glob: Find files by glob pattern, e.g. '**/*.md'. Paths are from your own folder, as the map shows them.\n"
        "- LS: List a folder by its path from your own folder, as the map shows it; without a path, your own folder.\n"
        "- Load: Load a file of the FILES map or SKILLS as it is, such as an image, a PDF or audio, so you see or hear it yourself.\n"
        "- lookup_order: Fetch an order by id. Returns the order as JSON.\n"
        "- now: The current UTC time\n"
        "- Delegate: Hand an input to another agent and get its reply.\n\n"
        "<<<AGENTS>>>\n- triage: Judges urgency and severity of a message\n\n"
        "<<<CONTEXT>>>\nAcme is on the enterprise plan."
    )
    assert messages == [*history, Message(role="user", content="Charged twice.")]
    assert tools_fingerprint(tools) == "b777cb12c847fb73"
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
            "name: a\nkind: decision\nmodel: judge\npreload: ['.']\n",
            "^tree: decision agent 'a' cannot declare preload:; a decision agent reads no files",
        ),
        (
            "name: a\nkind: generation\nmodel: strong\npreload: some\n",
            "^frontmatter: invalid agent .*AGENT.md",
        ),
    ],
)
def test_a_preload_that_is_not_a_list_and_a_decision_agent_s_preload_are_refused(
    tmp_path: Path, front: str, message: str
) -> None:
    (tmp_path / "a" / "questions").mkdir(parents=True)
    (tmp_path / "a" / "AGENT.md").write_text(f"---\n{front}---\n", encoding="utf-8")
    (tmp_path / "a" / "questions" / "q.json").write_text(
        '{"type": "check", "instructions": "Is it?"}'
    )
    with pytest.raises(ConfigurationError, match=message):
        Engine(tmp_path, models={"strong": Scripted(), "judge": Answering()})


async def test_a_preloaded_file_that_is_not_the_type_its_name_says_loads_and_load_refuses_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    (root / "support" / "knowledge" / "diagram.png").write_bytes(b"\0not a png")
    preload_all(root / "support")
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "x")
    assert isinstance(run, Generation)
    assert (
        Text(text="<<<knowledge/diagram.png>>>\nimage/png, 10 bytes; Load opens it") in run.preload
    )
    refused = await refusal(root, Scripted(), "knowledge/diagram.png")
    assert refused == "tools/builtins: diagram.png is not the image/png its name says"


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


async def test_a_file_over_load_s_cap_loads_the_tree_and_load_refuses_it_when_asked(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    with (root / "support" / "knowledge" / "archive.bin").open("wb") as file:
        file.truncate(20 * 1024 * 1024 + 1)
    preload_all(root / "support")
    run = await engine(root, Scripted(text_reply("ok"))).run("support", "acme", "x")
    assert isinstance(run, Generation)
    held = "<<<knowledge/archive.bin>>>\napplication/octet-stream, 20971521 bytes; Load opens it"
    assert Text(text=held) in run.preload
    load = await refusal(root, Scripted(), "knowledge/archive.bin")
    assert load == "tools/builtins: file exceeds 20971520 bytes: knowledge/archive.bin"


async def test_a_file_over_read_s_cap_is_in_the_fingerprint_though_the_preload_never_reads_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    big = root / "support" / "knowledge" / "notes.txt"
    big.write_text("a" * (5 * 1024 * 1024 + 1), encoding="utf-8")
    preload_all(root / "support")
    before = engine(root, Scripted())._agents["support"].fingerprint
    big.write_text("b" + "a" * (5 * 1024 * 1024), encoding="utf-8")
    assert engine(root, Scripted())._agents["support"].fingerprint != before


async def test_an_agent_without_load_is_not_told_load_opens_a_file(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    preload_all(support)
    agent = support / "AGENT.md"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace("\n---\n", "\ntools: [Read]\n---\n", 1),
        encoding="utf-8",
    )
    spec = engine(root, Scripted())._agents["support"]
    assert Text(text="<<<knowledge/diagram.png>>>\nimage/png, 15 bytes") in spec.preload
    assert not any("Load opens it" in getattr(part, "text", "") for part in spec.preload)


async def test_a_text_file_over_read_s_cap_goes_in_as_a_reference(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    name = "knowledge/notes.txt"
    (root / "support" / name).write_text("a" * (5 * 1024 * 1024 + 1), encoding="utf-8")
    read = await refusal(root, Scripted(), name, tool="Read")
    assert read == f"tools/builtins: file exceeds 5242880 bytes: {name}"
    preload_all(root / "support")
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "x")
    assert isinstance(run, Generation)
    assert Text(text=f"<<<{name}>>>\ntext/plain, 5242881 bytes; Load opens it") in run.preload
    loaded = await EngineTools(built, agent="support", scope="acme").dispatch(
        "Load", {"path": name}
    )
    assert loaded == [
        Document(
            media_type="text/plain", name="notes.txt", data=(root / "support" / name).read_bytes()
        )
    ]


async def test_a_file_the_model_does_not_take_loads_the_tree_and_load_refuses_it_when_asked(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    name = "knowledge/greeting.wav"
    (root / "support" / name).write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00")
    preload_all(root / "support")
    narrow = engine(root, PicturesAndDocuments(text_reply("ok")))
    run = await narrow.run("support", "acme", "x")
    assert isinstance(run, Generation)
    kind = mimetypes.guess_type(name)[0]
    assert Text(text=f"<<<{name}>>>\n{kind}, 18 bytes; Load opens it") in run.preload
    surface = EngineTools(narrow, agent="support", scope="acme")
    with pytest.raises(InputError) as refused:
        await surface.dispatch("Load", {"path": name})
    assert str(refused.value) == f"tools/builtins: the model does not take {kind} files: {name}"
    taking = engine(root, Scripted(text_reply("ok")))
    loaded = await EngineTools(taking, agent="support", scope="acme").dispatch(
        "Load", {"path": name}
    )
    assert isinstance(loaded, list) and len(loaded) == 1 and isinstance(loaded[0], Audio)


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
    preloaded += ["rules/chargebacks/disputed.md", "rules/refunds.md"]
    opens = {name: 1 + name.endswith(".md") for name in preloaded}
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
    ]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def agent_md(name: str, extra: str) -> str:
    return f"---\nname: {name}\nkind: generation\nmodel: strong\n{extra}---\nWork.\n"


def sent(spec: AgentSpec) -> list[str]:
    return [part.text.split("\n", 1)[0] for part in spec.preload if isinstance(part, Text)]


def load_error(root: Path) -> str:
    with pytest.raises(ConfigurationError) as caught:
        load_tree(root)
    return str(caught.value)


def test_preload_sends_what_it_names_in_list_order(tmp_path: Path) -> None:
    write(
        tmp_path / "coding/AGENT.md",
        agent_md("coding", "reads: ['../books']\npreload: ['./knowledge', './a.md', '../books']\n"),
    )
    write(tmp_path / "coding/knowledge/z.md", "Zed.\n")
    write(tmp_path / "coding/knowledge/b.md", "Bee.\n")
    write(tmp_path / "coding/a.md", "Own.\n")
    write(tmp_path / "coding/left-out.md", "Not named.\n")
    write(tmp_path / "books/x.md", "Shared.\n")
    assert load_tree(tmp_path)["coding"].preload == [
        Text(text="<<<knowledge/b.md>>>\nBee.\n"),
        Text(text="<<<knowledge/z.md>>>\nZed.\n"),
        Text(text="<<<a.md>>>\nOwn.\n"),
        Text(text="<<<../books/x.md>>>\nShared.\n"),
    ]


async def test_no_preload_sends_nothing(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", ""))
    write(tmp_path / "a/notes.md", "Own.\n")
    assert load_tree(tmp_path)["a"].preload == []
    run = await Engine(tmp_path, models={"strong": Scripted(text_reply("ok"))}).run(
        "a", "acme", "x"
    )
    assert isinstance(run, Generation) and run.preload == []


@pytest.mark.parametrize(
    ("entry", "names"),
    [
        (".", ["desk/guide.md", "knowledge/plans.md", "rules/refunds.md"]),
        ("./desk", ["desk/guide.md"]),
        ("./desk/writer", ["desk/writer/notes.md"]),
        ("./questions", ["questions/note.txt"]),
        (
            "./tools",
            [
                "tools/lookup_order.json",
                "tools/lookup_order.py",
                "tools/now.json",
                "tools/now.py",
            ],
        ),
    ],
)
def test_a_folder_entry_leaves_out_declarations_and_inner_agents(
    fixtures: Path, tmp_path: Path, entry: str, names: list[str]
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    write(support / "desk/guide.md", "The guide.\n")
    write(support / "desk/writer/AGENT.md", agent_md("writer", ""))
    write(support / "desk/writer/notes.md", "Inner.\n")
    write(support / "questions/note.txt", "A note.\n")
    agent = support / "AGENT.md"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace("\n---\n", f"\npreload: ['{entry}']\n---\n", 1),
        encoding="utf-8",
    )
    assert sent(engine(root, Scripted())._agents["support"]) == [f"<<<{name}>>>" for name in names]


@pytest.mark.parametrize(
    ("entry", "names"),
    [
        ("./skills/plain.md", ["skills/plain.md"]),
        ("./skills", ["skills/deep/SKILL.md", "skills/plain.md"]),
    ],
)
async def test_a_skill_named_directly_is_preloaded_and_still_listed(
    tmp_path: Path, entry: str, names: list[str]
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: ['{entry}']\n"))
    write(tmp_path / "a/skills/plain.md", "---\ndescription: Plain\n---\nDo it plainly.\n")
    write(tmp_path / "a/skills/deep/SKILL.md", "Go deep.\n")
    built = Engine(tmp_path, models={"strong": Scripted(text_reply("ok"))})
    run = await built.run("a", "acme", "x")
    assert isinstance(run, Generation)
    assert sent(built._spec("a")) == [f"<<<{name}>>>" for name in names]
    assert "skills/plain.md — Plain" in run.system
    assert built._spec("a").skills["plain"].path == "skills/plain.md"


@pytest.mark.parametrize("linked", [False, True], ids=["in-the-tree", "a-linked-agent"])
@pytest.mark.parametrize("entry", ["../skills/tone.md", "../skills"])
def test_an_inherited_skill_can_be_preloaded(tmp_path: Path, entry: str, linked: bool) -> None:
    root = tmp_path / "agents"
    write(root / "skills/tone.md", "Keep it warm.\n")
    place = tmp_path / "outside/desk" if linked else root / "desk"
    write(place / "AGENT.md", agent_md("desk", f"preload: ['{entry}']\n"))
    if linked:
        (root / "desk").symlink_to(place)
    spec = load_tree(root)["desk"]
    assert spec.preload == [Text(text="<<<../skills/tone.md>>>\nKeep it warm.\n")]
    assert spec.skills["tone"].path == "../skills/tone.md"


NOT_READ = "outside its folder and the folders it reads; add its folder to reads:, or move the file into the agent's folder"
HOLDS_THE_AGENT = "in a folder that holds the agent, and reads: cannot name such a folder; move the file into the agent's folder or into a folder it reads"


@pytest.mark.parametrize(
    ("written", "reason"),
    [
        ("../policy.md", HOLDS_THE_AGENT),
        ("..", HOLDS_THE_AGENT),
        ("../other/x.md", NOT_READ),
        ("../other", NOT_READ),
    ],
)
def test_a_path_outside_what_the_agent_reads_is_refused(
    tmp_path: Path, written: str, reason: str
) -> None:
    write(tmp_path / "policy.md", "Policy.\n")
    write(tmp_path / "other/x.md", "Other.\n")
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: ['{written}']\n"))
    assert load_error(tmp_path) == f"tree: agent 'a' preloads {written!r}, {reason}"


@pytest.mark.parametrize("written", ["./nope.md", "../books/nope.md", "./a.md/below", "./gone"])
def test_a_missing_path_is_refused(tmp_path: Path, written: str) -> None:
    write(
        tmp_path / "a/AGENT.md",
        agent_md("a", f"reads: ['../books']\npreload: ['{written}']\n"),
    )
    write(tmp_path / "a/a.md", "Own.\n")
    write(tmp_path / "books/x.md", "Shared.\n")
    (tmp_path / "a/gone").symlink_to(tmp_path / "a/removed.md")
    assert load_error(tmp_path) == (
        f"tree: agent 'a' preloads {written!r}, which does not exist; write its path from the agent's folder"
    )


@pytest.mark.parametrize(
    ("entries", "one", "other"),
    [
        ("['./knowledge', './knowledge/a.md']", "./knowledge", "./knowledge/a.md"),
        ("['./knowledge/a.md', './knowledge']", "./knowledge/a.md", "./knowledge"),
        ("['./knowledge', './knowledge/']", "./knowledge", "./knowledge/"),
        ("['./knowledge', '../a/knowledge']", "./knowledge", "../a/knowledge"),
        ("['./a.md', './a.md']", "./a.md", "./a.md"),
        ("['.', './knowledge/a.md']", ".", "./knowledge/a.md"),
    ],
)
def test_an_overlap_is_refused(tmp_path: Path, entries: str, one: str, other: str) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: {entries}\n"))
    write(tmp_path / "a/a.md", "Own.\n")
    write(tmp_path / "a/knowledge/a.md", "Known.\n")
    assert load_error(tmp_path) == (
        f"tree: agent 'a' preloads {one!r} and {other!r}, which send the same file; keep one"
    )


def test_a_root_agent_s_own_folder_overlaps_what_is_inside_it(tmp_path: Path) -> None:
    write(tmp_path / "AGENT.md", agent_md("lead", "preload: ['.', './notes.md']\n"))
    write(tmp_path / "notes.md", "Own.\n")
    assert load_error(tmp_path) == (
        "tree: agent 'lead' preloads '.' and './notes.md', which send the same file; keep one"
    )


@pytest.mark.parametrize(
    ("entries", "one", "other"),
    [
        ("['./v2', './latest']", "./v2", "./latest"),
        ("['./x.md', './y.md']", "./x.md", "./y.md"),
        ("['./k', './to-k.md']", "./k", "./to-k.md"),
        ("['../books', './to-books']", "../books", "./to-books"),
    ],
)
def test_two_entries_that_send_one_file_through_a_link_are_refused(
    tmp_path: Path, entries: str, one: str, other: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"reads: ['../books']\npreload: {entries}\n"))
    write(tmp_path / "a/v2/v.md", "V.\n")
    (tmp_path / "a/latest").symlink_to("v2")
    write(tmp_path / "a/x.md", "X.\n")
    (tmp_path / "a/y.md").symlink_to("x.md")
    write(tmp_path / "a/k/n.md", "N.\n")
    (tmp_path / "a/to-k.md").symlink_to("k/n.md")
    write(tmp_path / "books/b.md", "B.\n")
    (tmp_path / "a/to-books").symlink_to("../books")
    assert load_error(tmp_path) == (
        f"tree: agent 'a' preloads {one!r} and {other!r}, which send the same file; keep one"
    )


@pytest.mark.parametrize(
    ("entries", "names"),
    [
        ("['.', './skills/s.md']", ["notes.md", "sub/guide.md", "skills/s.md"]),
        ("['.', './tools/README.md']", ["notes.md", "sub/guide.md", "tools/README.md"]),
        ("['./sub', './sub/inner']", ["sub/guide.md", "sub/inner/notes.md"]),
    ],
)
def test_entries_that_send_no_file_in_common_all_go(
    tmp_path: Path, entries: str, names: list[str]
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: {entries}\n"))
    write(tmp_path / "a/notes.md", "Own.\n")
    write(tmp_path / "a/skills/s.md", "A skill.\n")
    write(tmp_path / "a/tools/README.md", "Tools.\n")
    write(tmp_path / "a/sub/guide.md", "Guide.\n")
    write(tmp_path / "a/sub/inner/AGENT.md", agent_md("inner", ""))
    write(tmp_path / "a/sub/inner/notes.md", "Inner.\n")
    assert sent(load_tree(tmp_path)["a"]) == [f"<<<{name}>>>" for name in names]


@pytest.mark.parametrize(
    ("entry", "folder", "names"),
    [
        ("./k", "a/k", ["k/link.md", "k/x.md"]),
        ("../books/k", "books/k", ["../books/k/link.md", "../books/k/x.md"]),
        ("../skills/deep", "skills/deep", ["../skills/deep/link.md", "../skills/deep/x.md"]),
    ],
)
def test_a_folder_entry_sends_a_link_the_map_lists_under_it(
    tmp_path: Path, entry: str, folder: str, names: list[str]
) -> None:
    root = tmp_path / "agents"
    write(root / "a/AGENT.md", agent_md("a", f"reads: ['../books']\npreload: ['{entry}']\n"))
    write(root / "books/other.md", "Other.\n")
    write(root / folder / "x.md", "X.\n")
    write(root / Path(folder).parent / "n.md", "N.\n")
    (root / folder / "link.md").symlink_to("../n.md")
    write(tmp_path / "outside.md", "Secret.\n")
    (root / folder / "out.md").symlink_to(tmp_path / "outside.md")
    assert load_tree(root)["a"].preload == [
        Text(text=f"<<<{names[0]}>>>\nN.\n"),
        Text(text=f"<<<{names[1]}>>>\nX.\n"),
    ]


def test_a_folder_link_entry_goes_under_the_link_s_name(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", "preload: ['./latest']\n"))
    write(tmp_path / "a/v2/x.md", "One.\n")
    (tmp_path / "a/latest").symlink_to("v2")
    assert load_tree(tmp_path)["a"].preload == [Text(text="<<<latest/x.md>>>\nOne.\n")]


def test_a_sub_folder_of_a_read_folder_is_preloaded(tmp_path: Path) -> None:
    write(
        tmp_path / "coding/AGENT.md",
        agent_md("coding", "reads: ['../books']\npreload: ['../books/sub']\n"),
    )
    write(tmp_path / "books/other.md", "Other.\n")
    write(tmp_path / "books/sub/x.md", "Inside.\n")
    assert load_tree(tmp_path)["coding"].preload == [
        Text(text="<<<../books/sub/x.md>>>\nInside.\n")
    ]


def test_an_entry_the_engine_cannot_reach_is_not_called_missing(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", "preload: ['./locked/x.md']\n"))
    write(tmp_path / "a/locked/x.md", "Locked.\n")
    locked = tmp_path / "a/locked"
    locked.chmod(0)
    try:
        assert load_error(tmp_path) == (
            f"tree: cannot read {(locked / 'x.md').resolve()}: Permission denied; make it readable to the engine"
        )
    finally:
        locked.chmod(0o755)


@pytest.mark.parametrize("value", ["[]", ""], ids=["empty-list", "no-value"])
def test_an_empty_preload_is_refused(tmp_path: Path, value: str) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: {value}".rstrip() + "\n"))
    assert load_error(tmp_path) == (
        f"frontmatter: invalid agent {tmp_path / 'a/AGENT.md'}; preload: it is empty; leave preload: out to send nothing, or name what to send, such as preload: ['./knowledge']"
    )


def test_preload_true_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", "preload: true\n"))
    assert load_error(tmp_path) == (
        f"frontmatter: invalid agent {tmp_path / 'a/AGENT.md'}; preload: input should be a valid list"
    )


@pytest.mark.parametrize(
    ("written", "reason"),
    [
        (
            "knowledge",
            "which is not a path from its folder; write it as './knowledge' or '../shared', with no leading /, no .. after a name, no hidden folder, no empty segment and no backslash",
        ),
        ("../../x.md", "above the root of the tree; a path in AGENT.md stays inside the tree"),
    ],
)
def test_an_entry_follows_the_rule_for_a_path_from_the_agent_s_folder(
    tmp_path: Path, written: str, reason: str
) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: ['{written}']\n"))
    assert load_error(tmp_path) == f"tree: agent 'a' preloads {written!r}, {reason}"


def test_a_preloaded_file_in_a_read_folder_counts_in_the_fingerprint(tmp_path: Path) -> None:
    write(
        tmp_path / "coding/AGENT.md",
        agent_md("coding", "reads: ['../books']\npreload: ['../books/x.md']\n"),
    )
    write(tmp_path / "books/x.md", "One.\n")
    spec = load_tree(tmp_path)["coding"]
    assert spec.preload == [Text(text="<<<../books/x.md>>>\nOne.\n")]
    write(tmp_path / "books/x.md", "Two.\n")
    assert load_tree(tmp_path)["coding"].fingerprint != spec.fingerprint


@pytest.mark.parametrize(
    ("written", "name"), [("./__pycache__/x.pyc", "__pycache__/x.pyc"), ("./pub", "pub/secret.md")]
)
def test_a_name_read_hides_is_refused(tmp_path: Path, written: str, name: str) -> None:
    write(tmp_path / "a/AGENT.md", agent_md("a", f"preload: ['{written}']\n"))
    write(tmp_path / "a/__pycache__/x.pyc", "cached\n")
    write(tmp_path / "a/.private/secret.md", "Secret.\n")
    (tmp_path / "a/pub").symlink_to(tmp_path / "a/.private")
    assert load_error(tmp_path) == (
        f"tree: agent 'a' cannot preload: tools/builtins: {name} is hidden; names starting with . and __pycache__ are not part of the agent's folder; fix that file, or drop it from preload:"
    )


@pytest.mark.parametrize(("written", "target"), [("./leak.md", "secret.md"), ("./leak", "secrets")])
def test_a_link_out_of_the_folder_is_refused(tmp_path: Path, written: str, target: str) -> None:
    root = tmp_path / "agents"
    write(root / "a/AGENT.md", agent_md("a", f"preload: ['{written}']\n"))
    write(tmp_path / "secret.md", "Secret.\n")
    write(tmp_path / "secrets/x.md", "Secret.\n")
    (root / "a" / written.removeprefix("./")).symlink_to(tmp_path / target)
    assert load_error(root) == (
        f"tree: agent 'a' cannot preload: jail: path outside the agent directory: {written.removeprefix('./')}; fix that file, or drop it from preload:"
    )
