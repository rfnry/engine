import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.compiled import load_folder
from rfnry_engine.engine import Engine
from rfnry_engine.errors import InputError
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

from .fakes import (
    Answering,
    MemoryBucket,
    Scripted,
    delegating,
    lead,
    said,
    text_reply,
)
from .trees import amend, provided, write_folder

PNG = b"\x89PNG\r\n\x1a\ndiagram"
PDF = b"%PDF-1.7 \xe2\x28\xa1 rates"
RULE = "A duplicate charge is refunded in full without approval.\n"


def tree(fixtures: Path, tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    shutil.copytree(fixtures, root)
    return root


def preload_all(root: Path) -> None:
    amend(root, "support", files={"preload": ["support/knowledge", "support/rules"]})


def furnish(support: Path) -> None:
    (support / "knowledge" / "diagram.png").write_bytes(PNG)
    (support / "knowledge" / "rates.pdf").write_bytes(PDF)
    (support / "rules" / "chargebacks").mkdir()
    (support / "rules" / "chargebacks" / "disputed.md").write_text(
        "A disputed charge waits for the bank.\n", encoding="utf-8"
    )
    (support / "rules" / ".draft.md").write_text("Not yet a rule.\n", encoding="utf-8")


def engine(root: Path, llm: GenerationModel) -> Engine:
    return Engine(root, providers=provided({"strong": llm, "judge": Answering()}))


def preloaded(root: Path) -> list[Part]:
    def text(name: str) -> Text:
        return Text(text=f"<<<{name}>>>\n" + (root / name).read_text(encoding="utf-8"))

    return [
        Text(text="<<<support/knowledge/diagram.png>>>\nimage/png, 15 bytes; Load opens it"),
        text("support/knowledge/plans.md"),
        Text(text="<<<support/knowledge/rates.pdf>>>\napplication/pdf, 18 bytes; Load opens it"),
        text("support/rules/chargebacks/disputed.md"),
        text("support/rules/refunds.md"),
    ]


async def test_preload_all_puts_every_rule_and_knowledge_file_before_the_model_in_path_order(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    plain = Scripted(text_reply("ok"))
    before = await engine(root, plain).run("support", "acme", "Charged twice.")
    preload_all(root)
    llm = Scripted(
        Reply(tool_calls=[ToolCall(id="c", name="now", args={})], stop_reason="tool_use"),
        text_reply("Refunded."),
    )
    async with engine(root, llm) as running:
        run = await running.run("support", "acme", "Charged twice.")
    assert isinstance(run, Generation) and isinstance(before, Generation)
    parts = preloaded(root)
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
    preload_all(root)
    llm = Scripted(text_reply("ok"))
    built = engine(root, llm)
    run = await built.run("support", "acme", "What does express cost?")
    assert isinstance(run, Generation)
    surface = EngineTools(built, agent="support", scope="acme")
    assert not any(isinstance(part, Image | Document) for part in run.preload)
    for line in (
        "<<<support/knowledge/diagram.png>>>\nimage/png, 15 bytes; Load opens it",
        "<<<support/knowledge/rates.pdf>>>\napplication/pdf, 18 bytes; Load opens it",
    ):
        assert Text(text=line) in run.preload
    for name, part in {
        "support/knowledge/rates.pdf": Document(
            media_type="application/pdf", name="rates.pdf", data=PDF
        ),
        "support/knowledge/diagram.png": Image(media_type="image/png", data=PNG),
    }.items():
        assert await surface.dispatch("Load", {"path": name}) == [part]


async def test_a_preloaded_link_goes_in_as_a_reference(fixtures: Path, tmp_path: Path) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    (support / "knowledge" / "notes.md").write_bytes(b"# Notes\n\x00 kept as bytes\n")
    (support / "knowledge" / "notes").symlink_to("notes.md")
    preload_all(root)
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "What does express cost?")
    assert isinstance(run, Generation)
    loaded = await EngineTools(built, agent="support", scope="acme").dispatch(
        "Load", {"path": "support/knowledge/notes"}
    )
    assert loaded == [
        Document(media_type="text/markdown", name="notes.md", data=b"# Notes\n\x00 kept as bytes\n")
    ]
    assert (
        Text(text="<<<support/knowledge/notes>>>\ntext/markdown, 24 bytes; Load opens it")
        in run.preload
    )


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
        read = ToolCall(
            id=f"r{len(self.calls)}", name="Read", args={"path": "support/rules/refunds.md"}
        )
        return Reply(tool_calls=[read], stop_reason="tool_use")


async def test_a_run_applies_a_rule_held_only_in_its_file_without_reading_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    assert (support / "rules" / "refunds.md").read_text(encoding="utf-8").endswith(RULE)
    assert (
        "without approval"
        not in Engine(root, providers=provided({"strong": Scripted(), "judge": None}))
        ._spec("support")
        .listing
    )
    reading = RuleFollower()
    on_demand = await engine(root, reading).run("support", "acme", "Charged twice.")
    assert isinstance(on_demand, Generation)
    assert [step.tool for step in on_demand.steps] == ["Read"]
    preload_all(root)
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
    await Engine(fixtures, providers=provided({"strong": llm, "judge": None})).run(
        "support",
        "acme",
        "Charged twice.",
        context="Acme is on the enterprise plan.",
        history=history,
    )
    [(system, messages, tools)] = llm.calls
    assert system == (
        "<<<AGENT>>>\nYou are the support desk. Be brief, cite the rule you apply.\n\n"
        "<<<FILES>>>\nsupport/\n  knowledge/\n    plans.md — The three plans\n  rules/\n"
        "    refunds.md — When a refund is owed\n\n"
        "<<<SKILLS>>>\nsupport/skills/escalate/SKILL.md — Hand a conversation to a person\n\n"
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


async def test_a_preloaded_file_that_is_not_the_type_its_name_says_loads_and_load_refuses_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    (root / "support" / "knowledge" / "diagram.png").write_bytes(b"\0not a png")
    preload_all(root)
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "x")
    assert isinstance(run, Generation)
    assert (
        Text(text="<<<support/knowledge/diagram.png>>>\nimage/png, 10 bytes; Load opens it")
        in run.preload
    )
    refused = await refusal(root, Scripted(), "support/knowledge/diagram.png")
    assert refused == "tools/builtins: diagram.png is not the image/png its name says"


async def test_the_preload_is_read_when_the_tree_loads_as_the_fingerprint_is_taken(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    preload_all(root)
    llm = Scripted(text_reply("a"), text_reply("b"))
    loaded = engine(root, llm)
    (support / "rules" / "refunds.md").write_text("Refunds need approval.\n", encoding="utf-8")
    old = await loaded.run("support", "acme", "x")
    new = await engine(root, Scripted(text_reply("c"))).run("support", "acme", "x")
    assert isinstance(old, Generation) and isinstance(new, Generation)
    held = f"<<<support/rules/refunds.md>>>\n---\ndescription: When a refund is owed\n---\n{RULE}"
    assert Text(text=held) in old.preload
    assert Text(text="<<<support/rules/refunds.md>>>\nRefunds need approval.\n") in new.preload
    assert old.fingerprint != new.fingerprint


async def test_a_conversation_passed_back_carries_no_second_copy_of_the_folder(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    preload_all(root)
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
    (root / "lead" / "rules").mkdir(parents=True)
    (root / "lead" / "rules" / "tone.md").write_text("Be kind.\n", encoding="utf-8")
    (root / "writer" / "knowledge").mkdir(parents=True)
    (root / "writer" / "knowledge" / "style.md").write_text("Short words.\n", encoding="utf-8")
    amend(root, "lead", files={"preload": ["lead"]})
    amend(root, "writer", files={"preload": ["writer"]})
    llm = Scripted(delegating("writer", "Draft it."), text_reply("Drafted."), text_reply("Sent."))
    run = await Engine(root, providers=provided({"strong": llm, "judge": Answering()})).run(
        "lead", "acme", "Go."
    )
    assert isinstance(run, Generation)
    [child] = run.children
    assert isinstance(child, Generation)
    assert run.preload == [Text(text="<<<lead/rules/tone.md>>>\nBe kind.\n")]
    assert child.preload == [Text(text="<<<writer/knowledge/style.md>>>\nShort words.\n")]
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
    preload_all(root)
    refinery = Refinery(storage)
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    built = Engine(
        root,
        refinery=refinery,
        providers=provided({"strong": Scripted(text_reply("Sorry.")), "judge": None}),
    )
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
    candidate = Engine(root, providers=provided({"strong": candidate_model, "judge": None}))
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
    preload_all(root)
    run = await engine(root, Scripted(text_reply("ok"))).run("support", "acme", "x")
    assert isinstance(run, Generation)
    held = "<<<support/knowledge/archive.bin>>>\napplication/octet-stream, 20971521 bytes; Load opens it"
    assert Text(text=held) in run.preload
    load = await refusal(root, Scripted(), "support/knowledge/archive.bin")
    assert load == "tools/builtins: file exceeds 20971520 bytes: support/knowledge/archive.bin"


async def test_a_file_over_read_s_cap_is_in_the_fingerprint_though_the_preload_never_reads_it(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    big = root / "support" / "knowledge" / "notes.txt"
    big.write_text("a" * (5 * 1024 * 1024 + 1), encoding="utf-8")
    preload_all(root)
    before = engine(root, Scripted())._agents["support"].fingerprint
    big.write_text("b" + "a" * (5 * 1024 * 1024), encoding="utf-8")
    assert engine(root, Scripted())._agents["support"].fingerprint != before


async def test_an_agent_without_load_is_not_told_load_opens_a_file(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    support = root / "support"
    furnish(support)
    preload_all(root)
    amend(root, "support", tools={"exclude": ["Grep", "Glob", "LS", "Load"]})
    spec = engine(root, Scripted())._agents["support"]
    assert Text(text="<<<support/knowledge/diagram.png>>>\nimage/png, 15 bytes") in spec.preload
    assert not any("Load opens it" in getattr(part, "text", "") for part in spec.preload)


async def test_a_text_file_over_read_s_cap_goes_in_as_a_reference(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    name = "support/knowledge/notes.txt"
    (root / name).write_text("a" * (5 * 1024 * 1024 + 1), encoding="utf-8")
    read = await refusal(root, Scripted(), name, tool="Read")
    assert read == f"tools/builtins: file exceeds 5242880 bytes: {name}"
    preload_all(root)
    built = engine(root, Scripted(text_reply("ok")))
    run = await built.run("support", "acme", "x")
    assert isinstance(run, Generation)
    assert Text(text=f"<<<{name}>>>\ntext/plain, 5242881 bytes; Load opens it") in run.preload
    loaded = await EngineTools(built, agent="support", scope="acme").dispatch(
        "Load", {"path": name}
    )
    assert loaded == [
        Document(media_type="text/plain", name="notes.txt", data=(root / name).read_bytes())
    ]


async def test_a_file_the_model_does_not_take_loads_the_tree_and_load_refuses_it_when_asked(
    fixtures: Path, tmp_path: Path
) -> None:
    root = tree(fixtures, tmp_path)
    name = "support/knowledge/greeting.wav"
    (root / name).write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00")
    preload_all(root)
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
    preload_all(root)
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
    preloaded = [
        "support/knowledge/diagram.png",
        "support/knowledge/plans.md",
        "support/knowledge/rates.pdf",
        "support/rules/chargebacks/disputed.md",
        "support/rules/refunds.md",
    ]
    opens = {name: 1 + name.endswith(".md") for name in preloaded}
    assert {name: opened.count(root / name) for name in preloaded} == opens
    assert spec.fingerprint == load_folder(root)["support"].fingerprint


async def test_preload_sends_what_it_names_in_list_order(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {
            "coding": {
                "kind": "generation",
                "model": "strong",
                "instructions": "coding.md",
                "files": {"preload": ["knowledge", "a.md", "books"]},
            }
        },
        files={
            "coding.md": "Work.\n",
            "knowledge/z.md": "Zed.\n",
            "knowledge/b.md": "Bee.\n",
            "a.md": "Own.\n",
            "left-out.md": "Not named.\n",
            "books/x.md": "Shared.\n",
        },
    )
    assert load_folder(tmp_path)["coding"].preload == [
        Text(text="<<<knowledge/b.md>>>\nBee.\n"),
        Text(text="<<<knowledge/z.md>>>\nZed.\n"),
        Text(text="<<<a.md>>>\nOwn.\n"),
        Text(text="<<<books/x.md>>>\nShared.\n"),
    ]


async def test_no_preload_sends_nothing(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {"kind": "generation", "model": "strong", "instructions": "a.md"}},
        files={"a.md": "Work.\n", "notes.md": "Own.\n"},
    )
    assert load_folder(folder)["a"].preload == []
    run = await Engine(folder, providers=provided({"strong": Scripted(text_reply("ok"))})).run(
        "a", "acme", "x"
    )
    assert isinstance(run, Generation) and run.preload == []
