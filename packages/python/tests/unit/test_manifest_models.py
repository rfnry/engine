import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.engine import Engine, check_folder
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.manifest import MANIFEST, ModelEntry, read_manifest
from rfnry_engine.models import Message, ModelChoice, Reply, ToolSpec
from rfnry_engine.records import Decision, Generation
from rfnry_engine.refinery.refinery import Refinery
from rfnry_engine.refinery.storage import Storage
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Answering, Scripted, delegating, text_reply
from .trees import provided, write_folder

PNG = b"\x89PNG\r\n\x1a\ndiagram"
GENERATION: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "a.md"}
DECISION: dict[str, Any] = {
    "kind": "decision",
    "model": "j",
    "instructions": "a.md",
    "questions": ["q/*.json"],
}
AREA = json.dumps(
    {"type": "choice", "instructions": "Which area?", "options": {"other": None, "billing": None}}
)
FILES = {"a.md": "Work.\n", "q/area.json": AREA}
ONE_ENTRY: dict[str, Any] = {"m": {"provider": "p"}}
NOT_HELD = r"^engine: model 'm' names provider 'ghost', which Engine\(providers=\) does not hold; it holds p$"
NO_GENERATE = r"^engine: agent 'a' is a generation agent, and model 'm' has no generate\(\)$"
NO_DECIDE = r"^engine: agent 't' is a decision agent, and model 'j' has no decide\(\)$"
UNFIT_PROVIDERS: list[Any] = [
    [("p", print)],
    {"": print},
    {" p": print},
    {1: print},
    {"p": "not callable"},
]


class Recording:
    def __init__(self, *adapters: Any) -> None:
        self.adapters = list(adapters)
        self.calls: list[tuple[str | None, str | None]] = []

    def __call__(self, model: str | None, effort: str | None, /) -> Any:
        self.calls.append((model, effort))
        return self.adapters.pop(0)


class Named(Scripted):
    def __init__(self, name: str, *replies: Reply) -> None:
        super().__init__(*replies)
        self.model = name


class NamedDecider(Answering):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.model = name


class Pictures(Scripted):
    media_types = {"image/*"}


class Documents(Scripted):
    media_types = {"application/pdf"}


class Lockstep:
    def __init__(self, name: str, step: asyncio.Barrier, *replies: Reply) -> None:
        self.name, self.step = name, step
        self.replies = list(replies)

    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        await self.step.wait()
        return self.replies.pop(0).model_copy(update={"model": self.name})


def _refused(root: Path, pattern: str) -> str:
    with pytest.raises(ConfigurationError) as raised:
        read_manifest(root)
    said = str(raised.value).replace(str(root), "<root>")
    assert said.startswith("manifest: ")
    assert re.search(pattern, said), said
    return said


def _one(root: Path, **entry: Any) -> Path:
    return write_folder(
        root, {"a": GENERATION}, models={"m": {"provider": "p", **entry}}, files=FILES
    )


def _crew(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {**GENERATION, "orchestration": {"delegate_to": ["triage", "writer"]}},
            "triage": {
                **DECISION,
                "orchestration": {"route": {"by": "area", "to": {"billing": "writer"}}},
            },
            "writer": {**GENERATION, "model": "w"},
        },
        models={
            "m": {"provider": "p", "model": "file-lead"},
            "j": {"provider": "p", "model": "file-judge"},
            "w": {"provider": "p", "model": "file-writer"},
        },
        files=FILES,
    )


def test_a_models_section_is_required(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).write_text(
        "version: 1\nagents:\n  a: {kind: generation, model: m, instructions: a.md}\n",
        encoding="utf-8",
    )
    _refused(tmp_path, r"models: field required")
    _refused(
        write_folder(tmp_path, {"a": GENERATION}, models={}),
        r"^manifest: models in <root>/rfnry\.yaml says nothing",
    )


def test_an_entry_takes_provider_model_and_effort(tmp_path: Path) -> None:
    assert read_manifest(_one(tmp_path, model="x", effort="high")).models == {
        "m": ModelEntry(provider="p", model="x", effort="high")
    }
    assert read_manifest(_one(tmp_path)).models["m"] == ModelEntry(provider="p")
    _refused(
        _one(tmp_path, api_key="x"),
        r"unknown key in <root>/rfnry\.yaml: models\.m\.api_key \(it takes provider, model, effort\)",
    )
    folder = write_folder(tmp_path, {"a": GENERATION}, models={"m": {"model": "x"}})
    _refused(folder, r"models\.m\.provider: field required")
    folder = write_folder(tmp_path, {"a": GENERATION}, models={"m": {}})
    _refused(folder, r"^manifest: models\.m in <root>/rfnry\.yaml says nothing")
    folder = write_folder(tmp_path, {"a": GENERATION}, models={"m": {"provider": " "}})
    _refused(folder, r"^manifest: models\.m\.provider in <root>/rfnry\.yaml says nothing")
    folder = write_folder(
        tmp_path, {"a": GENERATION}, models={**ONE_ENTRY, "x/y": {"provider": "p"}}
    )
    _refused(folder, r"^manifest: models\.x/y in <root>/rfnry\.yaml is not a plain name")


def test_an_agent_names_an_entry(tmp_path: Path) -> None:
    named = read_manifest(_one(tmp_path)).agents["a"]
    assert named.model == "m"
    _refused(
        write_folder(tmp_path, {"a": {**GENERATION, "model": "ghost"}}, models=ONE_ENTRY),
        r"^manifest: agents\.a\.model in <root>/rfnry\.yaml names 'ghost', which models: does not declare",
    )
    _refused(
        write_folder(
            tmp_path, {"a": {**GENERATION, "modes": {"x": {"model": "ghost"}}}}, models=ONE_ENTRY
        ),
        r"^manifest: agents\.a\.modes\.x\.model in <root>/rfnry\.yaml names 'ghost'",
    )


def test_an_entry_nothing_picks_is_refused(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path, {"a": GENERATION}, models={**ONE_ENTRY, "spare": {"provider": "p"}}
    )
    _refused(folder, r"^manifest: models\.spare in <root>/rfnry\.yaml is picked by nothing")


async def test_a_mode_picks_another_entry(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**GENERATION, "modes": {"quick": {"model": "q"}}}},
        models={
            "m": {"provider": "p", "model": "main"},
            "q": {"provider": "p", "model": "small", "effort": "low"},
        },
        files=FILES,
    )
    provider = Recording(Scripted(text_reply("one")), Scripted(text_reply("two")))
    engine = Engine(folder, providers={"p": provider})
    quick = await engine.run("a", "acme", "hi", mode="quick")
    plain = await engine.run("a", "acme", "hi")
    assert isinstance(quick, Generation) and isinstance(plain, Generation)
    assert (quick.reply, plain.reply) == ("one", "two")
    assert provider.calls == [("small", "low"), ("main", None)]


async def test_the_provider_gets_the_files_model_and_effort(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": GENERATION, "t": DECISION},
        models={
            "m": {"provider": "writing", "model": "claude-sonnet", "effort": "medium"},
            "j": {"provider": "judging", "model": "jev-latest", "effort": "high"},
        },
        files=FILES,
    )
    writing = Recording(Scripted(text_reply("one")), Scripted(text_reply("two")))
    judging = Recording(Answering())
    engine = Engine(folder, providers={"writing": writing, "judging": judging})
    assert (writing.calls, judging.calls) == ([], [])
    await engine.run("a", "acme", "hi")
    await engine.run("a", "acme", "hi")
    await engine.run("t", "acme", "hi")
    assert writing.calls == [("claude-sonnet", "medium"), ("claude-sonnet", "medium")]
    assert judging.calls == [("jev-latest", "high")]


async def test_a_delegated_run_on_the_same_entry_calls_the_provider_again(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "lead": {**GENERATION, "orchestration": {"delegate_to": ["writer"]}},
            "writer": GENERATION,
        },
        models=ONE_ENTRY,
        files=FILES,
    )
    lead, writer = Scripted(delegating("writer"), text_reply("done")), Scripted(text_reply("w"))
    provider = Recording(lead, writer)
    run = await Engine(folder, providers={"p": provider}).run("lead", "acme", "x")
    assert isinstance(run, Generation) and run.reply == "done"
    assert provider.calls == [(None, None), (None, None)]
    assert (len(lead.calls), len(writer.calls)) == (2, 1)


async def test_an_entry_of_only_a_provider_passes_none(tmp_path: Path) -> None:
    provider = Recording(Scripted(text_reply("ok")))
    await Engine(_one(tmp_path), providers={"p": provider}).run("a", "acme", "hi")
    assert provider.calls == [(None, None)]


@pytest.mark.parametrize(
    ("given", "holds"),
    [(None, "none"), ({}, "none"), ({"other": print}, "other")],
)
def test_a_missing_provider_is_refused_at_start(
    tmp_path: Path, given: dict[str, Any] | None, holds: str
) -> None:
    with pytest.raises(
        ConfigurationError,
        match=rf"^engine: model 'm' names provider 'p', which Engine\(providers=\) does not hold; it holds {holds}$",
    ):
        Engine(_one(tmp_path), providers=given)


def test_a_provider_may_go_unused(tmp_path: Path) -> None:
    spare = Recording()
    engine = Engine(_one(tmp_path), providers={"p": Recording(), "spare": spare})
    assert list(engine.agents) == ["a"] and spare.calls == []


def test_folder_check_needs_no_provider(tmp_path: Path) -> None:
    assert check_folder(_one(tmp_path))["a"].model == "m"


@pytest.mark.parametrize("providers", UNFIT_PROVIDERS)
def test_providers_that_do_not_fit_are_refused(tmp_path: Path, providers: Any) -> None:
    with pytest.raises(ConfigurationError, match="^engine: "):
        Engine(_one(tmp_path), providers=providers)


async def test_a_folder_takes_providers_and_a_tree_models(tmp_path: Path) -> None:
    file, tree = tmp_path / "file", tmp_path / "tree"
    file.mkdir()
    (tree / "a").mkdir(parents=True)
    (tree / "a/AGENT.md").write_text(
        "---\nname: a\nkind: generation\nmodel: m\ndescription: Works\n---\nWork.\n",
        encoding="utf-8",
    )
    _one(file)
    with pytest.raises(
        ConfigurationError,
        match=r"^engine: a folder with rfnry\.yaml takes providers=, not models=",
    ):
        Engine(file, models={"m": Scripted()})
    with pytest.raises(
        ConfigurationError,
        match=r"^engine: a folder with no rfnry\.yaml takes models=, \{\} for none, not providers=",
    ):
        Engine(tree, providers={"p": Recording()})
    with pytest.raises(ConfigurationError, match="^engine: a folder with no rfnry"):
        Engine(tree)
    assert Engine(tree, models={}).agents["a"].model == "m"
    with pytest.raises(ConfigurationError, match="has neither generate"):
        await Engine(tree, models={}).run("a", "acme", "x", models={"m": ModelChoice(model="x")})


def test_engine_models_holds_the_entries(tmp_path: Path) -> None:
    engine = Engine(_one(tmp_path, model="x"), providers={"p": Recording()})
    assert dict(engine.models) == {"m": ModelEntry(provider="p", model="x")}


async def test_a_choice_replaces_only_what_it_says(tmp_path: Path) -> None:
    provider = Recording(*(Scripted(text_reply("ok")) for _ in range(3)))
    engine = Engine(_one(tmp_path, model="file", effort="low"), providers={"p": provider})
    for choice in (
        ModelChoice(model="run"),
        ModelChoice(effort="high"),
        ModelChoice(model="run", effort="high"),
    ):
        await engine.run("a", "acme", "hi", models={"m": choice})
    assert provider.calls == [("run", "low"), ("file", "high"), ("run", "high")]


async def test_a_choice_reaches_delegated_and_routed_runs(tmp_path: Path) -> None:
    lead = Scripted(
        delegating("writer"), delegating("triage", call_id="d2"), text_reply("done", model="lead-1")
    )
    judge = NamedDecider("judge-1")
    writer = Scripted(
        text_reply("drafted", model="writer-1"), text_reply("routed", model="writer-2")
    )
    adapters: dict[str | None, Any] = {"run-lead": lead, "run-judge": judge, "run-writer": writer}
    built: list[tuple[str | None, str | None]] = []

    def provider(model: str | None, effort: str | None, /) -> Any:
        built.append((model, effort))
        return adapters[model]

    run = await Engine(_crew(tmp_path), providers={"p": provider}).run(
        "lead",
        "acme",
        "x",
        models={
            "m": ModelChoice(model="run-lead"),
            "j": ModelChoice(model="run-judge"),
            "w": ModelChoice(model="run-writer"),
        },
    )
    assert isinstance(run, Generation) and [s.is_error for s in run.steps] == [False, False]
    delegated, routing = run.children
    assert isinstance(delegated, Generation) and isinstance(routing, Decision)
    assert routing.routed_to is not None and routing.routed_to.model == "writer-2"
    assert built == [
        ("run-lead", None),
        ("run-writer", None),
        ("run-judge", None),
        ("run-writer", None),
    ]


async def test_a_choice_may_name_another_provider(tmp_path: Path) -> None:
    main, backup = Recording(), Recording(Scripted(text_reply("backed up")))
    run = await Engine(
        _one(tmp_path, model="file", effort="low"), providers={"p": main, "backup": backup}
    ).run("a", "acme", "hi", models={"m": ModelChoice(provider="backup")})
    assert isinstance(run, Generation) and run.reply == "backed up"
    assert main.calls == [] and backup.calls == [("file", "low")]


async def test_a_choice_naming_a_provider_nobody_gave_fails_before_any_model_call(
    tmp_path: Path, storage: Storage
) -> None:
    folder = write_folder(
        tmp_path, {"a": {**GENERATION, "description": "Works"}}, models=ONE_ENTRY, files=FILES
    )
    provider = Recording()
    engine = Engine(folder, providers={"p": provider})
    choice = {"m": ModelChoice(provider="ghost")}
    with pytest.raises(ConfigurationError, match=NOT_HELD):
        await engine.run("a", "acme", "hi", models=choice)
    with pytest.raises(ConfigurationError, match=NOT_HELD):
        engine.tool("a", scope="acme", models=choice)
    with pytest.raises(ConfigurationError, match=NOT_HELD):
        EngineTools(engine, agent="a", scope="acme", models=choice)
    with pytest.raises(ConfigurationError, match=NOT_HELD):
        Refinery(storage).replay("acme", engine, agent="a", models=choice)
    with pytest.raises(ConfigurationError, match=NOT_HELD):
        await Refinery(storage).label("acme", engine, models=choice)
    assert provider.calls == []


def test_an_empty_choice_is_refused() -> None:
    with pytest.raises(InputError, match="^models: a ModelChoice says nothing"):
        ModelChoice()


async def test_a_run_refuses_a_name_no_agent_runs_on(tmp_path: Path) -> None:
    provider = Recording()
    engine = Engine(_one(tmp_path), providers={"p": provider})
    with pytest.raises(
        ConfigurationError,
        match="^engine: no agent runs on model 'fsat'; the tree's agents run on m$",
    ):
        await engine.run("a", "acme", "hi", models={"fsat": ModelChoice(model="x")})
    assert provider.calls == []


async def test_a_run_may_hand_an_adapter_for_a_name(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {
            "lead": {**GENERATION, "orchestration": {"delegate_to": ["writer"]}},
            "writer": GENERATION,
        },
        models=ONE_ENTRY,
        files=FILES,
    )
    provider = Recording()
    engine = Engine(folder, providers={"p": provider})
    brought = Scripted(delegating("writer"), text_reply("drafted"), text_reply("done"))
    run = await engine.run("lead", "acme", "x", models={"m": brought})
    assert isinstance(run, Generation) and run.reply == "done"
    assert len(brought.calls) == 3 and provider.calls == []
    with pytest.raises(ConfigurationError, match="agent 'writer' is a generation agent"):
        await engine.run("writer", "acme", "x", models={"m": Answering()})


async def test_two_runs_at_once_keep_their_own_choices(tmp_path: Path) -> None:
    step = asyncio.Barrier(2)
    built: list[tuple[str | None, str | None]] = []

    def provider(model: str | None, effort: str | None, /) -> Lockstep:
        built.append((model, effort))
        assert model is not None
        who, name = model.split("-")
        replies = (
            [delegating("writer", name), text_reply(name)]
            if who == "lead"
            else [text_reply(f"drafted {name}")]
        )
        return Lockstep(model, step, *replies)

    folder = write_folder(
        tmp_path,
        {
            "lead": {**GENERATION, "orchestration": {"delegate_to": ["writer"]}},
            "writer": {**GENERATION, "model": "w"},
        },
        models={"m": {"provider": "p"}, "w": {"provider": "p"}},
        files=FILES,
    )
    engine = Engine(folder, providers={"p": provider})

    def own(name: str) -> dict[str, ModelChoice]:
        return {
            "m": ModelChoice(model=f"lead-{name}", effort=name),
            "w": ModelChoice(model=f"writer-{name}", effort=name),
        }

    async with asyncio.timeout(5):
        runs = await asyncio.gather(
            engine.run("lead", "acme", "x", models=own("one")),
            engine.run("lead", "acme", "x", models=own("other")),
        )
    for run, name in zip(runs, ("one", "other"), strict=True):
        assert isinstance(run, Generation)
        assert (run.reply, run.model) == (name, f"lead-{name}") and not run.steps[0].is_error
        child = run.children[0]
        assert isinstance(child, Generation)
        assert (child.reply, child.model) == (f"drafted {name}", f"writer-{name}")
    assert sorted(built) == [
        ("lead-one", "one"),
        ("lead-other", "other"),
        ("writer-one", "one"),
        ("writer-other", "other"),
    ]


async def test_the_kind_is_checked_on_what_the_provider_built(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION, "t": DECISION}, files=FILES)
    decider = Answering()
    writing, judging = Recording(decider, None), Recording(Scripted())
    engine = Engine(folder, providers={"m": writing, "j": judging})
    assert (writing.calls, judging.calls) == ([], [])
    with pytest.raises(ConfigurationError, match=NO_GENERATE):
        await engine.run("a", "acme", "hi")
    assert decider.calls == []
    with pytest.raises(ConfigurationError, match=NO_GENERATE):
        await engine.run("a", "acme", "hi")
    with pytest.raises(ConfigurationError, match=NO_DECIDE):
        await engine.run("t", "acme", "hi")


async def test_a_providers_own_exception_reaches_the_caller(tmp_path: Path) -> None:
    class Unavailable(Exception):
        pass

    problem = Unavailable("no key for this tenant")

    def provider(model: str | None, effort: str | None, /) -> Any:
        raise problem

    with pytest.raises(Unavailable) as raised:
        await Engine(_one(tmp_path), providers={"p": provider}).run("a", "acme", "hi")
    assert raised.value is problem


async def test_the_record_says_what_the_model_reports(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION, "t": DECISION}, files=FILES)
    writing = Recording(
        Named("built-1", text_reply("a", model="reported-1")),
        Named("built-2", text_reply("b")),
        Named("chosen", text_reply("c")),
    )
    engine = Engine(folder, providers={"m": writing, "j": Recording(NamedDecider("built-3"))})
    reported = await engine.run("a", "acme", "hi")
    assert reported.model == "reported-1"
    declared = await engine.run("a", "acme", "hi")
    assert declared.model == "built-2"
    decided = await engine.run("t", "acme", "hi")
    assert decided.model == "built-3"
    chosen = await engine.run("a", "acme", "hi", models={"m": ModelChoice(model="anything")})
    assert chosen.model == "chosen"


async def test_editing_models_leaves_the_fingerprint_alone(tmp_path: Path) -> None:
    async def fingerprint(**entry: Any) -> str:
        provider = Recording(Scripted(text_reply("ok")))
        run = await Engine(_one(tmp_path, **entry), providers={"p": provider}).run(
            "a", "acme", "hi"
        )
        return run.fingerprint.agent

    assert await fingerprint(model="one", effort="low") == await fingerprint(
        model="two", effort="high"
    )


async def test_engine_tools_read_what_the_provider_built(tmp_path: Path) -> None:
    folder = write_folder(
        tmp_path,
        {"a": {**GENERATION, "files": {"include": ["diagram.png"]}}},
        models=ONE_ENTRY,
        files=FILES,
    )
    (folder / "diagram.png").write_bytes(PNG)
    provider = Recording(Pictures(), Documents())
    engine = Engine(folder, providers={"p": provider})
    takes = await EngineTools(engine, agent="a", scope="acme").dispatch(
        "Load", {"path": "diagram.png"}
    )
    assert takes
    with pytest.raises(InputError, match="does not take image/png"):
        await EngineTools(
            engine, agent="a", scope="acme", models={"m": ModelChoice(model="words-only")}
        ).dispatch("Load", {"path": "diagram.png"})
    assert provider.calls == [(None, None), ("words-only", None)]


async def test_replay_and_label_take_choices(tmp_path: Path, storage: Storage) -> None:
    folder = write_folder(
        tmp_path,
        {"a": GENERATION, "judge": DECISION},
        models={"m": {"provider": "w", "model": "file"}, "j": {"provider": "d", "model": "file"}},
        files=FILES,
    )
    refinery = Refinery(storage)
    production = Engine(
        folder,
        refinery=refinery,
        providers=provided({"w": Scripted(text_reply("recorded")), "d": Answering()}),
    )
    await production.run("a", "acme", "Draft it")
    replaying = Recording(Scripted(text_reply("replayed", model="run-1")))
    candidate = Engine(folder, providers={"w": replaying, "d": Recording()})
    pairs = [
        pair
        async for pair in refinery.replay(
            "acme", candidate, agent="a", models={"m": ModelChoice(model="other", effort="high")}
        )
    ]
    assert [pair.error for pair in pairs] == [None]
    assert pairs[0].candidate is not None and pairs[0].candidate.model == "run-1"
    assert replaying.calls == [("other", "high")]
    judging = Recording(Answering())
    labeler = Engine(folder, providers={"w": Recording(), "d": judging})
    labeled = await refinery.label(
        "acme", labeler, judge="judge", models={"j": ModelChoice(model="better")}
    )
    assert (labeled.judged, labeled.errors) == (1, {})
    assert judging.calls == [("better", None)]
