from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from rfnry_engine import Engine
from rfnry_engine.errors import InputError
from rfnry_engine.manifest import ModelEntry
from rfnry_engine.models import Message, Text
from rfnry_engine.records import Generation
from rfnry_engine.tools.engine_tools import EngineTools

from .fakes import Scripted, delegating, text_reply
from .trees import provided, write_folder

PNG = b"\x89PNG\r\n\x1a\ndiagram"


class Pictures(Scripted):
    media_types = {"image/*"}


class Documents(Scripted):
    media_types = {"application/pdf"}


class Named(Scripted):
    def __init__(self, name: str, *replies: Any) -> None:
        super().__init__(*replies)
        self.model = name


LEAD: dict[str, Any] = {"kind": "generation", "model": "strong", "instructions": "lead.md"}
WRITER: dict[str, Any] = {"kind": "generation", "model": "fast", "instructions": "writer.md"}
BODIES = {"lead.md": "Work.\n", "writer.md": "Work.\n"}


def pair(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {**LEAD, "orchestration": {"delegate_to": ["writer"]}},
            "writer": WRITER,
        },
        files=BODIES,
    )


async def test_each_agent_runs_on_the_model_its_entry_names(tmp_path: Path) -> None:
    strong = Scripted(delegating("writer"), text_reply("done", model="strong-1"))
    fast = Scripted(text_reply("drafted", model="fast-1"))
    run = await Engine(pair(tmp_path), providers=provided({"strong": strong, "fast": fast})).run(
        "lead", "acme", "x"
    )
    assert isinstance(run, Generation)
    assert (run.reply, run.model, run.children[0].model) == ("done", "strong-1", "fast-1")
    child = run.children[0]
    assert isinstance(child, Generation)
    assert [m.model for m in child.conversation if m.role == "assistant"] == ["fast-1"]
    assert len(strong.calls) == 2 and len(fast.calls) == 1


def test_the_engines_models_are_one_view_built_with_the_engine(tmp_path: Path) -> None:
    engine = Engine(pair(tmp_path), providers=provided({"strong": Scripted(), "fast": Scripted()}))
    assert isinstance(engine.models, MappingProxyType)
    assert engine.models is engine.models
    with pytest.raises(TypeError):
        engine.models["strong"] = ModelEntry(provider="fast")  # type: ignore[index]
    with pytest.raises(AttributeError):
        engine.models = {}  # type: ignore[misc]


def with_picture(root: Path) -> Path:
    return write_folder(
        root,
        {
            "lead": {**LEAD, "files": {"preload": ["lead/diagram.png"]}},
            "writer": {**WRITER, "files": {"preload": ["writer/diagram.png"]}},
        },
        files=BODIES,
    )


def test_a_preload_loads_a_file_its_model_does_not_take(tmp_path: Path) -> None:
    (tmp_path / "lead").mkdir()
    (tmp_path / "writer").mkdir()
    (tmp_path / "lead/diagram.png").write_bytes(PNG)
    (tmp_path / "writer/diagram.png").write_bytes(PNG)
    root = with_picture(tmp_path)
    loaded = Engine(root, providers=provided({"strong": Pictures(), "fast": Documents()}))
    assert loaded._agents["writer"].preload == [
        Text(text="<<<writer/diagram.png>>>\nimage/png, 15 bytes; Load opens it")
    ]


async def test_load_follows_the_media_types_of_the_agent_s_model(tmp_path: Path) -> None:
    (tmp_path / "lead").mkdir()
    (tmp_path / "writer").mkdir()
    (tmp_path / "lead/diagram.png").write_bytes(PNG)
    (tmp_path / "writer/diagram.png").write_bytes(PNG)
    root = with_picture(tmp_path)
    engine = Engine(root, providers=provided({"strong": Pictures(), "fast": Documents()}))
    assert await EngineTools(engine, agent="lead", scope="acme").dispatch(
        "Load", {"path": "lead/diagram.png"}
    )
    with pytest.raises(InputError, match="does not take image/png"):
        await EngineTools(engine, agent="writer", scope="acme").dispatch(
            "Load", {"path": "writer/diagram.png"}
        )


def test_engine_models_holds_the_entries_and_there_is_no_generation(tmp_path: Path) -> None:
    engine = Engine(pair(tmp_path), providers=provided({"strong": Scripted(), "fast": Scripted()}))
    assert engine.models == {
        "strong": ModelEntry(provider="strong"),
        "fast": ModelEntry(provider="fast"),
    }
    assert not hasattr(engine, "generation") and not hasattr(engine, "decision")


async def test_the_fingerprint_is_the_same_whichever_adapter_a_name_maps_to(
    tmp_path: Path,
) -> None:
    root = pair(tmp_path)
    prints = []
    for strong in (Scripted(text_reply("a")), Named("other", text_reply("b"))):
        engine = Engine(root, providers=provided({"strong": strong, "fast": Pictures()}))
        run = await engine.run("lead", "acme", "x")
        assert isinstance(run, Generation)
        prints.append(run.fingerprint.agent)
    assert prints[0] == prints[1]


async def test_reasoning_does_not_cross_to_an_agent_on_another_model(tmp_path: Path) -> None:
    thought = {"thought": "plan"}
    strong = Named("model-a", text_reply("planned", reasoning=thought))
    fast = Named("model-b", text_reply("written"))
    root = pair(tmp_path)
    first = await Engine(
        root, providers=provided({"strong": strong, "fast": Named("model-b")})
    ).run("lead", "acme", "x")
    assert isinstance(first, Generation)
    history: list[Message] = first.conversation
    assert any(m.reasoning == thought for m in history)
    await Engine(root, providers=provided({"strong": Named("model-a"), "fast": fast})).run(
        "writer", "acme", "y", history=history
    )
    sent = [m for m in fast.calls[0][1] if m.role == "assistant"]
    assert sent and all(m.reasoning is None for m in sent)
