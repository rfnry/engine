import time
from pathlib import Path

import pytest

from rfnry_engine.errors import InputError
from rfnry_engine.jail import Jail
from rfnry_engine.models import Audio, Document, Image
from rfnry_engine.tools import builtins as builtins_module
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.tools import Tools

SUPPORT = Path(__file__).parents[1] / "fixtures" / "agents" / "support"


def _tools() -> Tools:
    return Tools(builtins(SUPPORT))


def test_the_five_builtins() -> None:
    assert _tools().names == ["Read", "Grep", "Glob", "LS", "Skill"]


async def test_read_inside_and_outside() -> None:
    assert "refunded in full" in await _tools().dispatch("Read", {"path": "rules/refunds.md"})
    with pytest.raises(InputError, match="outside the agent directory"):
        await _tools().dispatch("Read", {"path": "../billing/AGENT.md"})


async def test_grep_files_and_content() -> None:
    files = await _tools().dispatch("Grep", {"pattern": "refund", "case_insensitive": True})
    assert str(files).splitlines() == ["rules/_index.md", "rules/refunds.md"]
    lines = await _tools().dispatch("Grep", {"pattern": "refunded", "mode": "content"})
    assert lines == "rules/refunds.md:1:A duplicate charge is refunded in full without approval."


async def test_glob_and_ls() -> None:
    found = await _tools().dispatch("Glob", {"pattern": "**/*.md"})
    assert "tools/lookup_order.md" in str(found).splitlines()
    listing = await _tools().dispatch("LS", {})
    assert "rules/" in str(listing).splitlines() and "AGENT.md" in str(listing).splitlines()


async def test_skill_body_without_frontmatter() -> None:
    assert await _tools().dispatch("Skill", {"name": "escalate"}) == 'Say "escalating" and stop.'
    with pytest.raises(InputError, match="skill not found"):
        await _tools().dispatch("Skill", {"name": "ghost"})


async def test_bad_arguments_are_refused() -> None:
    with pytest.raises(InputError, match="offset"):
        await _tools().dispatch("Read", {"path": "AGENT.md", "offset": -1})


@pytest.fixture
def escape(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    (root / "skills").mkdir(parents=True)
    (root / "rules").mkdir()
    (root / "rules" / "inside.md").write_text("secret inside\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret outside\n")
    (root / "link.md").symlink_to(outside / "secret.md")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "skills" / "leak.md").symlink_to(outside / "secret.md")
    return root


def test_the_jail_refuses_dotdot_absolute_and_symlink_paths(escape: Path) -> None:
    jail = Jail(escape)
    assert jail.resolve("rules/inside.md") == escape.resolve() / "rules" / "inside.md"
    assert jail.resolve(".") == escape.resolve()
    for path in [
        "..",
        "../outside/secret.md",
        "rules/../../outside/secret.md",
        "/etc/passwd",
        str(escape.parent / "outside" / "secret.md"),
        "link.md",
        "linked/secret.md",
    ]:
        with pytest.raises(InputError, match="outside the agent directory"):
            jail.resolve(path)


async def test_no_builtin_reaches_through_a_symlink(escape: Path) -> None:
    tools = Tools(builtins(escape))
    for path in ["link.md", "linked/secret.md", "/etc/passwd"]:
        with pytest.raises(InputError, match="outside the agent directory"):
            await tools.dispatch("Read", {"path": path})
    with pytest.raises(InputError, match="outside the agent directory"):
        await tools.dispatch("Grep", {"pattern": "secret", "path": "linked"})
    with pytest.raises(InputError, match="outside the agent directory"):
        await tools.dispatch("LS", {"path": "linked"})
    assert await tools.dispatch("Grep", {"pattern": "secret"}) == "rules/inside.md"
    assert (
        await tools.dispatch("Grep", {"pattern": "secret", "mode": "content"})
        == "rules/inside.md:1:secret inside"
    )
    assert str(await tools.dispatch("Glob", {"pattern": "**/*.md"})).splitlines() == [
        "rules/inside.md"
    ]
    for pattern in ["../outside/*.md", "/etc/*"]:
        with pytest.raises(InputError, match="inside the agent directory"):
            await tools.dispatch("Glob", {"pattern": pattern})
    for name in ["leak", "../rules/inside", "../link"]:
        with pytest.raises(InputError):
            await tools.dispatch("Skill", {"name": name})


@pytest.fixture
def mixed(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    root.mkdir()
    (root / "notes.md").write_text("needle in text\n")
    (root / "blob.bin").write_bytes(b"needle\x00\x01\x02")
    (root / "latin.txt").write_bytes(b"needle caf\xe9\n")
    return root


async def test_read_refuses_binary_files_and_points_to_load(mixed: Path) -> None:
    tools = Tools(builtins(mixed))
    for path in ["blob.bin", "latin.txt"]:
        with pytest.raises(
            InputError,
            match=rf"^builtins: {path} is not a text file; Load gives you the file itself$",
        ):
            await tools.dispatch("Read", {"path": path})
    assert await tools.dispatch("Read", {"path": "notes.md"}) == "needle in text\n"


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
GIF = b"GIF89a\x01\x00\x01\x00\x00"
WEBP = b"RIFF\x1a\x00\x00\x00WEBPVP8 "
PDF = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n"
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00"
CSV = b"region,amount\nnorth,120\n"


@pytest.fixture
def media(tmp_path: Path) -> Path:
    root = tmp_path / "agent"
    (root / "knowledge").mkdir(parents=True)
    for name, data in {
        "diagram.png": PNG,
        "photo.jpg": JPEG,
        "anim.gif": GIF,
        "shot.webp": WEBP,
        "rates.pdf": PDF,
        "call.mp3": MP3,
        "sales.csv": CSV,
        "export.dat": b"\x00\x01\x02",
    }.items():
        (root / "knowledge" / name).write_bytes(data)
    (root / "knowledge" / "notes.md").write_text("plain\n")
    return root


async def test_load_hands_the_model_any_file_as_itself(media: Path) -> None:
    tools = Tools([load_tool(media)])
    expected = {
        "diagram.png": Image(media_type="image/png", data=PNG),
        "photo.jpg": Image(media_type="image/jpeg", data=JPEG),
        "anim.gif": Image(media_type="image/gif", data=GIF),
        "shot.webp": Image(media_type="image/webp", data=WEBP),
        "rates.pdf": Document(media_type="application/pdf", name="rates.pdf", data=PDF),
        "call.mp3": Audio(media_type="audio/mpeg", data=MP3),
        "sales.csv": Document(media_type="text/csv", name="sales.csv", data=CSV),
        "export.dat": Document(
            media_type="application/octet-stream", name="export.dat", data=b"\x00\x01\x02"
        ),
    }
    for name, part in expected.items():
        assert await tools.dispatch("Load", {"path": f"knowledge/{name}"}) == [part], name


async def test_read_refuses_what_load_takes(media: Path) -> None:
    tools = Tools(builtins(media))
    with pytest.raises(InputError, match="is not a text file; Load gives you the file itself"):
        await tools.dispatch("Read", {"path": "knowledge/diagram.png"})


async def test_a_name_and_bytes_that_disagree_are_refused(tmp_path: Path) -> None:
    (tmp_path / "about.md").write_bytes(b"%PDF-1.7 is a version of the format\n")
    (tmp_path / "broken.png").write_bytes(b"\x00\x01not a png")
    (tmp_path / "scan").write_bytes(PNG)
    tools = Tools([load_tool(tmp_path), *builtins(tmp_path)])
    assert await tools.dispatch("Load", {"path": "about.md"}) == [
        Document(
            media_type="text/markdown",
            name="about.md",
            data=b"%PDF-1.7 is a version of the format\n",
        )
    ]
    with pytest.raises(
        InputError, match=r"^builtins: broken.png is not the image/png its name says$"
    ):
        await tools.dispatch("Load", {"path": "broken.png"})
    assert await tools.dispatch("Load", {"path": "scan"}) == [
        Image(media_type="image/png", data=PNG)
    ]


async def test_load_keeps_to_what_the_model_takes(media: Path) -> None:
    tools = Tools([load_tool(media, media_types={"image/*", "application/pdf"})])
    assert await tools.dispatch("Load", {"path": "knowledge/photo.jpg"}) == [
        Image(media_type="image/jpeg", data=JPEG)
    ]
    with pytest.raises(
        InputError,
        match=r"^builtins: the model does not take audio/mpeg files: knowledge/call.mp3$",
    ):
        await tools.dispatch("Load", {"path": "knowledge/call.mp3"})


async def test_load_refuses_files_over_its_cap(
    media: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtins_module, "_LOAD_BYTES", 10)
    with pytest.raises(InputError, match=r"^builtins: file exceeds 10 bytes: knowledge/rates.pdf$"):
        await Tools([load_tool(media)]).dispatch("Load", {"path": "knowledge/rates.pdf"})


async def test_load_follows_the_jail(media: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(PDF)
    (media / "knowledge" / "link.pdf").symlink_to(outside)
    tools = Tools([load_tool(media)])
    for path in ["knowledge/link.pdf", "../outside.pdf", str(outside)]:
        with pytest.raises(InputError, match="outside the agent directory"):
            await tools.dispatch("Load", {"path": path})


async def test_load_reaches_the_workspace_when_the_run_has_one(media: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "chart.png").write_bytes(PNG)
    alone = Tools([load_tool(media)])
    [schema] = alone.definitions()
    assert "where" not in schema["input_schema"]["properties"]
    with pytest.raises(InputError, match=r"^builtins: not a file: chart.png$"):
        await alone.dispatch("Load", {"path": "chart.png", "where": "workspace"})
    both = Tools([load_tool(media, work)])
    assert await both.dispatch("Load", {"path": "chart.png", "where": "workspace"}) == [
        Image(media_type="image/png", data=PNG)
    ]
    assert await both.dispatch("Load", {"path": "knowledge/rates.pdf"}) == [
        Document(media_type="application/pdf", name="rates.pdf", data=PDF)
    ]
    with pytest.raises(InputError):
        await both.dispatch("Load", {"path": "../agent/knowledge/rates.pdf", "where": "workspace"})


async def test_grep_skips_media_and_glob_and_ls_list_it(media: Path) -> None:
    tools = Tools(builtins(media))
    assert await tools.dispatch("Grep", {"pattern": "PDF|PNG|plain"}) == "knowledge/notes.md"
    assert "knowledge/diagram.png" in str(await tools.dispatch("Glob", {"pattern": "knowledge/*"}))
    assert "rates.pdf" in str(await tools.dispatch("LS", {"path": "knowledge"}))


def test_read_and_load_tell_the_model_what_each_is_for() -> None:
    [read] = [d for d in _tools().definitions() if d["name"] == "Read"]
    assert "text file" in read["description"] and "use Load" in read["description"]
    [load] = Tools([load_tool(SUPPORT)]).definitions()
    assert "as it is" in load["description"] and "see or hear it yourself" in load["description"]


async def test_grep_skips_binary_files(mixed: Path) -> None:
    tools = Tools(builtins(mixed))
    assert await tools.dispatch("Grep", {"pattern": "needle"}) == "notes.md"


async def test_grep_returns_promptly_on_catastrophic_backtracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtins_module, "_GREP_SECONDS", 0.5)
    root = tmp_path / "agent"
    root.mkdir()
    (root / "evil.md").write_text("a" * 5000 + "b\n")
    started = time.monotonic()
    with pytest.raises(InputError, match="pattern took longer than 0.5s"):
        await Tools(builtins(root)).dispatch("Grep", {"pattern": "(a+)+$", "mode": "content"})
    assert time.monotonic() - started < 3


async def test_grep_stops_at_its_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtins_module, "_GREP_BYTES", 250)
    root = tmp_path / "agent"
    root.mkdir()
    for index in range(5):
        (root / f"f{index}.md").write_text("needle\n" + "x" * 93)
    result = str(await Tools(builtins(root)).dispatch("Grep", {"pattern": "needle"}))
    assert result.splitlines() == [
        "f0.md",
        "f1.md",
        "[truncated: scan budget of 250 bytes reached]",
    ]


async def test_grep_skips_files_over_the_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtins_module, "_MAX_BYTES", 50)
    root = tmp_path / "agent"
    root.mkdir()
    (root / "big.md").write_text("needle\n" + "x" * 100)
    (root / "small.md").write_text("needle\n")
    assert await Tools(builtins(root)).dispatch("Grep", {"pattern": "needle"}) == "small.md"


async def test_read_selects_lines_by_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("one\ntwo\nthree\nfour\n")
    tools = Tools(builtins(tmp_path))
    assert (
        await tools.dispatch("Read", {"path": "notes.md", "offset": 1, "limit": 2}) == "two\nthree"
    )
    assert await tools.dispatch("Read", {"path": "notes.md", "offset": 2}) == "three\nfour"
    assert await tools.dispatch("Read", {"path": "notes.md", "limit": 1}) == "one"
    with pytest.raises(InputError, match="limit"):
        await tools.dispatch("Read", {"path": "notes.md", "limit": 0})
