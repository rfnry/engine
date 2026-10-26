import time
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine import Local, Process
from rfnry_engine.errors import InputError
from rfnry_engine.jail import Jail
from rfnry_engine.models import Audio, Document, Image, Text
from rfnry_engine.tools import builtins as builtins_module
from rfnry_engine.tools.builtins import _contents, builtins, load_tool, preload_parts
from rfnry_engine.tools.tools import Tools

SUPPORT = Path(__file__).parents[1] / "fixtures" / "agents" / "support"


def _tools() -> Tools:
    return Tools(builtins(SUPPORT))


def test_the_four_builtins() -> None:
    assert _tools().names == ["Read", "Grep", "Glob", "LS"]


def test_every_builtin_takes_the_paths_of_the_files_map() -> None:
    described = {d["name"]: d["description"] for d in _tools().definitions()}
    [load] = Tools([load_tool(SUPPORT)]).definitions()
    for text in [*described.values(), load["description"]]:
        assert "FILES map" in text


async def test_read_inside_and_outside() -> None:
    assert "refunded in full" in await _tools().dispatch("Read", {"path": "rules/refunds.md"})
    with pytest.raises(InputError, match="outside the agent directory"):
        await _tools().dispatch("Read", {"path": "../billing/AGENT.md"})


async def test_grep_files_and_content() -> None:
    files = await _tools().dispatch("Grep", {"pattern": "refund", "case_insensitive": True})
    assert str(files).splitlines() == ["rules/refunds.md"]
    lines = await _tools().dispatch("Grep", {"pattern": "refunded", "mode": "content"})
    assert lines == "rules/refunds.md:4:A duplicate charge is refunded in full without approval."


async def test_glob_and_ls() -> None:
    found = await _tools().dispatch("Glob", {"pattern": "**/*.md"})
    assert "tasks/triage.md" in str(found).splitlines()
    listing = await _tools().dispatch("LS", {})
    assert "rules/" in str(listing).splitlines() and "AGENT.md" in str(listing).splitlines()


async def test_no_builtin_shows_a_bytecode_cache(tmp_path: Path) -> None:
    (tmp_path / "rules" / "__pycache__").mkdir(parents=True)
    (tmp_path / "rules" / "refunds.md").write_text("refund in full\n")
    (tmp_path / "rules" / "__pycache__" / "refunds.cpython-314.pyc").write_text("refund\n")
    (tmp_path / "__pycache__").mkdir()
    tools = Tools(builtins(tmp_path))
    assert str(await tools.dispatch("Glob", {"pattern": "**/*"})).splitlines() == [
        "rules/refunds.md"
    ]
    assert str(await tools.dispatch("Grep", {"pattern": "refund"})).splitlines() == [
        "rules/refunds.md"
    ]
    assert str(await tools.dispatch("LS", {})).splitlines() == ["rules/"]
    assert str(await tools.dispatch("LS", {"path": "rules"})).splitlines() == ["refunds.md"]


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
            match=rf"^tools/builtins: {path} is not a text file; Load gives you the file itself$",
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
        InputError, match=r"^tools/builtins: broken.png is not the image/png its name says$"
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
        match=r"^tools/builtins: the model does not take audio/mpeg files: knowledge/call.mp3$",
    ):
        await tools.dispatch("Load", {"path": "knowledge/call.mp3"})


async def test_load_refuses_files_over_its_cap(
    media: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtins_module, "_LOAD_BYTES", 10)
    with pytest.raises(
        InputError, match=r"^tools/builtins: file exceeds 10 bytes: knowledge/rates.pdf$"
    ):
        await Tools([load_tool(media)]).dispatch("Load", {"path": "knowledge/rates.pdf"})


async def test_load_follows_the_jail(media: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(PDF)
    (media / "knowledge" / "link.pdf").symlink_to(outside)
    tools = Tools([load_tool(media)])
    for path in ["knowledge/link.pdf", "../outside.pdf", str(outside)]:
        with pytest.raises(InputError, match="outside the agent directory"):
            await tools.dispatch("Load", {"path": path})


async def test_load_resolves_a_folder_file_once(
    media: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved: list[str | Path] = []
    resolve = Jail.resolve

    def counted(self: Jail, path: str | Path) -> Path:
        resolved.append(path)
        return resolve(self, path)

    monkeypatch.setattr(Jail, "resolve", counted)
    assert await Tools([load_tool(media)]).dispatch("Load", {"path": "knowledge/rates.pdf"}) == [
        Document(media_type="application/pdf", name="rates.pdf", data=PDF)
    ]
    assert resolved == ["knowledge/rates.pdf"]


async def test_load_names_and_types_a_link_by_the_file_it_opens(media: Path) -> None:
    knowledge = media / "knowledge"
    (knowledge / "report.pdf").write_bytes(PDF)
    (knowledge / "report").symlink_to("report.pdf")
    (knowledge / "notes").symlink_to("notes.md")
    (knowledge / "photo.png").symlink_to("photo.jpg")
    tools = Tools([load_tool(media)])
    for link, part in {
        "report": Document(media_type="application/pdf", name="report.pdf", data=PDF),
        "notes": Document(media_type="text/markdown", name="notes.md", data=b"plain\n"),
        "photo.png": Image(media_type="image/jpeg", data=JPEG),
    }.items():
        assert await tools.dispatch("Load", {"path": f"knowledge/{link}"}) == [part], link


async def test_load_names_a_file_by_its_path_where_the_system_cannot_name_it(
    media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (media / "knowledge" / "report").symlink_to("rates.pdf")
    monkeypatch.setattr(builtins_module, "_OPEN_FILES", str(tmp_path / "nowhere"))
    assert await Tools([load_tool(media)]).dispatch("Load", {"path": "knowledge/report"}) == [
        Document(media_type="application/pdf", name="report", data=PDF)
    ]


NOTES = b"# Notes\n\x00 kept as bytes\n"


@pytest.mark.parametrize(
    ("link", "target", "part"),
    [
        ("notes", "notes.md", Document(media_type="text/markdown", name="notes.md", data=NOTES)),
        ("call", "call.mp3", Audio(media_type="audio/mpeg", data=MP3)),
        ("photo.png", "photo.jpg", Image(media_type="image/jpeg", data=JPEG)),
    ],
)
async def test_preload_names_and_types_a_link_as_load_does(
    media: Path, link: str, target: str, part: Document | Audio | Image
) -> None:
    knowledge = media / "knowledge"
    (knowledge / "notes.md").write_bytes(NOTES)
    (knowledge / link).symlink_to(target)
    name = f"knowledge/{link}"
    parts, _ = preload_parts([name], None, Jail(media))
    assert parts == [Text(text=f"<<<{name}>>>"), part]
    assert await Tools([load_tool(media)]).dispatch("Load", {"path": name}) == [part]


@pytest.mark.parametrize("by", ["Load", "preload"])
async def test_a_file_removed_after_it_was_opened_keeps_its_own_name(
    media: Path, monkeypatch: pytest.MonkeyPatch, by: str
) -> None:
    opened = Jail.open

    def removing(self: Jail, path: str | Path, *args: Any) -> int:
        fd = opened(self, path, *args)
        (media / "knowledge" / "rates.pdf").unlink()
        return fd

    monkeypatch.setattr(Jail, "open", removing)
    part = Document(media_type="application/pdf", name="rates.pdf", data=PDF)
    name = "knowledge/rates.pdf"
    if by == "Load":
        assert await Tools([load_tool(media)]).dispatch("Load", {"path": name}) == [part]
    else:
        parts, _ = preload_parts([name], None, Jail(media))
        assert parts == [Text(text=f"<<<{name}>>>"), part]


async def test_a_file_whose_name_only_looks_removed_keeps_it(media: Path) -> None:
    (media / "knowledge" / "rates (deleted)").write_bytes(PDF)
    assert await Tools([load_tool(media)]).dispatch(
        "Load", {"path": "knowledge/rates (deleted)"}
    ) == [Document(media_type="application/pdf", name="rates (deleted)", data=PDF)]


async def test_load_reaches_the_workspace_when_the_run_has_one(media: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "chart.png").write_bytes(PNG)
    alone = Tools([load_tool(media)])
    [schema] = alone.definitions()
    assert "where" not in schema["input_schema"]["properties"]
    with pytest.raises(InputError, match=r"^tools/builtins: not a file: chart.png$"):
        await alone.dispatch("Load", {"path": "chart.png", "where": "workspace"})
    both = Tools([load_tool(media, Local(work))])
    assert await both.dispatch("Load", {"path": "chart.png", "where": "workspace"}) == [
        Image(media_type="image/png", data=PNG)
    ]
    assert await both.dispatch("Load", {"path": "knowledge/rates.pdf"}) == [
        Document(media_type="application/pdf", name="rates.pdf", data=PDF)
    ]
    with pytest.raises(
        InputError,
        match="^tools/workspace: path outside the workspace: ../agent/knowledge/rates.pdf$",
    ):
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


async def test_grep_notes_when_it_stops_at_its_walk_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builtins_module, "_MAX_WALK", 3)
    root = tmp_path / "agent"
    root.mkdir()
    for index in range(3):
        (root / f"f{index}.md").write_text("needle\n")
    tools = Tools(builtins(root))
    assert await tools.dispatch("Grep", {"pattern": "needle"}) == "f0.md\nf1.md\nf2.md"
    (root / "f3.md").write_text("needle\n")
    lines = str(await tools.dispatch("Grep", {"pattern": "needle"})).splitlines()
    assert len(lines) == 4 and lines[-1] == "[truncated: walk exceeded 3 entries]"


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


@pytest.fixture
def shelf(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    (root / "coding" / "notes").mkdir(parents=True)
    (root / "coding" / "notes" / "style.md").write_text("tabs never\n")
    (root / "books" / "python").mkdir(parents=True)
    (root / "books" / "python" / "async.md").write_text("await the future\n")
    (root / "books" / ".draft.md").write_text("await nothing\n")
    (root / "writing").mkdir()
    (root / "writing" / "voice.md").write_text("await warmly\n")
    return root


def reading(shelf: Path) -> Tools:
    books = (shelf / "books").resolve()
    coding = shelf / "coding"
    return Tools([*builtins(coding, {"../books": books}), load_tool(coding, None, None, [books])])


def test_the_jail_takes_more_places(shelf: Path) -> None:
    jail = Jail(shelf / "coding", also=[shelf / "books"])
    assert jail.resolve("../books/python/async.md") == (shelf / "books/python/async.md").resolve()
    assert jail.also == [(shelf / "books").resolve()]
    for path in ["../writing/voice.md", "..", "../books/../writing/voice.md"]:
        with pytest.raises(InputError, match="^jail: path outside the agent directory"):
            jail.resolve(path)


async def test_a_read_folder_is_read_by_its_path_beside_the_agent_s(shelf: Path) -> None:
    tools = reading(shelf)
    assert (
        await tools.dispatch("Read", {"path": "../books/python/async.md"}) == "await the future\n"
    )
    assert await tools.dispatch("Read", {"path": "notes/style.md"}) == "tabs never\n"
    assert await tools.dispatch("LS", {"path": "../books"}) == "python/"
    loaded = await tools.dispatch("Load", {"path": "../books/python/async.md"})
    assert isinstance(loaded, list) and len(loaded) == 1
    for name, args in [
        ("Read", {"path": "../writing/voice.md"}),
        ("LS", {"path": ".."}),
        ("Load", {"path": "../writing/voice.md"}),
    ]:
        with pytest.raises(InputError, match="outside the agent directory"):
            await tools.dispatch(name, args)


async def test_a_read_folder_reaches_nothing_beyond_itself(shelf: Path, tmp_path: Path) -> None:
    (tmp_path / "outside.md").write_text("await outside\n")
    (shelf / "books" / "out.md").symlink_to(tmp_path / "outside.md")
    (shelf / "books" / "w").symlink_to(shelf / "writing")
    (shelf / "books2").mkdir()
    (shelf / "books2" / "x.md").write_text("await x\n")
    tools = reading(shelf)
    for name, args in [
        ("Read", {"path": "../books/out.md"}),
        ("Load", {"path": "../books/out.md"}),
        ("Read", {"path": "../books/w/voice.md"}),
        ("LS", {"path": "../books/w"}),
        ("Read", {"path": "../books2/x.md"}),
        ("LS", {"path": "../books2"}),
        ("Grep", {"pattern": "await", "path": "../writing"}),
    ]:
        with pytest.raises(InputError, match="outside the agent directory"):
            await tools.dispatch(name, args)
    with pytest.raises(InputError, match="inside the agent directory"):
        await tools.dispatch("Glob", {"pattern": "../books2/*.md"})
    for listing in [
        await tools.dispatch("Grep", {"pattern": "await", "path": "../books"}),
        await tools.dispatch("Grep", {"pattern": "await", "path": "../books", "mode": "content"}),
        await tools.dispatch("Glob", {"pattern": "../books/**/*.md"}),
        await tools.dispatch("Glob", {"pattern": "../books/*"}),
    ]:
        assert "out.md" not in listing and "/w/" not in listing and "voice" not in listing


async def test_grep_and_glob_search_the_own_folder_unless_told(shelf: Path) -> None:
    tools = reading(shelf)
    assert await tools.dispatch("Grep", {"pattern": "await"}) == "no matches found"
    assert await tools.dispatch("Grep", {"pattern": "await", "path": "../books"}) == (
        "../books/python/async.md"
    )
    assert await tools.dispatch(
        "Grep", {"pattern": "await", "path": "../books", "mode": "content"}
    ) == ("../books/python/async.md:1:await the future")
    assert await tools.dispatch("Glob", {"pattern": "**/*.md"}) == "notes/style.md"
    assert await tools.dispatch("Glob", {"pattern": "../books/**/*.md"}) == (
        "../books/python/async.md"
    )
    for pattern in ["../writing/*.md", "../books/../writing/*", "../books", "../books/"]:
        with pytest.raises(InputError, match="inside the agent directory"):
            await tools.dispatch("Glob", {"pattern": pattern})


@pytest.mark.parametrize("pattern", [".", "../books/."])
async def test_glob_refuses_a_pattern_the_walk_cannot_take(shelf: Path, pattern: str) -> None:
    with pytest.raises(InputError, match="^tools/builtins: invalid glob pattern"):
        await reading(shelf).dispatch("Glob", {"pattern": pattern})


@pytest.mark.parametrize("glob", ["", "."])
async def test_grep_refuses_a_glob_the_match_cannot_take(shelf: Path, glob: str) -> None:
    with pytest.raises(InputError, match=f"^tools/builtins: invalid glob '{glob}': empty pattern$"):
        await reading(shelf).dispatch("Grep", {"pattern": "tabs", "glob": glob})


async def test_read_reaches_the_workspace_when_the_run_has_one(media: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "notes.txt").write_text("one\ntwo\nthree\n")
    (work / "chart.png").write_bytes(PNG)
    [alone] = [d for d in Tools(builtins(media)).definitions() if d["name"] == "Read"]
    assert "where" not in alone["input_schema"]["properties"]
    both = Tools(builtins(media, None, Local(work)))
    read = {"path": "notes.txt", "where": "workspace", "offset": 1, "limit": 1}
    assert await both.dispatch("Read", read) == "two"
    assert await both.dispatch("Read", {"path": "knowledge/notes.md"}) == "plain\n"
    with pytest.raises(
        InputError,
        match="^tools/builtins: chart.png is not a text file; Load gives you the file itself$",
    ):
        await both.dispatch("Read", {"path": "chart.png", "where": "workspace"})
    with pytest.raises(
        InputError,
        match="^tools/workspace: path outside the workspace: ../agent/knowledge/notes.md$",
    ):
        await both.dispatch("Read", {"path": "../agent/knowledge/notes.md", "where": "workspace"})


async def test_a_path_the_folder_lacks_says_how_to_reach_the_workspace(
    media: Path, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    (work / "task_folder").mkdir(parents=True)
    (work / "task_folder" / "a.csv").write_text("x\n")
    both = Tools([*builtins(media, None, Local(work)), load_tool(media, Local(work))])
    with pytest.raises(
        InputError,
        match="^tools/builtins: not a directory of your folder: task_folder; Shell lists your workspace$",
    ):
        await both.dispatch("LS", {"path": "task_folder"})
    for name, verb in [("Read", "reads"), ("Load", "loads")]:
        with pytest.raises(
            InputError,
            match=(
                f"^tools/builtins: not a file of your folder: task_folder/a.csv; "
                f'{name} with where="workspace" {verb} from your workspace$'
            ),
        ):
            await both.dispatch(name, {"path": "task_folder/a.csv"})
    with pytest.raises(InputError, match="^tools/builtins: not a directory: knowledge/notes.md$"):
        await both.dispatch("LS", {"path": "knowledge/notes.md"})
    with pytest.raises(InputError, match="^tools/builtins: not a file: knowledge$"):
        await both.dispatch("Read", {"path": "knowledge"})
    alone = Tools([*builtins(media), load_tool(media)])
    with pytest.raises(InputError, match="^tools/builtins: not a directory: task_folder$"):
        await alone.dispatch("LS", {"path": "task_folder"})
    for name in ["Read", "Load"]:
        with pytest.raises(InputError, match="^tools/builtins: not a file: task_folder/a.csv$"):
            await alone.dispatch(name, {"path": "task_folder/a.csv"})


async def test_read_and_load_go_through_any_backend_with_their_limits() -> None:
    class Files:
        def __init__(self) -> None:
            self.limits: list[int] = []

        async def spawn(self, command: str) -> Process:
            raise AssertionError("no command runs")

        async def read(self, path: str, limit: int) -> bytes:
            self.limits.append(limit)
            return PNG if path.endswith(".png") else b"text\n"

        async def write(self, path: str, data: bytes) -> None:
            raise AssertionError("nothing is written")

    files = Files()
    tools = Tools([*builtins(SUPPORT, None, files), load_tool(SUPPORT, files)])
    assert await tools.dispatch("Read", {"path": "a.txt", "where": "workspace"}) == "text\n"
    assert await tools.dispatch("Load", {"path": "b.png", "where": "workspace"}) == [
        Image(media_type="image/png", data=PNG)
    ]
    assert files.limits == [5 * 1024 * 1024, 20 * 1024 * 1024]


async def test_what_the_system_refuses_is_said_with_the_model_s_path(tmp_path: Path) -> None:
    root = tmp_path / "agent"
    (root / "locked").mkdir(parents=True)
    (root / "locked" / "a.md").write_text("a\n")
    (root / "closed.md").write_text("closed\n")
    (root / "locked").chmod(0o300)
    (root / "closed.md").chmod(0o000)
    tools = Tools([*builtins(root), load_tool(root)])
    try:
        with pytest.raises(
            InputError, match="^tools/builtins: cannot list locked: Permission denied$"
        ):
            await tools.dispatch("LS", {"path": "locked"})
        for name in ["Read", "Load"]:
            with pytest.raises(
                InputError, match="^tools/builtins: cannot read closed.md: Permission denied$"
            ):
                await tools.dispatch(name, {"path": "closed.md"})
        assert await tools.dispatch("Read", {"path": "locked/a.md"}) == "a\n"
    finally:
        (root / "locked").chmod(0o700)
        (root / "closed.md").chmod(0o600)


def test_a_file_far_over_the_limit_is_refused_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (tmp_path / "sparse.bin").open("wb") as file:
        file.truncate(1 << 40)

    def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("opened for reading")

    monkeypatch.setattr("rfnry_engine.tools.builtins.open", never, raising=False)
    with pytest.raises(InputError, match="^tools/builtins: file exceeds 10 bytes: sparse.bin$"):
        _contents(Jail(tmp_path), "sparse.bin", 10)


@pytest.fixture
def hidden(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text("TOKEN=1", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]", encoding="utf-8")
    (tmp_path / "tools" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tools" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (tmp_path / "plain.md").write_text("plain", encoding="utf-8")
    (tmp_path / "shown.md").symlink_to(tmp_path / ".env")
    return tmp_path


async def test_the_file_tools_refuse_a_hidden_path_named_directly(hidden: Path) -> None:
    tools = Tools(builtins(hidden))
    load = Tools([load_tool(hidden)])
    for tool, name, args in [
        (tools, "Read", {"path": ".env"}),
        (tools, "Read", {"path": "./tools/../.env"}),
        (load, "Load", {"path": "tools/__pycache__/x.pyc"}),
        (tools, "Grep", {"pattern": "core", "path": ".git/config"}),
        (tools, "LS", {"path": ".git"}),
    ]:
        with pytest.raises(InputError, match=r"^tools/builtins: .* is hidden"):
            await tool.dispatch(name, args)


async def test_a_visible_link_to_a_hidden_file_still_reads(hidden: Path) -> None:
    tools = Tools(builtins(hidden))
    assert await tools.dispatch("Read", {"path": "shown.md"}) == "TOKEN=1"
    assert await tools.dispatch("Read", {"path": "plain.md"}) == "plain"


async def test_an_absolute_path_is_judged_inside_the_folder_not_by_its_ancestors(
    tmp_path: Path,
) -> None:
    agent = tmp_path / ".h" / "agent"
    agent.mkdir(parents=True)
    (agent / "a.md").write_text("visible", encoding="utf-8")
    (agent / ".env").write_text("TOKEN=1", encoding="utf-8")
    tools = Tools(builtins(agent))
    assert await tools.dispatch("Read", {"path": str(agent / "a.md")}) == "visible"
    with pytest.raises(InputError, match="is hidden"):
        await tools.dispatch("Read", {"path": str(agent / ".env")})


async def test_an_absolute_path_outside_the_folder_is_outside_not_hidden(hidden: Path) -> None:
    other = hidden.parent / ".other" / "x.md"
    other.parent.mkdir(exist_ok=True)
    other.write_text("x", encoding="utf-8")
    with pytest.raises(InputError, match="outside the agent directory"):
        await Tools(builtins(hidden)).dispatch("Read", {"path": str(other)})


async def test_a_hidden_name_inside_a_reads_folder_is_refused(tmp_path: Path) -> None:
    agent, books = tmp_path / "agent", tmp_path / "books"
    agent.mkdir()
    books.mkdir()
    (books / ".x").write_text("x", encoding="utf-8")
    (books / "y.md").write_text("y", encoding="utf-8")
    tools = Tools(builtins(agent, {"books": books}))
    assert await tools.dispatch("Read", {"path": "../books/y.md"}) == "y"
    for path in ["../books/.x", str(books / ".x")]:
        with pytest.raises(InputError, match="is hidden"):
            await tools.dispatch("Read", {"path": path})


@pytest.fixture
def linked(tmp_path: Path) -> Path:
    agent = tmp_path / "real" / "agent"
    (agent / ".git").mkdir(parents=True)
    (agent / ".git" / "config").write_text("[core]", encoding="utf-8")
    (agent / ".env").write_text("TOKEN=1", encoding="utf-8")
    (agent / "plain.md").write_text("plain", encoding="utf-8")
    (agent / ".secret").mkdir()
    (agent / ".secret" / "up").symlink_to("..", target_is_directory=True)
    (agent / "shown.md").symlink_to(".env")
    (tmp_path / "real" / "books").mkdir()
    (tmp_path / "real" / "books" / ".x").write_text("x", encoding="utf-8")
    (tmp_path / "real" / ".other").mkdir()
    (tmp_path / "real" / ".other" / "x.md").write_text("x", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    return agent


async def test_an_absolute_path_through_a_linked_ancestor_is_judged_inside_the_folder(
    linked: Path, tmp_path: Path
) -> None:
    spelled = tmp_path / "link" / "agent"
    tools = Tools(builtins(linked, {"books": tmp_path / "real" / "books"}))
    load = Tools([load_tool(linked)])
    for tool, name, args in [
        (tools, "Read", {"path": str(spelled / ".env")}),
        (load, "Load", {"path": str(spelled / ".env")}),
        (tools, "Grep", {"pattern": "core", "path": str(spelled / ".git" / "config")}),
        (tools, "LS", {"path": str(spelled / ".git")}),
        (tools, "Read", {"path": str(spelled / ".secret" / "up" / "plain.md")}),
        (tools, "Read", {"path": str(tmp_path / "link" / "books" / ".x")}),
    ]:
        with pytest.raises(InputError, match=r"^tools/builtins: .* is hidden"):
            await tool.dispatch(name, args)
    assert await tools.dispatch("Read", {"path": str(spelled / "plain.md")}) == "plain"
    assert await tools.dispatch("Read", {"path": str(spelled / "shown.md")}) == "TOKEN=1"
    with pytest.raises(InputError, match="outside the agent directory"):
        await tools.dispatch("Read", {"path": str(tmp_path / "link" / ".other" / "x.md")})
