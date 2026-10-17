import os
import re
import stat
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from rfnry_engine import Jail
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.tools.builtins import builtins, load_tool
from rfnry_engine.tools.tools import Tools
from rfnry_engine.tree import load_agent

WRITE = os.O_WRONLY | os.O_CREAT


@pytest.fixture
def shelf(tmp_path: Path) -> Path:
    root = tmp_path / "agents"
    (root / "coding" / "notes").mkdir(parents=True)
    (root / "coding" / "notes" / "style.md").write_text("tabs never\n")
    (root / "books" / "python").mkdir(parents=True)
    (root / "books" / "python" / "async.md").write_text("await the future\n")
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    path = tmp_path / "outside"
    path.mkdir()
    (path / "style.md").write_text("secret\n")
    (path / "async.md").write_text("secret\n")
    return path


@pytest.fixture
def jail(shelf: Path) -> Jail:
    return Jail(shelf / "coding", also=[shelf / "books"])


def read(fd: int) -> bytes:
    try:
        return os.read(fd, 1000)
    finally:
        os.close(fd)


def refused(path: str) -> str:
    return f"^jail: path outside the agent directory: {re.escape(path)}$"


def swap_after_the_check(monkeypatch: pytest.MonkeyPatch, swap: Callable[[], None]) -> None:
    checked, opening = Jail.resolve, Jail.open.__code__

    def swapped(self: Jail, path: str | Path) -> Path:
        target = checked(self, path)
        if sys._getframe(1).f_code is opening:
            swap()
        return target

    monkeypatch.setattr(Jail, "resolve", swapped)


def linked(path: Path, to: Path) -> Callable[[], None]:
    def swap() -> None:
        if not path.is_symlink():
            path.rename(path.with_name(f"{path.name}.moved"))
            path.symlink_to(to)

    return swap


def test_open_hands_a_descriptor_to_a_file_of_any_place(jail: Jail, shelf: Path) -> None:
    assert read(jail.open("notes/style.md")) == b"tabs never\n"
    assert read(jail.open("../books/python/async.md")) == b"await the future\n"
    assert read(jail.open(shelf / "books" / "python" / "async.md")) == b"await the future\n"
    assert read(jail.open(str(shelf / "coding" / "notes" / "style.md"))) == b"tabs never\n"


def test_open_refuses_what_resolve_refuses(jail: Jail, outside: Path) -> None:
    for path in ["../outside/style.md", str(outside / "style.md"), "..", "../books/../x.md"]:
        with pytest.raises(InputError, match=refused(path)):
            jail.open(path)


def test_a_link_inside_a_place_still_opens(jail: Jail, shelf: Path) -> None:
    (shelf / "coding" / "alias").symlink_to("notes")
    (shelf / "coding" / "link.md").symlink_to("notes/style.md")
    (shelf / "books" / "py").symlink_to("python")
    assert read(jail.open("alias/style.md")) == b"tabs never\n"
    assert read(jail.open("link.md")) == b"tabs never\n"
    assert read(jail.open("../books/py/async.md")) == b"await the future\n"


@pytest.mark.parametrize(
    ("folder", "path", "flags"),
    [
        ("coding/notes", "notes/style.md", os.O_RDONLY),
        ("coding/notes", "notes/style.md", WRITE),
        ("coding/notes", "notes/new/new.md", WRITE),
        ("books/python", "../books/python/async.md", os.O_RDONLY),
        ("books/python", "../books/python/new.md", WRITE),
    ],
    ids=["read", "write", "create", "read-also", "create-also"],
)
def test_a_folder_swapped_for_a_link_after_the_check_is_refused(
    jail: Jail,
    shelf: Path,
    outside: Path,
    monkeypatch: pytest.MonkeyPatch,
    folder: str,
    path: str,
    flags: int,
) -> None:
    swap_after_the_check(monkeypatch, linked(shelf / folder, outside))
    with pytest.raises(InputError, match=refused(path)):
        jail.open(path, flags)
    assert (shelf / folder).is_symlink()
    assert sorted(p.name for p in outside.iterdir()) == ["async.md", "style.md"]
    assert {p.read_text() for p in outside.iterdir()} == {"secret\n"}


@pytest.mark.parametrize("flags", [os.O_RDONLY, WRITE], ids=["read", "write"])
@pytest.mark.parametrize(
    ("file", "path"),
    [
        ("coding/notes/style.md", "notes/style.md"),
        ("books/python/async.md", "../books/python/async.md"),
    ],
    ids=["root", "also"],
)
def test_a_file_swapped_for_a_link_after_the_check_is_refused(
    jail: Jail,
    shelf: Path,
    outside: Path,
    monkeypatch: pytest.MonkeyPatch,
    file: str,
    path: str,
    flags: int,
) -> None:
    swap_after_the_check(monkeypatch, linked(shelf / file, outside / "style.md"))
    with pytest.raises(InputError, match=refused(path)):
        jail.open(path, flags)
    assert (shelf / file).is_symlink()
    assert (outside / "style.md").read_text() == "secret\n"


@pytest.mark.parametrize(
    ("place", "path", "file"),
    [
        ("coding", "notes/style.md", "notes/style.md"),
        ("books", "../books/python/async.md", "python/async.md"),
    ],
    ids=["root", "also"],
)
def test_a_place_swapped_for_another_folder_after_it_was_made_is_refused(
    jail: Jail, shelf: Path, place: str, path: str, file: str
) -> None:
    (shelf / place).rename(shelf / f"{place}.moved")
    (shelf / place / file).parent.mkdir(parents=True)
    (shelf / place / file).write_text("other\n")
    with pytest.raises(InputError, match=refused(path)):
        jail.open(path)


def test_a_place_reached_through_a_link_still_opens(shelf: Path, tmp_path: Path) -> None:
    (tmp_path / "alias").symlink_to(shelf / "coding")
    assert read(Jail(tmp_path / "alias").open("notes/style.md")) == b"tabs never\n"


def test_o_creat_creates_the_missing_parents(jail: Jail, shelf: Path) -> None:
    fd = jail.open("drafts/2026/today.md", WRITE, 0o600)
    try:
        os.write(fd, b"hello")
    finally:
        os.close(fd)
    made = shelf / "coding" / "drafts" / "2026" / "today.md"
    assert made.read_bytes() == b"hello"
    assert stat.S_IMODE(made.stat().st_mode) == 0o600
    os.close(jail.open("../books/new/new.md", WRITE))
    assert (shelf / "books" / "new" / "new.md").is_file()


def test_what_the_system_refuses_reaches_the_caller_as_it_is(jail: Jail, shelf: Path) -> None:
    with pytest.raises(FileNotFoundError):
        jail.open("missing.md")
    with pytest.raises(FileNotFoundError):
        jail.open("drafts/today.md")
    with pytest.raises(NotADirectoryError):
        jail.open("notes/style.md/x")
    with pytest.raises(IsADirectoryError):
        jail.open("notes", os.O_WRONLY)
    assert not (shelf / "coding" / "drafts").exists()


def descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def test_no_descriptor_leaks_over_many_refused_calls(
    shelf: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coding = shelf / "coding"
    (coding / "out").symlink_to(outside)
    (coding / "notes" / "out.md").symlink_to(outside / "style.md")
    (coding / "notes" / "deep").mkdir()
    (coding / "notes" / "deep" / "out").symlink_to(outside)
    jail = Jail(coding)
    moved = Jail(shelf / "books")
    (shelf / "books").rename(shelf / "books.moved")
    (shelf / "books").mkdir()
    (shelf / "books" / "async.md").write_text("other\n")
    monkeypatch.setattr(Jail, "resolve", lambda self, path: self.root / path)
    calls: list[tuple[Jail, str, int, type[Exception]]] = [
        (jail, "out/style.md", os.O_RDONLY, InputError),
        (jail, "out/new/new.md", WRITE, InputError),
        (jail, "notes/out.md", os.O_RDONLY, InputError),
        (jail, "notes/deep/out/style.md", os.O_RDONLY, InputError),
        (jail, "missing.md", os.O_RDONLY, FileNotFoundError),
        (jail, "notes/missing/x.md", os.O_RDONLY, FileNotFoundError),
        (jail, "notes/style.md/x", os.O_RDONLY, NotADirectoryError),
        (jail, "notes", os.O_WRONLY, IsADirectoryError),
        (moved, "async.md", os.O_RDONLY, InputError),
    ]
    os.close(jail.open("notes/style.md"))
    before = descriptors()
    for _ in range(200):
        for place, path, flags, error in calls:
            with pytest.raises(error):
                place.open(path, flags)
    assert descriptors() == before
    assert sorted(p.name for p in outside.iterdir()) == ["async.md", "style.md"]


@pytest.mark.parametrize(
    ("tool", "path"), [("Read", "notes/style.md"), ("Load", "notes/style.md"), ("LS", "notes")]
)
async def test_a_builtin_refuses_a_folder_swapped_for_a_link_between_check_and_open(
    shelf: Path, outside: Path, monkeypatch: pytest.MonkeyPatch, tool: str, path: str
) -> None:
    coding = shelf / "coding"
    tools = Tools([*builtins(coding), load_tool(coding)])
    swap_after_the_check(monkeypatch, linked(coding / "notes", outside))
    with pytest.raises(InputError, match=refused(path)):
        await tools.dispatch(tool, {"path": path})
    assert (coding / "notes").is_symlink()


@pytest.mark.parametrize("mode", ["files", "content"])
async def test_grep_skips_a_file_whose_folder_is_swapped_for_a_link_between_check_and_open(
    shelf: Path, outside: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    coding = shelf / "coding"
    swap_after_the_check(monkeypatch, linked(coding / "notes", outside))
    grep = {"pattern": "secret|tabs", "mode": mode}
    assert await Tools(builtins(coding)).dispatch("Grep", grep) == "no matches found"
    assert (coding / "notes").is_symlink()


def test_preload_refuses_a_folder_swapped_for_a_link_between_check_and_open(
    shelf: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coding = shelf / "coding"
    (coding / "AGENT.md").write_text("---\nkind: generation\npreload: true\n---\nYou code.\n")
    swap_after_the_check(monkeypatch, linked(coding / "notes", outside))
    with pytest.raises(
        ConfigurationError,
        match=r"^tree: agent 'coding' cannot preload its folder: "
        r"jail: path outside the agent directory: notes/style\.md; move that file out of its folders, or drop preload: true$",
    ):
        load_agent(coding)
    assert (coding / "notes").is_symlink()
