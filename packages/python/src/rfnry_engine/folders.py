import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePath, PurePosixPath

from pathspec import GitIgnoreSpec
from pathspec.patterns.gitignore.spec import _DIR_MARK

from rfnry_engine.errors import InputError
from rfnry_engine.frontmatter import leniently_described
from rfnry_engine.jail import Jail

logger = logging.getLogger(__name__)

_warned: ContextVar[set[str] | None] = ContextVar("warned", default=None)


@contextmanager
def warns_once() -> Iterator[None]:
    token = _warned.set(set())
    try:
        yield
    finally:
        _warned.reset(token)


def unwarned(place: str | Path) -> bool:
    seen = _warned.get()
    if seen is None:
        return True
    real = os.path.realpath(place)
    if real in seen:
        return False
    seen.add(real)
    return True


def _past_stars(segments: Sequence[str], states: Iterable[int]) -> set[int]:
    passed = set()
    for state in states:
        passed.add(state)
        while state < len(segments) and segments[state] == "**":
            state += 1
            passed.add(state)
    return passed


def _states(segments: Sequence[str], path: PurePath) -> set[int]:
    states = _past_stars(segments, {0})
    for part in path.parts:
        hidden = part.startswith(".")
        moved = set()
        for state in states - {len(segments)}:
            segment = segments[state]
            if segment == "**":
                if not hidden:
                    moved.add(state)
            elif segment == part or (not hidden and PurePosixPath(part).full_match(segment)):
                moved.add(state + 1)
        states = _past_stars(segments, moved)
    return states


def pattern_reaches(pattern: str, path: PurePath) -> bool:
    segments = PurePosixPath(pattern).parts
    return len(segments) in _states(segments, path)


def pattern_covers(pattern: str, path: PurePath) -> bool:
    place = path
    while not pattern_reaches(pattern, place):
        if place.name.startswith(".") or place == place.parent:
            return False
        place = place.parent
    return True


def pattern_leads_into(pattern: str, folder: PurePath) -> bool:
    segments = PurePosixPath(pattern).parts
    return bool(_states(segments, folder) - {len(segments)}) or pattern_covers(pattern, folder)


def _inside(jail: Jail, entry: Path) -> bool:
    if not entry.is_symlink():
        return True
    try:
        jail.resolve(entry)
    except InputError:
        return False
    return True


def shown_link(folder: Path, entry: Path) -> bool:
    if not entry.is_symlink():
        return True
    parts = entry.resolve().relative_to(folder).parts
    return not any(part.startswith(".") or part == "__pycache__" for part in parts)


def leads_out(path: Path, folder: Path) -> Path | None:
    return None if _inside(Jail(folder), path) else path.resolve()


def _unreadable(exc: OSError) -> None:
    if unwarned(exc.filename):
        logger.warning("folders: cannot read %s: %s; it is left out", exc.filename, exc)


def readable(path: Path) -> bool:
    try:
        os.close(os.open(path, os.O_RDONLY))
    except OSError as exc:
        _unreadable(exc)
        return False
    return True


def listed(folder: Path) -> list[str]:
    try:
        return sorted(os.listdir(folder))
    except OSError as exc:
        _unreadable(exc)
        return []


def _matches_itself(regex: re.Pattern[str], rel: str, is_folder: bool) -> bool:
    found = regex.search(rel)
    if found is not None and found.groupdict().get(_DIR_MARK) is None:
        return True
    if not is_folder:
        return False
    marked = f"{rel}/"
    found = regex.search(marked, len(marked) - 1) or regex.search(marked)
    return (
        found is not None
        and found.groupdict().get(_DIR_MARK) is not None
        and found.end(_DIR_MARK) == len(marked)
    )


class Gitignored:
    def __init__(self, place: Path) -> None:
        self.place = place
        self.specs: dict[Path, GitIgnoreSpec] = {}

    def _decides(self, parts: tuple[str, ...], is_folder: bool) -> bool:
        for depth in range(len(parts) - 1, -1, -1):
            spec = self.specs.get(self.place.joinpath(*parts[:depth]))
            if spec is None:
                continue
            rel = "/".join(parts[depth:])
            for pattern in reversed(spec.patterns):
                if pattern.regex is not None and _matches_itself(pattern.regex, rel, is_folder):
                    return pattern.include is True
        return False

    def _names(self, path: Path) -> bool:
        if self.place not in path.parents:
            return False
        parts = path.relative_to(self.place).parts
        here = self.place
        for depth, part in enumerate(parts, start=1):
            here = here / part
            if self._decides(parts[:depth], here.is_dir() and not here.is_symlink()):
                return True
        return False

    def __call__(self, path: Path) -> bool:
        real = Path(os.path.realpath(path))
        return self._names(path) or (real != path and self._names(real))


def _rendered(
    dirs: Iterable[str],
    files: Sequence[tuple[str, Path]],
    describe: Callable[[Path], str | None],
) -> list[str]:
    children: dict[str, list[tuple[str, str, bool]]] = {}
    for rel in dirs:
        parent, _, name = rel.rpartition("/")
        children.setdefault(parent, []).append((name, f"{name}/", True))
    for rel, path in files:
        said = describe(path) if path.suffix == ".md" else None
        parent, _, name = rel.rpartition("/")
        children.setdefault(parent, []).append(
            (name, name if said is None else f"{name} — {said}", False)
        )
    lines: list[str] = []

    def render(parent: str, indent: int) -> None:
        for name, line, is_folder in sorted(children.get(parent, [])):
            lines.append("  " * indent + line)
            if is_folder:
                render(f"{parent}/{name}" if parent else name, indent + 1)

    render("", 0)
    return lines


def mapped(files: Sequence[tuple[str, Path]]) -> list[str]:
    folders = {
        parent.as_posix()
        for rel, _ in files
        for parent in PurePosixPath(rel).parents
        if parent != PurePosixPath(".")
    }
    return _rendered(folders, files, leniently_described)
