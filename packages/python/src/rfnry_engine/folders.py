import logging
import os
import re
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pathspec import GitIgnoreSpec
from pathspec.patterns.gitignore.spec import _DIR_MARK

from rfnry_engine.errors import InputError
from rfnry_engine.frontmatter import described
from rfnry_engine.jail import Jail

logger = logging.getLogger(__name__)

DECLARED = frozenset({"tools", "questions", "skills"})
DECLARATIONS = frozenset({"AGENT.md"})

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


def declares(folder: Path) -> bool:
    return (folder / "AGENT.md").is_file()


def _inside(jail: Jail, entry: Path) -> bool:
    if not entry.is_symlink():
        return True
    try:
        jail.resolve(entry)
    except InputError:
        return False
    return True


def _shown(folder: Path, entry: Path) -> bool:
    if not entry.is_symlink():
        return True
    parts = entry.resolve().relative_to(folder).parts
    return not any(part.startswith(".") or part == "__pycache__" for part in parts)


def leads_out(path: Path, folder: Path) -> Path | None:
    return None if _inside(Jail(folder), path) else path.resolve()


def _unreadable(exc: OSError) -> None:
    if unwarned(exc.filename):
        logger.warning("folders: cannot read %s: %s; it is left out", exc.filename, exc)


def _readable(path: Path) -> bool:
    try:
        os.close(os.open(path, os.O_RDONLY))
    except OSError as exc:
        _unreadable(exc)
        return False
    return True


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


def any_ignores(checks: Collection[Gitignored]) -> Callable[[Path], bool]:
    def ignored(path: Path) -> bool:
        return any(check(path) for check in checks)

    return ignored


def _walk(
    folder: Path,
    declared: bool,
    inner: bool = False,
    apart: Collection[Path] = (),
    within: Path | None = None,
    ignored: Callable[[Path], bool] | None = None,
) -> tuple[list[str], list[tuple[str, Path]]]:
    folder = folder.resolve()
    holder = folder if within is None else within.resolve()
    jail = Jail(holder)
    dirs: list[str] = []
    files: list[tuple[str, Path]] = []
    for directory, names, filenames in os.walk(folder, onerror=_unreadable):
        here = Path(directory)
        at = here.relative_to(folder).as_posix()
        top = at == "."
        names[:] = [
            name
            for name in names
            if not name.startswith(".")
            and name != "__pycache__"
            and not (declared and top and name in DECLARED)
            and here / name not in apart
            and (inner or not declares(here / name))
            and _inside(jail, here / name)
            and _shown(holder, here / name)
            and not (ignored is not None and ignored(here / name))
        ]
        dirs += [name if top else f"{at}/{name}" for name in names]
        for name in filenames:
            path = here / name
            if (
                name.startswith(".")
                or name == "__pycache__"
                or (declared and top and name in DECLARATIONS)
            ):
                continue
            if ignored is not None and ignored(path):
                continue
            if path.is_file() and _inside(jail, path) and _readable(path):
                files.append((name if top else f"{at}/{name}", path))
    return sorted(dirs), sorted(files)


def content(
    folder: Path,
    declared: bool,
    inner: bool = False,
    apart: Collection[Path] = (),
    within: Path | None = None,
    ignored: Callable[[Path], bool] | None = None,
) -> list[tuple[str, Path]]:
    return _walk(folder, declared, inner, apart, within, ignored)[1]


def entries(
    folder: Path,
    declared: bool,
    apart: Collection[Path] = (),
    ignored: Callable[[Path], bool] | None = None,
) -> list[str]:
    dirs, files = _walk(folder, declared, apart=apart, ignored=ignored)
    children: dict[str, list[tuple[str, str, bool]]] = {}
    for rel in dirs:
        parent, _, name = rel.rpartition("/")
        children.setdefault(parent, []).append((name, f"{name}/", True))
    for rel, path in files:
        said = described(path) if path.suffix == ".md" else None
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


def joined(own: list[str], reads: Mapping[str, list[str]]) -> str:
    lines = list(own)
    for name, shown in reads.items():
        lines += [f"{name}/", *(f"  {line}" for line in shown)]
    return "\n".join(lines)
