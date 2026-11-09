import logging
import os
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

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


def _walk(
    folder: Path, declared: bool, inner: bool = False, apart: Collection[Path] = ()
) -> tuple[list[str], list[tuple[str, Path]]]:
    folder = folder.resolve()
    jail = Jail(folder)
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
            and _shown(folder, here / name)
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
            if path.is_file() and _inside(jail, path) and _readable(path):
                files.append((name if top else f"{at}/{name}", path))
    return sorted(dirs), sorted(files)


def content(
    folder: Path, declared: bool, inner: bool = False, apart: Collection[Path] = ()
) -> list[tuple[str, Path]]:
    return _walk(folder, declared, inner, apart)[1]


def entries(folder: Path, declared: bool, apart: Collection[Path] = ()) -> list[str]:
    dirs, files = _walk(folder, declared, apart=apart)
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
