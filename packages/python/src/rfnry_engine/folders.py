import os
from collections.abc import Mapping
from pathlib import Path

from rfnry_engine.errors import InputError
from rfnry_engine.frontmatter import described
from rfnry_engine.jail import Jail

DECLARED = frozenset({"tools", "tasks", "questions"})
DECLARATIONS = frozenset({"AGENT.md", "TOOLS.md"})


def declares(folder: Path) -> bool:
    return (folder / "AGENT.md").is_file()


def _inside(jail: Jail, entry: Path) -> bool:
    if not (entry.is_symlink() or entry.is_junction()):
        return True
    try:
        jail.resolve(entry)
    except InputError:
        return False
    return True


def _walk(folder: Path, declared: bool) -> tuple[list[str], list[tuple[str, Path]]]:
    folder = folder.resolve()
    jail = Jail(folder)
    dirs: list[str] = []
    files: list[tuple[str, Path]] = []
    for directory, names, filenames in os.walk(folder):
        here = Path(directory)
        at = here.relative_to(folder).as_posix()
        top = at == "."
        names[:] = [
            name
            for name in names
            if not name.startswith(".")
            and not (declared and top and name in DECLARED)
            and _inside(jail, here / name)
        ]
        dirs += [name if top else f"{at}/{name}" for name in names]
        for name in filenames:
            path = here / name
            if name.startswith(".") or (declared and top and name in DECLARATIONS):
                continue
            if path.is_file() and _inside(jail, path):
                files.append((name if top else f"{at}/{name}", path))
    return sorted(dirs), sorted(files)


def content(folder: Path, declared: bool) -> list[tuple[str, Path]]:
    return _walk(folder, declared)[1]


def entries(folder: Path, declared: bool) -> list[str]:
    dirs, files = _walk(folder, declared)
    children: dict[str, list[tuple[str, str, bool]]] = {}
    for rel in dirs:
        parent, _, name = rel.rpartition("/")
        children.setdefault(parent, []).append((name, f"{name}/", True))
    for rel, path in files:
        said = described(path) if path.suffix == ".md" else None
        if said is not None:
            parent, _, name = rel.rpartition("/")
            children.setdefault(parent, []).append((name, f"{name} — {said}", False))
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
        lines += [f"../{name}/", *(f"  {line}" for line in shown)]
    return "\n".join(lines)


def listing(folder: Path, declared: bool, reads: Mapping[str, Path]) -> str:
    return joined(
        entries(folder, declared),
        {name: entries(read, declares(read)) for name, read in reads.items()},
    )
