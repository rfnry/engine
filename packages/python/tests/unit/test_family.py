import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
SIBLING = ROOT.parent / "memory"
SRC = "packages/python/src"
FAMILY = ("engine", "memory", "thread", "senses")

BLOCKS = {
    "parts and messages": (
        "rfnry_engine/models.py",
        "rfnry_memory/models.py",
        "class Text(BaseModel):",
        "    is_error: bool = False\n",
    ),
    "tools": (
        "rfnry_engine/tools/tools.py",
        "rfnry_memory/tools.py",
        "Result = ",
        "if n not in names})\n",
    ),
    "state": ("rfnry_engine/models.py", "rfnry_memory/models.py", "State = ", "list[Any]\n"),
    "generation and decision contract": (
        "rfnry_engine/models.py",
        "rfnry_memory/models.py",
        "class Usage(BaseModel):",
        "    ) -> ReplyLike: ...\n",
    ),
    "decision model": (
        "rfnry_engine/models.py",
        "rfnry_memory/models.py",
        "class DecisionModel(Protocol):",
        "| DecidedLike: ...\n",
    ),
}


def cut(path: Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    first = text.index(start)
    return text[first : text.index(end, first) + len(end)]


@pytest.mark.parametrize("name", BLOCKS)
def test_the_shared_block_is_byte_identical_with_the_sibling_library(name: str) -> None:
    own, theirs, start, end = BLOCKS[name]
    if not SIBLING.is_dir():
        pytest.skip(f"no sibling checkout at {SIBLING}")
    assert cut(ROOT / SRC / own, start, end) == cut(SIBLING / SRC / theirs, start, end)


def classes(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    return {
        node.name: "".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.parse(text).body
        if isinstance(node, ast.ClassDef)
    }


def providers(library: str) -> list[Path]:
    examples = ROOT.parent / library / "examples"
    found = [*examples.glob("*/providers.py"), *examples.glob("*/src/providers.py")]
    return sorted(path for path in found if ".venv" not in path.parts)


def test_every_provider_class_is_byte_identical_in_every_example_that_carries_it() -> None:
    copies: dict[str, set[str]] = {}
    for library in FAMILY:
        for path in providers(library):
            for name, text in classes(path).items():
                copies.setdefault(name, set()).add(text)
    assert {"TypeSafeProvider", "AnthropicProvider"} <= set(copies)
    assert sorted(name for name, texts in copies.items() if len(texts) > 1) == []
