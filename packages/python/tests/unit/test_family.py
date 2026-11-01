import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
SRC = "packages/python/src"


def sibling() -> Path:
    try:
        found = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        found = ""
    return (Path(found).parent if found else ROOT).parent / "memory"


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
    memory = sibling()
    if not memory.is_dir():
        pytest.fail(
            f"tests: no rfnry-memory checkout at {memory}; clone it beside the engine's checkout"
        )
    assert cut(ROOT / SRC / own, start, end) == cut(memory / SRC / theirs, start, end)
