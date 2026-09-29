import ast
import json
from pathlib import Path
from typing import Any

import pytest

from rfnry_engine.models import Choice

EXAMPLES = Path(__file__).parents[4] / "examples"
TAXONOMY = EXAMPLES / "decision_taxonomy"


@pytest.mark.parametrize(
    "path", sorted(EXAMPLES.glob("*/*.py")), ids=lambda p: p.parent.name + "/" + p.name
)
def test_every_example_compiles(path: Path) -> None:
    compile(
        path.read_text(encoding="utf-8"), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    )


def leaves(children: dict[str, Any], path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    criteria = ("what", "not_for", "examples")
    Choice(
        instructions="Which?",
        options={key: {c: node[c] for c in criteria} for key, node in children.items()},
    )
    found: set[tuple[str, ...]] = set()
    for key, node in children.items():
        found |= leaves(node["children"], (*path, key)) if "children" in node else {(*path, key)}
    return found


def test_the_taxonomy_example_labels_its_items_with_leaves_of_its_taxonomy() -> None:
    taxonomy = json.loads((TAXONOMY / "taxonomy.json").read_text(encoding="utf-8"))
    items = json.loads((TAXONOMY / "items.json").read_text(encoding="utf-8"))
    known = leaves(taxonomy)
    labelled = [tuple(item["path"]) for item in items if item["path"] is not None]
    assert len(known) == 18 and set(labelled) == known
    assert all(len(path) == 3 for path in known)
    assert sum(item["path"] is None for item in items) == 6
