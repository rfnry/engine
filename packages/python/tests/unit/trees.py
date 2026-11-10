from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from rfnry_engine.manifest import MANIFEST


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def write_folder(
    root: Path,
    agents: Mapping[str, Mapping[str, Any]],
    *,
    workers: Mapping[str, Any] | None = None,
    tools: Mapping[str, Any] | None = None,
    files: Mapping[str, str] | None = None,
) -> Path:
    document: dict[str, Any] = {"version": 1}
    if workers is not None:
        document["workers"] = workers
    if tools is not None:
        document["tools"] = tools
    document["agents"] = agents
    (root / MANIFEST).write_text(
        yaml.dump(document, Dumper=_NoAliases, sort_keys=False), encoding="utf-8"
    )
    for name, text in (files or {}).items():
        place = root / name
        place.parent.mkdir(parents=True, exist_ok=True)
        place.write_text(text, encoding="utf-8")
    return root
