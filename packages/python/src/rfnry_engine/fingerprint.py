from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from rfnry_engine.errors import InputError
from rfnry_engine.jail import Jail
from rfnry_engine.models import Question, ToolSpec

SHORT = 16


def _readable(root: Path) -> list[tuple[str, Path]]:
    jail = Jail(root)
    found: list[tuple[str, Path]] = []
    for directory, folders, files in os.walk(root):
        folders[:] = [name for name in folders if not name.startswith(".")]
        for name in files:
            path = Path(directory) / name
            if name.startswith(".") or not path.is_file():
                continue
            try:
                jail.resolve(path)
            except InputError:
                continue
            found.append((path.relative_to(root).as_posix(), path))
    return sorted(found, key=lambda item: item[0])


def folder_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name, path in _readable(root):
        with path.open("rb") as file:
            content = hashlib.file_digest(file, "sha256").digest()
        digest.update(name.encode("utf-8") + b"\0" + content)
    return digest.hexdigest()[:SHORT]


def tools_fingerprint(tools: list[ToolSpec]) -> str:
    canonical = json.dumps(
        [tool.model_dump(mode="json") for tool in sorted(tools, key=lambda tool: tool.name)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:SHORT]


def questions_fingerprint(questions: Mapping[str, Question]) -> str:
    canonical = json.dumps(
        {qid: questions[qid].model_dump(mode="json") for qid in sorted(questions)},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:SHORT]
