from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from rfnry_engine.models import Question, ToolSpec

SHORT = 16


def folder_fingerprint(files: list[tuple[str, Path]], read: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, path in files:
        if name in read:
            content = hashlib.sha256(read[name]).digest()
        else:
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
