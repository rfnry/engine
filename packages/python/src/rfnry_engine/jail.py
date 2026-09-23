from __future__ import annotations

from pathlib import Path

from rfnry_engine.errors import InputError


class Jail:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            target = candidate.resolve()
        except (ValueError, OSError) as exc:
            raise InputError(f"jail: path outside the agent directory: {path}") from exc
        if target != self.root and self.root not in target.parents:
            raise InputError(f"jail: path outside the agent directory: {path}")
        return target
