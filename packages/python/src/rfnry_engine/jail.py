from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rfnry_engine.errors import InputError


class Jail:
    def __init__(self, root: Path | str, *, also: Iterable[Path | str] = ()) -> None:
        self.root = Path(root).resolve()
        self.also = [Path(place).resolve() for place in also]

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            target = candidate.resolve()
        except (ValueError, OSError) as exc:
            raise InputError(f"jail: path outside the agent directory: {path}") from exc
        if not any(target == place or place in target.parents for place in (self.root, *self.also)):
            raise InputError(f"jail: path outside the agent directory: {path}")
        return target
