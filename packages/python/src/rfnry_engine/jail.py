import contextlib
import errno
import os
import stat
from collections.abc import Iterable
from pathlib import Path

from rfnry_engine.errors import InputError

PLACE = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY
FOLDER = PLACE | os.O_NOFOLLOW


class Jail:
    def __init__(self, root: Path | str, *, also: Iterable[Path | str] = ()) -> None:
        self.root = Path(root).resolve()
        self.also = [Path(place).resolve() for place in also]
        self._identities = {place: _identity(place) for place in (self.root, *self.also)}

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

    def open(self, path: str | Path, flags: int = os.O_RDONLY, mode: int = 0o666) -> int:
        target = self.resolve(path)
        place = next(p for p in (self.root, *self.also) if target == p or p in target.parents)
        *folders, name = target.relative_to(place).parts or (".",)
        parent = os.open(place, PLACE)
        try:
            found = os.fstat(parent)
            if (found.st_dev, found.st_ino) != self._identities.get(place):
                raise InputError(f"jail: path outside the agent directory: {path}")
            for folder in folders:
                if flags & os.O_CREAT:
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(folder, dir_fd=parent)
                parent, above = _at(path, folder, FOLDER, mode, parent), parent
                os.close(above)
            return _at(path, name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode, parent)
        finally:
            os.close(parent)


def _identity(place: Path) -> tuple[int, int] | None:
    try:
        found = os.stat(place)
    except OSError:
        return None
    return found.st_dev, found.st_ino


def _at(path: str | Path, name: str, flags: int, mode: int, parent: int) -> int:
    try:
        return os.open(name, flags, mode, dir_fd=parent)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR) and _link(name, parent):
            raise InputError(f"jail: path outside the agent directory: {path}") from None
        raise


def _link(name: str, parent: int) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode)
    except OSError:
        return False
