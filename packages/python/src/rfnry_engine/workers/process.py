import asyncio
import importlib.util
import inspect
import json
import os
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

PREFIX = "tool:"


def _error(exc: BaseException) -> str:
    try:
        return f"{type(exc).__name__}: {exc}"
    except BaseException:
        return f"{type(exc).__name__}: <exception str() failed>"


class _Source(Loader):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source

    def exec_module(self, module: ModuleType) -> None:
        code = compile(self.source, str(self.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


def _spec(name: str, loader: _Source) -> ModuleSpec:
    spec = ModuleSpec(name, loader, origin=str(loader.path))
    spec.has_location = True
    return spec


class _Helpers(MetaPathFinder):
    def __init__(self, folder: Path, sources: dict[str, str]) -> None:
        self.folder = folder
        self.sources = sources

    def find_spec(
        self, name: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if path is not None or name not in self.sources:
            return None
        return _spec(name, _Source(self.folder / f"{name}.py", self.sources[name]))


def _import(key: str, path: Path, source: str) -> ModuleType | str:
    name = f"{PREFIX}{key}"
    loader = _Source(path, source)
    module = importlib.util.module_from_spec(_spec(name, loader))
    sys.modules[name] = module
    try:
        loader.exec_module(module)
    except BaseException as exc:
        del sys.modules[name]
        return _error(exc)
    return module


def _inside(module: ModuleType, folder: Path) -> bool:
    file = getattr(module, "__file__", None)
    return isinstance(file, str) and Path(file).is_relative_to(folder)


def _before_paths() -> int:
    return next(
        (at for at, held in enumerate(sys.meta_path) if held is PathFinder), len(sys.meta_path)
    )


def _load(
    scripts: list[dict[str, str]], helpers: dict[str, dict[str, str]]
) -> dict[str, ModuleType | str]:
    folders: dict[Path, dict[str, tuple[Path, str]]] = {}
    for script in scripts:
        path = Path(script["path"])
        folders.setdefault(path.parent, {})[script["key"]] = (path, script["source"])
    loaded: dict[str, ModuleType | str] = {}
    for folder, members in folders.items():
        before = set(sys.modules)
        finder = _Helpers(folder, helpers.get(str(folder), {}))
        sys.meta_path.insert(_before_paths(), finder)
        for key, (path, source) in members.items():
            loaded[key] = _import(key, path, source)
        sys.meta_path.remove(finder)
        for name in set(sys.modules) - before:
            if not name.startswith(PREFIX) and _inside(sys.modules[name], folder):
                del sys.modules[name]
    return loaded


def _write(fd: int, text: str) -> None:
    data = memoryview(f"{text}\n".encode())
    while data:
        data = data[os.write(fd, data) :]


def _read(requests: BinaryIO, loop: asyncio.AbstractEventLoop, lines: asyncio.Queue[bytes]) -> None:
    for line in requests:
        loop.call_soon_threadsafe(lines.put_nowait, line)
    loop.call_soon_threadsafe(lines.put_nowait, b"")


async def _serve(
    modules: dict[str, ModuleType | str],
    threads: int,
    limit: int | None,
    requests: BinaryIO,
    replies: int,
) -> None:
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(threads)
    tasks: dict[int, asyncio.Task[None]] = {}
    works: dict[int, Future[Any]] = {}

    async def run(id: int, module: ModuleType, args: Any, call: Any) -> Any:
        if inspect.iscoroutinefunction(module.run):
            return await module.run(args, call)
        work = pool.submit(module.run, args, call)
        works[id] = work
        return await asyncio.wrap_future(work)

    async def handle(id: int, key: str, args: Any, call: Any) -> None:
        name = key.rsplit("/", 1)[-1]
        try:
            loaded = modules.get(key, f"workers/process: the worker has no tool {name}")
            if isinstance(loaded, str):
                text = json.dumps({"id": id, "error": loaded})
            else:
                result = await run(id, loaded, args, call)
                text = json.dumps({"id": id, "result": result}, allow_nan=False)
        except asyncio.CancelledError:
            text = json.dumps({"id": id, "error": "CancelledError: the call was cancelled"})
        except BaseException as exc:
            text = json.dumps({"id": id, "error": _error(exc)})
        if limit is not None and len(text) > limit:
            over = f"workers/process: {name} sent back {len(text)} bytes, over the limit of {limit} bytes"
            text = json.dumps({"id": id, "error": over})
        tasks.pop(id, None)
        works.pop(id, None)
        _write(replies, text)

    lines: asyncio.Queue[bytes] = asyncio.Queue()
    threading.Thread(target=_read, args=(requests, loop, lines), daemon=True).start()
    while line := await lines.get():
        message = json.loads(line)
        id = message["id"]
        if "cancel" in message:
            if id in works:
                works[id].cancel()
            elif id in tasks:
                tasks[id].cancel()
            continue
        task = asyncio.create_task(
            handle(id, message["key"], message["args"], message["call"]), eager_start=True
        )
        if not task.done():
            tasks[id] = task
    os._exit(0)


def _release(fds: tuple[int, ...]) -> None:
    nothing = os.open(os.devnull, os.O_RDWR)
    for fd in fds:
        os.dup2(nothing, fd, inheritable=False)
    os.close(nothing)


def main() -> None:
    sys.dont_write_bytecode = True
    requests = os.fdopen(os.dup(0), "rb")
    replies = os.dup(1)
    os.register_at_fork(after_in_child=partial(_release, (requests.fileno(), replies)))
    nothing = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nothing, 0)
    os.close(nothing)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    start = json.loads(requests.readline())
    modules = _load(start["scripts"], start["helpers"])
    asyncio.run(_serve(modules, start["threads"], start["limit"], requests, replies))


if __name__ == "__main__":
    main()
