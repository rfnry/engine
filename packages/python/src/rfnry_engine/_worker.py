import asyncio
import importlib.util
import inspect
import json
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

PREFIX = "tool:"


def _error(exc: BaseException) -> str:
    try:
        return f"{type(exc).__name__}: {exc}"
    except BaseException:
        return f"{type(exc).__name__}: <exception str() failed>"


def _import(key: str, path: Path) -> ModuleType | str:
    name = f"{PREFIX}{key}"
    loader = SourceFileLoader(name, str(path))
    spec = ModuleSpec(name, loader, origin=str(path))
    spec.has_location = True
    module = importlib.util.module_from_spec(spec)
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


def _load(scripts: list[dict[str, str]]) -> dict[str, ModuleType | str]:
    folders: dict[Path, dict[str, Path]] = {}
    for script in scripts:
        path = Path(script["path"])
        folders.setdefault(path.parent, {})[script["key"]] = path
    loaded: dict[str, ModuleType | str] = {}
    for folder, members in folders.items():
        before = set(sys.modules)
        sys.path.insert(0, str(folder))
        for key, path in members.items():
            loaded[key] = _import(key, path)
        sys.path.remove(str(folder))
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
    modules: dict[str, ModuleType | str], threads: int, requests: BinaryIO, replies: int
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
        try:
            loaded = modules.get(key, f"workers: the worker has no tool {key.rsplit('/', 1)[-1]}")
            if isinstance(loaded, str):
                text = json.dumps({"id": id, "error": loaded})
            else:
                result = await run(id, loaded, args, call)
                text = json.dumps({"id": id, "result": result}, allow_nan=False)
        except asyncio.CancelledError:
            text = json.dumps({"id": id, "error": "CancelledError: the call was cancelled"})
        except BaseException as exc:
            text = json.dumps({"id": id, "error": _error(exc)})
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


def main() -> None:
    sys.dont_write_bytecode = True
    requests = os.fdopen(os.dup(0), "rb")
    replies = os.dup(1)
    nothing = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nothing, 0)
    os.close(nothing)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    start = json.loads(requests.readline())
    asyncio.run(_serve(_load(start["scripts"]), start["threads"], requests, replies))


if __name__ == "__main__":
    main()
