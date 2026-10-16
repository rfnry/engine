import ast
from pathlib import Path

import rfnry_engine

SRC = Path(rfnry_engine.__file__).parent

LOG_CALLS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

ALLOWED: dict[tuple[str, str], str] = {
    ("tools/workspace.py", "failed[0]"): "a backend's kill() error reaches the caller as it is",
    ("tools/workspace.py", "error"): "a backend's kill() error reaches the caller as it is",
    ("tools/workspace.py", "cancelled"): "a cancel that came during a kill goes on once it ends",
    ("stop.py", "cancelled"): "a cancel that came while a run's calls ended goes on once they have",
}


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _leading(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        text = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                text += part.value
            else:
                return text + "{"
        return text
    return None


def _raises(path: Path) -> list[tuple[int, str, str | None]]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        first = exc.args[0] if isinstance(exc, ast.Call) and exc.args else None
        found.append(
            (node.lineno, ast.unparse(exc), _leading(first) if first is not None else None)
        )
    return found


def _logs(path: Path) -> list[tuple[int, str, str | None]]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Name) and owner.id in {"logger", "logging"}):
            continue
        if node.func.attr not in LOG_CALLS:
            continue
        args = node.args[1:] if node.func.attr == "log" else node.args
        found.append((node.lineno, ast.unparse(node), _leading(args[0]) if args else None))
    return found


def _key(path: Path, source: str) -> tuple[str, str]:
    return str(path.relative_to(SRC)), source


def test_every_error_message_starts_with_its_module_name() -> None:
    wrong = []
    for path in _sources():
        prefix = f"{path.stem}: "
        for line, source, text in _raises(path):
            if text is None:
                if _key(path, source) not in ALLOWED:
                    wrong.append(f"{path.name}:{line}: no literal message: {source}")
                continue
            rest = text[len(prefix) :]
            if not text.startswith(prefix) or not rest or rest[0].isupper() or rest[0].isspace():
                wrong.append(f"{path.name}:{line}: {text!r}")
    assert wrong == []


def test_every_log_message_is_lowercase() -> None:
    wrong = []
    for path in _sources():
        for line, source, text in _logs(path):
            if text is None or not text or text[0].isupper():
                wrong.append(f"{path.name}:{line}: {source}")
    assert wrong == []


def test_the_allowlist_names_only_raises_that_exist() -> None:
    present = {
        _key(path, source)
        for path in _sources()
        for _, source, text in _raises(path)
        if text is None
    }
    assert set(ALLOWED) <= present
