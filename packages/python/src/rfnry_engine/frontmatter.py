import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ValidationError

from rfnry_engine.errors import ConfigurationError

_FRONTMATTER = re.compile(r"^---\n(?:(.*?)\n)?---[ \t]*(?:\n(.*))?\Z", re.DOTALL)
_BOOL = "tag:yaml.org,2002:bool"
_MERGE = "tag:yaml.org,2002:merge"


if TYPE_CHECKING:
    _Base = yaml.SafeLoader
else:
    _Base = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class _Loader(_Base):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: list[Any] = []
        for key_node, _ in node.value:
            if key_node.tag == _MERGE:
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.append(key)
        return super().construct_mapping(node, deep=deep)


_Loader.yaml_implicit_resolvers = {
    first: [(tag, pattern) for tag, pattern in resolvers if tag != _BOOL]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    _BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def split_frontmatter(raw: str) -> tuple[str, str] | None:
    match = _FRONTMATTER.match(raw.removeprefix("\ufeff"))
    if match is None:
        return None
    return match.group(1) or "", match.group(2) or ""


def body_of(raw: str) -> str:
    split = split_frontmatter(raw)
    return (raw.removeprefix("\ufeff") if split is None else split[1]).strip()


_HEAD = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


_STARTED = (
    "while scanning a quoted scalar",
    "while scanning a simple key",
    "while parsing a flow",
)
_FIXES = (
    ("tab character", "indent with spaces, not tabs"),
    ("character '\\t'", "indent with spaces, not tabs"),
    ("unacceptable character", "remove the control character"),
    ("quoted scalar", "close the quote"),
    ("could not find expected ':'", "write each entry as key: value"),
    ("expected ',' or ']'", "close the [ with ]"),
    ("expected ',' or '}'", "close the { with }"),
    (
        "mapping values are not allowed",
        "quote a value that holds ': ', or line the key up with the keys above it",
    ),
    ("duplicate key", "remove one of them"),
    ("did not find expected key", "line the key up with the keys around it"),
    ("expected <block end>", "line the key up with the keys around it"),
)


def _unreadable(exc: yaml.YAMLError, text: str) -> tuple[str, str, str]:
    problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    context = getattr(exc, "context", None) or ""
    mark = getattr(exc, "problem_mark", None)
    if context.startswith(_STARTED):
        mark = getattr(exc, "context_mark", None) or mark
    character = getattr(exc, "character", None)
    position = text.find(chr(character)) if isinstance(character, int) else -1
    if mark is not None:
        line, column = mark.line, mark.column
    elif position >= 0:
        line = text.count("\n", 0, position)
        column = position - (text.rfind("\n", 0, position) + 1)
    else:
        line = column = -1
    place = "" if line < 0 else f" at line {line + 2}, column {column + 1}"
    lines = text.split("\n")
    at = lines[line][column : column + 1] if 0 <= line < len(lines) else ""
    said = "tab character" if at == "\t" else f"{problem} {context}"
    fix = next((fix for sign, fix in _FIXES if sign in said), "fix the YAML there")
    return place, problem, fix


def described(path: Path) -> str | None:
    try:
        raw = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None
    return description(raw, path)


def description(raw: str, path: Path) -> str | None:
    match = _HEAD.match(raw.removeprefix("\ufeff"))
    if match is None:
        return None
    try:
        data = yaml.load(match.group(1), Loader=_Loader)
    except yaml.YAMLError as exc:
        place, problem, fix = _unreadable(exc, match.group(1))
        raise ConfigurationError(
            f"frontmatter: invalid YAML in {path}{place}: {problem}; {fix}"
        ) from exc
    said = data.get("description") if isinstance(data, dict) else None
    if said is not None and not isinstance(said, str):
        raise ConfigurationError(
            f"frontmatter: description must be a string in {path}; write it as one line of text"
        )
    collapsed = " ".join(said.split()) if said else ""
    return collapsed or None


def _said(message: str) -> str:
    message = message.removeprefix("Value error, ").removeprefix("tree: ")
    return message[:1].lower() + message[1:]


def plainly(exc: ValidationError, whole: str = "the file") -> str:
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or whole}: {_said(error['msg'])}"
        for error in exc.errors(include_url=False)
    )


def parse_frontmatter_doc[T: BaseModel](
    raw: str,
    *,
    model: type[T],
    source: Path | str,
    label: str,
) -> tuple[T, str]:
    split = split_frontmatter(raw)
    if split is None:
        raise ConfigurationError(
            f"frontmatter: missing in {source}; start the file with a block of YAML between two --- lines"
        )
    front_raw, body = split
    try:
        data = yaml.load(front_raw, Loader=_Loader) or {}
    except yaml.YAMLError as exc:
        place, problem, fix = _unreadable(exc, front_raw)
        raise ConfigurationError(
            f"frontmatter: invalid YAML in {source}{place}: {problem}; {fix}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"frontmatter: {source} holds no keys between its --- lines; write key: value lines between the two --- lines, or start the file without ---"
        )
    try:
        config = model.model_validate(data)
    except ValidationError as exc:
        unknown = [
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors()
            if error["type"] == "extra_forbidden"
        ]
        if unknown:
            others = [key for key in unknown if key != "name"]
            fix = "; ".join(
                [
                    *(["its file names it, so remove the name: line"] if "name" in unknown else []),
                    *(
                        []
                        if not others
                        else ["check its spelling"]
                        if any("." in key for key in others)
                        else [f"it takes {', '.join(model.model_fields)}"]
                    ),
                ]
            )
            raise ConfigurationError(
                f"frontmatter: unknown key {', '.join(map(repr, unknown))} in {source}; {fix}"
            ) from exc
        raise ConfigurationError(f"frontmatter: invalid {label} {source}; {plainly(exc)}") from exc
    return config, body.lstrip("\n")
