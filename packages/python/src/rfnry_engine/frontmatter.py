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


_HEAD = re.compile(r"\A---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def described(path: Path) -> str | None:
    try:
        raw = path.read_bytes().decode("utf-8").removeprefix("\ufeff")
    except UnicodeDecodeError:
        return None
    match = _HEAD.match(raw)
    if match is None:
        return None
    try:
        data = yaml.load(match.group(1), Loader=_Loader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"frontmatter: invalid YAML in {path}: {exc}") from exc
    said = data.get("description") if isinstance(data, dict) else None
    if said is not None and not isinstance(said, str):
        raise ConfigurationError(f"frontmatter: description must be a string in {path}")
    collapsed = " ".join(said.split()) if said else ""
    return collapsed or None


def parse_frontmatter_doc[T: BaseModel](
    raw: str,
    *,
    model: type[T],
    source: Path | str,
    label: str,
) -> tuple[T, str]:
    split = split_frontmatter(raw)
    if split is None:
        raise ConfigurationError(f"frontmatter: missing in {source}")
    front_raw, body = split
    try:
        data = yaml.load(front_raw, Loader=_Loader) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"frontmatter: invalid YAML in {source}: {exc}") from exc
    try:
        config = model.model_validate(data)
    except ValidationError as exc:
        unknown = [
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors()
            if error["type"] == "extra_forbidden"
        ]
        if unknown:
            raise ConfigurationError(
                f"frontmatter: unknown key {', '.join(map(repr, unknown))} in {source}"
            ) from exc
        raise ConfigurationError(f"frontmatter: invalid {label} {source}: {exc}") from exc
    return config, body.lstrip("\n")
