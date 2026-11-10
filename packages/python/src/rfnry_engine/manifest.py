from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.frontmatter import (
    MERGE_TAG,
    YAML_FAILURES,
    StrictLoader,
    plainly,
    unreadable,
)

MANIFEST = "rfnry.yaml"
Kind = Literal["generation", "decision"]


def _from_the_root(value: str) -> str:
    if value.startswith("/") or "\\" in value or ".." in value.split("/"):
        raise ValueError(
            f"manifest: {value!r} is not written from the folder root; write it with no leading /, no .. and no backslash"
        )
    return value


_FolderPath = Annotated[str, AfterValidator(_from_the_root)]
_Count = Annotated[int, Field(ge=1)]
_Seconds = Annotated[float, Field(gt=0)]


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
    to: dict[str, str]
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerEntry(_Closed):
    processes: _Count = 1
    threads: _Count = 32
    max_result_bytes: _Count | None = None


class FilesSection(_Closed):
    include: list[_FolderPath] | None = None
    exclude: list[_FolderPath] | None = None
    preload: list[_FolderPath] | None = None
    gitignore: bool = False


class ToolsSection(_Closed):
    include: list[str] | None = None
    exclude: list[str] | None = None
    timeout: _Seconds | None = None
    max_parallel: _Count | None = None


class SkillsSection(_Closed):
    include: list[_FolderPath] | None = None
    exclude: list[_FolderPath] | None = None
    preload: list[_FolderPath] | None = None
    public: list[str] | Literal["all"] | None = None

    @field_validator("public", mode="before")
    @classmethod
    def _public_is_all_or_names(cls, value: object) -> object:
        if value == "all" or (
            isinstance(value, list) and all(isinstance(name, str) for name in value)
        ):
            return value
        raise ValueError(f"manifest: write all or a list of names, got {value!r}")


class Orchestration(_Closed):
    delegate_to: list[str] | None = None
    max_depth: _Count | None = None
    max_parallel: _Count | None = None
    route: RouteSpec | None = None


class ModeEntry(_Closed):
    description: str | None = None
    instructions: _FolderPath | None = None
    model: str | None = None
    max_model_calls: _Count | None = None
    worker: str | None = None
    files: FilesSection | None = None
    tools: ToolsSection | None = None
    skills: SkillsSection | None = None
    orchestration: Orchestration | None = None
    workspace: bool | None = None
    questions: list[_FolderPath] | None = None

    @field_validator("model")
    @classmethod
    def _a_model_is_not_padded(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError(
                f"manifest: model must not be padded, got {value!r}; write the name of an entry under models:, such as model: strong"
            )
        return value


class AgentEntry(ModeEntry):
    model: str
    kind: Kind
    modes: dict[str, ModeEntry] = Field(default_factory=dict)


class Resolved(ModeEntry):
    model: str
    kind: Kind
    files: FilesSection = Field(default_factory=FilesSection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    skills: SkillsSection = Field(default_factory=SkillsSection)
    orchestration: Orchestration = Field(default_factory=Orchestration)


class ModelEntry(_Closed):
    provider: str
    model: str | None = None
    effort: str | None = None


class ToolEntry(_Closed):
    module: _FolderPath
    function: str = "run"
    description: str | None = None
    schema_file: _FolderPath | None = Field(default=None, alias="schema")
    timeout: _Seconds | None = None


class Manifest(_Closed):
    version: Literal[1]
    workers: dict[str, WorkerEntry] = Field(default_factory=dict)
    models: dict[str, ModelEntry]
    tools: dict[str, ToolEntry] = Field(default_factory=dict)
    agents: dict[str, AgentEntry]

    @field_validator("version", mode="before")
    @classmethod
    def _version_is_the_number_1(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError(f"manifest: write version: 1, got {value!r}")
        return value


_SECTIONS = ("files", "tools", "skills", "orchestration")
_LEFT_OUT: dict[Kind, tuple[frozenset[str], str]] = {
    "generation": (
        frozenset({"questions", "orchestration.route"}),
        "a generation agent takes everything but questions and orchestration.route",
    ),
    "decision": (
        frozenset(
            {
                "max_model_calls",
                "files",
                "tools",
                "skills",
                "workspace",
                "orchestration.delegate_to",
                "orchestration.max_depth",
                "orchestration.max_parallel",
            }
        ),
        "a decision agent takes only kind, model, description, instructions, worker, questions, orchestration.route and modes",
    ),
}


class _ManifestLoader(StrictLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        for key_node, _ in node.value:
            if key_node.tag == MERGE_TAG:
                raise yaml.constructor.ConstructorError(
                    None, None, "a merge key is refused", key_node.start_mark
                )
        return super().construct_mapping(node, deep=deep)


def _text_of(root: Path, path: Path) -> str:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"manifest: {root} holds no {MANIFEST}; write one there, or point the engine at the folder that holds it"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"manifest: cannot read {path}: {exc.strerror}") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"manifest: {path} is not UTF-8; save it as UTF-8") from exc


def _parsed(text: str, path: Path) -> object:
    try:
        for event in yaml.parse(text, Loader=_ManifestLoader):
            if isinstance(event, yaml.NodeEvent) and event.anchor is not None:
                mark = event.start_mark
                place = "" if mark is None else f" at line {mark.line + 1}"
                quote = (
                    "quote it if it is a pattern, or " if isinstance(event, yaml.AliasEvent) else ""
                )
                raise ConfigurationError(
                    f"manifest: an anchor or alias in {path}{place} is refused; {quote}write the value out where it is used"
                )
        return yaml.load(text, Loader=_ManifestLoader)
    except YAML_FAILURES as exc:
        place, problem, fix = unreadable(exc, text, first_line=1)
        raise ConfigurationError(
            f"manifest: invalid YAML in {path}{place}: {problem}; {fix}"
        ) from exc


def _says_nothing(value: object) -> bool:
    return (
        value is None
        or value == []
        or value == {}
        or (isinstance(value, str) and not value.strip())
    )


def _walk_for_nothing(held: object, at: str, path: Path) -> None:
    inner: Iterable[tuple[object, object]]
    if isinstance(held, dict):
        inner = held.items()
    elif isinstance(held, list):
        inner = enumerate(held)
    else:
        return
    for key, value in inner:
        where = f"{at}.{key}" if at else str(key)
        if _says_nothing(value):
            raise ConfigurationError(
                f"manifest: {where} in {path} says nothing; write what it holds, or remove the key"
            )
        _walk_for_nothing(value, where, path)


def _refuse_what_says_nothing(data: dict[Any, Any], path: Path) -> None:
    try:
        _walk_for_nothing(data, "", path)
    except RecursionError as exc:
        raise ConfigurationError(
            f"manifest: {path} is nested too deeply; no key of this file holds that many levels"
        ) from exc


def _keys_at(loc: tuple[int | str, ...]) -> list[str]:
    shape: type[BaseModel] = Manifest
    parts = iter(loc[:-1])
    for part in parts:
        annotation = shape.model_fields[str(part)].annotation
        if get_origin(annotation) is dict:
            next(parts)
        shape = next(
            arg
            for arg in get_args(annotation)
            if isinstance(arg, type) and issubclass(arg, BaseModel)
        )
    return [field.alias or name for name, field in shape.model_fields.items()]


def _validated(data: dict[Any, Any], path: Path) -> Manifest:
    try:
        manifest = Manifest.model_validate(data, strict=True)
    except ValidationError as exc:
        unknown = [error["loc"] for error in exc.errors() if error["type"] == "extra_forbidden"]
        if unknown:
            named = "; ".join(
                f"{'.'.join(map(str, loc))} (it takes {', '.join(_keys_at(loc))})"
                for loc in unknown
            )
            raise ConfigurationError(f"manifest: unknown key in {path}: {named}") from exc
        _refuse_what_says_nothing(data, path)
        raise ConfigurationError(f"manifest: invalid {path}; {plainly(exc)}") from exc
    _refuse_what_says_nothing(data, path)
    return manifest


def _refuse_names_that_are_not_plain(manifest: Manifest, path: Path) -> None:
    maps: list[tuple[str, Iterable[str]]] = [
        ("agents", manifest.agents),
        ("workers", manifest.workers),
        ("models", manifest.models),
        ("tools", manifest.tools),
        *((f"agents.{name}.modes", agent.modes) for name, agent in manifest.agents.items()),
    ]
    for at, names in maps:
        for name in names:
            if not name or name != name.strip() or name.startswith(".") or "/" in name:
                raise ConfigurationError(
                    f"manifest: {at}.{name} in {path} is not a plain name, got {name!r}; write it with no blank, no padding, no leading . and no /"
                )


def _layers(name: str, agent: AgentEntry) -> Iterator[tuple[str, ModeEntry]]:
    yield f"agents.{name}", agent
    for mode, entry in agent.modes.items():
        yield f"agents.{name}.modes.{mode}", entry


def _said_keys(entry: ModeEntry) -> set[str]:
    orchestration = entry.orchestration.model_fields_set if entry.orchestration else set()
    return entry.model_fields_set | {f"orchestration.{key}" for key in orchestration}


def _refuse_what_its_kind_leaves_out(name: str, agent: AgentEntry, path: Path) -> None:
    left_out, takes = _LEFT_OUT[agent.kind]
    for at, entry in _layers(name, agent):
        taken = sorted(_said_keys(entry) & left_out)
        if taken:
            raise ConfigurationError(f"manifest: {at}.{taken[0]} in {path}: {takes}; remove it")


def _refuse_missing_instructions(name: str, agent: AgentEntry, path: Path) -> None:
    if agent.instructions is not None:
        return
    without = (
        [
            f"agents.{name}.modes.{mode}"
            for mode, entry in agent.modes.items()
            if entry.instructions is None
        ]
        if agent.modes
        else [f"agents.{name}"]
    )
    if without:
        raise ConfigurationError(
            f"manifest: {without[0]}.instructions is missing in {path}; instructions is required on the entry or on every mode"
        )


def _refuse_loose_workers(manifest: Manifest, path: Path) -> None:
    picked: set[str] = set()
    for name, agent in manifest.agents.items():
        for at, entry in _layers(name, agent):
            if entry.worker is None:
                continue
            if entry.worker not in manifest.workers:
                raise ConfigurationError(
                    f"manifest: {at}.worker in {path} names {entry.worker!r}, which workers: does not declare; declare it there, or leave worker: out to use the default"
                )
            picked.add(entry.worker)
        if agent.worker is None and (
            agent.instructions is not None
            or any(mode.worker is None for mode in agent.modes.values())
        ):
            picked.add("default")
    loose = sorted(set(manifest.workers) - picked)
    if loose:
        raise ConfigurationError(
            f"manifest: workers.{loose[0]} in {path} is picked by nothing; pick it with worker: on an agent or a mode, or remove it"
        )


def _refuse_loose_models(manifest: Manifest, path: Path) -> None:
    picked: set[str] = set()
    for name, agent in manifest.agents.items():
        for at, entry in _layers(name, agent):
            if entry.model is None:
                continue
            if entry.model not in manifest.models:
                raise ConfigurationError(
                    f"manifest: {at}.model in {path} names {entry.model!r}, which models: does not declare; declare it there, or pick one of {', '.join(sorted(manifest.models))}"
                )
            picked.add(entry.model)
    loose = sorted(set(manifest.models) - picked)
    if loose:
        raise ConfigurationError(
            f"manifest: models.{loose[0]} in {path} is picked by nothing; pick it with model: on an agent or a mode, or remove it"
        )


def read_manifest(root: Path) -> Manifest:
    path = root / MANIFEST
    data = _parsed(_text_of(root, path), path)
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"manifest: {path} holds no keys; write version: 1, models: and agents: in it"
        )
    manifest = _validated(data, path)
    _refuse_names_that_are_not_plain(manifest, path)
    for name, agent in manifest.agents.items():
        _refuse_what_its_kind_leaves_out(name, agent, path)
        _refuse_missing_instructions(name, agent, path)
    _refuse_loose_workers(manifest, path)
    _refuse_loose_models(manifest, path)
    return manifest


def _joined(below: object, above: object) -> object:
    if not isinstance(above, list):
        return above
    if below == "all":
        return below
    earlier = below if isinstance(below, list) else []
    return list(dict.fromkeys([*earlier, *above]))


def _layered(below: BaseModel | None, above: BaseModel | None) -> dict[str, object]:
    joined: dict[str, object] = {}
    for layer in (below, above):
        if layer is not None:
            for name in layer.model_fields_set:
                joined[name] = _joined(joined.get(name), getattr(layer, name))
    return joined


def overlaid(entry: AgentEntry, mode: str | None) -> Resolved:
    above = ModeEntry() if mode is None else entry.modes[mode]
    sections = {name: _layered(getattr(entry, name), getattr(above, name)) for name in _SECTIONS}
    joined = _layered(entry, above)
    joined.pop("modes", None)
    return Resolved.model_validate(joined | sections)
