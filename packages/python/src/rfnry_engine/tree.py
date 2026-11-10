import json
import os
import re
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Literal

from pathspec import GitIgnoreSpec
from pathspec.patterns.gitignore.spec import GitIgnoreSpecPattern
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from rfnry_engine.errors import ConfigurationError
from rfnry_engine.folders import Gitignored
from rfnry_engine.frontmatter import plainly
from rfnry_engine.manifest import Kind, RouteSpec
from rfnry_engine.models import Check, Choice, Part, Question, Score, Text
from rfnry_engine.tools.folder import ToolDef, load_scripts, project_of
from rfnry_engine.workers.pool import Script

OUTPUT_TOOL = "Output"


class QuestionFront(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "score", "check"]
    instructions: str | dict[str, JsonValue] | list[JsonValue] | None = None
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None] | None = None
    levels: list[str | dict[str, JsonValue] | list[JsonValue]] | None = None
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None

    @field_validator("instructions")
    @classmethod
    def _instructions_say_something(
        cls, value: str | dict[str, JsonValue] | list[JsonValue] | None
    ) -> str | dict[str, JsonValue] | list[JsonValue] | None:
        if value is not None and not (value.strip() if isinstance(value, str) else value):
            raise ValueError("tree: instructions must say something; write the question")
        return value


QUESTION_TYPES: dict[str, type[Choice] | type[Score] | type[Check]] = {
    "choice": Choice,
    "score": Score,
    "check": Check,
}
SHAPE = "type, instructions and its options, levels or yes and no"
TAKES = {"choice": ("options",), "score": ("levels",), "check": ("yes", "no")}
TAKEN = "a choice takes options, a score levels and a check yes and no"
QUESTION_FILES = frozenset({".json", ".py", ".md"})


def foreign(kind: JsonValue, data: Mapping[str, JsonValue]) -> str | None:
    own = TAKES.get(kind) if isinstance(kind, str) else None
    if own is None:
        return None
    others = {field for fields in TAKES.values() for field in fields} - set(own)
    return next((key for key in sorted(data) if key in others), None)


def as_question(data: Mapping[str, JsonValue]) -> Question:
    front = QuestionFront.model_validate(data)
    return QUESTION_TYPES[front.type].model_validate(front.model_dump(exclude_unset=True))


class Skill(BaseModel):
    path: str
    place: Path
    description: str
    body: str


def skill_text(skill: Skill) -> Text:
    return Text(text=f"<<<SKILL {skill.path}>>>\n{skill.body}")


class AgentSpec(BaseModel):
    root: Path
    name: str
    kind: Kind
    model: str
    description: str
    body: str
    agents: list[str]
    route: RouteSpec | None
    questions: dict[str, Question]
    keep: list[str]
    fingerprint: str
    granted: frozenset[str]
    workspace: bool = False
    scripted: dict[str, tuple[dict[str, JsonValue], Script]] = Field(default_factory=dict)
    preload: list[Part] = Field(default_factory=list)
    preload_digest: str
    listing: str = ""
    reachable: list[str] = Field(default_factory=list)
    skills: dict[str, Skill] = Field(default_factory=dict)
    offered: frozenset[str] = frozenset()
    public: tuple[str, ...] = ()
    folder_tools: dict[str, ToolDef] = Field(default_factory=dict)
    modes: dict[str, AgentSpec] = Field(default_factory=dict)
    needs_mode: bool = False
    worker: str = "default"
    withheld: Callable[[Path], bool]
    withheld_when_invoked: dict[str, Callable[[Path], bool]] = Field(default_factory=dict)
    named: frozenset[Path] = frozenset()
    included_folders: list[str] = Field(default_factory=list)
    max_model_calls: int | None = None
    tool_timeout: float | None = None
    max_depth: int | None = None
    tool_parallel: int | None = None
    delegate_parallel: int | None = None
    mode: str | None = None


def no_such_skill(spec: AgentSpec, skill: str) -> str:
    if spec.skills:
        have = f"its skills are {', '.join(spec.skills)}"
    else:
        have = (
            "it has no skills; name a folder holding SKILL.md in skills.include or skills.preload"
        )
    return f"agent {spec.name!r} has no skill {skill!r}; {have}"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"tree: {path} is not UTF-8 text; save it as UTF-8") from exc
    except OSError as exc:
        raise ConfigurationError(
            f"tree: cannot read {path}: {exc.strerror or exc}; make it readable to the engine"
        ) from exc


def _object(path: Path) -> dict[str, JsonValue]:
    try:
        data = json.loads(_read(path))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"tree: {path} is not valid JSON: {exc}; fix it in that file"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"tree: {path} is not a JSON object; a question is an object with {SHAPE}"
        )
    unknown = sorted(key for key in data if key not in QuestionFront.model_fields)
    if unknown:
        raise ConfigurationError(
            f"tree: {path} has the unknown key {unknown[0]!r}; a question takes type, instructions, options, levels, yes and no"
        )
    held = foreign(data.get("type"), data)
    if held is not None:
        raise ConfigurationError(
            f"tree: {path} is a {data['type']} question holding {held!r}; {TAKEN}, so drop {held!r} or change its type"
        )
    return data


def _question(path: Path) -> Question:
    try:
        return as_question(_object(path))
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}; {plainly(exc)}") from exc


def _front(path: Path) -> dict[str, JsonValue]:
    try:
        return QuestionFront.model_validate(_object(path)).model_dump(exclude_unset=True)
    except ValidationError as exc:
        raise ConfigurationError(f"tree: invalid question {path}; {plainly(exc)}") from exc


def _declared(folder: Path) -> list[Path]:
    try:
        with os.scandir(folder) as entries:
            found = []
            for entry in entries:
                try:
                    entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                found.append(Path(entry.path))
    except OSError as exc:
        raise ConfigurationError(
            f"tree: cannot read {folder}: {exc.strerror or exc}; make it readable to the engine"
        ) from exc
    return sorted(found)


def _compiled(line: str) -> GitIgnoreSpecPattern | None:
    written = line.rstrip()
    try:
        return GitIgnoreSpecPattern(f"{written}*/" if written.endswith("/**/") else line)
    except ValueError, re.PatternError:
        return None


def _gitignore(path: Path) -> GitIgnoreSpec:
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ConfigurationError(
            f"tree: cannot read {path}: {reason}; make it readable to the engine"
        ) from exc
    compiled = (_compiled(line) for line in text.split("\n"))
    return GitIgnoreSpec([pattern for pattern in compiled if pattern is not None])


class Gitignores:
    def __init__(self) -> None:
        self._read: dict[Path, GitIgnoreSpec] = {}

    def _spec(self, path: Path) -> GitIgnoreSpec:
        if path not in self._read:
            self._read[path] = _gitignore(path)
        return self._read[path]

    def held_in(self, here: Path, check: Gitignored) -> None:
        gitignore = here / ".gitignore"
        if gitignore.is_file() and not gitignore.is_symlink():
            check.specs[here] = self._spec(gitignore)


def question_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        p
        for p in _declared(directory)
        if p.is_file() and not p.name.startswith(("_", ".")) and p.suffix in QUESTION_FILES
    ]


def question_parts(directory: Path, files: Collection[Path]) -> tuple[dict[str, Path], list[str]]:
    for path in files:
        if path.suffix == ".md":
            raise ConfigurationError(
                f"tree: question {path} is markdown; write it as {path.stem}.json with {SHAPE}"
            )
    definitions = {p.stem: p for p in files if p.suffix == ".json"}
    scripts = [p.stem for p in files if p.suffix == ".py"]
    for name in scripts:
        if name not in definitions:
            raise ConfigurationError(
                f"tree: {directory / name}.py has no {name}.json beside it; a question is {name}.json, and {name}.py fills it from data"
            )
    return definitions, scripts


def read_questions(
    directory: Path,
    definitions: Mapping[str, Path],
    scripts: Collection[str],
    root: Path,
) -> tuple[dict[str, Question], dict[str, tuple[dict[str, JsonValue], Script]]]:
    fixed = {name: _question(path) for name, path in definitions.items() if name not in scripts}
    fronts = {name: _front(definitions[name]) for name in scripts}
    held = {directory.resolve()}
    loaded = load_scripts(directory, scripts, project_of(root), root, held) if scripts else {}
    return fixed, {name: (fronts[name], loaded[name]) for name in scripts}


def utf8_name(name: str, path: Path) -> None:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        raise ConfigurationError(
            f"tree: the name of {str(path)!r} is not UTF-8; rename it"
        ) from None
