import json
import os
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from pydantic import JsonValue

from rfnry_engine.chain import preload_digest
from rfnry_engine.errors import ConfigurationError, InputError
from rfnry_engine.fingerprint import folder_fingerprint
from rfnry_engine.folders import (
    Gitignored,
    listed,
    mapped,
    pattern_covers,
    pattern_leads_into,
    pattern_reaches,
    readable,
    shown_link,
    warns_once,
)
from rfnry_engine.frontmatter import body_of, description
from rfnry_engine.jail import Jail
from rfnry_engine.manifest import (
    MANIFEST,
    AgentEntry,
    FilesSection,
    Resolved,
    overlaid,
    read_manifest,
)
from rfnry_engine.models import Choice, Part, Question
from rfnry_engine.tools.builtins import BUILTINS, preload_parts
from rfnry_engine.tree import (
    QUESTION_FILES,
    AgentSpec,
    Gitignores,
    question_files,
    question_parts,
    read_questions,
    utf8_name,
)
from rfnry_engine.workers.pool import Script


class _Asked(NamedTuple):
    questions: dict[str, Question]
    scripted: dict[str, tuple[dict[str, JsonValue], Script]]
    files: dict[str, Path]


class _Reach(NamedTuple):
    on_the_map: dict[str, Path]
    files: dict[str, Path]
    preloaded: list[str]


def _who(agent: str, mode: str | None) -> str:
    return f"agent {agent!r}" if mode is None else f"mode {mode!r} of agent {agent!r}"


def _instructions(root: Path, who: str, written: str) -> tuple[bytes, str]:
    try:
        raw = (root / written).read_bytes()
    except (OSError, ValueError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ConfigurationError(
            f"compiled: {who} cannot read its instructions {written!r}: {reason}; write the path of a readable file from the folder root"
        ) from exc
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(
            f"compiled: {who} has instructions {written!r} that are not UTF-8 text; save the file as UTF-8"
        ) from exc


def _reached_by(root: Path, who: str, written: str) -> list[Path]:
    found = [
        path
        for path in root.glob(written)
        if path.is_file()
        and path.suffix in QUESTION_FILES
        and not path.name.startswith("_")
        and pattern_reaches(written, path.relative_to(root))
    ]
    if not found:
        raise ConfigurationError(
            f"compiled: {who} lists the questions {written!r}, which reach no question file; write a path or a glob from the folder root, such as triage/*.json"
        )
    return found


def _beside(directory: Path, reached: Collection[Path]) -> list[Path]:
    stems = {path.stem for path in reached}
    return [
        path
        for path in question_files(directory)
        if path.suffix != ".md"
        and (
            path.stem in stems or (path.suffix == ".py" and not path.with_suffix(".json").is_file())
        )
    ]


def _questions(root: Path, who: str, written: list[str]) -> _Asked:
    folders: dict[Path, set[Path]] = {}
    for pattern in written:
        for path in _reached_by(root, who, pattern):
            folders.setdefault(path.parent, set()).add(path)
    questions: dict[str, Question] = {}
    scripted: dict[str, tuple[dict[str, JsonValue], Script]] = {}
    held: dict[str, Path] = {}
    reached: list[Path] = []
    for directory, paths in sorted(folders.items()):
        definitions, scripts = question_parts(
            directory, sorted({*paths, *_beside(directory, paths)})
        )
        for name, path in definitions.items():
            if name in held:
                raise ConfigurationError(
                    f"compiled: {who} asks {name!r} twice, in {held[name].relative_to(root)} and {path.relative_to(root)}; a question is named by its file, so rename one"
                )
            held[name] = path
        fixed, filled = read_questions(directory, definitions, scripts, root, ())
        questions |= fixed
        scripted |= filled
        reached.extend(definitions.values())
        for _, script in filled.values():
            reached.extend([script.path, *(helper.path for helper in script.helpers.values())])
    return _Asked(
        questions, scripted, {path.relative_to(root).as_posix(): path for path in reached}
    )


def _check_what_it_reaches(root: Path, who: str, files: Mapping[str, Path]) -> None:
    manifest = (root / MANIFEST).resolve()
    for name, path in files.items():
        utf8_name(name, path)
        target = path.resolve()
        if target == manifest:
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which is {MANIFEST}; a model never reads that file, so point at another"
            )
        if not target.is_relative_to(root):
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which links to {target}, outside the folder root; move the file into the folder"
            )


def _in_reach(root: Path, patterns: Collection[str], gitignore: bool) -> dict[str, Path]:
    manifest = (root / MANIFEST).resolve()
    gitignores = Gitignores()
    gitignored = Gitignored(root) if gitignore else None
    found: dict[str, Path] = {}

    def visit(folder: Path, rel: PurePosixPath, trail: frozenset[Path]) -> None:
        if gitignored is not None:
            gitignores.held_in(folder, gitignored)
        for name in listed(folder):
            path, below = folder / name, rel / name
            target = Path(os.path.realpath(path))
            if (
                name == "__pycache__"
                or target == manifest
                or target in trail
                or not target.is_relative_to(root)
                or (gitignored is not None and gitignored(path))
            ):
                continue
            if path.is_dir():
                if shown_link(root, path) and any(
                    pattern_leads_into(pattern, below) for pattern in patterns
                ):
                    visit(path, below, trail | {target})
            elif (
                any(pattern_covers(pattern, below) for pattern in patterns)
                and path.is_file()
                and readable(path)
            ):
                found[below.as_posix()] = path

    visit(root, PurePosixPath("."), frozenset({root}))
    return found


def _matching(
    who: str, key: str, patterns: Iterable[str], candidates: Mapping[str, Path]
) -> dict[str, Path]:
    matched: dict[str, Path] = {}
    for pattern in patterns:
        if PurePosixPath(pattern) == PurePosixPath(MANIFEST):
            raise ConfigurationError(
                f"compiled: {who} lists {pattern!r} in {key}, which is {MANIFEST}; a model never reads that file, so point at another"
            )
        found = {
            name: path
            for name, path in candidates.items()
            if pattern_covers(pattern, PurePosixPath(name))
        }
        if not found:
            raise ConfigurationError(
                f"compiled: {who} lists {pattern!r} in {key}, which reaches nothing; write a folder, a file or a glob from the folder root that reaches a file, or remove it"
            )
        matched |= found
    return matched


def _own_files(resolved: Resolved) -> set[str]:
    if resolved.instructions is None:
        return set()
    return {PurePosixPath(resolved.instructions).as_posix()}


def _reach(root: Path, who: str, files: FilesSection, own: Collection[str]) -> _Reach:
    include = files.include or []
    preload = files.preload or []
    candidates = _in_reach(root, [*include, *preload], files.gitignore)
    included = _matching(who, "files.include", include, candidates)
    excluded = _matching(who, "files.exclude", files.exclude or [], included)
    kept = {name: path for name, path in included.items() if name not in excluded}
    preloaded = _matching(who, "files.preload", preload, candidates)
    return _Reach(
        on_the_map={name: path for name, path in kept.items() if name not in own},
        files={**kept, **preloaded},
        preloaded=list(preloaded),
    )


def _with_folders(root: Path, files: Iterable[Path]) -> set[Path]:
    held = {root}
    for file in files:
        held.add(file)
        held.update(parent for parent in file.parents if parent.is_relative_to(root))
    return held


def _outside(root: Path, reached: Mapping[str, Path]) -> Callable[[Path], bool]:
    spelled = _with_folders(root, (root / name for name in reached))
    real = _with_folders(root, (path.resolve() for path in reached.values()))

    def withheld(path: Path) -> bool:
        return (
            Path(os.path.normpath(path)) not in spelled or Path(os.path.realpath(path)) not in real
        )

    return withheld


def _hidden_names(root: Path, names: Iterable[str]) -> frozenset[Path]:
    held: set[Path] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        held.update(
            root.joinpath(*parts[: depth + 1])
            for depth, part in enumerate(parts)
            if part.startswith(".")
        )
    return frozenset(held)


def _file_tools(resolved: Resolved) -> list[str]:
    if resolved.files.include is None and resolved.files.preload is None:
        return []
    return sorted(BUILTINS - set(resolved.tools.exclude or ()))


def _preloaded(
    root: Path, who: str, names: list[str], named: frozenset[Path], loadable: bool
) -> tuple[list[Part], dict[str, bytes]]:
    try:
        return preload_parts(names, Jail(root, named=named), loadable)
    except InputError as exc:
        raise ConfigurationError(
            f"compiled: {who} cannot preload: {exc}; fix that file, or drop it from files.preload"
        ) from exc


def _fingerprint(
    root: Path,
    agent: str,
    mode: str | None,
    resolved: Resolved,
    files: Mapping[str, Path],
    read: Mapping[str, bytes],
) -> str:
    entry = f"{MANIFEST}#{agent}" if mode is None else f"{MANIFEST}#{agent}:{mode}"
    canonical = json.dumps(
        resolved.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return folder_fingerprint(
        sorted([(entry, root / MANIFEST), *files.items()]), {**read, entry: canonical}
    )


def _spec(root: Path, agent: str, entry: AgentEntry, mode: str | None) -> AgentSpec:
    resolved = overlaid(entry, mode)
    who = _who(agent, mode)
    asked = _questions(root, who, resolved.questions or [])
    files = dict(asked.files)
    read: dict[str, bytes] = {}
    body = ""
    said = resolved.description
    if resolved.instructions is not None:
        read[resolved.instructions], text = _instructions(root, who, resolved.instructions)
        files[resolved.instructions] = root / resolved.instructions
        body = body_of(text)
        said = said or description(text, root / resolved.instructions)
    reach = (
        _Reach({}, {}, [])
        if resolved.instructions is None
        else _reach(root, who, resolved.files, _own_files(resolved))
    )
    files |= reach.files
    _check_what_it_reaches(root, who, files)
    keep = _file_tools(resolved)
    named = _hidden_names(root, reach.files)
    preload, loaded = _preloaded(root, who, reach.preloaded, named, "Load" in keep)
    read |= loaded
    return AgentSpec(
        root=root,
        name=agent,
        path=".",
        kind=resolved.kind,
        model=resolved.model,
        description=said or "",
        body=body,
        agents=resolved.orchestration.delegate_to or [],
        route=resolved.orchestration.route,
        questions=asked.questions,
        keep=keep,
        fingerprint=_fingerprint(root, agent, mode, resolved, files, read),
        workspace=bool(resolved.workspace),
        scripted=asked.scripted,
        preload=preload,
        preload_digest=preload_digest(preload),
        listing="\n".join(mapped(sorted(reach.on_the_map.items()))),
        needs_mode=resolved.instructions is None,
        worker=resolved.worker or "default",
        withheld=_outside(root, reach.files) if reach.files else None,
        named=named,
        included_folders=[
            pattern for pattern in resolved.files.include or [] if (root / pattern).is_dir()
        ],
        max_model_calls=resolved.max_model_calls,
        tool_timeout=resolved.tools.timeout,
        max_depth=resolved.orchestration.max_depth,
        tool_parallel=resolved.tools.max_parallel,
        delegate_parallel=resolved.orchestration.max_parallel,
        mode=mode,
    )


def _agent(root: Path, name: str, entry: AgentEntry) -> AgentSpec:
    spec = _spec(root, name, entry, None)
    modes = {mode: _spec(root, name, entry, mode) for mode in entry.modes}
    return spec.model_copy(update={"modes": modes})


def _refuse_targets(agents: Mapping[str, AgentSpec], who: str, spec: AgentSpec) -> None:
    routed = spec.route.to.values() if spec.route is not None else ()
    for name in (*spec.agents, *routed):
        target = agents.get(name)
        if target is None:
            raise ConfigurationError(
                f"compiled: {who} names {name!r} in delegate_to or route, and {MANIFEST} holds no such agent; its agents are {', '.join(sorted(agents))}"
            )
        if target.needs_mode:
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which needs a mode; a delegated or routed run takes none, so give {name!r} instructions on its entry or reach another agent"
            )


def _check_route(agents: Mapping[str, AgentSpec], who: str, spec: AgentSpec) -> None:
    route = spec.route
    if route is None or spec.needs_mode:
        return
    if route.by in spec.scripted:
        script = spec.scripted[route.by][1].path.relative_to(spec.root).as_posix()
        raise ConfigurationError(
            f"compiled: {who} routes by {route.by!r}, whose options come from {script}; route by a question whose options are fixed"
        )
    question = spec.questions.get(route.by)
    if not isinstance(question, Choice):
        raise ConfigurationError(
            f"compiled: {who} routes by {route.by!r}, which is not a choice question in its questions:; write by: as the name of a choice question it asks"
        )
    for option, name in route.to.items():
        if option not in question.options:
            raise ConfigurationError(
                f"compiled: {who} routes the option {option!r}, which {route.by!r} does not have; its options are {', '.join(question.options)}"
            )
        target = agents[name]
        if target.kind == "decision" and not target.questions and not target.scripted:
            raise ConfigurationError(
                f"compiled: {who} routes to decision agent {name!r}, which has no questions: to ask; give it questions: on its entry, since a routed run takes only the input and no mode"
            )


def _check_orchestration(agents: Mapping[str, AgentSpec]) -> None:
    for agent in agents.values():
        for spec in (agent, *agent.modes.values()):
            who = _who(agent.name, spec.mode)
            _refuse_targets(agents, who, spec)
            _check_route(agents, who, spec)


@warns_once()
def load_folder(root: Path) -> dict[str, AgentSpec]:
    root = Path(root).resolve()
    manifest = read_manifest(root)
    agents = {name: _agent(root, name, manifest.agents[name]) for name in sorted(manifest.agents)}
    _check_orchestration(agents)
    return agents
