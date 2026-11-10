import json
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from pydantic import JsonValue

from rfnry_engine.chain import preload_digest
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.fingerprint import folder_fingerprint
from rfnry_engine.frontmatter import body_of, description
from rfnry_engine.manifest import MANIFEST, AgentEntry, Resolved, overlaid, read_manifest
from rfnry_engine.models import Question
from rfnry_engine.tree import (
    QUESTION_FILES,
    AgentSpec,
    question_files,
    question_parts,
    read_questions,
    utf8_name,
    validate_tree,
)
from rfnry_engine.workers.pool import Script

_NOTHING_PRELOADED = preload_digest([])


class _Asked(NamedTuple):
    questions: dict[str, Question]
    scripted: dict[str, tuple[dict[str, JsonValue], Script]]
    files: dict[str, Path]


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
    named = {part for part in PurePosixPath(written).parts if part.startswith(".")}
    found = [
        path
        for path in root.glob(written)
        if path.is_file()
        and path.suffix in QUESTION_FILES
        and not path.name.startswith("_")
        and all(part in named or not part.startswith(".") for part in path.relative_to(root).parts)
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
        fixed, filled = read_questions(directory, definitions, scripts, root, (root,))
        questions |= fixed
        scripted |= filled
        reached.extend(definitions.values())
        for _, script in filled.values():
            reached.extend([script.path, *(helper.path for helper in script.helpers.values())])
    return _Asked(
        questions, scripted, {path.relative_to(root).as_posix(): path for path in reached}
    )


def _check_what_it_reaches(root: Path, who: str, files: Mapping[str, Path]) -> None:
    for name, path in files.items():
        utf8_name(name, path)
        target = path.resolve()
        if not target.is_relative_to(root):
            raise ConfigurationError(
                f"compiled: {who} reaches {name!r}, which links to {target}, outside the folder root; move the file into the folder"
            )


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
    _check_what_it_reaches(root, who, files)
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
        keep=[],
        fingerprint=_fingerprint(root, agent, mode, resolved, files, read),
        workspace=bool(resolved.workspace),
        scripted=asked.scripted,
        preload_digest=_NOTHING_PRELOADED,
        needs_mode=resolved.instructions is None,
        worker=resolved.worker or "default",
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


def _refuse_targets(agents: Mapping[str, AgentSpec]) -> None:
    for agent in agents.values():
        for spec in (agent, *agent.modes.values()):
            who = _who(agent.name, spec.mode)
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


def load_folder(root: Path) -> dict[str, AgentSpec]:
    root = Path(root).resolve()
    manifest = read_manifest(root)
    agents = {name: _agent(root, name, manifest.agents[name]) for name in sorted(manifest.agents)}
    _refuse_targets(agents)
    validate_tree(agents)
    return agents
