import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from rfnry_engine.compiled import load_folder
from rfnry_engine.errors import ConfigurationError
from rfnry_engine.manifest import (
    MANIFEST,
    FilesSection,
    Orchestration,
    SkillsSection,
    ToolsSection,
    overlaid,
    read_manifest,
)

from .trees import write_folder

GENERATION: dict[str, Any] = {"kind": "generation", "model": "m", "instructions": "a.md"}
DECISION: dict[str, Any] = {"kind": "decision", "model": "m", "instructions": "a.md"}
MODELS = "models:\n  m: {provider: p}\n"
MINIMAL = f"version: 1\n{MODELS}agents:\n  a: {{kind: generation, model: m, instructions: a.md}}\n"
ROUTE: dict[str, Any] = {"by": "category", "to": {"billing": "support"}}

DESIGN_EXAMPLE = """\
version: 1

workers:                              # optional; without it, one shared worker of one process
  customers:
    processes: 2                      # tool processes of this worker
    threads: 32                       # plain `def run` calls at once, per process
    max_result_bytes: 33554432        # the largest result a tool may send back
  live:
    processes: 1

models:                               # each model once; agents pick them by name
  strong:                             # the name agents use
    provider: anthropic               # required: a name the host gives in Engine(providers=)
    model: claude-sonnet-5-5          # optional: the default; a run can choose another
    effort: medium                    # optional: the default; a run can choose another
  decision: {provider: typesafe, model: jev-latest}

tools:                                # each tool once; agents pick them by name
  lookup_order:                       # the name the model sees; the file can be called anything
    module: tools/lookup_order.py     # required: the Python file, from the folder root
    function: run                     # optional; run is the default
    description: Finds an order       # else the description in the schema file
    schema: tools/lookup_order.json   # optional: description, input_schema, output_schema
    timeout: 20                       # optional: seconds per call
  refund: {module: tools/refund.py, schema: tools/refund.json}

runtime_tools: [handoff, verify_customer]   # names a run gives through tools=

agents:
  support:
    kind: generation                  # required: generation, decision or voice
    model: strong                     # required: a name under models:
    description: Handles support      # else the description: in the instructions file's frontmatter
    instructions: support/support.md  # any markdown file; frontmatter is stripped
    max_model_calls: 24               # per run; a delegated run counts its own
    worker: customers                 # which worker runs its tools

    files:
      allow: [knowledge]              # what the model may open: folders, files or globs
      deny: [knowledge/drafts/**]     # taken back out
      preload: [support/rules]        # sent with every model call; also readable
      gitignore: true                 # what a .gitignore ignores is not there

    tools:
      allow: ["*", handoff]           # names or patterns: declared above, built in, or given by the run
      deny: [refund]
      timeout: 30                     # seconds per call
      max_parallel: 4                 # tool calls of one run at once; absent means no cap

    skills:
      allow: [support/skills/*]       # listed for the model to open
      deny: [internal]
      preload: [support/skills/tone]  # full text sent with every model call
      public: [escalate]              # people may type /escalate; a list or the word all

    orchestration:
      delegate_to: [drafter]          # agents the Delegate tool may reach; one way
      max_depth: 1                    # levels of delegation below this agent
      max_parallel: 2                 # Delegate calls of one run at once; absent means no cap

    workspace: true                   # gives a shell and writable files; the host passes a Workspace

    modes:
      email:
        skills:
          preload: [support/channels/email]
      chat:
        worker: live
        tools:
          allow: [verify_customer]
        skills:
          preload: [support/channels/live, support/channels/chat]

  triage:
    kind: decision
    model: decision
    modes:
      email: {instructions: triage/email.md, questions: [triage/email/*.json]}
      chat: {instructions: triage/chat.md, questions: [triage/chat/*.json]}
    orchestration:
      route: {by: category, to: {billing: support}, min_confidence: 0.7}
"""


def _file(root: Path, text: str) -> Path:
    (root / MANIFEST).write_text(text, encoding="utf-8")
    return root


def _refused(root: Path, pattern: str) -> str:
    with pytest.raises(ConfigurationError) as raised:
        read_manifest(root)
    said = str(raised.value).replace(str(root), "<root>")
    assert said.startswith("manifest: ")
    assert re.search(pattern, said), said
    return said


def _nested(dotted: str, value: Any) -> dict[str, Any]:
    held = value
    for part in reversed(dotted.split(".")):
        held = {part: held}
    return dict(held)


def _agent(dotted: str, value: Any, on_mode: bool, base: dict[str, Any]) -> tuple[Any, str]:
    if on_mode:
        return (
            {**base, "modes": {"m": _nested(dotted, value)}},
            f"agents.a.modes.m.{dotted}",
        )
    return ({**base, **_nested(dotted, value)}, f"agents.a.{dotted}")


def test_a_minimal_file_loads(tmp_path: Path) -> None:
    manifest = read_manifest(write_folder(tmp_path, {"a": GENERATION}))
    assert manifest.version == 1
    assert list(manifest.agents) == ["a"]
    assert manifest.agents["a"].kind == "generation"
    assert manifest.workers == {}
    assert manifest.tools == {}
    assert manifest.agents["a"].modes == {}


def test_write_folder_writes_no_aliases_for_a_shared_entry(tmp_path: Path) -> None:
    assert list(
        read_manifest(write_folder(tmp_path, {"a": GENERATION, "b": GENERATION})).agents
    ) == ["a", "b"]


def test_the_example_of_the_design_parses_as_written(tmp_path: Path) -> None:
    manifest = read_manifest(_file(tmp_path, DESIGN_EXAMPLE))
    assert manifest.workers["customers"].processes == 2
    assert manifest.workers["customers"].max_result_bytes == 33554432
    assert manifest.workers["live"].threads == 32
    assert manifest.models["strong"].effort == "medium"
    assert manifest.models["decision"].model == "jev-latest"
    assert manifest.tools["lookup_order"].schema_file == "tools/lookup_order.json"
    assert manifest.tools["lookup_order"].timeout == 20
    assert manifest.tools["refund"].function == "run"
    support = manifest.agents["support"]
    assert support.files is not None and support.files.gitignore is True
    assert support.tools is not None and support.tools.allow == ["*", "handoff"]
    assert support.skills is not None and support.skills.public == ["escalate"]
    assert support.orchestration is not None and support.orchestration.delegate_to == ["drafter"]
    assert support.workspace is True
    assert overlaid(support, "chat").worker == "live"
    assert overlaid(support, "chat").tools.allow == ["*", "handoff", "verify_customer"]
    assert manifest.runtime_tools == ["handoff", "verify_customer"]
    triage = manifest.agents["triage"]
    assert triage.orchestration is not None and triage.orchestration.route is not None
    assert triage.orchestration.route.min_confidence == 0.7
    assert triage.modes["email"].questions == ["triage/email/*.json"]


def test_a_missing_file_names_the_root(tmp_path: Path) -> None:
    _refused(tmp_path, "manifest: <root> holds no rfnry.yaml; ")


def test_a_manifest_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).mkdir()
    _refused(tmp_path, "cannot read")


@pytest.mark.parametrize("text", ["", "# only a comment\n", "- a\n", "just words\n"])
def test_a_file_that_holds_no_keys_is_refused(tmp_path: Path, text: str) -> None:
    _refused(_file(tmp_path, text), "manifest: <root>/rfnry.yaml holds no keys")


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("version: 1\nagents:\n\ta: 1\n", 3),
        ("version: 1\nagents: [\n", 3),
        ('version: 1\nagents: "open\n', 2),
    ],
)
def test_broken_yaml_names_the_line_of_the_file(tmp_path: Path, text: str, line: int) -> None:
    _refused(_file(tmp_path, text), rf"manifest: invalid YAML in .* at line {line}\b")


def test_an_unknown_key_names_its_path_and_what_is_allowed(tmp_path: Path) -> None:
    agent = {**GENERATION, "tools": {"dney": ["x"]}}
    _refused(
        write_folder(tmp_path, {"a": agent}),
        r"agents\.a\.tools\.dney \(it takes allow, deny, timeout, max_parallel\)",
    )


@pytest.mark.parametrize("empty", [[], None, "", "  "])
def test_an_unknown_key_with_no_value_names_what_is_allowed(tmp_path: Path, empty: Any) -> None:
    agent = {**GENERATION, "tools": {"dney": empty}}
    _refused(
        write_folder(tmp_path, {"a": agent}),
        r"agents\.a\.tools\.dney \(it takes allow, deny, timeout, max_parallel\)",
    )

    tool = {"module": "tools/x.py", "timout": empty}
    _refused(
        write_folder(tmp_path, {"a": GENERATION}, tools={"x": tool}),
        r"tools\.x\.timout \(it takes module, function, description, schema, timeout\)",
    )


def test_an_unknown_key_at_the_top_names_the_top_keys(tmp_path: Path) -> None:
    _refused(
        _file(tmp_path, f"versoin: 1\n{MINIMAL}"),
        r"versoin \(it takes version, workers, models, tools, runtime_tools, agents\)",
    )


@pytest.mark.parametrize(
    ("agent", "path", "allowed"),
    [
        ({**GENERATION, "dney": 1}, r"agents\.a\.dney", "kind"),
        (
            {**GENERATION, "modes": {"m": {"tools": {"dney": ["x"]}}}},
            r"agents\.a\.modes\.m\.tools\.dney",
            "allow",
        ),
        (
            {**GENERATION, "modes": {"m": {"kind": "decision"}}},
            r"agents\.a\.modes\.m\.kind",
            "questions",
        ),
        ({**GENERATION, "files": {"alow": ["x"]}}, r"agents\.a\.files\.alow", "gitignore"),
        ({**GENERATION, "skills": {"publik": ["x"]}}, r"agents\.a\.skills\.publik", "public"),
        (
            {**DECISION, "orchestration": {"rout": ROUTE}},
            r"agents\.a\.orchestration\.rout",
            "delegate_to",
        ),
        (
            {**DECISION, "orchestration": {"route": {**ROUTE, "minimum": 1}}},
            r"agents\.a\.orchestration\.route\.minimum",
            "min_confidence",
        ),
    ],
)
def test_an_unknown_key_is_named_at_every_level(
    tmp_path: Path, agent: dict[str, Any], path: str, allowed: str
) -> None:
    _refused(write_folder(tmp_path, {"a": agent}), rf"{path} \(it takes [^)]*\b{allowed}\b")


def test_an_unknown_key_of_a_worker_names_the_keys_of_a_worker(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, workers={"default": {"proceses": 2}})
    _refused(folder, r"workers\.default\.proceses \(it takes [^)]*max_result_bytes")


def test_every_unknown_key_is_named_at_once(tmp_path: Path) -> None:
    agent = {**GENERATION, "dney": 1, "tools": {"dney": ["x"]}}
    said = _refused(write_folder(tmp_path, {"a": agent}), r"agents\.a\.dney")
    assert "agents.a.tools.dney" in said


@pytest.mark.parametrize("old", ["include", "exclude"])
@pytest.mark.parametrize("section", ["files", "tools", "skills"])
def test_the_old_key_of_a_section_is_an_unknown_key(tmp_path: Path, section: str, old: str) -> None:
    write_folder(
        tmp_path,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                section: {old: ["x"]},
            }
        },
        files={"a.md": "Work.\n"},
    )
    with pytest.raises(
        ConfigurationError, match=rf"unknown key in .*: agents\.a\.{section}\.{old} \(it takes "
    ):
        load_folder(tmp_path)


def test_deny_wins_over_allow_in_files(tmp_path: Path) -> None:
    write_folder(
        tmp_path,
        {
            "a": {
                "kind": "generation",
                "model": "strong",
                "instructions": "a.md",
                "files": {"allow": ["k"], "deny": ["k/secret.md"]},
            }
        },
        files={"a.md": "Work.\n", "k/open.md": "o\n", "k/secret.md": "s\n"},
    )
    assert set(load_folder(tmp_path)["a"].reachable) == {"k/open.md"}


GENERATION_LISTS = [
    "files.allow",
    "files.deny",
    "files.preload",
    "tools.allow",
    "tools.deny",
    "skills.allow",
    "skills.deny",
    "skills.preload",
    "skills.public",
    "orchestration.delegate_to",
]


@pytest.mark.parametrize("on_mode", [False, True])
@pytest.mark.parametrize("empty", [[], None])
@pytest.mark.parametrize(
    ("key", "base"),
    [*((key, GENERATION) for key in GENERATION_LISTS), ("questions", DECISION)],
)
def test_an_empty_list_says_nothing(
    tmp_path: Path, key: str, base: dict[str, Any], empty: Any, on_mode: bool
) -> None:
    agent, path = _agent(key, empty, on_mode, base)
    _refused(write_folder(tmp_path, {"a": agent}), re.escape(path) + ".* says nothing")


@pytest.mark.parametrize("empty", [None, {}])
@pytest.mark.parametrize("key", ["files", "tools", "skills", "orchestration", "modes"])
def test_a_section_with_no_value_says_nothing(tmp_path: Path, key: str, empty: Any) -> None:
    agent = {**GENERATION, key: empty}
    _refused(write_folder(tmp_path, {"a": agent}), rf"agents\.a\.{key}\b.* says nothing")


@pytest.mark.parametrize("empty", ["", " {}"])
@pytest.mark.parametrize("key", ["workers", "tools"])
def test_a_top_level_mapping_with_no_value_says_nothing(
    tmp_path: Path, key: str, empty: str
) -> None:
    _refused(
        _file(tmp_path, f"{key}:{empty}\n{MINIMAL}"), rf"manifest: {key} in <root>.* says nothing"
    )


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("model", None),
        ("description", ""),
        ("instructions", ""),
        ("files.allow", [""]),
        ("files.allow", [None]),
    ],
)
def test_a_value_that_says_nothing_is_refused(tmp_path: Path, dotted: str, value: Any) -> None:
    agent = {**GENERATION, **_nested(dotted, value)}
    _refused(
        write_folder(tmp_path, {"a": agent}), re.escape(f"agents.a.{dotted}") + r".* says nothing"
    )


def test_a_mode_with_no_value_says_nothing(tmp_path: Path) -> None:
    agent = {**GENERATION, "modes": {"m": None}}
    _refused(write_folder(tmp_path, {"a": agent}), r"agents\.a\.modes\.m\b.* says nothing")


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("version: 1\nagents:\n  a: &base {kind: generation, model: m, instructions: a.md}\n", 3),
        (
            "version: 1\nagents:\n  a: {kind: generation, model: m, instructions: a.md}\n  b: *base\n",
            4,
        ),
        (
            "version: 1\nagents:\n  a:\n    kind: generation\n    <<: {model: m}\n    instructions: a.md\n",
            5,
        ),
        ("version: 1\nagents:\n  &name a: {kind: generation, model: m, instructions: a.md}\n", 3),
        (
            "version: 1\nagents:\n  a: {kind: generation, model: m, instructions: a.md}\n  *name : {kind: generation, model: m, instructions: b.md}\n",
            4,
        ),
        (
            "version: 1\nagents:\n  a:\n    kind: generation\n    model: m\n    instructions: a.md\n    modes:\n      m:\n        <<: {model: n}\n",
            9,
        ),
        (
            "version: 1\ntools:\n  x:\n    <<: {module: x.py}\nagents:\n  a: {kind: generation, model: m, instructions: a.md}\n",
            4,
        ),
        (
            "version: 1\nworkers:\n  w:\n    <<: {processes: 1}\nagents:\n  a: {kind: generation, model: m, instructions: a.md, worker: w}\n",
            4,
        ),
    ],
)
def test_anchors_aliases_and_merge_keys_are_refused(tmp_path: Path, text: str, line: int) -> None:
    _refused(_file(tmp_path, text), rf"at line {line}\b")


def test_a_merge_key_says_to_write_the_keys_out(tmp_path: Path) -> None:
    text = "version: 1\nagents:\n  a:\n    <<: {kind: generation}\n"
    _refused(_file(tmp_path, text), "merge key.*write each key out")


def test_an_anchored_list_item_is_refused(tmp_path: Path) -> None:
    text = (
        "version: 1\nagents:\n  a:\n    kind: generation\n    model: m\n    instructions: a.md\n"
        "    tools:\n      allow: [&one x]\n"
    )
    _refused(_file(tmp_path, text), r"at line 8\b")


def test_a_bare_star_starts_an_alias_and_is_refused(tmp_path: Path) -> None:
    text = (
        "version: 1\nagents:\n  a:\n    kind: generation\n    model: m\n    instructions: a.md\n"
        "    tools: {allow: [*]}\n"
    )
    _refused(_file(tmp_path, text), r"at line 7\b.*quote a value")


def test_an_alias_says_to_quote_a_pattern(tmp_path: Path) -> None:
    text = _with("tools: {allow: [*_lookup]}")
    _refused(_file(tmp_path, text), r"at line 3 is refused; quote it if it is a pattern")


def test_an_anchor_says_nothing_of_patterns(tmp_path: Path) -> None:
    said = _refused(
        _file(tmp_path, _with("description: &x text")),
        r"at line 3 is refused; write the value out where it is used",
    )
    assert "pattern" not in said


def test_a_bare_ampersand_says_to_quote_it(tmp_path: Path) -> None:
    _refused(
        _file(tmp_path, _with("description: & more")),
        r"at line 3\b.*quote a value that starts with \* or &",
    )


def test_a_quoted_star_is_a_pattern(tmp_path: Path) -> None:
    text = (
        f"version: 1\n{MODELS}agents:\n  a:\n    kind: generation\n    model: m\n    instructions: a.md\n"
        '    tools: {allow: ["*"]}\n'
    )
    tools = read_manifest(_file(tmp_path, text)).agents["a"].tools
    assert tools is not None and tools.allow == ["*"]


def _with(keys: str) -> str:
    return f"version: 1\nagents:\n  a: {{kind: generation, model: m, instructions: a.md, {keys}}}\n{MODELS}"


@pytest.mark.parametrize(
    ("text", "said"),
    [
        (
            "version: 1\nagents:\n  a: {kind: generation, model: m, instructions: a.md}\n  a: {kind: generation, model: m, instructions: b.md}\n",
            r"duplicate key .* remove one of them",
        ),
        (f"version: 1\n{MINIMAL}", r"duplicate key .* remove one of them"),
        (_with("kind: decision"), r"duplicate key .* remove one of them"),
        (_with("workspace: yes"), r"agents\.a\.workspace"),
        (_with("workspace: no"), r"agents\.a\.workspace"),
        (_with("workspace: on"), r"agents\.a\.workspace"),
        (_with("files: {gitignore: off}"), r"agents\.a\.files\.gitignore"),
    ],
)
def test_duplicate_keys_and_yes_no_booleans_stay_strict(
    tmp_path: Path, text: str, said: str
) -> None:
    _refused(_file(tmp_path, text), said)


@pytest.mark.parametrize(
    "value",
    [
        "2026-13-45",
        "9" * 5000,
        "!!int nope",
        '!!int ""',
        "!!float nope",
        "!!bool nope",
        "!!timestamp nope",
        "!!set [1]",
    ],
)
def test_a_value_yaml_cannot_build_is_invalid_yaml(tmp_path: Path, value: str) -> None:
    _refused(
        _file(tmp_path, _with(f"description: {value}")),
        r"manifest: invalid YAML in <root>/rfnry\.yaml: a value cannot be read: .*; quote it",
    )


@pytest.mark.parametrize("innermost", ["x", ""])
def test_deep_nesting_is_refused(tmp_path: Path, innermost: str) -> None:
    deep = "[" * 3000 + innermost + "]" * 3000
    _refused(
        _file(tmp_path, _with(f"description: {deep}")),
        r"manifest: (invalid YAML in )?<root>.* nested too deeply",
    )


def test_deep_nesting_under_an_unknown_key_names_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_deep(*args: Any, **kwargs: Any) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("rfnry_engine.manifest._walk_for_nothing", too_deep)
    _refused(_file(tmp_path, _with("extra: [x]")), r"agents\.a\.extra \(it takes")


def test_a_document_too_deep_to_load_is_invalid_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_deep(*args: Any, **kwargs: Any) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(yaml, "load", too_deep)
    _refused(_file(tmp_path, MINIMAL), r"manifest: invalid YAML in <root>.* nested too deeply")


def test_true_and_false_are_the_booleans(tmp_path: Path) -> None:
    entry = read_manifest(
        _file(tmp_path, _with("workspace: true, files: {gitignore: false}"))
    ).agents["a"]
    assert entry.workspace is True
    assert entry.files is not None and entry.files.gitignore is False


@pytest.mark.parametrize("version", ["2", '"1"', "true", "1.0", "0"])
def test_the_version_must_be_1(tmp_path: Path, version: str) -> None:
    text = MINIMAL.replace("version: 1", f"version: {version}")
    _refused(_file(tmp_path, text), r"version: write version: 1, got")


def test_the_version_is_required(tmp_path: Path) -> None:
    _refused(_file(tmp_path, MINIMAL.replace("version: 1\n", "")), r"version: field required")


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("version: 1\n", r"agents: field required"),
        ("version: 1\nagents:\n", r"manifest: agents in <root>.* says nothing"),
        ("version: 1\nagents: {}\n", r"manifest: agents in <root>.* says nothing"),
    ],
)
def test_at_least_one_agent_is_required(tmp_path: Path, text: str, said: str) -> None:
    _refused(_file(tmp_path, text), said)


DECISION_REFUSES: list[tuple[str, Any, str]] = [
    ("files.allow", ["x"], "files"),
    ("files.gitignore", True, "files"),
    ("tools.allow", ["x"], "tools"),
    ("skills.allow", ["x"], "skills"),
    ("max_model_calls", 3, "max_model_calls"),
    ("workspace", True, "workspace"),
    ("orchestration.delegate_to", ["x"], "orchestration.delegate_to"),
    ("orchestration.max_depth", 1, "orchestration.max_depth"),
    ("orchestration.max_parallel", 1, "orchestration.max_parallel"),
]


@pytest.mark.parametrize("on_mode", [False, True])
@pytest.mark.parametrize(("key", "value", "refused"), DECISION_REFUSES)
def test_decision_agents_take_no_files_tools_or_skills(
    tmp_path: Path, key: str, value: Any, refused: str, on_mode: bool
) -> None:
    agent, _ = _agent(key, value, on_mode, DECISION)
    at = "agents.a.modes.m" if on_mode else "agents.a"
    said = _refused(write_folder(tmp_path, {"a": agent}), re.escape(f"{at}.{refused} in "))
    assert "a decision agent takes only" in said
    assert "orchestration.route" in said


@pytest.mark.parametrize("on_mode", [False, True])
@pytest.mark.parametrize(
    ("key", "value"),
    [("questions", ["q.json"]), ("orchestration.route", ROUTE)],
)
def test_generation_agents_take_no_questions_or_route(
    tmp_path: Path, key: str, value: Any, on_mode: bool
) -> None:
    agent, path = _agent(key, value, on_mode, GENERATION)
    said = _refused(write_folder(tmp_path, {"a": agent}), re.escape(path))
    assert "a generation agent takes everything but" in said


def test_a_decision_agent_takes_its_questions_and_a_route(tmp_path: Path) -> None:
    agent = {**DECISION, "questions": ["q/*.json"], "orchestration": {"route": ROUTE}}
    entry = read_manifest(write_folder(tmp_path, {"a": agent})).agents["a"]
    assert entry.questions == ["q/*.json"]
    assert entry.orchestration is not None
    assert entry.orchestration.route is not None
    assert entry.orchestration.route.by == "category"


def test_instructions_are_required_on_the_entry_or_every_mode(tmp_path: Path) -> None:
    bare = {"kind": "generation", "model": "m"}
    _refused(write_folder(tmp_path, {"a": bare}), r"agents\.a\.instructions")

    partly = {**bare, "modes": {"one": {"instructions": "o.md"}, "two": {"model": "n"}}}
    said = _refused(write_folder(tmp_path, {"a": partly}), r"agents\.a\.modes\.two\.instructions")
    assert "agents.a.modes.one" not in said

    whole = {**bare, "modes": {"one": {"instructions": "o.md"}, "two": {"instructions": "t.md"}}}
    entry = read_manifest(write_folder(tmp_path, {"a": whole})).agents["a"]
    assert entry.instructions is None
    assert overlaid(entry, "two").instructions == "t.md"


def test_instructions_on_the_entry_serve_every_mode(tmp_path: Path) -> None:
    agent = {**GENERATION, "modes": {"m": {"model": "n"}}}
    entry = read_manifest(write_folder(tmp_path, {"a": agent})).agents["a"]
    assert overlaid(entry, "m").instructions == "a.md"


def test_a_model_is_required_on_the_entry(tmp_path: Path) -> None:
    agent = {"kind": "generation", "instructions": "a.md", "modes": {"m": {"model": "n"}}}
    _refused(write_folder(tmp_path, {"a": agent}), r"agents\.a\.model: field required")


PATHS: list[tuple[str, dict[str, Any]]] = [
    ("instructions", GENERATION),
    ("files.allow", GENERATION),
    ("files.deny", GENERATION),
    ("files.preload", GENERATION),
    ("skills.allow", GENERATION),
    ("skills.deny", GENERATION),
    ("skills.preload", GENERATION),
    ("questions", DECISION),
]


@pytest.mark.parametrize("bad", ["/x", "a/../b", "../x", "a\\b"])
@pytest.mark.parametrize(("key", "base"), PATHS)
def test_a_path_is_written_from_the_root(
    tmp_path: Path, key: str, base: dict[str, Any], bad: str
) -> None:
    value = bad if key == "instructions" else [bad]
    agent, path = _agent(key, value, False, base)
    said = _refused(write_folder(tmp_path, {"a": agent}), re.escape(path))
    assert "from the folder root" in said
    assert "manifest: " not in said.removeprefix("manifest: ")


@pytest.mark.parametrize("bad", ["/x", "a/../b", "a\\b"])
def test_a_path_on_a_mode_is_written_from_the_root(tmp_path: Path, bad: str) -> None:
    agent = {**GENERATION, "modes": {"m": {"instructions": bad}}}
    _refused(write_folder(tmp_path, {"a": agent}), r"agents\.a\.modes\.m\.instructions")


def test_two_dots_inside_a_name_are_not_a_path_going_up(tmp_path: Path) -> None:
    agent = {**GENERATION, "instructions": "a..b/c..d.md", "files": {"allow": ["x/**"]}}
    entry = read_manifest(write_folder(tmp_path, {"a": agent})).agents["a"]
    assert entry.instructions == "a..b/c..d.md"


def _with_name(root: Path, where: str, name: str) -> tuple[Path, str]:
    if where == "agents":
        return write_folder(root, {name: GENERATION}), f"agents.{name}"
    if where == "workers":
        picking = {**GENERATION, "worker": name} if name.strip() else GENERATION
        return write_folder(
            root, {"a": picking}, workers={name: {"processes": 1}}
        ), f"workers.{name}"
    if where == "tools":
        return write_folder(
            root, {"a": GENERATION}, tools={name: {"module": "x.py"}}
        ), f"tools.{name}"
    if where == "models":
        plain = name == name.strip() and name and not name.startswith(".") and "/" not in name
        picking = {**GENERATION, "model": name} if plain else GENERATION
        models = {name: {"provider": "p"}, **({} if plain else {"m": {"provider": "p"}})}
        return write_folder(root, {"a": picking}, models=models), f"models.{name}"
    agent = {**GENERATION, "modes": {name: {"model": "n"}}}
    return write_folder(root, {"a": agent}), f"agents.a.modes.{name}"


@pytest.mark.parametrize("bad", ["", " ", " a", "a ", ".a", "a/b"])
@pytest.mark.parametrize("where", ["agents", "workers", "models", "tools", "modes"])
def test_a_name_is_one_plain_word(tmp_path: Path, where: str, bad: str) -> None:
    folder, path = _with_name(tmp_path, where, bad)
    said = _refused(folder, re.escape(f"manifest: {path} in <root>") + ".* is not a plain name")
    assert repr(bad) in said


@pytest.mark.parametrize("bad", [" a", "a ", ".a", "a/b"])
def test_a_runtime_tool_is_one_plain_word(tmp_path: Path, bad: str) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, runtime_tools=[bad])
    _refused(
        folder,
        rf"^manifest: runtime_tools in <root>/rfnry\.yaml lists {re.escape(repr(bad))}, which is not a plain name",
    )


@pytest.mark.parametrize("name", ["a", "a.b", "a-b", "a_b", "a b"])
@pytest.mark.parametrize("where", ["agents", "workers", "models", "tools", "modes"])
def test_a_plain_name_loads(tmp_path: Path, where: str, name: str) -> None:
    folder, _ = _with_name(tmp_path, where, name)
    assert read_manifest(folder).version == 1


@pytest.mark.parametrize("on_mode", [False, True])
@pytest.mark.parametrize("model", [" m", "m ", " m "])
def test_a_padded_model_is_refused(tmp_path: Path, model: str, on_mode: bool) -> None:
    agent, path = _agent("model", model, on_mode, GENERATION)
    _refused(write_folder(tmp_path, {"a": agent}), re.escape(path) + ": model must not be padded")


@pytest.mark.parametrize("on_mode", [False, True])
@pytest.mark.parametrize("model", ["", "  "])
def test_a_blank_model_says_nothing(tmp_path: Path, model: str, on_mode: bool) -> None:
    agent, path = _agent("model", model, on_mode, GENERATION)
    _refused(write_folder(tmp_path, {"a": agent}), re.escape(path) + r" in <root>.* says nothing")


@pytest.mark.parametrize(
    ("dotted", "value", "base"),
    [
        ("instructions", "  ", GENERATION),
        ("files.allow", ["  "], GENERATION),
        ("files.deny", ["  "], GENERATION),
        ("files.preload", ["  "], GENERATION),
        ("skills.allow", ["  "], GENERATION),
        ("skills.deny", ["  "], GENERATION),
        ("skills.preload", ["  "], GENERATION),
        ("skills.public", ["  "], GENERATION),
        ("tools.allow", ["  "], GENERATION),
        ("tools.deny", ["  "], GENERATION),
        ("orchestration.delegate_to", ["  "], GENERATION),
        ("worker", "  ", GENERATION),
        ("questions", ["  "], DECISION),
    ],
)
def test_a_whitespace_only_path_or_name_says_nothing(
    tmp_path: Path, dotted: str, value: Any, base: dict[str, Any]
) -> None:
    agent = {**base, **_nested(dotted, value)}
    _refused(
        write_folder(tmp_path, {"a": agent}), re.escape(f"agents.a.{dotted}") + r".* says nothing"
    )


@pytest.mark.parametrize("key", ["module", "schema", "function"])
def test_a_whitespace_only_field_of_a_tool_says_nothing(tmp_path: Path, key: str) -> None:
    tool = {"module": "tools/x.py", key: "  "}
    folder = write_folder(tmp_path, {"a": GENERATION}, tools={"x": tool})
    _refused(folder, rf"tools\.x\.{key} in <root>.* says nothing")


def test_a_tool_or_skill_name_is_not_a_path(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "tools": {"allow": ["a/../b"], "deny": ["/x"]},
        "skills": {"public": ["a\\b"]},
    }
    entry = read_manifest(write_folder(tmp_path, {"a": agent})).agents["a"]
    assert entry.tools is not None and entry.tools.deny == ["/x"]
    assert entry.skills is not None and entry.skills.public == ["a\\b"]


@pytest.mark.parametrize("on_mode", [False, True])
@pytest.mark.parametrize("picked", ["x", "default"])
def test_a_worker_nothing_declares_is_refused(tmp_path: Path, picked: str, on_mode: bool) -> None:
    agent, path = _agent("worker", picked, on_mode, GENERATION)
    said = _refused(write_folder(tmp_path, {"a": agent}), re.escape(path))
    assert repr(picked) in said


def test_a_worker_nothing_declares_is_refused_beside_declared_ones(tmp_path: Path) -> None:
    agents = {"a": {**GENERATION, "worker": "x"}, "b": {**GENERATION, "worker": "y"}}
    said = _refused(
        write_folder(tmp_path, agents, workers={"x": {"processes": 1}}), r"agents\.b\.worker"
    )
    assert "'y'" in said


def test_a_declared_worker_nothing_picks_is_refused(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, workers={"spare": {"processes": 1}})
    _refused(folder, r"workers\.spare")


def test_a_worker_picked_by_a_mode_is_used(tmp_path: Path) -> None:
    agent, _ = _agent("worker", "x", True, GENERATION)
    manifest = read_manifest(write_folder(tmp_path, {"a": agent}, workers={"x": {"threads": 4}}))
    assert overlaid(manifest.agents["a"], "m").worker == "x"
    assert overlaid(manifest.agents["a"], None).worker is None


def test_an_agent_with_none_runs_beside_declared_workers(tmp_path: Path) -> None:
    agents = {"a": {**GENERATION, "worker": "x"}, "b": GENERATION}
    manifest = read_manifest(write_folder(tmp_path, agents, workers={"x": {"processes": 2}}))
    assert manifest.agents["b"].worker is None


def test_a_declared_default_is_used_implicitly(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, workers={"default": {"processes": 3}})
    assert read_manifest(folder).workers["default"].processes == 3


def test_a_declared_default_a_mode_without_a_worker_uses_is_kept(tmp_path: Path) -> None:
    agent = {
        "kind": "generation",
        "model": "m",
        "modes": {
            "one": {"instructions": "o.md", "worker": "x"},
            "two": {"instructions": "t.md"},
        },
    }
    workers = {"default": {"processes": 2}, "x": {"processes": 1}}
    assert read_manifest(write_folder(tmp_path, {"a": agent}, workers=workers)).workers


def test_a_declared_default_everything_names_past_is_refused(tmp_path: Path) -> None:
    workers = {"default": {"processes": 2}, "x": {"processes": 1}}
    folder = write_folder(tmp_path, {"a": {**GENERATION, "worker": "x"}}, workers=workers)
    _refused(folder, r"workers\.default")


def test_a_declared_default_no_run_can_reach_is_refused(tmp_path: Path) -> None:
    agent = {
        "kind": "generation",
        "model": "m",
        "modes": {"one": {"instructions": "o.md", "worker": "x"}},
    }
    workers = {"default": {"processes": 2}, "x": {"processes": 1}}
    _refused(write_folder(tmp_path, {"a": agent}, workers=workers), r"workers\.default")


def test_a_tool_entry_takes_what_the_design_lists(tmp_path: Path) -> None:
    manifest = read_manifest(
        write_folder(tmp_path, {"a": GENERATION}, tools={"x": {"module": "tools/x.py"}})
    )
    entry = manifest.tools["x"]
    assert (entry.module, entry.function) == ("tools/x.py", "run")
    assert (entry.description, entry.schema_file, entry.timeout) == (None, None, None)

    full = {
        "module": "tools/x.py",
        "function": "find",
        "description": "Finds",
        "schema": "tools/x.json",
        "timeout": 20,
    }
    entry = read_manifest(write_folder(tmp_path, {"a": GENERATION}, tools={"x": full})).tools["x"]
    assert (entry.function, entry.description) == ("find", "Finds")
    assert (entry.schema_file, entry.timeout) == ("tools/x.json", 20.0)

    _refused(
        write_folder(tmp_path, {"a": GENERATION}, tools={"x": {"function": "f"}}),
        r"tools\.x\.module",
    )

    said = _refused(
        write_folder(tmp_path, {"a": GENERATION}, tools={"x": {"modul": "tools/x.py"}}),
        r"tools\.x\.modul\b",
    )
    assert "function, description, schema, timeout" in said
    assert "schema_file" not in said


def test_the_field_name_of_the_schema_is_not_a_key(tmp_path: Path) -> None:
    tool = {"module": "tools/x.py", "schema_file": "tools/x.json"}
    folder = write_folder(tmp_path, {"a": GENERATION}, tools={"x": tool})
    _refused(folder, r"tools\.x\.schema_file")


@pytest.mark.parametrize(
    ("value", "shown"), [(5, "5"), (1.5, "1.5"), (True, "True"), (["a"], "['a']")]
)
def test_a_tool_written_as_neither_a_path_nor_an_object_is_refused_naming_both(
    tmp_path: Path, value: Any, shown: str
) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, tools={"fetch": value})
    said = (
        f"tools: tool 'fetch' is neither a path nor an object, got {shown}; write the path of its "
        "module, such as tools/fetch.py, or an object with module:"
    )
    _refused(folder, rf"^manifest: invalid <root>/rfnry\.yaml; {re.escape(said)}$")


def test_a_tool_with_no_value_says_nothing(tmp_path: Path) -> None:
    folder = write_folder(tmp_path, {"a": GENERATION}, tools={"fetch": None})
    _refused(folder, r"^manifest: tools\.fetch in <root>/rfnry\.yaml says nothing")


@pytest.mark.parametrize("bad", ["/x.py", "a/../b.py", "a\\b.py"])
@pytest.mark.parametrize("key", ["module", "schema"])
def test_a_tools_paths_are_written_from_the_root(tmp_path: Path, key: str, bad: str) -> None:
    tool = {"module": "tools/x.py", key: bad}
    folder = write_folder(tmp_path, {"a": GENERATION}, tools={"x": tool})
    _refused(folder, rf"tools\.x\.{key}: .*from the folder root")


def test_a_bom_is_accepted_and_non_utf8_is_refused(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).write_bytes(b"\xef\xbb\xbf" + MINIMAL.encode())
    assert read_manifest(tmp_path).agents["a"].kind == "generation"

    (tmp_path / MANIFEST).write_bytes(b"version: 1\n# caf\xe9\n")
    _refused(tmp_path, "UTF-8")


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("max_model_calls", "5"),
        ("max_model_calls", True),
        ("tools.timeout", "3"),
        ("tools.max_parallel", 1.5),
        ("orchestration.max_depth", "1"),
        ("description", 3),
        ("instructions", ["a.md"]),
    ],
)
def test_a_value_of_the_wrong_type_is_refused_by_its_path(
    tmp_path: Path, dotted: str, value: Any
) -> None:
    agent = {**GENERATION, **_nested(dotted, value)}
    _refused(write_folder(tmp_path, {"a": agent}), re.escape(f"agents.a.{dotted}"))


@pytest.mark.parametrize("value", [True, "2"])
def test_a_worker_takes_numbers(tmp_path: Path, value: Any) -> None:
    agent = {**GENERATION, "worker": "x"}
    folder = write_folder(tmp_path, {"a": agent}, workers={"x": {"processes": value}})
    _refused(folder, r"workers\.x\.processes")


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("max_model_calls", 0),
        ("tools.timeout", 0),
        ("tools.timeout", -1),
        ("tools.max_parallel", 0),
        ("orchestration.max_depth", 0),
        ("orchestration.max_parallel", 0),
    ],
)
def test_a_limit_on_an_agent_is_positive(tmp_path: Path, dotted: str, value: Any) -> None:
    agent = {**GENERATION, **_nested(dotted, value)}
    _refused(write_folder(tmp_path, {"a": agent}), re.escape(f"agents.a.{dotted}"))


@pytest.mark.parametrize("key", ["processes", "threads", "max_result_bytes"])
def test_a_limit_on_a_worker_is_positive(tmp_path: Path, key: str) -> None:
    agent = {**GENERATION, "worker": "x"}
    folder = write_folder(tmp_path, {"a": agent}, workers={"x": {key: 0}})
    _refused(folder, rf"workers\.x\.{key}")


def test_a_timeout_on_a_tool_is_positive(tmp_path: Path) -> None:
    tool = {"module": "tools/x.py", "timeout": 0}
    _refused(write_folder(tmp_path, {"a": GENERATION}, tools={"x": tool}), r"tools\.x\.timeout")


@pytest.mark.parametrize("value", ["some", 5, [1], True])
def test_public_is_all_or_a_list_of_names(tmp_path: Path, value: Any) -> None:
    agent = {**GENERATION, "skills": {"public": value}}
    said = _refused(write_folder(tmp_path, {"a": agent}), r"agents\.a\.skills\.public: write all")
    assert "list[str]" not in said


def test_public_takes_the_word_all(tmp_path: Path) -> None:
    agent = {**GENERATION, "skills": {"public": "all"}}
    skills = read_manifest(write_folder(tmp_path, {"a": agent})).agents["a"].skills
    assert skills is not None and skills.public == "all"


def _entry(tmp_path: Path, agent: dict[str, Any]) -> Any:
    return read_manifest(write_folder(tmp_path, {"a": agent})).agents["a"]


def test_a_mode_adds_to_lists_and_replaces_single_values(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "tools": {"deny": ["a"]},
        "modes": {"m": {"model": "m2", "tools": {"deny": ["b"]}}},
    }
    resolved = overlaid(_entry(tmp_path, agent), "m")
    assert resolved.tools.deny == ["a", "b"]
    assert resolved.model == "m2"
    assert resolved.kind == "generation"


def test_a_list_is_added_once_in_entry_then_mode_order(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "files": {"allow": ["b", "a"]},
        "modes": {"m": {"files": {"allow": ["a", "c"]}}},
    }
    assert overlaid(_entry(tmp_path, agent), "m").files.allow == ["b", "a", "c"]


def test_every_list_field_adds(tmp_path: Path) -> None:
    decision = {
        **DECISION,
        "questions": ["q/one.json"],
        "modes": {"m": {"questions": ["q/two.json"]}},
    }
    assert overlaid(_entry(tmp_path, decision), "m").questions == ["q/one.json", "q/two.json"]

    generation = {
        **GENERATION,
        "orchestration": {"delegate_to": ["x"]},
        "skills": {"preload": ["p/1"], "public": ["one"]},
        "modes": {
            "m": {
                "orchestration": {"delegate_to": ["y"]},
                "skills": {"preload": ["p/2"], "public": ["two"]},
            }
        },
    }
    resolved = overlaid(_entry(tmp_path, generation), "m")
    assert resolved.orchestration.delegate_to == ["x", "y"]
    assert resolved.skills.preload == ["p/1", "p/2"]
    assert resolved.skills.public == ["one", "two"]


def test_no_mode_is_the_entry_with_every_section_present(tmp_path: Path) -> None:
    resolved = overlaid(_entry(tmp_path, GENERATION), None)
    assert resolved.kind == "generation"
    assert resolved.instructions == "a.md"
    assert resolved.files == FilesSection()
    assert resolved.tools == ToolsSection()
    assert resolved.skills == SkillsSection()
    assert resolved.orchestration == Orchestration()


def test_a_mode_with_no_section_keeps_the_entrys(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "files": {"allow": ["k"], "gitignore": True},
        "tools": {"timeout": 9},
        "modes": {"m": {"description": "Other"}},
    }
    resolved = overlaid(_entry(tmp_path, agent), "m")
    assert resolved.files.allow == ["k"]
    assert resolved.files.gitignore is True
    assert resolved.tools.timeout == 9
    assert resolved.description == "Other"


def test_a_section_only_on_the_mode_is_taken_whole(tmp_path: Path) -> None:
    agent = {**GENERATION, "modes": {"m": {"tools": {"allow": ["x"], "max_parallel": 2}}}}
    resolved = overlaid(_entry(tmp_path, agent), "m")
    assert resolved.tools.allow == ["x"]
    assert resolved.tools.max_parallel == 2


def test_a_mode_that_says_nothing_of_gitignore_leaves_it(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "files": {"gitignore": True},
        "modes": {"m": {"files": {"allow": ["x"]}}},
    }
    assert overlaid(_entry(tmp_path, agent), "m").files.gitignore is True


def test_a_mode_can_turn_off_what_the_entry_turned_on(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "workspace": True,
        "files": {"gitignore": True},
        "modes": {"m": {"workspace": False, "files": {"gitignore": False}}},
    }
    resolved = overlaid(_entry(tmp_path, agent), "m")
    assert resolved.workspace is False
    assert resolved.files.gitignore is False


def test_single_values_of_a_section_are_replaced(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "tools": {"timeout": 30, "max_parallel": 4},
        "modes": {"m": {"tools": {"timeout": 5}}},
    }
    resolved = overlaid(_entry(tmp_path, agent), "m")
    assert resolved.tools.timeout == 5
    assert resolved.tools.max_parallel == 4


def test_a_route_is_a_single_value_and_is_replaced_whole(tmp_path: Path) -> None:
    agent = {
        **DECISION,
        "orchestration": {"route": {**ROUTE, "min_confidence": 0.7}},
        "modes": {"m": {"orchestration": {"route": {"by": "other", "to": {"x": "y"}}}}},
    }
    route = overlaid(_entry(tmp_path, agent), "m").orchestration.route
    assert route is not None
    assert (route.by, route.to, route.min_confidence) == ("other", {"x": "y"}, None)


@pytest.mark.parametrize(
    ("entry_public", "mode_public", "resolved"),
    [
        (["x"], "all", "all"),
        ("all", ["x"], "all"),
        ("all", "all", "all"),
    ],
)
def test_public_all_is_never_narrowed(
    tmp_path: Path, entry_public: Any, mode_public: Any, resolved: Any
) -> None:
    agent = {
        **GENERATION,
        "skills": {"public": entry_public},
        "modes": {"m": {"skills": {"public": mode_public}}},
    }
    assert overlaid(_entry(tmp_path, agent), "m").skills.public == resolved


def test_a_text_that_reads_all_is_an_ordinary_single_value(tmp_path: Path) -> None:
    agent = {**GENERATION, "description": "all", "modes": {"m": {"description": "some"}}}
    assert overlaid(_entry(tmp_path, agent), "m").description == "some"


def test_overlaying_leaves_the_entry_as_it_was(tmp_path: Path) -> None:
    agent = {
        **GENERATION,
        "tools": {"deny": ["a"]},
        "modes": {"m": {"tools": {"deny": ["b"]}}},
    }
    entry = _entry(tmp_path, agent)
    overlaid(entry, "m")
    assert entry.tools is not None and entry.tools.deny == ["a"]
    assert entry.modes["m"].tools is not None and entry.modes["m"].tools.deny == ["b"]
