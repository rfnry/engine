# rfnry-engine API

The reference: every public name, its signature, what it returns and what it raises. How and why to use them is in [the guide](engine.md).

## Models

### `GenerationModel`

```python
@runtime_checkable
class GenerationModel(Protocol):
    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ReplyLike: ...
```

A model that writes text and calls tools: any object with this method, matched by shape. It is handed copies, made for each call, of the messages — to their parts, their tool calls' `args` and their `reasoning` — and of the tool specs, to their `input_schema`, so an adapter that changes them changes neither the record, the next call, nor another run. It returns a `Reply`, or anything with its fields; a result whose fields do not fit makes the run raise `ModelError`. An adapter may also carry `media_types`, the file types its model takes, such as `{"image/*", "application/pdf"}`, so `Load` refuses the rest, and `model`, the name of the model it calls, a `str` the run records as its model when no reply reports one, and `settings`, a JSON object of what it sends its vendor besides the conversation, which the run records beside the model when a reply reports none. Settings are stored and exported, so a credential or an endpoint never goes in them.

### `StreamingGenerationModel`

```python
@runtime_checkable
class StreamingGenerationModel(Protocol):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: OnDelta,
    ) -> ReplyLike: ...
```

A generation model that also streams. When a run has `on_event`, the engine calls `generate_stream` instead of `generate`; the adapter awaits `on_delta(text)` with each piece of the reply's text as it arrives and `on_delta(text, reasoning=True)` with each piece of the model's reasoning its vendor streams, and returns the whole reply, as `generate` does. An adapter that never passes `reasoning=` streams text alone.

### `OnDelta`

```python
class OnDelta(Protocol):
    async def __call__(self, text: str, *, reasoning: bool = False) -> None: ...
```

What `generate_stream` is handed. Text goes out as a `TextDelta`, reasoning as a `ReasoningDelta`, each with the turn. An adapter that annotates `on_delta` as `Callable[[str], Awaitable[None]]` still fits, since it never passes `reasoning`.

### `DecisionModel`

```python
@runtime_checkable
class DecisionModel(Protocol):
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

A model that answers typed questions about a state in one call: any object with this method, matched by shape. It returns an answer per question id, or a `Decided` when it can report what the call cost and which model answered. An adapter may also carry `model`, the name of the model it calls, a `str` the run records as its model when the call reports none, `settings`, a JSON object of what it sends its vendor besides the conversation, recorded when the call reports none (stored and exported, so a credential or an endpoint never goes in them), and `media_types`, the media its model reads in a state, such as `{"image/*"}`: a run's parts with that media reach it as the state, a list of parts; without it the state is text, and parts it does not declare are refused before it is called. It is handed copies of the state, to its parts, and of the questions, so an adapter that changes them changes neither the record, the input the run was given, nor the questions of the folder, which every tenant's runs ask. Answers that do not fit — a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels — make the run raise `ModelError`.

### `VoiceModel`

```python
@runtime_checkable
class VoiceModel(Protocol):
    async def open(self, *, tools: list[ToolSpec], instructions: str | None) -> Any: ...
```

A live model, one that hears and speaks itself: any object with this method, matched by shape, the shape of rfnry-senses' `LiveModel`. The engine does not import senses and never calls `open`: `Engine.talk` hands the model to your session in `VoiceStart.model`. An adapter may also carry `model`, a `str` the run records when the session reports none. A `voice` agent's `model:` names one, and the kind check is that it has `open`.

### `ReplyLike`

```python
class ReplyLike(Protocol):
    text: str
    tool_calls: Sequence[ToolCallLike]
    usage: UsageLike | None
    stop_reason: StopReason
    model: str | None
    settings: Mapping[str, JsonValue] | None
    reasoning: Any
```

The shape a generation model's result is read by: the read-only fields of `Reply`. `ToolCallLike` is a protocol of `id: str`, `name: str` and `args: dict[str, Any]`, `UsageLike` is one of `input`, `output`, `cache_creation` and `cache_read`, each an `int`, and `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `DecidedLike`

```python
class DecidedLike(Protocol):
    answers: Mapping[str, AnswerLike]
    usage: UsageLike | None
    model: str | None
    settings: Mapping[str, JsonValue] | None
```

The shape a decision model's result is read by when it is more than the answers: the read-only fields of `Decided`. `UsageLike` is a protocol of `input`, `output`, `cache_creation` and `cache_read`, each an `int`.

### `AnswerLike`

```python
AnswerLike = ChoiceAnswerLike | ScoreAnswerLike | CheckAnswerLike
```

The shape an answer is read by: the fields of `ChoiceAnswer`, `ScoreAnswer` or `CheckAnswer` without `confidence`, which the engine computes. `ChoiceAnswerLike` is a protocol of `type: Literal["choice"]`, `choice: str` and `probabilities: dict[str, float]`; `ScoreAnswerLike` is one of `type: Literal["score"]`, `score: float` and `probabilities: list[float]`; `CheckAnswerLike` is one of `type: Literal["check"]` and `probability: float`.

### `QuestionLike`

```python
QuestionLike = ChoiceLike | ScoreLike | CheckLike
```

The shape of the questions a decision model is handed: the fields of `Choice`, `Score` or `Check`. `ChoiceLike` is a protocol of `type: Literal["choice"]`, `instructions: str | dict[str, JsonValue] | list[JsonValue]` and `options: Mapping[str, str | dict[str, JsonValue] | list[JsonValue] | None]`; `ScoreLike` is one of `type: Literal["score"]`, `instructions: str | dict[str, JsonValue] | list[JsonValue]` and `levels: Sequence[str | dict[str, JsonValue] | list[JsonValue]]`; `CheckLike` is one of `type: Literal["check"]`, `instructions: str | dict[str, JsonValue] | list[JsonValue]`, `yes: str | dict[str, JsonValue] | list[JsonValue] | None` and `no: str | dict[str, JsonValue] | list[JsonValue] | None`. An adapter branches on `question.type`; one annotated with `QuestionLike` from rfnry-engine or rfnry-memory serves both libraries.

### `Message`

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Content = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False
    prefix: str | None = None
    usage: Usage | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

One turn of a conversation. An assistant turn carries its text in `content`, its calls in `tool_calls` and its `reasoning` as the model returned it; a tool result has `role="tool"`, the `tool_call_id` it answers, and `is_error` when the call failed.

`prefix`, `usage`, `model` and `settings` are the engine's record of an assistant message it made from a reply, never sent to a model and outside the vocabulary shared with rfnry-memory: `prefix` is the first 16 hex characters of a sha256 over everything sent before the message when it was written — the model, as its adapter names it, the system prompt, the tools, the preloaded files and the earlier messages as sent — and its `reasoning` is sent again only while what precedes it is the same; `usage` is what the call that wrote it reported, and `model` the model that wrote it — the one the reply named, else the adapter's `model` attribute — so a run whose adapter changed models mid-run says which wrote each reply, where the run's `model` names the last; `settings` are the JSON object the reply reported, else the adapter's `settings` attribute, and stay out of the `prefix` hash. `prefix` is set on an assistant message whose reply carried reasoning, and is `None` otherwise. A `usage`, `prefix`, `model` or `settings` that does not fit these fields — another library's usage, a prefix that is not 16 hex characters, a model that is not a string, settings that are not a JSON object — reads as `None`, so a message from elsewhere is taken without them; only a `prefix` that is dropped costs its `reasoning`, which is not sent again.

### `HistoryLike`

```python
class HistoryLike(Protocol):
    async def read(self, start: int, stop: int | None = None) -> Sequence[object]: ...
```

A conversation kept anywhere, read by position: the messages at positions `start` up to `stop`, every one from `start` when `stop` is `None`, each read by the fields of `Message`. A position is a message's index in the conversation from 0 and always holds the same message. `Engine.run` and `EngineTools` take one as `history`, as they take a list.

### `ToolSpec`

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
```

A tool as a generation model is offered it: its name, its description and the JSON schema of its input.

### `ToolCall`

```python
class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = {}
```

A call a generation model makes: its id, the tool's name and the arguments.

### `Reply`

```python
class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None
    stop_reason: StopReason = "end_turn"
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
    reasoning: Any = None
```

What a generation model returns. A reply whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty asks the engine to run those calls and call the model again; anything else is final, and a call it holds is answered `NotRun` and never run. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `Usage`

```python
class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0

    def __add__(self, other: Usage) -> Usage: ...
```

The tokens of one model call, or summed over a run; `a + b` adds them field by field. It is frozen, and refuses a negative count or an unknown field.

### `Text`

```python
class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str
```

A text part.

### `Image`

```python
class Image(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: bytes | None = None
    url: str | None = None
```

An image part. It carries exactly one of `data`, whose bytes travel as base64 in JSON, or `url`, which the library never fetches.

Raises `InputError` when given both `data` and `url`, or neither.

### `Audio`

```python
class Audio(BaseModel):
    type: Literal["audio"] = "audio"
    media_type: str
    data: bytes | None = None
    url: str | None = None
```

An audio part, carrying `data` or `url` as `Image` does.

Raises `InputError` when given both `data` and `url`, or neither.

### `Document`

```python
class Document(BaseModel):
    type: Literal["document"] = "document"
    media_type: str
    data: bytes | None = None
    url: str | None = None
    name: str | None = None
```

A document part, such as a PDF, carrying `data` or `url` as `Image` does, and optionally its file name.

Raises `InputError` when given both `data` and `url`, or neither.

### `Part`

```python
Part = Annotated[Text | Image | Audio | Document, Field(discriminator="type")]
```

Any of the four parts, told apart by `type`. A part from another family library is read by its fields.

### `Content`

```python
Content = str | list[Part]
```

What a message or a tool result carries: a string or a list of parts.

### `State`

```python
State = str | dict[str, Any] | list[Any]
```

What a run is given: a string, a JSON object or a list, a list of parts included. It is a generation run's input and what a decision run's questions are about.

### `Choice`

```python
class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]
```

A question answered with one of `options`, each key with what it means, or `None` when the key says enough. It takes at least two options. `instructions` and what an option means are a string, or a JSON object or list for a question with parts or data of its own; `JsonValue` is pydantic's, any value JSON can hold.

### `Score`

```python
class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    levels: list[str | dict[str, JsonValue] | list[JsonValue]]
```

A question answered with a level, `levels` listed in order. It takes at least two levels. `instructions` and each level are a string, or a JSON object or list.

### `Check`

```python
class Check(BaseModel):
    type: Literal["check"] = "check"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None
```

A yes-or-no question; `yes` and `no` may say what each answer means. `instructions`, `yes` and `no` are a string, or a JSON object or list.

### `Question`

```python
Question = Choice | Score | Check
```

Any of the three questions, told apart by `type`.

### `ChoiceAnswer`

```python
class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.0
```

The answer to a `Choice`: the option chosen and a probability per option. The engine sets `confidence` from `probabilities`; an adapter leaves it out.

### `ScoreAnswer`

```python
class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0
```

The answer to a `Score`: a position in `levels`, from 0 to the last index, and a probability per level. The engine sets `confidence` from `probabilities`.

### `CheckAnswer`

```python
class CheckAnswer(BaseModel):
    type: Literal["check"] = "check"
    probability: float
    confidence: float = 0.0
```

The answer to a `Check`: the probability of yes, between 0 and 1. The engine sets `confidence` from it.

### `Answer`

```python
Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer
```

Any of the three answers, told apart by `type`.

### `Decided`

```python
class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

What `decide` returns instead of the answers alone when the adapter can report what the call cost and which model answered. The run's record carries the usage and model the call reported, and its settings; the adapter's `model` and `settings` attributes stand in for what the call does not report.

### `confidence`

```python
def confidence(probabilities: list[float]) -> float: ...
```

Returns how peaked a distribution is, `(n·p_max − 1)/(n − 1)`, at least 0, and 1.0 for fewer than two probabilities. The engine sets every answer's `confidence` with it, so a threshold means the same thing for every adapter and question type.

### `ModelChoice`

```python
@dataclass(frozen=True)
class ModelChoice:
    provider: str | None = None
    model: str | None = None
    effort: str | None = None
```

What a run changes about a model the folder declares, as a value of `models=` on `Engine.run`, `Engine.tool`, `EngineTools`, `Refinery.replay` and `Refinery.label`. Each field set replaces the entry's, and the rest stays the entry's; the engine interprets neither `model` nor `effort`, and `provider` must be one `Engine(providers=)` holds. Raises `InputError` when none of the three is set (`models: a ModelChoice says nothing; give it a provider, a model or an effort`), or one is blank or not text (`models: a ModelChoice's model must be non-blank text or left out; got ''`).

### `Provider`

```python
Provider = Callable[[str | None, str | None], GenerationModel | DecisionModel | VoiceModel]
```

What `Engine(providers=)` maps a provider's name to: your function, called as `provider(model, effort)`, positionally, so `functools.cache` works on it. It holds whatever the adapter needs and returns the adapter. The engine calls it at the start of each run and of each run it delegates or routes to, and at each `Load` on an `EngineTools` outside a run, to read the adapter's `media_types`, and keeps nothing it built, so you cache and close what it builds. It is a plain function: one that returns a coroutine, an `async def`, is a `ConfigurationError` at that run's start, the coroutine closed unawaited (`engine: provider 'anthropic' returned a coroutine; a provider is a plain function that returns an adapter, so write it with def, not async def`).

### `ModelEntry`

```python
class ModelEntry(BaseModel):
    provider: str
    model: str | None = None
    effort: str | None = None
```

An entry of the `models:` section of an `rfnry.yaml`, frozen, in `Engine.models`: the `provider` the host gave a function for in `Engine(providers=)`, and the `model` and `effort` it builds with, each a default a run can replace.

## Engine

### `Engine`

```python
class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        providers: Mapping[str, Provider] | None = None,
        refinery: Recorder | None = None,
        max_model_calls: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
        max_worker_processes: int = 16,
        worker_result_bytes: int | None = 33_554_432,
        history_walk: int = 10_000,
        history_chars: int = 100_000,
    ) -> None: ...

    agents: dict[str, AgentInfo]
    models: Mapping[str, ModelEntry]
```

`root` holds an `rfnry.yaml`, the only thing the engine reads to find its agents: they are that file's entries, each mode compiled to a full agent at load. Its `models:` section declares each model, an agent's or a mode's `model:` naming one of its entries, and `providers` maps each entry's `provider` to the `Provider` that builds it: a provider no entry names is allowed since a run's `ModelChoice` may name it, and `models` on the engine is the file's entries, a read-only view, where `ModelEntry` is an entry's `provider`, `model` and `effort`, frozen. What a provider returns is checked for its agent's kind when that run starts, a provider returning `None` failing that check, and a provider's own exception reaches the caller unchanged, before that run's first model call. In a delegated run its text becomes the tool's result, which the delegating model and the record hold, and in a routed run the router's record holds it, so a provider must keep keys out of it. `models:` is outside every fingerprint. An entry's `files` decides what its model may open, every path written from the folder root: what `allow` reaches and what `preload` sends with every model call, each less `deny`, and anything else reads as missing to the file tools, a link that stays inside the folder read under its own name, even when what it leads to is outside `allow` or hidden, unless what it leads to is denied, ignored by a `.gitignore` under `files.gitignore: true`, or inside a skill the entry holds but neither offers nor preloads, which reads as missing; that reach is fixed when the engine loads, and a pattern reaching nothing is a `ConfigurationError` naming the agent, the key and the pattern. A tool is declared once under `tools:`, its key the name the model sees, as an object whose `function` is called as `function(args, call)`, `run` when left out, or as the path of its module alone, which is `run` with the `.json` of the same name beside it as its `schema`, a file it then requires, and an entry picks it by name through `tools.allow`, an entry holding `*` being a pattern over the names under `tools:` and any other a declared tool, then one its sections grant (`Read`, `Grep`, `Glob`, `LS` and `Load` with `files`, `Delegate` with `orchestration.delegate_to`, `Shell`, `Job`, `Jobs`, `Write` and `Edit` with `workspace: true`), then one the run gives through `tools=`, which also replaces a declared tool of the same name; `tools.deny` drops exactly the tool it names, a tool a section grants included; each call's deadline is the smallest of the tool's `timeout`, the entry's `tools.timeout` and `tool_timeout`. A module, function or schema that is not there, the JSON beside a tool written as a path included, a module that is not a `.py` file or is named like a standard-library module, a pattern matching nothing, a declared tool no entry or mode picks, one named like a tool the engine gives, a name in `tools.allow` whose section is missing, a name in `tools.allow` or `tools.deny` that differs only by case from a declared tool or one the engine gives (`compiled: agent 'billing' lists 'read' in tools.allow; did you mean 'Read'? A tool's name matches exactly, so write it as it is declared or as the engine gives it`), `Output` or `History`, in any case, in `tools.allow` or `tools.deny`, one or two of `Shell`, `Job` and `Jobs` in `tools.deny` without the rest, a package folder beside a tool, an import of a name below a `.py` file beside it, and an import that resolves to nothing and sits under neither `try/except ImportError` nor `if TYPE_CHECKING` are a `ConfigurationError`. An entry's `skills` names folders holding `SKILL.md`, each named by its folder: `allow` and `preload` take a path or a glob reaching such folders, `deny` a skill's name, its folder or a glob, and `public` names or `all`; what is offered is `allow` less `deny`, a denied skill is not sent either, a preloaded skill is sent as its text without frontmatter, `skill` reaches any skill the entry or a mode names, and `/name` only `public`. The folder of an offered or preloaded skill is open to the file tools on every run, and that of the skill a run invokes on that run; `files.deny` holds inside a skill's folder, for everything but its `SKILL.md`. The frontmatter of an instructions file, a `SKILL.md` or a markdown file on the map is read only for `description`, and YAML that does not parse is no error: a top-level `description:` line is still read, its matching quotes stripped, and a `description:` that parses as anything but text is no description. A path or a glob reaching no skill, a `deny` naming none of the agent's skills, a `preload` that `deny` empties, a `public` name not offered or holding whitespace, and two folders giving one name are a `ConfigurationError`. The files an agent is made of are read once, when the engine loads: one it cannot read — its instructions, a skill's `SKILL.md`, a question's `.json`, a tool's script, schema or a module beside it, a question's script or its helper — is a `ConfigurationError` naming it (`compiled: agent 'support' cannot read its instructions 'support/instructions.md': Permission denied; write the path of a readable file from the folder root`, `tree: cannot read agents/triage/q/urgent.json: Permission denied; make it readable to the engine`, or `tools/folder:` for a tool's or a question's script), while a folder or a content file it cannot read is left out, with one warning naming it each time the folder loads, and a file removed while its folder is listed is left out as though it was never there. The tools' scripts, and the scripts beside its decision agents' questions, run in worker processes the engine starts, each when its first call comes, and keeps until `close`. A `def run` past `tool_timeout` holds its worker's thread until it returns, while an `async def run` past it is cancelled and holds none, unless it blocks the worker's event loop: once a call sent after its cancel passes its deadline too, with that cancel still unanswered, each of those calls holds a thread. The worker goes by the `run` it loaded: one a decorator made a coroutine function runs on its event loop, any other on a thread, and an awaitable that one returns is awaited on the event loop. The workers follow the event loop that calls them, one loop at a time, so a call from another loop stops those the first loop started and starts its own. Workers run on POSIX systems only, Linux and macOS, as `Local` does. A `workers:` section of an `rfnry.yaml` declares the workers, each a named set of processes with `processes` (1 unless given), `threads` (32 unless given, the plain `def run` calls at once in each process) and `max_result_bytes`, and `worker:` on an entry or a mode picks one, a call going to its process with the fewest calls in flight. An agent with no `worker:` uses the worker called `default`, one process unless declared; a `worker:` naming no declared worker, and a declared worker nothing picks, are a `ConfigurationError`. A worker holds the tools and question scripts of the agents and modes that pick it, a tool picked through two workers loading in both, and a delegated or routed run uses its own agent's `worker:`. `max_worker_processes` is the ceiling on the processes of all the workers: a declared one always counts its `processes`, and `default`, unless declared, counts only when an entry or a mode that uses it holds a tool or a question's script. `Recorder` is a protocol of `async def record(self, run: Run) -> None`, which a `Refinery` meets; it is handed the finished record itself — the object `Engine.run` then returns, unless the run raised — which nothing else of the engine's holds. `tool_timeout` is each tool call's deadline in seconds, `None` for none. `worker_result_bytes` caps what a worker sends back for one call — its result or its error as JSON — 32 MiB by default, `None` for no limit, and a worker's `max_result_bytes` lowers it for that worker and never raises it; past it the call fails with `WorkerError` (`workers/process: lookup_order sent back 40000023 bytes, over the limit of 33554432 bytes`) and the result never reaches the server; the bytes are the JSON as the worker writes it, a non-ASCII character escaped to six, and a result under the limit still costs the server about three times its size while it is read. `history_walk` is how many messages one `History` search walks back, `history_chars` how many characters one `History` result holds. An agent named like a mode of another, `a.b` beside mode `b` of agent `a`, is a `ConfigurationError` when the engine loads, since both would export under one key (`compiled: agent 'a.b' and mode 'b' of agent 'a' would export under one key, 'a.b'; rename one`). `agents` maps each agent's name to its `AgentInfo`, in name order. Once loaded, it logs `engine: agents loaded [lead, triage, writer]` at INFO on the `rfnry_engine.engine` logger.

An entry's `max_model_calls`, `orchestration.max_depth` and `tools.timeout` can lower `Engine(max_model_calls=)`, `Engine(max_depth=)` and `Engine(tool_timeout=)` and never raise them: the smaller value holds, and `tool_timeout=None` leaves the entry's. `max_model_calls` counts one run's model calls, a delegated run counting its own. `orchestration.max_depth` counts levels of delegation below the agent, its descendants included, and `Engine(max_depth=)` counts the whole stack, the first agent included. `tools.max_parallel` caps a run's tool calls at once and `orchestration.max_parallel` its `Delegate` calls, which `tools.max_parallel` does not count; absent means no cap. A call takes its slot after `approve` and its deadline starts once it has one. A call a command makes through the tool bridge takes a slot of `tools.max_parallel`, and a `Shell` on a run with a bridge takes none.

A `kind: voice` entry holds a spoken conversation on a live model, and `Engine.talk` runs it. It takes only `kind`, `model`, `description`, `instructions`, `worker`, `tools` and `modes`; `files`, `skills`, `orchestration`, `workspace`, `questions` and `max_model_calls`, on the entry or a mode, are a `ConfigurationError` naming the dotted path (`manifest: agents.voice.files in rfnry.yaml: a voice agent takes only kind, model, description, instructions, worker, tools and modes; remove it`). Its `tools.allow` names declared tools and tools the run gives: a name a section grants is a `ConfigurationError` (`compiled: agent 'voice' lists 'Read' in tools.allow, which a voice agent has no section to grant; remove the name`), as `Output` and `History` are for any agent. An entry or a mode whose `delegate_to` or `route` names a voice agent is a `ConfigurationError` when the engine loads (`compiled: agent 'lead' reaches 'voice', a voice agent; only engine.talk runs one, so reach another agent`).

Raises `ConfigurationError` when `root` is not a directory or holds no `rfnry.yaml` (`manifest: agents holds no rfnry.yaml; write one there, or point the engine at the folder that holds it`), `providers` is not a mapping, a key of it is blank, padded or not text, or a value of it cannot be called, an entry under `models:` names a provider `providers` does not hold (`engine: model 'strong' names provider 'anthropic', which Engine(providers=) does not hold; it holds typesafe`), `models:` is missing or says nothing, an entry of it takes a key but `provider`, `model` and `effort` or has no `provider`, a name in it is not plain, an agent's or a mode's `model:` names no entry (`manifest: agents.a.model in rfnry.yaml names 'ghost', which models: does not declare; declare it there, or pick one of strong`), an entry nothing picks (`manifest: models.spare in rfnry.yaml is picked by nothing; pick it with model: on an agent or a mode, or remove it`), a file or folder an agent reaches has a name that is not UTF-8 (`tree: the name of '/srv/agents/support/caf\udce9.md' is not UTF-8; rename it`), a question's `<id>.json` is not valid JSON or a JSON object, holds a key but `type`, `instructions`, `options`, `levels`, `yes` and `no`, holds a key of another type (`tree: agents/triage/q/team.json is a choice question holding 'levels'; a choice takes options, a score levels and a check yes and no, so drop 'levels' or change its type`), has missing or empty instructions or does not fit its type — a choice with fewer than two options, a score with fewer than two levels, unless a script beside it fills them — a question's `<id>.py` sits without its `<id>.json` and no `.py` beside it imports it, whichever agent or mode asks the questions there (`tree: agents/matcher/q/order.py has no order.json beside it, and no script beside it imports it; a question is order.json, and order.py fills it from data`), a question's script or its helper — a `.py` module beside it that it imports, by the rule a tool's follows — links out of the folder root (`tools/folder: agents/matcher/q/order.py links to /srv/order.py, outside the folder root; move the file into the folder root, or link the whole folder`), has no `def run` or `async def run` at its top level, is named like a standard-library module, is not valid Python, imports `rfnry_engine`, a module relatively, a helper inside a function or class, a package folder beside it, a name below a module beside it, a module that resolves to nothing outside `try/except ImportError` and `if TYPE_CHECKING`, or a module whose file lies in the project that holds the folder — the nearest folder at or above `root` with a `pyproject.toml` — outside the standard library, the installed packages — those of the server's site folders and of the site folders the workers' interpreter adds with `site.addsitedir` as it starts, a site folder a `.pth` file adds included, but not a folder a `.pth` path line puts on the path; loading runs that interpreter once, for at most ten seconds, only when a script imports a module of the project the server's own site folders do not hold — and the modules beside it, a tool's schema file is not an object, holds a key but `description`, `input_schema` and `output_schema`, has no description in text when the entry gives none, an `input_schema` whose top level is not `"type": "object"`, or a schema that does not compile, a `.gitignore` of an agent with `files.gitignore: true` cannot be read or decoded (`tree: cannot read agents/support/.gitignore: Permission denied; make it readable to the engine`), a file a `files.preload` entry sends is one `Read` would refuse or cannot be read (`compiled: agent 'support' cannot preload: jail: path outside the folder: notes/style.md; fix that file, or drop it from files.preload`), an entry or mode names an agent that needs a mode in `delegate_to` or `route` (`compiled: agent 'lead' reaches 'a', which needs a mode; a delegated or routed run takes none, so give 'a' instructions on its entry or reach another agent`), the workers run more processes in all than `max_worker_processes` (`engine: the workers of this folder run 17 processes in all, over max_worker_processes=16; lower processes: under workers:, or raise Engine(max_worker_processes=)`), or `max_model_calls`, `max_depth` or `max_worker_processes` is not a positive integer, `tool_timeout`, `history_walk` or `history_chars` is not positive, or `worker_result_bytes` is not a positive integer or `None` (`engine: worker_result_bytes must be a positive number of bytes, or None for no limit`).

### `Engine.run`

```python
async def run(
    self,
    agent: str,
    scope: str,
    input: State,
    *,
    mode: str | None = None,
    skill: str | None = None,
    history: Sequence[object] | HistoryLike | None = None,
    start: int = 0,
    context: str | None = None,
    expect: type[BaseModel] | None = None,
    on_event: EventHandler | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    record: bool = True,
    stop: Stop | None = None,
    inbox: asyncio.Queue[Any] | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
    models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
) -> Generation | Decision: ...
```

Runs `agent` for the tenant `scope` on `input` and returns its record: a `Generation` for a generation agent, a `Decision` for a decision agent. `mode` selects one of the modes an `rfnry.yaml` entry declares: the run is that mode's overlay on the entry, its instructions, tools, skills and questions as compiled at load, `deny` winning over a mode's `allow`, and the record's `mode` names it. A delegated or routed run takes no mode and runs on its agent's entry, whatever mode its caller selected. `models` changes what a model's name means for this run and every run it delegates or routes to: a value is a `ModelChoice`, merged field by field over the file's entry, or a built adapter that takes the entry's place, each provider called at its run's start. A key left out stays as it was, the engine's own mapping does not change, two runs at once each keep their own, and `run.model` records what the reply reports, else the `model` attribute of the adapter the run used. A generation agent's `skill` reaches any skill its entry or one of its modes names, offered to the model, preloaded or denied alike, and puts its `SKILL.md` in front of `input`, in the same user message, as a `Text` of `<<<SKILL <path>>>>` on a line of its own followed by the file's body without its frontmatter; the input follows as its own parts, and an empty input adds none. Delegated and routed runs never take the caller's skill. An `input` that starts with `/<name>`, `name` being one of the skills its entry's `skills.public` names — a string, or a list whose first part is a `Text`, `/` its very first character — invokes that skill the same way, and the run's `input` is the text after that first word, its leading whitespace stripped (`"/escalate A third complaint."` records `skill` `escalate` and `input` `"A third complaint."`; in a list that text replaces the first part, or the part is dropped when nothing follows, and a list left with no part is `""`). Any other input stays as it came, and is never an error, a path, an unknown name and a skill the agent has but does not make public included, so absent `skills.public` no input is ever a command. `engine.tool`, `Delegate`, routes and a replay never read `/name`. A decision agent asks the questions its entry's or its mode's `questions:` reaches, every `<id>.json`; one with an `<id>.py` beside it is first filled by that script's `run(input, call)`, in the agent's worker and under `tool_timeout`, the scripts of a run at once — `input` the state as JSON, `call` the run's `scope`, `run_id`, `agent` and `tags` and the `question` id — whose returned object fills what the JSON leaves out, never its `type` nor another key it sets, and is checked as a question; questions composed at run time, by a model or by a runtime of yours, reach it through `Delegate` or `engine.tool`, read by their fields and asked with the folder's in the same call, and a recorded run's are re-asked by `replay`. `context` is data for this run, such as the contract under review or today's date, never instructions. `expect` is the model the output must fit, its descriptions included, which reach the model in the `Output` tool's schema; of a reply's `Output` calls the first that fits is taken, each before it is answered with what is wrong, each after it is answered `NotRun` as any call after it is, and the reply's other calls before it run. `history` is a list of messages or a `HistoryLike`, and the run sends its messages from position `start` on — `start=0` sends them all; a routed run is given both, a delegated run neither. A route's run never gets the router's runtime questions. With a refinery attached, a run is recorded unless `record=False` — one that raises after its first model call, or from its `RunStarted` handler, too, as far as it got and with its `error`, before the exception reaches you — and a recording that fails is logged at WARNING on the `rfnry_engine.engine` logger, `engine: refinery.record failed for run <run_id>`, never raised, as it is when a model's tool call nests its `args` past 254 levels of lists and objects, `args` itself the first, or fewer in a delegated or routed run (`ValueError: Circular reference detected (depth exceeded)`). `EventHandler` is `Callable[[Event], Awaitable[None]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets, and `tools` rides this run only, a tool of it replacing a declared tool of the same name and offered only when the agent's `tools.allow` names it; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`, called with the agent's name and a copy of the call, for a call to a tool the run offers: a call to any other name is answered `InputError: tools: unknown tool 'ghost'` without asking, and leaves an `is_error` step. A tool, `approve`, each `on_event` handler and the `expect` model's validators are handed their own copy of a call's `args` — every list and object in them copied, any other value handed as it is — so one that changes what it was handed changes neither the record nor what the model is sent next.

Cancelling the task that runs it cancels the model call in flight and every tool call still running, a `Delegate` call and the run it started included, and re-raises the cancellation once all of them have ended, as `asyncio.TaskGroup` does; a tool that ignores cancellation holds it. Each further cancel meanwhile, two that land at the same moment counted as two, does not end the wait: it cancels once more everything still running, so a workspace command still being stopped is killed at once, with no grace period.

Raises `InputError` when the engine was closed (`engine: this engine was closed; build a new Engine`), `scope` is empty, `workspace` belongs to another scope, a decision run has no question to ask, a decision agent is given media its model does not declare in `media_types`, `tags` are outside their limits, `history` does not fit the contract or breaks a tool call from its result (a result whose call is not in it, or a call with no result), `start` is negative, past the end of a list `history`, or given without a `history`, `input` is not a string, a JSON object or a list, or holds anything but strings, finite numbers, booleans, `None`, lists, objects with string keys and, as items of the list itself, parts — a set, a tuple, a non-string key or a cycle among them (`engine: input must be a string, a JSON object or a list; got set at input['a']`), or `input` mixes parts with other values. Raises `InputError` when a decision agent is given a `skill` (`engine: decision agent 'grader' takes no skill; skills are for generation agents`), or a run is given both `skill` and an `input` that starts with `/<a public skill>` (`engine: a run invokes one skill, by skill= or by an input that starts with /<name>; got skill='escalate' and /close`). Raises `InputError`, before any model is called, when `mode` is left out on an agent whose entry has no `instructions` (`engine: agent 'a' needs a mode; its modes are chat, email`), or the agent is a voice agent (`engine: agent 'voice' is a voice agent; engine.talk runs it`). Raises `NotFoundError` when the folder has no agent `agent`, the agent has no mode `mode` (`engine: agent 'a' has no mode 'ghost'; its modes are chat, email`, or `it has no modes`), or the agent holds no skill `skill`, before any model is called; the message names the skills it holds. Raises `ConfigurationError`, before any model is called, when a `ModelChoice` names a provider `Engine(providers=)` does not hold (`engine: model 'strong' names provider 'backup', which Engine(providers=) does not hold; it holds anthropic`), when the model the agent runs on has not its kind's method (`agent 'writer' is a generation agent, and model 'fast' has no generate()`; a delegated or routed run is checked the same way when it starts), `models` is not a mapping, a key of it is blank, padded, not text or a key no agent or mode of the folder runs on (`no agent runs on model 'fsat'; the agents of rfnry.yaml run on judge, strong`), or a value of it has none of `generate`, `decide` and `open` and is no `ModelChoice`, an agent with `workspace: true` runs without a `workspace`, `tools` repeats a name of its own or of a tool the engine gives the agent, or holds a tool named `History`, the agent's `tools.allow` names a tool neither declared, granted by its sections nor given in `tools` (`tools/engine_tools: agent 'support' lists 'refnd' in tools.allow, which this run does not offer; its tools are Glob, Grep, LS, Load, Read, refund`), or `approve` returns anything but `True` or `False`. Raises `ModelError` when an adapter's result does not fit the contract, or a question's script returns anything but an object, returns a key its `<id>.json` sets — `type` always among them, so a script never changes its question's kind (`engine: order.py of decision agent 'matcher' returned 'type', which question 'order' sets in order.json; a script returns only what its question's JSON leaves out, such as {"options": {...}}`) — or a key its type does not take (`engine: order.py of decision agent 'matcher' returned 'levels', which question 'order', a choice, does not take; a choice takes options, a score levels and a check yes and no`), or fills a question that does not fit (`engine: question 'order' of decision agent 'matcher' does not fit once order.py fills it: the question: models: a choice needs at least two options`). Raises `WorkerError` when a question's script raises or its worker stops, naming the script and its question before what it raised (`engine: order.py of decision agent 'matcher' could not fill question 'order': KeyError: 'customer'`), and `TimeoutError` when it passes `tool_timeout` (`workers/pool: order timed out after 30s`); a question's script that fails ends its run before the decision model is called, recorded with its `error`. Raises `RunawayError` when the loop passes `max_model_calls`. Raises `DelegationLoopError` when a route reaches an agent already in the same chain of routes, and `DelegationDepthError` when a route goes deeper than `max_depth`.

### `Engine.tool`

```python
def tool(
    self,
    agent: str,
    *,
    scope: str,
    mode: str | None = None,
    name: str | None = None,
    context: str | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
    on_event: EventHandler | None = None,
    stop: Callable[[], Stop | None] | None = None,
    models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
) -> Tools: ...
```

Returns a `Tools` of one tool that runs `agent` as a top-level run under `scope`, recorded like any run: `mode` selects one of the agent's modes, as in `Engine.run`, and the record's `mode` names it; without it the run is on the entry. The tool is named `name` or the agent's name, described by the agent's description, or its mode's, from its entry or its instructions file's frontmatter, taking `{"input": str}` — with optional `questions` when it is a decision agent, as `Delegate` takes them, each a question's JSON or a `Choice`, `Score` or `Check` — and returning a generation agent's reply or a decision agent's answers as a dict. `context`, `tools`, `workspace`, `tags`, `approve` and `models` ride every call: `models`, as in `Engine.run`, for that call's run and what it delegates or routes to, and a run that calls the tool lends it none of its own. A call made inside a run started with `record=False`, such as a replay's, is not recorded. Called inside a run, each call's run is in that run's foreground: it takes the run's `stop` and `on_event`, as a delegated run does, so the run's `stop` reaches it instead of cancelling the call. `on_event`, given, rides every call instead; `stop`, given, is called at each call and returns the `Stop` of that call's run, or `None`, so a tool that lives as long as a session stops only the turn a person stops. Inside a run, a call whose run a stop ended, or the run it routed to, is answered `stopped by the user`, as a stopped `Delegate` call is; outside one, `dispatch` returns what the stopped run produced. Given inside a run, `on_event` and `stop` win: a `stop` of your own, or `None` from the function, leaves the call out of the run's foreground, and the run's stop cancels it as it cancels any tool. `EventHandler` is `Callable[[Event], Awaitable[None]]`. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, `tags` are outside their limits, the agent is a voice agent (`engine: agent 'voice' is a voice agent; engine.talk runs it`), or the agent needs a mode and `mode` is left out (`engine: agent 'a' needs a mode; its modes are chat, email`). Raises `NotFoundError` when the folder has no agent `agent` or the agent has no mode `mode` (`engine: agent 'a' has no mode 'ghost'; its modes are chat, email`), before any model is called, and `ConfigurationError` when the agent has no description (`engine: agent 'billing' has no description; write description: on its entry in rfnry.yaml, or in its instructions file's frontmatter, to use it as a tool`), or `models` does not fit as in `Engine.run`. A call raises what `Engine.run` raises, and `InputError` when `questions` does not fit the contract, holds an empty id or an id the run already asks, or is given to a generation agent.

### `Engine.talk`

```python
async def talk(
    self,
    agent: str,
    scope: str,
    session: Callable[[VoiceStart], Awaitable[Spoken]],
    *,
    mode: str | None = None,
    context: str | None = None,
    tools: ToolsLike | None = None,
    models: Mapping[str, VoiceModel | ModelChoice] | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
    on_event: EventHandler | None = None,
    record: bool = True,
) -> Voice: ...
```

Runs a `voice` agent for the tenant `scope`, for as long as your `session` holds the conversation, and returns its record, a `Voice`. The engine declares the agent, resolves its mode, gates its tools and records the run; `session` moves the audio, with rfnry-senses or anything else, and the engine imports neither. `session` is called once with a `VoiceStart`, the run's `run_id` minted before it opens, and returns a `Spoken`: its `transcript`, its `model`, recorded over the adapter's `model` attribute, and its `usage`, `None` when the host has none, which a consumer must not read as zero cost. `at` is when the run started and `duration_ms` covers the whole session.

`start.instructions` is the mode's instructions file, frontmatter stripped; `context`, when there is one, follows it as a `<<<CONTEXT>>>` section. A host that fills in per-session values renders them from that text. `start.tools` is one `Tools`, the `ToolsLike` a live session takes, holding the tools `tools.allow` picks as for any agent, `tools` riding this run as it does in `Engine.run`; a bare name the run does not give fails before `session` is called. Each call passes `approve`, takes its slot of `tools.max_parallel` and its deadline, runs on the path a generation run's call runs on, in its agent's worker for a declared tool, and becomes a `Step`, and `on_event` receives its `ToolCallStarted` and `ToolResultReady`. An allowed call's result reaches the session exactly as the tool returned it, and the step holds it as a run's step does. A call that is refused, fails or passes its deadline raises `InputError` to the session with the step's text (`engine: NotApproved: hang_up was not approved`), and the step has `is_error`, and `approved` when `approve` was asked. A name the agent was not offered is refused before `approve` is asked: the session gets `InputError` (`engine: InputError: tools: unknown tool 'ghost'`), and it leaves an `is_error` step whose `approved` is `None`. Cancelling the task that runs a `dispatch`, because the person changed what they asked, cancels the tool's body and raises `CancelledError` in that task; the session goes on, and the step is kept with `stopped` set, `stopped by the user` as its result and what `approve` answered in `approved`. A declared tool's `async def run` is cancelled in its worker, while a plain `def run` goes on holding its worker's thread until it returns, since a thread cannot be cancelled, and its step reads `stopped` all the same. A call cancelled after its tool answered, during its `ToolResultReady` handler, keeps the step its tool earned. An exception that comes out of the call's path, from an `approve` that raises or returns anything but a `bool` or from an `on_event` handler, reaches the session unchanged. A call that has no step by then leaves an `is_error` step with `"<Type>: <message>"` as its result, and a step that exists stands, so after a `ToolResultReady` handler that raises, the step says what the tool did while the session hears the handler's exception. A call still running when the session ends is cancelled the same way before the record is made, and a `dispatch` made once the session has ended raises `InputError` (`engine: the session of agent 'voice' has ended; a tool call cannot start after it`), runs nothing and leaves no step.

`models` changes what a model's name means for this session, as in `Engine.run`, and its typical value is a live model built for one session; a `ModelChoice` works as it does there too, and a key no agent or mode of the folder runs on is refused. Whatever the name resolves to must have `open`, checked before the session opens. With a refinery attached, the run is recorded unless `record=False`: a session that raises is recorded with its `error`, the steps it took and an empty transcript, and the exception reaches you; a recording that fails is logged as in `Engine.run`, never raised. Cancelling the task that runs `talk` cancels the session and the calls still running, and records nothing, as for any run. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets; `EventHandler` is `Callable[[Event], Awaitable[None]]`; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`.

Raises `InputError` when the engine was closed, `scope` is empty, `tags` are outside their limits, the agent is another kind (`engine: agent 'writer' is a generation agent; engine.run runs it, and engine.talk only a voice agent`) or needs a mode, or `session` returns anything but a `Spoken` (`engine: the session of agent 'voice' returned a value of type dict, not a Spoken; return Spoken(transcript=...)`), which is recorded as a failed session. Raises `NotFoundError` when the folder has no agent `agent` or the agent has no mode `mode`. Raises `ConfigurationError`, before the session opens, when `models` does not fit as in `Engine.run`, the model has no `open` (`engine: agent 'voice' is a voice agent, which needs a live model, and model 'live' has no open()`), or `tools.allow` keeps a name this run does not offer (`tools/engine_tools: agent 'voice' lists 'consult' in tools.allow, which this run does not offer; its tools are none`). Any other exception `session` raises reaches you unchanged.

### `Engine.close`

```python
async def close(self) -> None: ...
```

Stops every worker process the engine started, and whatever they started, and returns once each worker has ended, its pipes closed and its process reaped — but for the pipes of one still starting on a loop that stopped, below — whether it is called from the event loop its workers serve or from another, in this thread or another; a tool call still in flight in one fails with `WorkerError` (`workers/pool: the worker for lookup_order stopped: it was closed`). An error stopping a worker raised reaches the caller only once every worker has been stopped: the first, with the others logged at ERROR on the `rfnry_engine.workers.pool` logger. A worker a call is still starting counts: `close` waits for it to come up and stops it, and raises an error stopping another worker only once that one has ended too; when the loop it starts on stops before it comes up, `close` kills and reaps its process there and then, raises an error stopping another worker as it would have, and leaves its three pipes to that loop: they close when the loop runs again, or, when it never does, once the call that was starting it is garbage collected. A worker's process group is signalled once, so a process the engine reaped is never signalled again. A cancel of `close` reaches the caller once the workers on its own loop are stopped, without waiting for one still starting, which is stopped as it comes up; workers on another loop are handed to that loop to stop, and once a cancel has landed `close` hands the rest over without waiting for their loop, so a blocked loop never holds the cancel. A stop that fails on another loop after a cancel is logged at ERROR when that loop gets to it, even when the loop stops before a worker it is starting comes up. The engine is closed for good: `run`, a call of an `engine.tool`, `EngineTools`, and a `replay` or `label` over it raise `InputError` (`engine: this engine was closed; build a new Engine`) before any model is called, so nothing is recorded — a replay's pairs and a label's `errors` carry it. A run already in flight goes on, but starts no other run: its later folder tool calls fail as above, a `Delegate` reads as that tool's error (`InputError: engine: this engine was closed; build a new Engine`), and a route fails the decision run with that `InputError`, before the model of the agent it would reach, generation or decision, is called. Leaving `async with Engine(…) as engine:` calls it. Safe to call more than once.

### `Stop`

```python
class Stop:
    def __init__(self) -> None: ...
```

A person's handle on a running run, passed as `run(stop=)`; it reaches the runs it delegates and routes to, and the runs its calls of an `engine.tool` start. A stop set before the run starts ends it before the first model call; one set after it ended does nothing.

### `Stop.now`

```python
def now(self) -> None: ...
```

Cancels the model call in flight and every tool call still running, except a call with a generation run going under the same stop — one a `Delegate` call or an `engine.tool` started, from the call's task or one it made — which is waited for while that run takes the stop, and then cut, with whatever else it was doing, as any call is; a `Shell` call whose command's script started such a run through the tool bridge is cancelled and its command killed, while the script's call is waited for as that run takes the stop and then cut, so that run keeps its record and the `Shell` call ends after it, unless the run is cancelled or that call's deadline passes; a call waiting on a decision run, one model call, is cancelled as a model call in flight is; a call it finds not yet running never runs, so a stop from an `on_event` handler of a call's `ToolCallStarted`, or of a reply's `TextDelta`, comes before the tool runs. A handler still running when a stop cancels its call, or a streamed `TextDelta` or `ReasoningDelta` handler, is cancelled with that work, so a kill switch does its own work, such as an audit, before it calls `stop.now()`. The run returns with `stopped="now"`, and each call it cut off is answered with the error `stopped by the user`. A call that already has its result keeps it, and the stop waits for an `on_event` still handling its `ToolResultReady` to return, so a slow handler delays it; cancelling the run cancels the handler. The run returns once the work the stop cancelled has ended, so a tool that ignores cancellation holds the stop, as it holds a cancel of the run. `now()` is safe to call any number of times: each call after the first, while the run is stopping, cancels once more every tool call still running — not a call with a generation run under the same stop, whose run takes that `now()` itself, nor a call whose `ToolResultReady` handler the stop waits for — so a workspace command still being stopped is killed at once, with no grace period, as it is when the stopping run is cancelled.

### `Stop.after_step`

```python
def after_step(self) -> None: ...
```

Lets the current tool calls finish and ends the run before its next model call, with `stopped="after_step"`. After `now()` it does nothing.

## Records

### `AgentInfo`

```python
class AgentInfo(BaseModel):
    name: str
    kind: Kind
    model: str
    description: str
    delegate_to: list[str]
    files: list[str]
    tools: list[str]
    run_tools: list[str]
    skills: dict[str, str]
    public: dict[str, str]
    worker: str
    modes: dict[str, AgentInfo]
```

An agent as the engine loaded it, in `Engine.agents`: its name, its kind, its `model` — the key its entry gives — its description, the agents it may `delegate_to`, the `files` its model may open, by their paths from the root, the `tools` it picks from `tools:`, by name — the built-ins and the tools a run gives are not listed — its `run_tools`, the names its `tools.allow` expects a run to give through `tools=`, in name order, its `skills`, every skill it holds, each to its `description:` — `""` for one without; a decision agent has none — its `public`, the ones people may call by typing `/name`, in name order, each to its `description:` — `{}` when there are none, its `worker`, and its `modes`, each mode's own `AgentInfo` by the mode's name, its `name` being its agent's, `{}` for a mode itself and for an agent without modes. Each read of `Engine.agents` builds them anew, so changing one changes nothing the engine holds. For an agent of an `rfnry.yaml`, `skills` holds every skill its entry or a mode names and `public` those its `skills.public` names. `Kind` is `Literal["generation", "decision", "voice"]`.

### `Generation`

```python
class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    call: CallInfo | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    mode: str | None = None
    skill: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    settings: dict[str, JsonValue] | None = None
    fingerprint: Fingerprint
    input: State
    system: str
    preload: list[Part] = []
    start: int = 0
    messages: list[Message]
    conversation: list[Message]
    tools: list[ToolSpec] = []
    steps: list[Step]
    calls: int
    reply: str
    usage: Usage
    children: list[Generation | Decision] = []
    stopped: Stopped | None = None
    stop_reason: StopReason | None = None
    error: str | None = None
    added: list[Message]
```

The record of a generation run, `mode` naming the mode it selected and `None` for a run on the entry: the `input` exactly as it was given — read back, a list is the parts carried by the user message the input became, the skill's part aside, when their JSON equals the stored input, and the stored JSON otherwise, so parts, text-only parts and a JSON list shaped like parts each read back as they were given, while a part-shaped class of your own is stored as it dumps itself and reads back as parts only when that JSON equals the engine part's — the `skill` it was run with, the `context` it was given, when it started (`at`), which agent entry, files and tools produced it (`fingerprint`), the system prompt as sent, the messages as sent to the first model call — the history, then the input — the whole `conversation`, the tools offered, every tool call as a `Step`, the model calls and their summed usage, the reply, and the runs it delegated to in `children`, in step order, and in a run that failed, after them, the delegated runs it had not answered yet, in the order they started. `model` is the model the replies reported; when none did, as in a run stopped during its first model call, it is the adapter's `model` attribute if that is a non-empty string, else `None`. `settings` are those of the call that wrote the last assistant message: the settings its reply reported, else the adapter's `settings` attribute, `None` when neither is a JSON object. An accepted `Output` makes `reply` its JSON. `preload` is what `files.preload` and `skills.preload` sent, `[]` without them: the files `files.preload` reaches less `files.deny`, pattern by pattern in list order, a file `Read` returns as a `Text` of `<<<path>>>` on a line of its own followed by what `Read` returns, any other file, or one over 5 MB, as a `Text` of `<<<path>>>` and a line naming its type and size, such as `image/png, 15 bytes; Load opens it`, which `Load` opens when the model asks and nothing about which is refused when the folder loads (an agent whose tools leave out `Load` gets the line without `; Load opens it`), then each preloaded skill's text without its frontmatter; every model call receives it as one user message ahead of the messages, and neither `messages` nor `conversation` holds it. `Stopped` is `Literal["now", "after_step"]`; `stopped` is `None` for a run that ended on its own. `start` is the position in the history the run began at, `0` when it was sent whole. `added` is computed from the record: the input and every message after it, `conversation` from the input on — what a caller keeps. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`; `stop_reason` is the last reply's `stop_reason` — `"max_tokens"` for a reply the model's limit cut off, `"tool_use"` after an accepted `Output` — and `None` when no reply came back. `error` is `None` for a run that returned; a run that raised after its first model call, or from its `RunStarted` handler, is recorded with `error` as `"<Type>: <message>"`, `reply` `""` and what it had done by then. `call` is the tool call the run worked for, as its `RunStarted.call` within one scope: the `Delegate` call for a delegated run, the caller's call for an `engine.tool` consult made inside a run, the call a tool body was serving for a run that body or a task it started began, the router's call for a routed run, and `None` outside any tool call or for a call made in another scope.

### `Decision`

```python
class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    call: CallInfo | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    mode: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    settings: dict[str, JsonValue] | None = None
    fingerprint: Fingerprint
    state: State
    questions: dict[str, Question]
    runtime: list[str] = []
    answers: dict[str, Answer]
    usage: Usage
    routed_to: Generation | Decision | None = None
    error: str | None = None
    added: list[Message]
```

The record of a decision run, `mode` as in `Generation`: the state as the model was handed it — a list of parts when the adapter's `media_types` took them; read back, a list state holding media reads back as parts only when every item is written exactly as a part, all its fields and no other, and as JSON otherwise, so a JSON list written that way reads back as parts — every question asked, from the folder — a scripted one as its script filled it — and at run time, with `runtime` naming the ids that came at run time, an answer per question id with its `confidence`, the usage, model and settings the adapter reported — its `model` and `settings` attributes when the call reports none — and what produced it (`fingerprint`, whose `tools` is the fingerprint of no tools and whose `questions` covers `questions`). `routed_to` is the run its route started, if any. `added` is the `added` of the run its route started, `[]` when it routed nowhere. `error` is set, and `answers` empty, when the decision model's call or a question's script raised; when a script raised, `questions` holds only the questions that need no script, the runtime ones included; a routed run that failed sits in `routed_to` with its own `error`, or `routed_to` is `None` when the routed run was refused before its first model call, and the router carries the same `error`; a routed run whose `RunStarted` handler raised had started, and sits in `routed_to`. Its `call` is the same as a generation's.

### `Voice`

```python
class Voice(BaseModel):
    kind: Literal["voice"] = "voice"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    mode: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    fingerprint: Fingerprint
    instructions: str
    tools: list[ToolSpec] = []
    steps: list[Step]
    transcript: list[SpokenLine]
    usage: Usage | None = None
    error: str | None = None
```

The record of a voice session, what `Engine.talk` returns: `mode` as in `Generation`, `context` as it was given, `instructions` the text the session was handed, `tools` the tools it was offered, `steps` its tool calls in the order they finished, the same `Step` a generation run writes, and the `transcript` the session returned. `parent_run_id` is always `None`, since nothing delegates or routes to a voice agent. `model` is the session's, else the adapter's `model` attribute. `usage` is the session's and `None` when the host has none, which a consumer must not read as zero cost. `error` is `None` for a session that returned; one that raised is recorded with `error` as `"<Type>: <message>"`, its `steps` as far as it got and an empty `transcript`. A `fingerprint`'s `tools` covers `tools`, and its `questions` is `None`. `Refinery.run` reads it back equal, a `usage` of `None` included. `Refinery.review` takes a stance and a comment on it and no correction, `Refinery.outcome` takes it as any run, `Refinery.export` leaves it out, and `Refinery.replay` refuses it.

### `SpokenLine`

```python
class SpokenLine(BaseModel):
    speaker: Literal["person", "assistant"]
    text: str
    at: datetime | None = None
```

One line of a `Voice.transcript`: who said it, what they said and when, `None` when the host does not know.

### `Spoken`

```python
class Spoken(BaseModel):
    transcript: list[SpokenLine]
    model: str | None = None
    usage: Usage | None = None
```

What a session function returns to `Engine.talk`: the transcript, the model that ran, and the usage when the host has one.

### `VoiceStart`

```python
@dataclass(frozen=True)
class VoiceStart:
    run_id: str
    model: VoiceModel
    instructions: str
    tools: Tools
    mode: str | None
```

What `Engine.talk` hands a session function: the run's `run_id`, minted before the session opens, the live model it resolved, the instructions, the one `Tools` whose calls pass `approve`, the deadline and the `Step` path, and the `mode`.

### `Run`

```python
Run = Generation | Decision | Voice
```

A record of any kind, told apart by `kind`: `Engine.run` returns a `Generation` or a `Decision`, and `Engine.talk` a `Voice`. A record holds objects of its own: what it was given — the input, the history, what the inbox held, the questions asked at run time, the tools' specs and results, each reply's tool calls and reasoning, each answer — is copied into it, a routed run's input apart from its router's state, every list and object, any other value as it is, and so are the preload and the questions it took from the folder, so changing a record changes nothing the folder shares with every tenant's runs, nothing of another record, and nothing your code still holds, and changing what your code holds changes no record.

### `Fingerprint`

```python
class Fingerprint(BaseModel):
    agent: str
    tools: str
    questions: str | None
```

Which agent, which tools and which questions produced a run, each the first 16 hex characters of a sha256. `agent` covers the agent's entry in `rfnry.yaml` — the entry with the mode it runs on overlaid, and the entries of the tools it picks, as canonical JSON, `models:` left out — and every file it reaches, by its path from the folder root and its content: its instructions, its questions with their scripts and the modules beside them, the scripts, modules and schema files of the tools it picks, the files `files.allow` and `files.preload` reach, and every file of each skill the entry or a mode names; it is computed once, when the `Engine` is built. `tools` covers the tools the model was offered, `Run.tools` sorted by name as canonical JSON: the builtins, `History` when the run's `start` is above 0, the declared tools it picks, a workspace's, `Delegate` with the agents it may call, the run's `tools=` and `Output` when `expect=` asked for one. `questions` covers the questions a decision run asked, `Decision.questions` as JSON ordered by question id, each question's fields in order and its options and levels as given, runtime questions included; it is `None` for a generation run. None covers the input, the history, `context`, `tags`, the model or the code behind a tool a run passes; a folder tool's script is a file of the agent's.

### `Step`

```python
class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result: Content
    is_error: bool
    at: datetime
    duration_ms: int
    child_run_id: str | None = None
    stopped: bool = False
    approved: bool | None = None
    by: str | None = None
```

One tool call of a generation or voice run: the tool, its arguments, its result, whether it failed, when it finished and how long it took. `child_run_id` names the run a `Delegate` started, `stopped` marks a call `Stop.now()` cut off, one whose `Delegate` or `engine.tool` run a stop ended, or one a voice session cancelled, and `approved` is what `approve` answered, kept on a call cut after it answered, and `None` when it gave no answer, because nobody was asked or the call was cut first. `by` is the id of the model's `Shell` call whose command made this call through the workspace's tool bridge, `None` for a call the model made itself; such a step finished first, so it sits in `Generation.steps` before the `Shell` step that made it, and the conversation holds no message for it. A call still running when its command ended is recorded with `stopped` set and `stopped by the user` as its result.

### `RunStarted`

```python
class RunStarted(BaseModel):
    run_id: str
    agent: str
    call: CallInfo | None
```

Sent to `on_event` once by every generation run, before anything else it sends — also by a run that a stop set before it started ends before its first model call. `call` is the tool call the run works for, as `current_call` sees it when the run starts: the caller's `Delegate` call for a delegated run, the caller's call for a run an `engine.tool` call started inside a run, the call that started the decision run for a run it routes to, `None` for a run started outside any tool call. It names the call even when it was made in another scope, where the run's record holds `None`. Every later event with this `run_id` belongs under `call`, so sub-agents nest at any depth. A decision run sends no events; the `ToolCallStarted` and `ToolResultReady` of the call that asked it show it. A stop waits for its handler, and one the handler sets ends the run before its first model call; a handler that raises fails the run, recorded with that `error` and no model call.

### `TextDelta`

```python
class TextDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str
```

Text from the model, sent to `on_event` as it is produced; from a model that does not stream, the whole text of each reply. `turn` is the model call of the run that said it, from 0: text of a turn that ended in tool calls was said beside them, and a run that ends in text replies with its last turn's text. A delegated or routed run sends its own `run_id` and counts its own turns.

### `ReasoningDelta`

```python
class ReasoningDelta(BaseModel):
    run_id: str
    agent: str
    turn: int
    text: str
```

The model's reasoning, sent to `on_event` as a streaming adapter passes it with `on_delta(text, reasoning=True)`; `turn` as in `TextDelta`. It is not part of the reply or the conversation: a reply's reasoning is recorded as the adapter returned it, in `Message.reasoning`, and reasoning streamed before `Stop.now()` is not kept. A model that does not stream, or does not stream its reasoning, sends none.

### `MessagesTaken`

```python
class MessagesTaken(BaseModel):
    run_id: str
    agent: str
    turn: int
    messages: list[Message]
```

Sent to `on_event` when a run takes messages from its `inbox`, at a step boundary: `messages` are copies, the handler's own to their parts, of the `role="user"` messages it appended to the conversation, in the order the inbox held them, and `turn` is the model call about to read them. The inbox is first in, first out and the run takes all it holds, so they are the first `len(messages)` items put on it that no earlier event reported. Nothing taken, an inbox not read because `Stop.after_step()` is pending, and messages left when the run ends send none. A stop waits for its handler, and one the handler sets ends the run before that model call, the messages already in its conversation.

### `ToolCallStarted`

```python
class ToolCallStarted(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str
    args: dict[str, Any]
    by: str | None = None
```

Sent to `on_event` when a tool call starts, before any approval, with a copy of the arguments the model gave it, the handler's own. `by` is the id of the `Shell` call whose command made the call through the workspace's tool bridge, its `call_id` then `<Shell call id>.<n>`, n counting from 1 for each command; `None` for a call of the model's. A handler that calls `Stop.now()` stops the call before its tool runs, and a handler still running when a stop cancels its call is cancelled with it, so a kill switch does its own work, such as an audit, before it calls `Stop.now()`.

### `ToolResultReady`

```python
class ToolResultReady(BaseModel):
    run_id: str
    agent: str
    call_id: str
    step: Step
    by: str | None = None
```

Sent to `on_event` once, when a tool call is done: `step` is a copy of the `Step` the run records for it, the handler's own, equal to its entry in `Generation.steps`, sent on every way a call ends — a result, an error, a timeout, a call `approve` refused, a delegation, a call `Stop.now()` cut off, whether or not it had started, an `Output` included. Any other call the engine answers without running it — an `Output` it accepts or refuses, a call after an accepted `Output`, a call in a reply that ended for a reason other than tool use, answered `NotRun` — sends neither `ToolCallStarted` nor `ToolResultReady`; its step is in `Generation.steps`. A cancelled run raises and is not recorded, so a call it cancels sends no `ToolResultReady`, a call whose tool returns after the cancel included: it ends with the cancellation before any handler sees its result. `by` is `step.by`, as the call's `ToolCallStarted` has it. A handler that raises on one call of a reply fails the run once the reply's other calls are answered: their events still reach the handler, and the run raises the first failure, logging each further one as an error on the `rfnry_engine.engine` logger.

### `CommandOutput`

```python
class CommandOutput(BaseModel):
    run_id: str
    agent: str
    call_id: str
    stream: Pipe
    text: str
```

Sent to `on_event` while a foreground `Shell` command of a run with a `Workspace` runs: one event for what the command writes, as it arrives, with chunks that queued while the handler was busy joined into one event per stream, with the `call_id` of the `Shell` call and the `stream` it was written to. `text` is UTF-8 decoded per stream, so a character written in two pieces arrives whole. Every event of a command comes before the `ToolResultReady` of its call, in order per stream; `stdout` and `stderr` are two pipes, with no order between them. The handler never slows the command or changes what the model reads: the result is the same with or without `on_event`. Chunks queue up to 1 MiB while the handler is busy; past that, a chunk is dropped from the events, as is a single read of more than 1 MiB even while the handler is idle, and once a stream overflows its output is dropped until the handler catches up; the next event of that stream starts with `[... <n> bytes not shown ...]` and a newline, counting all the bytes dropped. A command started with `background=true` sends none. A handler that raises on a `CommandOutput` fails the run, once the command has ended. `Pipe` is `Literal["stdout", "stderr"]`.

### `Event`

```python
Event = (
    RunStarted
    | TextDelta
    | ReasoningDelta
    | MessagesTaken
    | ToolCallStarted
    | ToolResultReady
    | CommandOutput
)
```

What `on_event` receives.

### `Review`

```python
class Review(BaseModel):
    stance: Literal["approval", "objection"] | None = None
    comment: str | None = None
    correction: str | dict[str, Any] | None = None
```

A person's label on a run, for `Refinery.review`: a stance, a comment, a correction, or several of them; it needs at least one. A generation run takes a text correction, a decision run a value per question id, and a voice run none.

### `Outcome`

```python
class Outcome(BaseModel):
    status: Literal["succeeded", "failed"]
    detail: str | None = None
```

What the world reported about a run, for `Refinery.outcome`.

### `Label`

```python
class Label(BaseModel):
    label_id: str
    run_id: str
    scope: str
    at: datetime
    source: Literal["human", "world", "judge", "teacher"]
    review: Review | None = None
    outcome: Outcome | None = None
    answers: dict[str, Answer] | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

A stored label: its `source` and what it says — a person's `review`, the world's `outcome`, or a judge's or a teacher's `answers`. A judge's or a teacher's label also names the `model` and `settings` of the run that made it; a person's and the world's leave them `None`.

### `Labeled`

```python
class Labeled(BaseModel):
    judged: int = 0
    taught: int = 0
    errors: dict[str, str] = {}
```

What `Refinery.label` returns: how many runs the judge and the teacher labeled, and each run that could not be labeled, by run id, with why.

### `Exported`

```python
class Exported(BaseModel):
    keys: list[str]
    rows: dict[str, int]
    watermark: datetime | None
```

What `Refinery.export` returns: the keys written, the rows written per dataset, and the `watermark` to pass as the next export's `since`.

### `Conflict`

```python
class Conflict(BaseModel):
    run_id: str
    corrections: list[tuple[str, datetime, str]]
```

A run that carries two different human corrections: each as `(label_id, at, text)`, oldest first.

### `Replay`

```python
class Replay(BaseModel):
    recorded: Run
    labels: list[Label]
    attempt: int = 0
    candidate: Run | None = None
    error: str | None = None
```

One pair from `Refinery.replay`: the recorded run as `Refinery.run` reads it back, every label on it oldest first, which of the `repeat` replays of that run this is, from 0, and the candidate's run on what the recorded run was given, never recorded. When the candidate failed, `candidate` is `None` and `error` is the failure as `"TypeName: message"`.

## Tools

### `Tool`

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    body: Body | None = None
    args: type[BaseModel] | None = None
    input_schema: dict[str, Any] | None = None
```

One tool: a name, a description, an input schema and a body. `Body` is `Callable[[dict[str, Any]], Awaitable[Result]]`, an async function of the arguments, where `Result` is `str | dict[str, Any] | list[Part]`; `args` validates the arguments and gives the schema, its descriptions and docstring included, unless `input_schema` does; a tool given only `input_schema` receives its arguments as the model sent them. `description` is what the model reads about the tool, and its first line is what the prompt's list of tools shows. It runs in the consumer's process and rides a run, in `run(tools=)`, where it wins over a folder tool of the same name, and nothing checks it for a body or a description before it is called.

### `Tool.schema`

```python
def schema(self) -> dict[str, Any]: ...
```

Returns the input schema: `input_schema`, else the JSON schema of `args`, else an object with no properties.

### `Tool.definition`

```python
def definition(self) -> dict[str, Any]: ...
```

Returns the tool as a model is offered it, `{name, description, input_schema}`.

### `Tool.call`

```python
async def call(self, args: dict[str, Any]) -> Result: ...
```

Validates `args` against `args` when the tool has one, awaits the body, and returns what it returns. `Result` is `str | dict[str, Any] | list[Part]`.

Raises `ConfigurationError` when the tool has no body. Raises `InputError` when the arguments do not fit `args`.

### `Tools`

```python
class Tools:
    def __init__(self, tools: Iterable[Tool | ToolsLike] = ()) -> None: ...

    names: list[str]

    def __add__(self, other: Tools | ToolsLike) -> Tools: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
```

A set of tools with unique names: what a model receives, and what code can call the same way. It takes `Tool`s and anything else with its two verbs: `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which another `Tools` meets. `names` lists the tools in order and `a + b` is the union.

Raises `ConfigurationError` when two tools share a name, on `+` as well.

### `Tools.definitions`

```python
def definitions(self) -> list[dict[str, Any]]: ...
```

Returns each tool as `{name, description, input_schema}`, in order.

### `Tools.dispatch`

```python
async def dispatch(self, name: str, args: dict[str, Any]) -> Result: ...
```

Calls the tool `name` with `args` and returns its result. `Result` is `str | dict[str, Any] | list[Part]`.

Raises `InputError` when there is no tool `name`. A tool's own errors pass through.

### `Tools.only`

```python
def only(self, *names: str) -> Tools: ...
```

Returns a `Tools` of the named tools, in the order named.

Raises `InputError` when a name is not in the set.

### `Tools.without`

```python
def without(self, *names: str) -> Tools: ...
```

Returns a `Tools` without the named tools; a name not in the set is ignored.

### `EngineTools`

```python
class EngineTools(Tools):
    def __init__(
        self,
        engine: Engine,
        *,
        agent: str,
        scope: str,
        mode: str | None = None,
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
        history: Sequence[object] | HistoryLike | None = None,
        start: int = 0,
        on_event: EventHandler | None = None,
        stop: Callable[[], Stop | None] | None = None,
        invoked: str | None = None,
        models: Mapping[str, RunModel] | None = None,
    ) -> None: ...
```

`RunModel` is `GenerationModel | DecisionModel | VoiceModel | ModelChoice`. The tools `agent` runs with under `scope`, in the mode `mode` selects, as a `Tools` to drive without a model, exactly as a run with these arguments offers them: the builtins `Read`, `Grep`, `Glob`, `LS` and `Load` when its `files` or `skills` opens a file (a path of `Read`, `Load`, or the `path` of `Grep` or `LS`, judged where it leads, its links and `..` resolved, from the folder root — is an `InputError` when the fingerprint leaves it out, `tools/builtins: .env is hidden; names starting with . and __pycache__ are not part of the folder of the agents`: a file whose folder has a name starting with `.` or named `__pycache__` on its way there, or whose own name as written is one — a visible link to a hidden file of the same folder reads — and a folder that leads to such a name, a visible link to a hidden folder included, and `Glob`, `Grep` and `LS` list nothing so hidden, nor a link that leads outside those folders, while `LS` lists a broken or looping link inside them by its name, as it lists a file, and `Grep` given a file labels its matches with the path as given, a file link under its own name; a part written after a file names a folder that is not there, as POSIX has it, so `Read`, `Load`, `Grep` and `LS` of `notes.md/`, `notes.md/.`, `notes.md/..` or `notes.md/x` are `tools/builtins: not a folder: notes.md/`, the path as written; a file link that comes from outside those folders, or leads from one of them into another, is judged by the file it leads to as well as by its own name, and a path that leads outside them is the jail's `InputError`, `jail: path outside the folder: <path>`, unless it leaves them through a hidden name, as a link in a hidden folder that leads out does, which is hidden; a path outside what `files` and the skills it is offered or preloads reach — a path a `.gitignore` ignores under `files.gitignore: true`, and the folder of a skill the agent holds but neither offers nor preloads, included — reads exactly as a file that does not exist and listings leave it out, whatever the spelling, unless it is a file of the `invoked` skill), `History` when `start` is above 0, the declared tools it picks — but one `tools` holds a tool of the same name, which wins — the workspace's tools when it declares `workspace: true`, `Delegate` when it has `orchestration.delegate_to`, and the tools of `tools` its `tools.allow` names, each less what `tools.deny` drops, and a `Delegate` dispatched through it hands its child the whole of `tools`, which the child's own `tools.allow` narrows. A folder tool's call has its arguments checked against its `input_schema` — a mismatch an `InputError`, `tools/declared: invalid arguments for lookup_order: 5 is not of type "string" at order_id` — and goes to the agent's worker with `call` holding `scope` and `tags` — inside a run's tool call of this scope, the run's tags with the call's `run_id`, `agent`, `tool`, `call_id` and `by`; outside one, the `tags` given here, `{}` without — under the engine's `tool_timeout`, which counts the wait for a worker too (`TimeoutError: workers/pool: lookup_order timed out after 30s`); arguments JSON cannot hold, a `datetime` or a `set` passed from code, are an `InputError` before any worker sees them (`workers/pool: the arguments for lookup_order are not JSON: Object of type datetime is not JSON serializable`); its result is checked against its `output_schema`, a mismatch a `ModelError`, and a list whose every item has the `type` of a part comes back as parts, its media `data` read from base64. What the script raised, a result JSON cannot hold, and a worker that stopped are a `WorkerError`. The workers follow the event loop that calls them, one loop at a time: a call from another loop stops the workers the first one started and starts its own. `Delegate` takes `agent` and `input`, and optional `questions` — ids to objects with the fields of `Choice`, `Score` or `Check` — when one is a decision agent; its description lists each agent's name, description and kind. A `Delegate` dispatched through it runs the child as a top-level run — not recorded when it is dispatched inside a run started with `record=False` — with this scope, `tools`, `workspace`, `tags` and `approve`, and the `questions` it is called with, its `on_event` and `stop` as `engine.tool` takes them (`EventHandler` is `Callable[[Event], Awaitable[None]]`), and raises what `Engine.run` raises, and `InputError` when `questions` does not fit the contract, holds an empty id or an id the run already asks, or is given to a generation agent; inside a run the same refusals reach the model as tool errors. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. `History` reads `history` before position `start`: a page of up to `limit` messages — 20, at most 100 — before `before`, which defaults to `start` and never passes it, each under a JSON header with its `position`, `role`, and its `tool_calls`, or `tool_call_id` and `is_error`, in the order they happened; with `pattern`, a regular expression (`case_insensitive`), only the messages whose text matches, walking back at most the engine's `history_walk` messages. An image, audio or document shows as a line naming it — its type, its media type, a document's `name` and its `url` when it has one — never as the file. A result holds at most `history_chars` characters of messages and, when there is more, ends with where to go on; the engine's `tool_timeout` bounds a search's pattern. `invoked` is the skill this run invoked, whose folder these tools see too, its other files included, though the skill is neither offered nor preloaded; a run passes its own, and `None` leaves the folders of the offered and the preloaded skills open, as on every run.

Raises `InputError` when the engine was closed (`engine: this engine was closed; build a new Engine`), `scope` is empty, `workspace` belongs to another scope, `start` is negative, `start` is above 0 without a `history`, `start` is past the end of a list `history`, or the agent needs a mode and `mode` is left out. Raises `NotFoundError` when the folder has no agent `agent`, the agent has no mode `mode`, or the agent holds no skill `invoked` (`tools/engine_tools: agent 'support' has no skill 'typo'; its skills are close`). `models`, as in `Engine.run`, decides what `Load` takes and what `Delegate` runs on. Raises `ConfigurationError` when `models` does not fit as in `Engine.run`, the agent declares `workspace: true` and no `workspace` is given, two tools of `tools` share a name, one shares a name with a tool the engine gives the agent, or the agent's `tools.allow` names a tool neither declared, granted by its sections nor given in `tools` (`tools/engine_tools: agent 'support' lists 'refnd' in tools.allow, which this run does not offer; its tools are Glob, Grep, LS, Load, Read, refund`). A tool named `History` in `tools` is a `ConfigurationError` whatever `start` is: the name is the builtin's on every run.

### `current_call`

```python
def current_call() -> CallInfo | None: ...
```

The tool call the code runs inside: a tool's body, the run's `approve` and `on_event` handlers for that call, and the tasks the body starts — they keep answering for it after it ends. `None` outside a run's tool call: a `Tools.dispatch` or an `EngineTools` call of your own. A delegated run's calls, and the calls of a run an `engine.tool` call starts, name their own run; the rest of such a run — its model calls and its text handlers — runs inside the caller's call and sees that one.

### `CallInfo`

```python
class CallInfo(BaseModel):
    run_id: str
    agent: str
    scope: str
    tool: str
    call_id: str
    by: str | None = None
    tags: Mapping[str, str] = {}
```

What `current_call` returns, and what `RunStarted.call` holds: the run's id, its agent and scope, the tool called and the id of the call, as `ToolCall.id` carries it. `by` is the id of the `Shell` call whose command made the call through the workspace's tool bridge, `None` for a call of the model's, so `approve` and a tool can tell a call a script made without reading its id. `tags` are the tags of the run whose call it is, read-only, `{}` for a run without tags, so a tool bound to what the run is about — the ticket, the customer — reads it from the call, set by your code and never by the model. Frozen, so it can be kept, compared and put in a set.

### `Jail`

```python
class Jail:
    def __init__(
        self,
        root: Path | str,
        *,
        also: Iterable[Path | str] = (),
        withheld: Callable[[Path], bool] | None = None,
        named: Collection[Path] = (),
    ) -> None: ...

    root: Path
    also: list[Path]
    withheld: Callable[[Path], bool] | None
    named: frozenset[Path]
```

Keeps paths inside one directory, `root`, or inside one of `also`, all resolved; for a tool of your own that reads the folder of the agents, as the builtins do. It notes each place's device and inode when it is made and checks a place by them before it opens anything in it. The check covers `root` and every folder of `also`, so it guards each of its places as `Local` guards its root; it compares directory inode numbers, so a filesystem where they are not stable — overlayfs without `xino`, FUSE without `use_ino` — is not supported for any place, and a `Jail` made before its place exists refuses every open in it. `withheld`, which the engine gives the builtins for every path outside an agent's `files` and the skills it is offered or invokes on that run, a path a `.gitignore` ignores under `files.gitignore: true` included, is asked by `open` about the path as written, before the links in it are followed: a path it answers true for raises the `FileNotFoundError` of a file that does not exist, and the builtins leave such a path out of what they list. `named`, which the engine gives the builtins, holds the hidden paths that file names: the builtins treat a hidden name held there as any other, and a hidden name below one stays hidden.

### `Jail.resolve`

```python
def resolve(self, path: str | Path) -> Path: ...
```

Returns `path`, taken relative to `root` unless it is absolute, resolved with its symlinks.

Raises `InputError` when it resolves outside `root` and every place in `also`.

### `Jail.open`

```python
def open(self, path: str | Path, flags: int = 0, mode: int = 0o666) -> int: ...
```

Opens `path`, resolved as `resolve` does, and returns its file descriptor, which the caller closes. `flags` and `mode` are those of `os.open`; `flags` is `os.O_RDONLY` unless given. It opens the place that holds the target, checks it is still the folder the `Jail` was made with, and walks down from it one folder at a time without following a link, so a link swapped in after the check is refused; with `os.O_CREAT` it creates the missing folders on the way. The last name is opened non-blocking, so a pipe never holds the call. The builtins and `Local` read and write through it.

Raises `InputError` when `path` resolves outside `root` and every place in `also`, when the walk meets a link, or when the place is no longer the folder it was. What the system refuses, such as a missing file, is raised as the `OSError` it is, and a path `withheld` answers true for is raised as a missing file.

### `JobInfo`

```python
class JobInfo(BaseModel):
    job: str
    command: str
    started: datetime
    running: bool
    exit_code: int | None
```

A job of a workspace, as `Workspace.jobs` returns it: its id, its command, when it started (timezone-aware), whether it still runs, and its exit code once it has ended.

### `Workspace`

```python
class Workspace:
    def __init__(
        self,
        backend: Backend,
        *,
        scope: str,
        timeout: float = 120.0,
        output_bytes: int = 30_000,
        bridge: bool = True,
        tree: str | None = None,
    ) -> None: ...

    backend: Backend
    scope: str
    timeout: float
    output_bytes: int
    bridge: bool
    tree: str | None
```

Where an agent with `workspace: true` writes code and runs it, passed as `run(workspace=)`: the library's tools over `backend`, for the tenant `scope`. `timeout` bounds every `Shell`, `Job`, `Write` and `Edit` call in seconds; a command still running at the limit, or when its call is cancelled, is stopped with its process's `kill()`, and so is anything a command leaves running when it ends, after which what remains of its output is read for at most a second. Only that limit is the workspace's: a `TimeoutError` the backend raises itself, from `read`, `write` or a process's `wait`, reaches the caller as it is, never as `timed_out: true` or the workspace's `tools/workspace: Write did not finish within 120s`. On `Local`, whose `read` and `write` each run in a thread a cancel cannot stop, a `Write` or `Edit` past `timeout`, like a workspace `Read` or `Load` past `Engine(tool_timeout=)`, returns its `TimeoutError` only once the thread has ended. Every kill runs to its end and the call that started it waits for it, however many cancels come meanwhile, so a stop or a cancel of `Shell` or `Job(id, stop=true)` returns once the kill has returned. A kill counts as the first stop, and two cancels that land at the same moment count as two: the next cancel after the one that started it, or the second during a kill the workspace started on its own — for a command past `timeout` or that has ended, `Job(id, stop=true)` or `close` — cancels that kill, once, so `kill()` ends the command at once. A `kill()` error the call cannot raise, because it is already leaving with a cancellation or with the error of a `wait()` that raised, is logged as an error, `tools/workspace: could not stop the command: …`; one from the kill that follows a job's end is in the job's output. `output_bytes` caps `stdout` and `stderr` each, keeping their first and last half. Jobs belong to the workspace and outlive the runs that started them; each is read in the background until it ends or `close` stops it. A workspace and its jobs live in one event loop: a job whose loop ends is stopped.

`tree` is where the folder of the agents is mounted in the sandbox, as commands see it, such as `/srv/agents`, kept as `posixpath.normpath` writes it, so `/srv/agents/` is `/srv/agents`. With it, the commands of a run get `RFNRY_TREE`, the folder's path, and the `Shell` description tells the model where it is, in a variable and never in a path, along with `$RFNRY_TREE/<folder>` for each folder its `files.allow` names; a command started with `background=true` gets it too. A `Shell` that code dispatches outside a run, or from inside another tool, gets neither. Without `tree`, nothing is added.

`bridge` lets a foreground `Shell` command of a run call the run's tools. When it is on, the backend has a `root` attribute — the bridge's contract with a backend: a local directory its commands see at the same path, as `Local`'s, where the socket is created — and the run offers at least one tool the bridge can carry, the `Shell` description tells the model so, in variables, never a path, and each foreground command gets `RFNRY_TOOLS`, the path of a Unix socket `<root>/.rfnry-<8 hex>.sock`, and `RFNRY_TOOLS_KEY`, a key of its own, as private as the sandbox keeps the command's environment. The socket is created just before the command starts and closed once it has ended and its output is read, then removed if its path still holds it, and every request it answers runs as a model's call of the run: its events, `approve`, its deadline, the stop and its `Step`, with `by` the `Shell` call's id. It carries every tool the run offers but `Shell`, `Job`, `Jobs`, `Write`, `Edit` and any tool named `Output` or `Delegate`, a run's own `tools=` included, and at most 64 calls run at once per command, a connection past that closed at once after a `tools/bridge:` error line, which the script may see as a broken pipe or a reset instead; another name is answered `InputError: tools: unknown tool '<name>'` and runs nothing. A command makes at most 10,000 calls — every request that reaches the run's tools, an unknown name's included — and sends at most 2 MiB of request lines, each line read whole counted as it is read, before it is parsed, one the bridge refuses included, and a call's `args` nest at most 128 levels of lists and objects, `args` itself the first, read from the line before it is parsed, its strings aside and the request itself the first level, so any field nested past 129 levels is past it, not only `args`; past each, the request is answered `tools/bridge: this command made its 10000 calls; no more are answered`, `tools/bridge: this command sent its 2097152 bytes of calls; no more are answered` or `tools/bridge: a call's args nest deeper than 128 levels` before it is parsed, and it runs nothing and leaves no step and no event, and a connection closed at the cap of 64, refused before its line is read, counts toward neither budget. A request is checked in this order and answered with the error of the first check it fails: a line over 1 MiB (`tools/bridge: a request is at most 1048576 bytes`), which is not read whole and counts toward neither budget; then, the line counted toward the 2 MiB, the 10,000 calls and then the 2 MiB; then the depth; then a line that is not one JSON object; then the command's key (`tools/bridge: the key is not this command's`); then a string `tool` and an object `args`. So a line over 1 MiB gets the size error whatever else is wrong with it, a line past a budget gets the budget's error even when it is nested too deep or malformed, and a line nested past 129 levels gets the depth error even when its key is not the command's or it is no JSON at all. The budgets are each command's own: commands a run has going at once each get theirs, so what a run's commands send in all grows with how many it runs at once. The `Shell` description states both budgets. When the socket cannot be made, a path too long for one included, the command runs without it and its `stderr` ends with `tools/workspace: no tool bridge for this command: <why>`. A `Shell` with `bridge=False`, on a backend without `root`, started with `background=true`, or dispatched by code outside a run's own `Shell` call, gets no socket.

Raises `ConfigurationError` when `backend` lacks `spawn`, `read` or `write`, `scope` is empty, `timeout` or `output_bytes` is not positive, or `tree` is not an absolute path.

### `Workspace.jobs`

```python
def jobs(self) -> list[JobInfo]: ...
```

The jobs the workspace lists now, in the order they started, for a frontend that shows them. It reads memory and changes nothing: the model's `Jobs()` lists the same jobs, and a job leaves the list once the model reads its last output.

### `Workspace.output`

```python
def output(self, job: str) -> str: ...
```

Everything `job` has printed so far, in the cap of `output_bytes`: its first and last half, with the bytes between omitted, as the model's `Job` would show it. It does not take the output from the model: what `Job` returns next is unchanged. While the job runs, a character not yet complete is left out. Raises `InputError` when the workspace lists no job `job`.

### `Workspace.tools`

```python
def tools(self) -> list[Tool]: ...
```

Returns the five tools that an agent with `workspace: true` gets whenever a workspace is passed: `Shell(command, background=false)`, which returns `exit_code`, `stdout`, `stderr` and `timed_out`, or with `background=true` starts a job and returns `{"job": id}` at once; `Job(id, wait=0, stop=false)`, which waits up to `wait` seconds for the job to end, then returns its `output` since the last read, `running`, and `exit_code` once it has ended, and with `stop=true` stops a job still running after the wait; `Jobs()`, which lists the jobs as `{"jobs": [{job, command, started, running, exit_code}]}`; `Write(path, content)`, which writes a file exactly; and `Edit(path, old, new, all=false)`, which replaces one exact occurrence, or every occurrence with `all=true`, in a UTF-8 file of at most 5 MB, and says how many it replaced. They go through the backend, and its errors reach the caller as they are. `Job` raises `InputError` for a `wait` outside 0 to `timeout`, or an id with no job: a job leaves the list once its last output is read.

### `Workspace.close`

```python
async def close(self) -> None: ...
```

Stops every job of the workspace with its process's `kill()`, waits for each to end, and empties the list; a job whose `kill()` raised stays listed, so a second `close` or `Job(id, stop=true)` can reach it. The workspace stays usable and its ids keep counting. Leaving `async with Workspace(…) as workspace:` calls it. Raises the first error a `kill()` raised, after asking every job, and logs the others. A cancelled `close` returns once every `kill()` has returned — a second cancel cancels each `kill()` still running, once — and logs the `kill()` errors it cannot raise; cancelled while its kills run, it still empties the list of the jobs they stopped and waits for those to end, as an uncancelled `close` does, before it raises. Each job still ending takes the cancels of `close` as its own — a further one that came before the wait, as the wait starts — so a first lets the kill that follows the job's end run its course, and a further one cancels that kill, once.

### `Backend`

```python
@runtime_checkable
class Backend(Protocol):
    async def spawn(self, command: str, env: Mapping[str, str]) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...
```

Where a workspace's commands run and its files live, passed to `Workspace`. `spawn` starts `command` in the workspace's directory with no input, with `env` — its own copy, made for each command — added to the command's environment, and returns at once, and must return promptly: a stop or a cancel that comes while it runs waits for it to return, then for the kill of what it started, however many cancels come meanwhile, and a second one cancels that kill as soon as it starts; with `background=true` it waits for the job to be listed, so `close` stops it. `read` returns a file's bytes and refuses one larger than `limit`; `write` writes a file whole, creating its parent directories. Paths are the model's, relative to the workspace, and what they can reach is the backend's to decide. An error a backend raises reaches the model as the tool's error. A backend's description, if it has one, is ignored: what the environment holds is told in the agent's markdown. A backend gets the tool bridge when it has a `root` attribute: a local directory its commands see at the same path, where each foreground command's socket is created; a backend whose `root` means anything else gets a socket created at that path, so one without a shared directory has no `root`.

### `Process`

```python
class Process(Protocol):
    stdout: Stream
    stderr: Stream

    async def wait(self) -> int: ...

    async def kill(self) -> None: ...
```

A command a backend started. `stdout` and `stderr` are read until they return `b""`; `Stream` is a protocol of `async def read(self, n: int) -> bytes`, which an `asyncio.StreamReader` meets. `wait` returns the exit code once the command has exited, even while something it started still holds its output open. `kill` ends the command and everything it started and returns once they have ended or cannot be stopped further; it may be called after the command exited, to end what it left behind, and more than once. The workspace waits for every `kill` it starts, through any cancel, so it must return promptly, and it does not retry one that raised: a `kill` owns its escalation, as `Local` sends SIGKILL a second after SIGTERM. A second stop — a cancel that comes while the workspace waits for a `kill`, after the one that started it, or the second during a kill the workspace started on its own — cancels it, once, and the workspace still waits for it: a cancelled `kill` must still end the command and everything it started, at once, before it returns, as `Local` sends SIGKILL in a `finally`. Its error is the error of the call that asked for it, is logged when that call has another error to raise or is leaving with a cancellation, and is in the job's output when it follows a job's end.

### `Local`

```python
class Local:
    def __init__(
        self,
        root: Path | str,
        *,
        shell: Sequence[str] = ("bash", "-c"),
        env: dict[str, str] | None = None,
    ) -> None: ...

    root: Path
    shell: list[str]
    env: dict[str, str]

    async def spawn(self, command: str, env: Mapping[str, str]) -> Process: ...

    async def read(self, path: str, limit: int) -> bytes: ...

    async def write(self, path: str, data: bytes) -> None: ...
```

A `Backend` on this machine. Commands run as `[*shell, command]` in `root`, in a new session, with `env` as their whole environment, and what `spawn` is given added over it — by default `PATH`, `HOME` set to `root`, and `LANG`; a sandbox goes in front of the shell, as in `shell=["bwrap", …, "bash", "-c"]`. `kill` sends the command's process group SIGTERM and, a second later or at once when it is cancelled, SIGKILL. A process that leaves the group with `setsid` outlives the kill, but not the command's pipes: once the command has exited and the workspace no longer reads its output, `Local` closes them, so that process holds none of the engine's files and what it writes there after is lost. `read` and `write` stay inside `root`, symlinks resolved, and each runs in a thread: a call cancelled meanwhile returns once its thread has ended, however many cancels come, so a `Write` or `Edit` cancelled while it writes has written its file when the cancel returns.

Raises `ConfigurationError` when `root` is not a directory, `shell` is not a non-empty list of non-empty strings, or its first word is not on the `PATH` of `env`. `spawn` raises `InputError` when the command cannot start, as when `root` was removed: `tools/workspace: cannot start the command: No such file or directory`. `read` and `write` raise `InputError` for a path outside `root`, through a link that leads out of it or one swapped in while they walk to the file, and for every path once `root` is no longer the folder it was when `Local` was made — replaced, or reached through a link swapped in for it or a folder above it; `read` for a path that is not a file — missing, a directory, a FIFO or a socket, or a path through a file, such as `notes.txt/x` — or a file larger than `limit`; `write` for a directory, a FIFO or a socket, and a path through a file as `tools/workspace: cannot write notes.txt/x: Not a directory`. Any other error the system gives is `tools/workspace: cannot read {path}: {reason}` or `tools/workspace: cannot write {path}: {reason}`. Every message names the path as the model gave it, never the path on this machine. The root check compares directory inode numbers, so a filesystem where they are not stable — overlayfs without `xino`, FUSE without `use_ino` — refuses a valid root and is not supported as a workspace root.

## Storage

### `Storage`

```python
@runtime_checkable
class Storage(Protocol):
    async def check(self) -> None: ...

    async def put_runs(self, scope: str, rows: list[RunRow]) -> None: ...

    async def get_run(self, scope: str, run_id: str) -> RunRow | None: ...

    async def runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        kind: str | None = None,
        since: datetime | None = None,
        unlabeled_by: str | None = None,
        limit: int | None = 1000,
    ) -> list[RunRow]: ...

    async def put_label(self, scope: str, row: LabelRow) -> LabelRow: ...

    async def labels(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        source: str | None = None,
        since: datetime | None = None,
    ) -> list[LabelRow]: ...

    async def run_tree(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> list[RunRow]: ...

    async def delete_runs(
        self,
        scope: str,
        *,
        run_ids: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> int: ...

    async def delete_scope(self, scope: str) -> int: ...

    async def count_scope(self, scope: str) -> int: ...
```

The one storage protocol, as the refinery calls it; `PostgresStorage` implements exactly these methods, as a storage of your own does. Every method but `check` takes `scope` first and reads and writes that scope only: `check` only reads, and raises `ConfigurationError` when the storage does not fit this version; `put_runs` stores every row of one recorded tree in one transaction, replacing rows with the same ids; a row of another scope raises `StorageError` and nothing is stored; `put_label` stores a row unless the scope already holds a label with its id, which it never replaces, and returns the row stored under that id, so calls racing on one id all get the same row; `get_run` returns a run or `None`; `runs` returns runs newest first by `at`, at most `limit` or all with `None`, narrowed by `run_ids`, `kind`, `since` (stored after it, by `recorded`) and `unlabeled_by`, which keeps the runs no label from that source names; `labels` returns labels oldest first, narrowed by `run_ids`, `source` and `since`; `run_tree` returns the runs `run_ids` or `tags` (by containment) match and every run under them, oldest first; `delete_runs` removes those runs and every label on them, and `delete_scope` every run and label in the scope, each returning the number of rows removed; `count_scope` counts the scope's runs and labels. The rows `put_runs` and `put_label` are handed are made for the call, and what the refinery builds from the rows a storage returns copies them, so a storage may keep the rows it is handed or returns, and changing a run or a label the refinery returned changes no row. A storage raises `StorageError` when a row's scope is not the call's, and `PostgresStorage` raises it whenever the database refuses.

### `PostgresStorage`

```python
class PostgresStorage:
    def __init__(self, pool: Any) -> None: ...
```

Implements `Storage` in Postgres, over an asyncpg pool you built, in the tables `engine_runs` and `engine_labels` of the schema first on the pool's `search_path`, which the `rfnry-engine` command creates and upgrades. `check` raises `ConfigurationError` naming the command to run when that schema holds no engine tables (naming tables `initialize()` made in another schema to move with `upgrade --from`), is behind or ahead of this version, holds tables `initialize()` made in 0.1.4 or earlier, or is one whose schema or tables the pool's role cannot use; every `json` and `jsonb` value is sent and read as text, cast in its SQL, so it works with any `json` or `jsonb` codec set on the pool, which never sees what it stores or reads; `put_runs` stores its rows in one transaction and holds each body's JSON at most twice over while it is sent, passing it as a `bytea` argument that is an object with the buffer protocol, not `bytes`, which keeps no copy once it has handed the JSON over and encodes it again from the body each further time it is asked, so a `bytea` codec set on the pool must take any such object, as `bytes(value)` does — one that takes only `bytes` fails every `put_runs` with `StorageError` (`refinery/postgres: storage rejected put_runs: invalid input for query argument $10: …`); `delete_runs` and `delete_scope` each run in one transaction; `put_label` inserts unless the id is taken and then reads the stored row back, so processes racing on one id store one row and all get it.

### `RunRow`

```python
class RunRow(BaseModel):
    scope: str
    run_id: str
    parent_run_id: str | None
    agent: str
    kind: str
    at: datetime
    recorded: datetime
    duration_ms: int
    model: str | None
    mode: str | None
    body: dict[str, Any]
```

A stored run. `body` is the run's record as JSON with its children and routed run left out, each a row of its own joined by `parent_run_id`, and a generation's `messages` replaced by `sent`, their count, since they are the start of its `conversation`; read a run back with `Refinery.run`; `at` is when the run started and `recorded` when the refinery stored it.

### `LabelRow`

```python
class LabelRow(BaseModel):
    scope: str
    label_id: str
    run_id: str
    source: str
    at: datetime
    body: dict[str, Any]
```

A stored label; `body` is the `Label` as JSON.

### `Refinery`

```python
class Refinery:
    def __init__(self, storage: Storage) -> None: ...

    storage: Storage
```

Records runs, stores labels on them, exports datasets and replays recorded runs on a candidate, all through `storage`, which stays readable to read the rows back yourself. Passed as `Engine(refinery=)`, it records every run that returns. A storage's `StorageError` passes through every verb.

### `Refinery.check`

```python
async def check(self) -> None: ...
```

Checks the storage: for `PostgresStorage`, that its schema fits this version; it only reads. Raises `ConfigurationError` naming the command that fixes it.

### `Refinery.record`

```python
async def record(self, run: Run) -> None: ...
```

Stores `run` and every run delegated or routed under it, all stamped with one `recorded` time: what an engine with a refinery does after each run not made with `record=False`.

Raises `InputError` when the run's scope is empty, or a generation's `messages` are not the start of its `conversation`; nothing is stored then.

### `Refinery.run`

```python
async def run(self, scope: str, run_id: str) -> Run: ...
```

Returns the recorded run as `Engine.run` or `Engine.talk` returned it, its delegated and routed runs back in place; a child's id returns that child and the runs under it.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.review`

```python
async def review(
    self,
    scope: str,
    run_id: str,
    review: Review,
    *,
    label_id: str | None = None,
) -> Label: ...
```

Stores a person's review of the run and returns the label, with `source="human"`. Without `label_id` the label gets a fresh id. With it the label is stored under `label_id`, once: when the scope already holds a label with that id on this run, that stored label is returned and nothing is written, so the same call made again — after a crash, or by every process that heard the same event, at the same moment or not — stores one label.

Raises `InputError` when `scope` or `label_id` is empty, when the correction does not fit the run — a dict for a generation run, a string for a decision run, a question id the run was not asked, or any correction on a voice run (`refinery: a voice run takes a stance and a comment, not a correction`) — or when the scope's label `label_id` is on another run. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.outcome`

```python
async def outcome(
    self,
    scope: str,
    run_id: str,
    outcome: Outcome,
    *,
    label_id: str | None = None,
) -> Label: ...
```

Stores what the world reported about the run and returns the label, with `source="world"`. `label_id` works as in `Refinery.review`: the label is stored under it once, and a later call with it on the same run returns the stored label and writes nothing.

Raises `InputError` when `scope` or `label_id` is empty, or when the scope's label `label_id` is on another run. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.conflicts`

```python
async def conflicts(self, scope: str) -> list[Conflict]: ...
```

Returns the runs that carry two different human corrections. A dataset can hold either, not both; the library reports rather than guesses.

Raises `InputError` when `scope` is empty.

### `Refinery.export`

```python
async def export(
    self,
    scope: str,
    bucket: Bucket,
    *,
    since: datetime | None = None,
    holdout_percent: int = 20,
    mode: str | None = None,
) -> Exported: ...
```

Writes every run recorded or labeled after `since`, with all its labels, to `bucket` as JSONL under `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, or `{scope}/{agent}.{mode}/{dataset}/{timestamp}.jsonl` for a run in a mode, so two modes never share a key, in the datasets `generation`, `preference` and `decision`. Given `mode`, only that mode's runs are written and the watermark covers only them. Every row carries, after `run_id`, its run's `parent_run_id` and its `call` as `Run.call` holds it, `holdout`, from a hash of its run id and `holdout_percent`, its run's `mode`, and its run's `fingerprint` as `{"agent", "tools", "questions"}`. A generation row carries the run's `stopped` and `stop_reason`. A run whose `error` is set is in no dataset, except a decision run that answered before the run it routed to raised: its rows are written. A voice run is in no dataset, and neither it nor a label on it counts toward the watermark. Returns the keys, the rows per dataset and the watermark for the next `since`.

Raises `InputError` when `scope` is empty or `holdout_percent` is outside 0 to 100, before anything is read or written.

### `Refinery.label`

```python
async def label(
    self,
    scope: str,
    engine: Engine,
    *,
    judge: str | None = None,
    teacher: str | None = None,
    limit: int = 500,
    models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
) -> Labeled: ...
```

Runs `judge`, a decision agent, over the generation runs it has not labeled, and `teacher`, a generation agent, over the decision runs it has not labeled — at most `limit` runs each, their own runs skipped — and stores each answer as a label. A run that failed is labeled like any other; the judge is told its `error`. A run that could not be labeled is named in `errors` and stays unlabeled for the next call. `models`, as `Engine.run` takes them, are what the judge and the teacher run on, over the engine's by key.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the folder has no agent `judge` or `teacher`. Raises `ConfigurationError` when `judge` is not a decision agent, `teacher` not a generation agent, or `models` does not fit as in `Engine.run`; each check happens before any run is labeled.

### `Refinery.replay`

```python
def replay(
    self,
    scope: str,
    candidate: Engine,
    *,
    agent: str,
    holdout: bool = False,
    labeled: bool = False,
    tags: dict[str, str] | None = None,
    fingerprint: Fingerprint | None = None,
    limit: int | None = None,
    holdout_percent: int = 20,
    expect: type[BaseModel] | None = None,
    approve: Approve | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    concurrency: int = 1,
    repeat: int = 1,
    conversation: Callable[[Run], Sequence[object] | HistoryLike] | None = None,
    models: Mapping[str, GenerationModel | DecisionModel | ModelChoice] | None = None,
) -> AsyncGenerator[Replay]: ...
```

Runs `candidate` on what the top-level runs of `agent` in `scope` were given and yields a `Replay` for each, oldest first by when the run started. `holdout` keeps the held-out runs, by the same hash and `holdout_percent` as `export`; `labeled` the runs with a `human` or `world` label; `tags` the runs whose tags contain these, as `delete_runs` matches; `fingerprint` the runs whose fingerprint equals it, all three parts; `limit` the first so many. Each replay is a run of the candidate, as `candidate.run(agent, scope, …, record=False)` would be but never reading `/name`, with the recorded input — parts read back as parts — the history (the messages sent before the input) with `start=0`, so without `History` — or, given `conversation`, what it returns for the recorded run, cut before the recorded input, with the recorded `start`, so the candidate has the same window and `History` over everything before it — the skill, which the candidate reads from its own folder, and the `context` — a run invoked by `/name` recorded that skill and the rest of its input, and the replay sends that pair, `/name` never read again, so a rest that itself starts with `/<skill>` replays as text, as does a recorded input of `/close x` whose `close` has become public on the candidate since; a decision run replays its state and its runtime questions as recorded, the rest of its questions coming from the candidate's folder, its scripts filling them anew, and when its route reached a generation run, that run's input, history and context. `expect`, `approve`, `tools`, `workspace` and `models` are passed to every replay, `models` as `Engine.run` takes them. `conversation` is called once per replay, with the recorded top-level run, and applies to the generation run it replays; a decision run that reaches no generation run never calls it, and a `conversation` that raises, or that returns a conversation ending before the recorded input, gives the pair its `error`. It gives the recorded window only when the run was sent its conversation with `start`: a run recorded with `start=0` on a list the app had already cut replays on the conversation's first messages instead. `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. The candidate's tools run for real. Each run is replayed `repeat` times, each attempt a run of its own yielded as a pair of its own whose `attempt` counts from 0 — its `recorded` run and `labels` are copies of their own, so changing one attempt's pair changes neither another's nor what a later attempt is run on; the attempts of a run come one after another, in attempt order, and `limit` counts recorded runs, not attempts. At most `concurrency` replays run at once, attempts included, and while a pair is being read at most `concurrency - 1` run, so with the default a replay starts when its pair is asked for. Pairs are yielded in that order either way. Closing the generator, or cancelling its reader while it waits for a pair, cancels each replay still running and waits for them to end. That close stands for the reader's first cancel: each cancel of the reader beyond its first, two at the same moment counted as two, cancels them once more, so a second ends a replay's workspace command at once. The close returns once they have ended, and a cancelled reader then raises `CancelledError`. A candidate that raises yields its pair with `error` and the replay goes on, the other attempts of the same run included; each replay runs in the mode its top-level run recorded, and a mode the candidate no longer has is such an error (`InputError: refinery: run 'r1' ran in mode 'chat', which agent 'a' of the candidate does not have`), as is a recorded voice run, whatever the candidate's agent is, since a person's side of a conversation cannot be sent again (`InputError: refinery: run 'r1' is a voice run; a spoken conversation cannot be replayed`). Nothing is recorded and no label is written.

Raises `InputError` when `scope` is empty, `workspace` belongs to another scope, `holdout_percent` is outside 0 to 100, `tags` is empty, or `limit`, `concurrency` or `repeat` is below 1. Raises `NotFoundError` when the candidate has no agent `agent`. Raises `ConfigurationError` when `models` does not fit as in `Engine.run`. Each check happens when `replay` is called, before anything is read; a storage's `StorageError` is raised by the first pair.

### `Refinery.delete_scope`

```python
async def delete_scope(self, scope: str) -> int: ...
```

Removes every run and label in the scope, and returns the number of rows removed.

Raises `InputError` when `scope` is empty.

### `Refinery.delete_runs`

```python
async def delete_runs(
    self,
    scope: str,
    *,
    run_ids: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> int: ...
```

Removes the runs `run_ids` or `tags` match, a tag by containment, every run delegated or routed from them, and every label on any of them, and returns the number of rows removed.

Raises `InputError` when `scope` is empty, when given neither selector or both, or when the one given is empty.

### `Refinery.verify_absent`

```python
async def verify_absent(
    self,
    scope: str,
    *,
    run_ids: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> bool: ...
```

Returns `True` when nothing the selector matches is left: no run, no run under one, and for `run_ids` no label. Without a selector it checks that the scope holds nothing.

Raises `InputError` when `scope` is empty, when given both selectors, or when the one given is empty.

### `Bucket`

```python
@runtime_checkable
class Bucket(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...
```

Where `Refinery.export` writes: any object with this method. Keys never repeat, so an export never overwrites an earlier one.

### `is_holdout`

```python
def is_holdout(run_id: str, *, holdout_percent: int = 20) -> bool: ...
```

Returns whether the run is held out, from a hash of its id: the same answer every export gives its rows.

Raises `InputError` when `holdout_percent` is outside 0 to 100.

## Errors

### `EngineException`

```python
class EngineException(Exception): ...
```

The root of every error the library raises. Every message names the module that raised it, then says what went wrong in lowercase: `engine: agent 'ghost' was not found`. Inside a run, a tool's error is the model's to read, with its type in front, and is never raised.

### `ConfigurationError`

```python
class ConfigurationError(EngineException): ...
```

A folder, a model, a tool or a workspace the engine cannot use, found when it is built or when a run needs it.

### `NotFoundError`

```python
class NotFoundError(EngineException): ...
```

An agent the tree does not have, a judge or a teacher included, a skill the agent does not have, or a run the refinery does not.

### `InputError`

```python
class InputError(EngineException): ...
```

A call the library refuses: an empty scope, a workspace of another scope, tags outside their limits, runtime questions that do not fit or that the run already asks, `questions` given to a generation agent, a `skill` given to a decision agent, a run given a skill both by `skill` and by an input that starts with `/<a public skill>`, a decision run with nothing to ask, media a decision model does not declare, a media part with both or neither of `data` and `url`, a correction of the wrong shape, a tool called with bad arguments or a path outside the folder.

### `ModelError`

```python
class ModelError(EngineException): ...
```

An adapter broke the contract: a result whose fields do not fit, a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels. Or a script did: a folder tool's result that does not match its `output_schema`, or a question's script that returns anything but an object or fills a question that does not fit.

### `RunawayError`

```python
class RunawayError(EngineException): ...
```

A generation run passed its limit of model calls without ending: the smaller of `Engine(max_model_calls=)` and its entry's `max_model_calls`.

### `DelegationLoopError`

```python
class DelegationLoopError(EngineException): ...
```

A route would send its input to an agent that already decided it in the same chain of routes. It raises.

### `DelegationDepthError`

```python
class DelegationDepthError(EngineException): ...
```

Nesting went deeper than `Engine(max_depth=)`, or more levels below an agent than its entry's `orchestration.max_depth`. Through `Delegate` it reaches the model as a tool error; through a route it raises.

### `StorageError`

```python
class StorageError(EngineException): ...
```

The storage refused: the database failed, or a row's scope is not the scope of the call. `PostgresStorage`, and a step the `rfnry-engine` command runs, raise it for an error of the database or the driver (asyncpg's), of the connection (`OSError`) or of a timeout (`TimeoutError`, an `OSError`), its message ending in the error's text or, when the error has none, its type's name (`refinery/postgres: storage rejected get_run: TimeoutError`); any other error, such as a row body that is not JSON or a defect in a step's own code, reaches the caller as itself.

### `WorkerError`

```python
class WorkerError(EngineException): ...
```

A tool or a question's script that runs in a worker process failed there. Its message is what the script raised, `ValueError: no order A-1` — `SystemExit: 3` for a `sys.exit(3)` — or what its import raised, a question's script's after the script and the question it fills (`engine: order.py of decision agent 'matcher' could not fill question 'order': KeyError: 'customer'`); a result JSON cannot hold, `TypeError: Object of type datetime is not JSON serializable`; a result or error over `worker_result_bytes`, `workers/process: lookup_order sent back 40000023 bytes, over the limit of 33554432 bytes`; a worker that stopped while the call was in flight — it exited, even with a process it forked still holding its pipes (`workers/pool: the worker for lookup_order stopped: exited with code 3`), it sent back a line that is not a reply (`it sent a line that is not a reply`), a signal ended it (`killed by signal 9`), the next call retired it once every thread was held by a call past its deadline, a `def run` or an `async def run` blocking the worker's event loop (`every thread was held by a call past its deadline`), or a call came from another event loop (`a call came from another event loop`); or a call in flight when `Engine.close` stopped its worker, `workers/pool: the worker for lookup_order stopped: it was closed`. Inside a run, the model reads a tool's as a tool error; a question's script's ends its decision run, before the decision model is called.
