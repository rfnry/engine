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

A model that writes text and calls tools: any object with this method, matched by shape. It returns a `Reply`, or anything with its fields; a result whose fields do not fit makes the run raise `ModelError`. An adapter may also carry `media_types`, the file types its model takes, such as `{"image/*", "application/pdf"}`, so `Load` refuses the rest.

### `StreamingGenerationModel`

```python
@runtime_checkable
class StreamingGenerationModel(Protocol):
    async def generate_stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], Awaitable[None]],
    ) -> ReplyLike: ...
```

A generation model that also streams. When a run has `on_event`, the engine calls `generate_stream` instead of `generate` and the adapter awaits `on_delta` with each piece of text as it arrives; it returns the whole reply, as `generate` does.

### `DecisionModel`

```python
@runtime_checkable
class DecisionModel(Protocol):
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

A model that answers typed questions about a state in one call: any object with this method, matched by shape. It returns an answer per question id, or a `Decided` when it can report what the call cost and which model answered. An adapter may also carry `media_types`, the media its model reads in a state, such as `{"image/*"}`: a run's parts with that media reach it as the state, a list of parts; without it the state is text, and parts it does not declare are refused before it is called. Answers that do not fit — a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels — make the run raise `ModelError`.

### `ReplyLike`

```python
class ReplyLike(Protocol):
    text: str
    tool_calls: Sequence[ToolCallLike]
    usage: UsageLike | None
    stop_reason: StopReason
    model: str | None
    reasoning: Any
```

The shape a generation model's result is read by: the read-only fields of `Reply`. `ToolCallLike` is a protocol of `id: str`, `name: str` and `args: dict[str, Any]`, `UsageLike` is one of `input`, `output`, `cache_creation` and `cache_read`, each an `int`, and `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `DecidedLike`

```python
class DecidedLike(Protocol):
    answers: Mapping[str, AnswerLike]
    usage: UsageLike | None
    model: str | None
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
```

One turn of a conversation. An assistant turn carries its text in `content`, its calls in `tool_calls` and its `reasoning` as the model returned it; a tool result has `role="tool"`, the `tool_call_id` it answers, and `is_error` when the call failed.

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
    reasoning: Any = None
```

What a generation model returns. A reply whose `stop_reason` is `"tool_use"` and whose `tool_calls` is non-empty asks the engine to run those calls and call the model again; anything else is final. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

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
```

What `decide` returns instead of the answers alone when the adapter can report what the call cost and which model answered. The run's record carries both.

### `confidence`

```python
def confidence(probabilities: list[float]) -> float: ...
```

Returns how peaked a distribution is, `(n·p_max − 1)/(n − 1)`, at least 0, and 1.0 for fewer than two probabilities. The engine sets every answer's `confidence` with it, so a threshold means the same thing for every adapter and question type.

## Engine

### `Engine`

```python
class Engine:
    def __init__(
        self,
        root: Path | str,
        *,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        tools: list[Tool] | None = None,
        refinery: Recorder | None = None,
        max_steps: int = 200,
        max_depth: int = 5,
        tool_timeout: float | None = 30.0,
    ) -> None: ...

    agents: dict[str, AgentInfo]
```

Loads the tree of agents under `root` and checks it against the models and tool bodies given; `tools` is one list for the whole tree, and `Recorder` is a protocol of `async def record(self, run: Run) -> None`, which a `Refinery` meets. `tool_timeout` is each tool call's deadline in seconds, `None` for none. `agents` maps each agent's name to its `AgentInfo`.

Raises `ConfigurationError` when `root` is not a directory, when the tree cannot be used (no agent under `root`, a bad `AGENT.md`, question or tool file, an unknown frontmatter key, an agent in `agents:` or `route:` the tree does not have or a decision agent there with no `questions/`, a route not by a choice question of the router's `questions/`, a decision agent with a file straight in `tasks/` or a routing one with `tasks/`), when a tool is given that no agent declares or, with a generation model, a declared tool has no body, when two tools share a name, or when `max_steps`, `max_depth` or `tool_timeout` is not positive.

### `Engine.run`

```python
async def run(
    self,
    agent: str,
    scope: str,
    input: State,
    *,
    task: str | None = None,
    questions: Mapping[str, QuestionLike] | None = None,
    history: Sequence[object] | None = None,
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
) -> Run: ...
```

Runs `agent` for the tenant `scope` on `input` and returns its record: a `Generation` for a generation agent, a `Decision` for a decision agent. A generation agent's `task` puts `tasks/<task>.md` in its system prompt; a decision agent's asks the questions in `tasks/<task>/` instead of `questions/`. `questions` are a decision agent's runtime questions, read by their fields and asked with the folder's in the same call; the routed run does not get them. With a refinery attached, a run that returns is recorded unless `record=False`, and a recording that fails is logged, never raised. `EventHandler` is `Callable[[Event], Awaitable[None]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets, and `tools` rides this run only; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`, called with the agent's name and the call.

Raises `InputError` when `scope` is empty, a generation agent is given `questions`, a runtime question does not fit the contract or has an empty id or one the folder already asks, a decision run has no question to ask, a decision agent is given media its model does not declare in `media_types`, `tags` are outside their limits, `history` does not fit the contract, or `input` mixes parts with other values. Raises `NotFoundError` when the tree has no agent `agent`, or the agent has no task `task`, before any model is called. Raises `ConfigurationError` when the model for the agent's kind is absent, a decision agent routes to a generation agent without a generation model, an agent with `workspace: true` runs without a `workspace`, `tools` repeats a name the agent already has, a tool named `Output` collides with `expect`, or `approve` returns anything but `True` or `False`. Raises `ModelError` when an adapter's result does not fit the contract. Raises `RunawayError` when the loop passes `max_steps`. Raises `DelegationLoopError` or `DelegationDepthError` when a route loops or goes deeper than `max_depth`.

### `Engine.tool`

```python
def tool(
    self,
    agent: str,
    *,
    scope: str,
    name: str | None = None,
    description: str | None = None,
    context: str | None = None,
    tools: ToolsLike | None = None,
    workspace: Workspace | None = None,
    tags: Mapping[str, str] | None = None,
    approve: Approve | None = None,
) -> Tools: ...
```

Returns a `Tools` of one tool that runs `agent` as a top-level run under `scope`, recorded like any run: named `name` or the agent's name, described by `description` or the agent's, taking `{"input": str}` and returning a generation agent's reply or a decision agent's answers as a dict. `context`, `tools`, `workspace`, `tags` and `approve` ride every call. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets; `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`.

Raises `InputError` when `scope` is empty or `tags` are outside their limits. Raises `NotFoundError` when the tree has no agent `agent`. A call raises what `Engine.run` raises.

### `Stop`

```python
class Stop:
    def __init__(self) -> None: ...
```

A person's handle on a running run, passed as `run(stop=)`; it reaches the runs it delegates and routes to. A stop set before the run starts ends it before the first model call; one set after it ended does nothing.

### `Stop.now`

```python
def now(self) -> None: ...
```

Cancels the model call in flight and every tool call still running. The run returns with `stopped="now"`, and each call it cut off is answered with the error `stopped by the user`.

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
    description: str
    agents: list[str]
```

An agent as the engine loaded it, in `Engine.agents`: its name, its kind, its description and the agents it may call. `Kind` is `Literal["generation", "decision"]`.

### `Generation`

```python
class Generation(BaseModel):
    kind: Literal["generation"] = "generation"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    task: str | None = None
    context: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    fingerprint: Fingerprint | None = None
    input: State
    system: str
    messages: list[Message]
    conversation: list[Message]
    tools: list[ToolSpec] = []
    steps: list[Step]
    calls: int
    reply: str
    usage: Usage
    children: list[Run] = []
    stopped: Stopped | None = None
```

The record of a generation run: the `context` it was given, when it started (`at`), which agent folder and tools produced it (`fingerprint`), the system prompt as sent, the messages as sent to the first model call — the history, then the input — the whole `conversation`, the tools offered, every tool call as a `Step`, the model calls and their summed usage, the reply, and the runs it delegated to in `children`, in step order. An accepted `Output` makes `reply` its JSON. `Stopped` is `Literal["now", "after_step"]`; `stopped` is `None` for a run that ended on its own. A run recorded before `context` or `fingerprint` was kept reads back with `None`.

### `Decision`

```python
class Decision(BaseModel):
    kind: Literal["decision"] = "decision"
    run_id: str
    parent_run_id: str | None = None
    agent: str
    scope: str
    tags: dict[str, str] = {}
    task: str | None = None
    at: datetime
    duration_ms: int
    model: str | None
    fingerprint: Fingerprint | None = None
    state: State
    questions: dict[str, Question]
    runtime: list[str] = []
    answers: dict[str, Answer]
    usage: Usage
    routed_to: Run | None = None
```

The record of a decision run: the task it was given, the state as the model was handed it — a list of parts when the adapter's `media_types` took them, read back as parts — every question asked, from the folder or the task and at run time, with `runtime` naming the ids that came at run time, an answer per question id with its `confidence`, the usage and model the adapter reported, and what produced it (`fingerprint`, whose `tools` is the fingerprint of no tools and whose `questions` covers `questions`). `routed_to` is the run its route started, if any. A run recorded before `task`, `runtime` or `fingerprint` was kept reads back with `None`, `[]` and `None`.

### `Run`

```python
Run = Generation | Decision
```

What `Engine.run` returns, told apart by `kind`.

### `Fingerprint`

```python
class Fingerprint(BaseModel):
    agent: str
    tools: str
    questions: str | None = None
```

Which agent, which tools and which questions produced a run, each the first 16 hex characters of a sha256. `agent` covers every file in the agent's directory as the engine loaded it, by relative path and content, leaving out hidden files and folders (a name starting with `.`) and links that lead out of the directory; it is computed once, when the `Engine` is built. `tools` covers the tools the model was offered, `Run.tools` sorted by name as canonical JSON: the folder's, the code's, the run's `tools=` and `Output` when `expect=` asked for one. `questions` covers the questions a decision run asked, `Decision.questions` as JSON ordered by question id, each question's fields in order and its options and levels as given, runtime questions included; it is `None` for a generation run and for a run recorded before it was kept. None covers the input, the history, `context`, `tags`, the model or the code behind a tool.

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
```

One tool call of a generation run: the tool, its arguments, its result, whether it failed, when it finished and how long it took. `child_run_id` names the run a `Delegate` started, `stopped` marks a call `Stop.now()` cut off, and `approved` is what `approve` answered, `None` when nobody was asked.

### `TextDelta`

```python
class TextDelta(BaseModel):
    run_id: str
    agent: str
    text: str
```

Text from the model, sent to `on_event` as it is produced; from a model that does not stream, the whole text of each reply.

### `ToolCallStarted`

```python
class ToolCallStarted(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str
```

Sent to `on_event` when a tool call starts, before any approval.

### `ToolResultReady`

```python
class ToolResultReady(BaseModel):
    run_id: str
    agent: str
    call_id: str
    tool: str
    is_error: bool
```

Sent to `on_event` when a tool call has its result, with whether it failed.

### `Event`

```python
Event = TextDelta | ToolCallStarted | ToolResultReady
```

What `on_event` receives.

### `Review`

```python
class Review(BaseModel):
    stance: Literal["approval", "objection"] | None = None
    comment: str | None = None
    correction: str | dict[str, Any] | None = None
```

A person's label on a run, for `Refinery.review`: a stance, a comment, a correction, or several of them; it needs at least one. A generation run takes a text correction, a decision run a value per question id.

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
```

A stored label: its `source` and what it says — a person's `review`, the world's `outcome`, or a judge's or a teacher's `answers`.

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
    candidate: Run | None = None
    error: str | None = None
```

One pair from `Refinery.replay`: the recorded run as `Refinery.run` reads it back, every label on it oldest first, and the candidate's run on what the recorded run was given, never recorded. When the candidate failed, `candidate` is `None` and `error` is the failure as `"TypeName: message"`.

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

One tool: a name, a description, an input schema and a body. `Body` is `Callable[[dict[str, Any]], Awaitable[Result]]`, an async function of the arguments, where `Result` is `str | dict[str, Any] | list[Part]`; `args` validates the arguments and gives the schema unless `input_schema` does. In `Engine(tools=)` it is the body behind a tool an agent declares, and a declaration with only a name adopts its description and schema.

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
        tools: ToolsLike | None = None,
        workspace: Workspace | None = None,
        tags: Mapping[str, str] | None = None,
        approve: Callable[[str, ToolCall], Awaitable[bool]] | None = None,
    ) -> None: ...
```

The tools `agent` runs with under `scope`, as a `Tools` to drive without a model: the builtins, the tools it declares, the workspace's tools when it declares `workspace: true`, `Delegate` when it names `agents:`, and `tools`. A `Delegate` dispatched through it runs the child as a top-level run with this scope, `tools`, `workspace`, `tags` and `approve`. `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the tree has no agent `agent`. Raises `ConfigurationError` when the agent declares `workspace: true` and no `workspace` is given, a tool it declares has no body, or two tools share a name.

### `Jail`

```python
class Jail:
    def __init__(self, root: Path | str) -> None: ...

    root: Path
```

Keeps paths inside one directory, `root`, resolved; for a tool of your own that reads the agent's directory, as the builtins do.

### `Jail.resolve`

```python
def resolve(self, path: str | Path) -> Path: ...
```

Returns `path`, taken relative to `root` unless it is absolute, resolved with its symlinks.

Raises `InputError` when it resolves outside `root`.

### `Workspace`

```python
class Workspace:
    def __init__(
        self,
        root: Path | str,
        *,
        sandbox: Sequence[str] | None = None,
        unsandboxed: bool = False,
        shell: Sequence[str] = ("bash", "-c"),
        timeout: float = 120.0,
        output_bytes: int = 30_000,
        env: dict[str, str] | None = None,
    ) -> None: ...
```

A directory where an agent with `workspace: true` writes code and runs it, passed as `run(workspace=)`. It takes exactly one of `sandbox`, the command each `Shell` call runs inside, and `unsandboxed=True`; `timeout` bounds every `Shell`, `Write` and `Edit` call in seconds, `output_bytes` caps `stdout` and `stderr` each, and `env` is the only environment commands see, by default `PATH`, `HOME` set to the directory, and `LANG`.

Raises `ConfigurationError` when `root` is not a directory, when neither or both of `sandbox` and `unsandboxed=True` are given, when `sandbox` or `shell` is not a non-empty list of strings, when `timeout` or `output_bytes` is not positive, or when an unsandboxed `shell` is not on `PATH`.

### `Workspace.tools`

```python
def tools(self) -> list[Tool]: ...
```

Returns the three tools an agent with `workspace: true` gets: `Shell(command)`, which returns `exit_code`, `stdout`, `stderr` and `timed_out`; `Write(path, content)`, which writes a file exactly; and `Edit(path, old, new)`, which replaces one exact occurrence. `Write` and `Edit` refuse a path outside the directory with an `InputError`.

## Storage

### `Storage`

```python
@runtime_checkable
class Storage(Protocol):
    async def initialize(self) -> None: ...

    async def put_run(self, scope: str, row: RunRow) -> None: ...

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

    async def put_label(self, scope: str, row: LabelRow) -> None: ...

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

The one storage protocol, as the refinery calls it; `InMemoryStorage` and `PostgresStorage` implement exactly these methods. Every method but `initialize` takes `scope` first and reads and writes that scope only: `initialize` creates what the storage needs and is safe to run again; `put_run` and `put_label` store a row, replacing the one with the same id; `get_run` returns a run or `None`; `runs` returns runs newest first by `at`, at most `limit` or all with `None`, narrowed by `run_ids`, `kind`, `since` (stored after it, by `recorded`) and `unlabeled_by`, which keeps the runs no label from that source names; `labels` returns labels oldest first, narrowed by `run_ids`, `source` and `since`; `run_tree` returns the runs `run_ids` or `tags` (by containment) match and every run under them, oldest first; `delete_runs` removes those runs and every label on them, and `delete_scope` every run and label in the scope, each returning the number of rows removed; `count_scope` counts the scope's runs and labels. Both storages raise `StorageError` when a row's scope is not the call's, and `PostgresStorage` raises it whenever the database refuses.

### `InMemoryStorage`

```python
class InMemoryStorage:
    def __init__(self) -> None: ...
```

Implements `Storage` in the process's memory, for tests and small things; `initialize` does nothing, and nothing outlives the object.

### `PostgresStorage`

```python
class PostgresStorage:
    def __init__(self, pool: Any) -> None: ...
```

Implements `Storage` in Postgres, over an asyncpg pool you built. `initialize` creates the tables `engine_runs` and `engine_labels` and their indexes, so one database can hold other libraries' tables beside them; `delete_runs` and `delete_scope` each run in one transaction.

### `RunRow`

```python
class RunRow(BaseModel):
    scope: str
    run_id: str
    parent_run_id: str | None
    agent: str
    kind: str
    task: str | None
    at: datetime
    recorded: datetime
    duration_ms: int
    model: str | None
    body: dict[str, Any]
```

A stored run. `body` is the run's record as JSON with its children and routed run left out, each a row of its own joined by `parent_run_id`; `at` is when the run started and `recorded` when the refinery stored it.

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

### `Refinery.initialize`

```python
async def initialize(self) -> None: ...
```

Initializes the storage: for `PostgresStorage`, creates its tables. Safe to run again.

### `Refinery.record`

```python
async def record(self, run: Run) -> None: ...
```

Stores `run` and every run delegated or routed under it, all stamped with one `recorded` time: what an engine with a refinery does after each run not made with `record=False`.

Raises `InputError` when the run's scope is empty.

### `Refinery.run`

```python
async def run(self, scope: str, run_id: str) -> Run: ...
```

Returns the recorded run as `Engine.run` returned it, its delegated and routed runs back in place; a child's id returns that child and the runs under it.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.review`

```python
async def review(self, scope: str, run_id: str, review: Review) -> Label: ...
```

Stores a person's review of the run and returns the label, with `source="human"`.

Raises `InputError` when `scope` is empty, or when the correction does not fit the run: a dict for a generation run, a string for a decision run, or a question id the run was not asked. Raises `NotFoundError` when the scope has no run `run_id`.

### `Refinery.outcome`

```python
async def outcome(self, scope: str, run_id: str, outcome: Outcome) -> Label: ...
```

Stores what the world reported about the run and returns the label, with `source="world"`.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the scope has no run `run_id`.

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
) -> Exported: ...
```

Writes every run recorded or labeled after `since`, with all its labels, to `bucket` as JSONL under `{scope}/{agent}/{dataset}/{timestamp}.jsonl`, in the datasets `generation`, `preference` and `decision`. Every row carries `holdout`, from a hash of its run id and `holdout_percent`, and its run's `fingerprint` as `{"agent", "tools"}`, `null` for a run recorded before fingerprints were kept. Returns the keys, the rows per dataset and the watermark for the next `since`.

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
) -> Labeled: ...
```

Runs `judge`, a decision agent, over the generation runs it has not labeled, and `teacher`, a generation agent, over the decision runs it has not labeled — at most `limit` runs each, their own runs skipped — and stores each answer as a label. A run that could not be labeled is named in `errors` and stays unlabeled for the next call.

Raises `InputError` when `scope` is empty. Raises `NotFoundError` when the tree has no agent `judge` or `teacher`. Raises `ConfigurationError` when `judge` is not a decision agent or `teacher` not a generation agent; each check happens before any run is labeled.

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
) -> AsyncGenerator[Replay, None]: ...
```

Runs `candidate` on what the top-level runs of `agent` in `scope` were given and yields a `Replay` for each, oldest first by when the run started. `holdout` keeps the held-out runs, by the same hash and `holdout_percent` as `export`; `labeled` the runs with a `human` or `world` label; `tags` the runs whose tags contain these, as `delete_runs` matches; `fingerprint` the runs whose fingerprint equals it, all three parts; `limit` the first so many. Each replay is `candidate.run(agent, scope, …, record=False)` with the recorded input — parts read back as parts — the history (the messages sent before the input), the task and the `context`; a decision run replays its state, its task and its runtime questions as recorded, the rest of its questions coming from the candidate's folder, and when its route reached a generation run, that run's input, history and context. `expect`, `approve`, `tools` and `workspace` are passed to every replay. `Approve` is `Callable[[str, ToolCall], Awaitable[bool]]`; `ToolsLike` is a protocol of `def definitions(self) -> list[dict[str, Any]]` and `async def dispatch(self, name: str, args: dict[str, Any])`, which a `Tools` meets. The candidate's tools run for real. At most `concurrency` replays run at once, and while a pair is being read at most `concurrency - 1` run, so with the default a replay starts when its pair is asked for. Pairs are yielded in that order either way, and closing the generator cancels the replays still running. A candidate that raises yields its pair with `error` and the replay goes on. Nothing is recorded and no label is written.

Raises `InputError` when `scope` is empty, `holdout_percent` is outside 0 to 100, `tags` is empty, or `limit` or `concurrency` is below 1. Raises `NotFoundError` when the candidate has no agent `agent`. Each check happens when `replay` is called, before anything is read; a storage's `StorageError` is raised by the first pair.

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

A tree, a model, a tool or a workspace the engine cannot use, found when it is built or when a run needs it.

### `NotFoundError`

```python
class NotFoundError(EngineException): ...
```

An agent the tree does not have, a judge or a teacher included, a task the agent does not have, or a run the refinery does not.

### `InputError`

```python
class InputError(EngineException): ...
```

A call the library refuses: an empty scope, tags outside their limits, runtime questions that do not fit or that the folder already asks, `questions` given to a generation agent, a decision run with nothing to ask, media a decision model does not declare, a media part with both or neither of `data` and `url`, a correction of the wrong shape, a tool called with bad arguments or a path outside its directory.

### `ModelError`

```python
class ModelError(EngineException): ...
```

An adapter broke the contract: a result whose fields do not fit, a missing answer, probabilities that do not cover the options, lie outside 0 and 1 or do not sum to 1, a score outside its levels.

### `RunawayError`

```python
class RunawayError(EngineException): ...
```

A generation run passed `max_steps` model calls without ending.

### `DelegationLoopError`

```python
class DelegationLoopError(EngineException): ...
```

A run would start an agent already running above it. Through `Delegate` it reaches the model as a tool error; through a route it raises.

### `DelegationDepthError`

```python
class DelegationDepthError(EngineException): ...
```

Nesting went deeper than `max_depth`. Through `Delegate` it reaches the model as a tool error; through a route it raises.

### `StorageError`

```python
class StorageError(EngineException): ...
```

The storage refused: the database failed, or a row's scope is not the scope of the call.
