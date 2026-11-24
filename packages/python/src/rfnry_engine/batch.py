import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from rfnry_engine.models import Refused, ToolCall
from rfnry_engine.records import Step

STOPPED = "stopped by the user"

logger = logging.getLogger("rfnry_engine.engine")


def _now() -> datetime:
    return datetime.now(UTC)


def _refusal(name: str, answer: bool | Refused) -> str:
    said = f"NotApproved: {name} was not approved"
    reason = answer.reason.strip() if isinstance(answer, Refused) else ""
    return f"{said}: {reason}" if reason else said


def _raise_first(failures: Sequence[BaseException]) -> None:
    for error in failures[1:]:
        logger.error("engine: a further call of the same batch failed: %s", error, exc_info=error)
    if failures:
        raise failures[0]


def unrun(call: ToolCall, why: str) -> Step:
    return Step(
        tool=call.name,
        args=call.args,
        result=f"NotRun: {call.name} was not run; {why}",
        is_error=True,
        at=_now(),
        duration_ms=0,
    )
