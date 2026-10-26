import hashlib

from rfnry_engine.errors import InputError

HOLDOUT_PERCENT = 20


def is_holdout(run_id: str, *, holdout_percent: int = HOLDOUT_PERCENT) -> bool:
    if not 0 <= holdout_percent <= 100:
        raise InputError(f"refinery/holdout: holdout_percent must be 0-100, got {holdout_percent}")
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100 < holdout_percent
