"""Display elapsed time without counting time spent viewing a finished job."""
from datetime import datetime, timezone

from .models import RunStatus


def _timestamp(value: str | None) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def galaxy_terminal_timestamp(job: dict) -> str | None:
    """Use Galaxy's terminal update, never a local refresh or cache timestamp."""
    if str(job.get("state") or "").lower() not in {"ok", "error", "deleted", "deleted_new", "cancelled"}:
        return None
    value = job.get("update_time") or job.get("updated_utc")
    try:
        return _timestamp(value).isoformat()
    except (TypeError, ValueError):
        return None


def run_elapsed_label(record, *, now: datetime | None = None) -> str:
    terminal = (record.analysis_status or record.status) in {
        RunStatus.OK, RunStatus.ERROR, RunStatus.CANCELLED,
    }
    if terminal and not record.galaxy_terminal_utc:
        return "Finished · duration unavailable"
    try:
        start = _timestamp(record.created_utc)
        end = _timestamp(record.galaxy_terminal_utc) if terminal else (now or datetime.now(timezone.utc))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        elapsed = (end - start).total_seconds()
        if elapsed < 0:
            return "Duration unavailable"
        return f"{elapsed:.0f} s from submission" if terminal else f"{elapsed:.0f} s elapsed"
    except (TypeError, ValueError):
        return "Duration unavailable"
