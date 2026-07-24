"""Shared UTC-storage / JST-presentation helpers for Anchor timestamps."""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")


def parse_stored_utc(value: str) -> datetime:
    """Parse Anchor's historical naive-UTC or an offset-aware ISO timestamp."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def utc_storage_to_jst(value: str) -> str:
    """Render a stored UTC instant as a naive JST ISO string for local UI."""
    if not value:
        return value
    local = parse_stored_utc(value).astimezone(JST).replace(tzinfo=None)
    return local.isoformat()


def jst_input_to_utc_storage(value: str) -> str:
    """Convert a JST editor value to Anchor's existing naive-UTC storage form."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("empty timestamp")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JST)
    utc = dt.astimezone(UTC).replace(tzinfo=None)
    return utc.isoformat()


def jst_today() -> str:
    return datetime.now(JST).date().isoformat()


def jst_day_bounds_utc(date_str: str) -> tuple[str, str]:
    """Return [start,end) as naive UTC strings for one JST calendar day."""
    day = date.fromisoformat(date_str)
    start = datetime.combine(day, time.min, tzinfo=JST).astimezone(UTC)
    end = start + timedelta(days=1)
    return (
        start.replace(tzinfo=None).isoformat(),
        end.replace(tzinfo=None).isoformat(),
    )


def jst_range_bounds_utc(start_date: str, end_date: str) -> tuple[str, str]:
    """Return UTC bounds for an inclusive JST date range."""
    start, _ = jst_day_bounds_utc(start_date)
    _, end = jst_day_bounds_utc(end_date)
    return start, end
