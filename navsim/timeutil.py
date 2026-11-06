"""One timestamp parser for every external source.

Planet writes acquisition times with however many fractional-second digits it
likes -- ``2026-08-18T16:31:09.82394Z`` -- and Python 3.10's
``datetime.fromisoformat`` accepts only three or six.  Before this module the
failure was swallowed and the scene was stamped with the *current* time, so
seven of thirty Norfolk dates silently became "today", scrambling the berth
timeline.  Everything that reads a timestamp now comes through here, and an
unparseable one is an error, never a silent substitute.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

_FRAC = re.compile(r"(\.\d+)")


def parse_time(raw) -> datetime | None:
    """Parse an ISO-8601-ish timestamp into an aware UTC datetime."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00").replace("z", "+00:00")
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    m = _FRAC.search(s)
    if m:
        frac = m.group(1)[1:]
        s = s[:m.start()] + "." + (frac + "000000")[:6] + s[m.end():]
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


#: Russian month abbreviations, independent of the system locale.
RU_MONTHS = ("янв", "фев", "мар", "апр", "мая", "июн",
             "июл", "авг", "сен", "окт", "ноя", "дек")
#: Nominative forms for axis ticks (a month on its own, not "5 мая").
RU_MONTHS_NOM = ("янв", "фев", "мар", "апр", "май", "июн",
                 "июл", "авг", "сен", "окт", "ноя", "дек")


def ru_date(dt: datetime, fmt: str = "%d %b %Y") -> str:
    """``strftime`` with ``%b`` replaced by a Russian month abbreviation.

    Never relies on the system locale, so it reads the same everywhere.
    """
    return dt.strftime(fmt.replace("%b", RU_MONTHS[dt.month - 1]))
