"""One list of every stale-cache fallback taken this process.

Both `nflverse._cached` and `sleeper_api.get_players` prefer yesterday's
cache to a failed pull, and both print a `[warn]` line when they do. The
line scrolls past. A scheduled run that is about to be relayed by something
that did not watch the terminal needs to *ask* afterwards whether anything
it read was stale — `report.freshness_issues` does, through this list.
"""

from __future__ import annotations

FALLBACKS: list[dict] = []


def record(name: str, error: str, age_hours: float) -> None:
    FALLBACKS.append({"name": name, "error": error, "age_hours": age_hours})


def reset() -> None:
    FALLBACKS.clear()
