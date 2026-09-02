"""Sleeper API client.

Read-only, no auth, no API key. Sleeper's documented guidance is to stay under
1000 requests/minute; we stay far below that and cache aggressively.

Reference: https://docs.sleeper.com
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE = "https://api.sleeper.app/v1"

# Sleeper allows ~1000/min. We cap far lower: nothing here is latency-critical
# except the live draft poll, which is a single endpoint on a fixed interval.
_MIN_INTERVAL_S = 0.08

_last_call = 0.0


def _throttle() -> None:
    global _last_call
    delta = time.monotonic() - _last_call
    if delta < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - delta)
    _last_call = time.monotonic()


# Transient failures worth one more try: rate limiting, server errors, and
# network timeouts. A 4xx other than 429 is a real answer and is not retried.
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


def _get(path: str, *, timeout: float = 15.0,
         attempts: int = _MAX_ATTEMPTS) -> Any:
    """GET a Sleeper endpoint. Returns parsed JSON, or None for an empty body.

    Sleeper returns `null` (not 404) for a valid-shaped request with no data —
    e.g. a user with no drafts in a season. Callers must handle None.

    Retries with backoff on 429 / 5xx / timeouts, honouring `Retry-After`.
    Three attempts is enough to ride out a hiccup and short enough that the
    live draft poll (5s interval) still notices a real outage promptly.
    """
    url = f"{BASE}/{path.lstrip('/')}"
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        _throttle()
        try:
            resp = requests.get(url, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt == attempts:
                raise
            time.sleep(min(8.0, 0.5 * 2 ** attempt))
            continue
        if resp.status_code in _RETRY_STATUSES and attempt < attempts:
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else 0.5 * 2 ** attempt
            except ValueError:
                wait = 0.5 * 2 ** attempt
            time.sleep(min(8.0, wait))
            continue
        resp.raise_for_status()
        if not resp.content:
            return None
        return resp.json()
    raise RuntimeError(f"sleeper: {url} failed after {attempts} attempts") \
        from last_exc


def epoch_ms_to_iso(ms: int | None) -> str | None:
    """Sleeper returns epoch milliseconds. Repo convention is UTC ISO 8601."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Users, leagues, drafts
# --------------------------------------------------------------------------

def get_user(username_or_id: str) -> dict | None:
    return _get(f"user/{username_or_id}")


def get_user_leagues(user_id: str, season: int, sport: str = "nfl") -> list[dict]:
    return _get(f"user/{user_id}/leagues/{sport}/{season}") or []


def get_user_drafts(user_id: str, season: int, sport: str = "nfl") -> list[dict]:
    return _get(f"user/{user_id}/drafts/{sport}/{season}") or []


def get_league(league_id: str) -> dict | None:
    return _get(f"league/{league_id}")


def get_league_users(league_id: str) -> list[dict]:
    return _get(f"league/{league_id}/users") or []


def get_rosters(league_id: str) -> list[dict]:
    return _get(f"league/{league_id}/rosters") or []


def get_matchups(league_id: str, week: int) -> list[dict]:
    return _get(f"league/{league_id}/matchups/{week}") or []


def get_winners_bracket(league_id: str) -> list[dict]:
    return _get(f"league/{league_id}/winners_bracket") or []


def get_losers_bracket(league_id: str) -> list[dict]:
    return _get(f"league/{league_id}/losers_bracket") or []


def get_transactions(league_id: str, week: int) -> list[dict]:
    return _get(f"league/{league_id}/transactions/{week}") or []


def get_draft(draft_id: str) -> dict | None:
    return _get(f"draft/{draft_id}")


def get_draft_picks(draft_id: str) -> list[dict]:
    return _get(f"draft/{draft_id}/picks") or []


def get_trending(kind: str = "add", lookback_hours: int = 24, limit: int = 50,
                 sport: str = "nfl") -> list[dict]:
    """Waiver-demand signal. `kind` is 'add' or 'drop'."""
    if kind not in ("add", "drop"):
        raise ValueError("kind must be 'add' or 'drop'")
    return _get(
        f"players/{sport}/trending/{kind}"
        f"?lookback_hours={lookback_hours}&limit={limit}"
    ) or []


# --------------------------------------------------------------------------
# Player map (large — cache hard)
# --------------------------------------------------------------------------

def get_players(sport: str = "nfl", cache_dir: Path | None = None,
                max_age_hours: float = 12.0) -> dict[str, dict]:
    """Full Sleeper player map (~5MB).

    Sleeper's docs ask that this be called at most once per day. We cache to
    disk and only refetch past `max_age_hours`. Never call this mid-draft —
    load it before the draft starts.

    A stale cache beats a failed pull, for the same reason `nflverse._cached`
    says so: losing the board on draft morning because Sleeper is briefly
    down is worse than a board built on yesterday's injury statuses. The
    fallback is printed, never silent.
    """
    cache: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"sleeper_players_{sport}.json"
        if cache.exists():
            age_h = (time.time() - cache.stat().st_mtime) / 3600
            if age_h < max_age_hours:
                return json.loads(cache.read_text(encoding="utf-8"))

    try:
        data = _get(f"players/{sport}") or {}
    except Exception as exc:
        if cache is not None and cache.exists():
            age_h = (time.time() - cache.stat().st_mtime) / 3600
            print(f"  [warn] sleeper players: pull failed ({exc}); using "
                  f"cache aged {age_h:.1f}h")
            return json.loads(cache.read_text(encoding="utf-8"))
        raise

    if cache is not None:
        cache.write_text(json.dumps(data), encoding="utf-8")
    return data


# Sleeper injury_status values that make a player undraftable / unclaimable.
# A hard gate, not a discount — never spend a pick or a waiver claim here.
UNAVAILABLE_STATUSES = {"IR", "Out", "PUP", "Sus", "NA", "DNR"}


def is_unavailable(player: dict) -> bool:
    status = (player.get("injury_status") or "").strip()
    return status in UNAVAILABLE_STATUSES


# --------------------------------------------------------------------------
# Derived: standings and final placements
# --------------------------------------------------------------------------

@dataclass
class TeamSeason:
    roster_id: int
    owner_id: str | None
    display_name: str | None
    wins: int
    losses: int
    ties: int
    points_for: float
    points_against: float
    regular_season_seed: int | None = None
    final_rank: int | None = None
    made_playoffs: bool = False


def _pts(whole: Any, decimal: Any) -> float:
    """Sleeper splits points into integer and decimal parts."""
    return float(whole or 0) + float(decimal or 0) / 100.0


def get_standings(league_id: str) -> list[TeamSeason]:
    """Regular-season standings, sorted by wins then points-for.

    Sleeper's own tiebreak is wins -> points_for, which is what we replicate.
    """
    rosters = get_rosters(league_id)
    users = {u["user_id"]: u for u in get_league_users(league_id)}

    teams = []
    for r in rosters:
        s = r.get("settings", {}) or {}
        owner = users.get(r.get("owner_id"), {})
        teams.append(TeamSeason(
            roster_id=r["roster_id"],
            owner_id=r.get("owner_id"),
            display_name=owner.get("display_name"),
            wins=int(s.get("wins", 0)),
            losses=int(s.get("losses", 0)),
            ties=int(s.get("ties", 0)),
            points_for=_pts(s.get("fpts"), s.get("fpts_decimal")),
            points_against=_pts(s.get("fpts_against"), s.get("fpts_against_decimal")),
        ))

    teams.sort(key=lambda t: (-t.wins, -t.points_for))
    for i, t in enumerate(teams, start=1):
        t.regular_season_seed = i
    return teams


def resolve_final_standings(league_id: str) -> list[TeamSeason]:
    """Full final ordering, with playoff placements resolved from the bracket.

    Sleeper's winners bracket carries a `p` field on placement matches: `p=1` is
    the championship game, `p=3` the third-place game, and so on. The winner
    takes rank `p`, the loser `p+1`. Teams absent from the bracket keep their
    regular-season order beneath the playoff field.
    """
    teams = get_standings(league_id)
    by_roster = {t.roster_id: t for t in teams}

    placed: dict[int, int] = {}
    for match in get_winners_bracket(league_id):
        p = match.get("p")
        if p is None:
            continue
        if match.get("w") is not None:
            placed[int(match["w"])] = int(p)
        if match.get("l") is not None:
            placed[int(match["l"])] = int(p) + 1

    for roster_id, rank in placed.items():
        if roster_id in by_roster:
            by_roster[roster_id].final_rank = rank
            by_roster[roster_id].made_playoffs = True

    # Non-playoff teams fill the remaining ranks in regular-season order.
    next_rank = (max(placed.values()) + 1) if placed else 1
    for t in teams:
        if t.final_rank is None:
            t.final_rank = next_rank
            next_rank += 1

    teams.sort(key=lambda t: t.final_rank or 999)
    return teams


def get_champion(league_id: str) -> TeamSeason | None:
    finals = resolve_final_standings(league_id)
    return finals[0] if finals else None
