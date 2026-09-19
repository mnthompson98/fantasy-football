"""Who the rest of the league just dropped, and what that costs to pick up.

The waiver report already evaluates everyone unrostered, so a player another
team dropped on Wednesday is *in* the candidate pool by construction. What
it does not say is that he was just dropped — which matters twice over:

- **It is the freshest information on the wire.** A manager cut him to make
  room for a claim; that is a decision about him made this week by someone
  who watched him play, and it is worth a separate look rather than a line
  buried among 400 free agents.
- **He is not free yet.** Under this league's settings a dropped player sits
  on waivers for `waiver_clear_days` (2) before becoming a free agent, so for
  two days after Wednesday's processing picking him up is a *claim* — it
  costs your place in the queue — and after that it is free. The date that
  flips is the whole question, so it is on every row.

Sleeper's transaction log (`/league/<id>/transactions/<leg>`) is the source.
It is indexed by `leg` (NFL week) and a Wednesday-morning processing run
lands in the leg that just ended or the one starting, so both are read.
Only `status == "complete"` counts; a failed claim dropped nobody.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from ..ingest.player_ids import normalize_position
from ..ingest.sleeper_api import get_transactions

COLUMNS = ["player_key", "player_id", "player_name", "position", "nfl_team",
           "dropped_by", "dropped_at", "via", "on_waivers_until"]


def _utc(ms) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def drops_from(transactions: list[dict], *, since: datetime,
               waiver_clear_days: int, owner_names: dict[int, str],
               my_roster_id: int | None = None) -> list[dict]:
    """Every completed drop at or after `since`, as plain dicts.

    Pure — takes the payload, returns rows keyed by Sleeper `player_id` —
    so the shape of what Sleeper returns can be pinned by a test. Drops by
    our own roster are left out: you know what you dropped.
    """
    rows: list[dict] = []
    for tx in transactions:
        if str(tx.get("status")) != "complete":
            continue
        when = _utc(tx.get("status_updated") or tx.get("created"))
        if when is None or when < since:
            continue
        for player_id, roster_id in (tx.get("drops") or {}).items():
            if my_roster_id is not None and roster_id == my_roster_id:
                continue
            rows.append({
                "player_id": str(player_id),
                "dropped_by": owner_names.get(roster_id, str(roster_id)),
                "dropped_at": when,
                "via": str(tx.get("type") or "unknown"),
                "on_waivers_until": when + timedelta(days=int(waiver_clear_days)),
            })
    return rows


def recent_drops(league_id: str, legs: list[int], cw, *,
                 since: datetime, waiver_clear_days: int,
                 owner_names: dict[int, str],
                 my_roster_id: int | None = None,
                 still_rostered: set | None = None) -> pd.DataFrame:
    """Players other teams dropped since `since`, resolved to `player_key`.

    A player dropped and then picked up again by someone else is not
    available, so `still_rostered` (the league's current roster set) removes
    him. A dropped player the crosswalk cannot resolve is kept under a
    synthetic key and reported by name — nothing is dropped silently.
    """
    raw: list[dict] = []
    for leg in sorted(set(int(l) for l in legs if l and l > 0)):
        try:
            raw += get_transactions(league_id, leg) or []
        except Exception as exc:
            print(f"  [warn] transactions leg {leg}: {exc}")
    rows = drops_from(raw, since=since, waiver_clear_days=waiver_clear_days,
                      owner_names=owner_names, my_roster_id=my_roster_id)

    out: list[dict] = []
    seen: set[str] = set()
    for r in sorted(rows, key=lambda r: r["dropped_at"], reverse=True):
        key = cw.resolve(sleeper_id=r["player_id"]) or f"SLEEPER_{r['player_id']}"
        if key in seen:
            continue                        # keep the most recent drop only
        seen.add(key)
        if still_rostered and key in still_rostered:
            continue
        meta = cw.meta.get(key, {})
        out.append({
            "player_key": key,
            "player_id": r["player_id"],
            "player_name": meta.get("player_name") or r["player_id"],
            "position": normalize_position(meta.get("position")) or "?",
            "nfl_team": meta.get("nfl_team"),
            **{k: r[k] for k in ("dropped_by", "dropped_at", "via",
                                 "on_waivers_until")},
        })
    return pd.DataFrame(out, columns=COLUMNS)


def annotate(drops: pd.DataFrame, moves: list, *,
             stream_threshold: float,
             now: datetime | None = None) -> pd.DataFrame:
    """Join each drop to its waiver evaluation and say what a pickup costs
    *right now*: a claim while he is still on waivers, free after.

    `moves` is `waivers.evaluate()` over the dropped players against your
    roster, run with the stream floor disabled so that every one of them
    gets a number — the floor is re-applied here. "Worth it" then means:
    clears the priority bar if he is still on waivers, or clears the stream
    floor if he has become a free agent. A dropped player who does not
    improve your lineup is the answer for most of them, and is printed as
    such rather than left out.
    """
    now = now or datetime.now(timezone.utc)
    by_key = {m.add_key: m for m in moves}
    rows = []
    for r in drops.itertuples(index=False):
        m = by_key.get(r.player_key)
        on_waivers = r.on_waivers_until is not None and now < r.on_waivers_until
        rows.append({
            **r._asdict(),
            "net": m.net if m else 0.0,
            "add_points": m.add_points if m else float("nan"),
            "drop": m.drop if m else None,
            "on_waivers": on_waivers,
            "burns_priority": bool(m and m.burns_priority),
            "worth_it": bool(m) and m.net >= stream_threshold
                        and (m.burns_priority or not on_waivers),
        })
    df = pd.DataFrame(rows, columns=list(drops.columns) + [
        "net", "add_points", "drop", "on_waivers", "burns_priority", "worth_it"])
    return df.sort_values("net", ascending=False, kind="stable")


def describe(row) -> str:
    """One line per dropped player, in the language the decision is made in."""
    when = row.dropped_at.strftime("%a %H:%M UTC") if row.dropped_at else "?"
    until = (row.on_waivers_until.strftime("%a %H:%M UTC")
             if row.on_waivers_until is not None else "?")
    cost = (f"on waivers until {until} — a CLAIM (burns priority)"
            if row.on_waivers else "cleared waivers — free agent")
    pts = f"{row.add_points:.1f}" if row.add_points == row.add_points else "unranked"
    verdict = ("worth it" if row.worth_it
               else ("not worth priority" if row.on_waivers and row.net > 0
                     else "does not improve the lineup"))
    line = (f"  {row.position:<4}{str(row.player_name)[:24]:<25}"
            f"proj {pts:>5}  lineup {row.net:+5.1f}  {verdict}")
    line += f"\n       dropped {when} by {row.dropped_by} ({row.via}); {cost}"
    if row.drop and row.net > 0:
        line += f"; drop {row.drop}"
    return line
