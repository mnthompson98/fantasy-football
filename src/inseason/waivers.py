"""Waiver decisions under **rolling priority** — not FAAB.

This is the part of the league most systems get wrong, because most systems
assume FAAB. Under FAAB a claim costs money and you can spend a little on a
marginal player. Under rolling priority a successful claim costs your *position
in the queue*: you drop to last and stay there until everyone else has claimed.
There is no small bid. Every claim is all-in on that one player.

Two consequences drive everything here:

1. **A claim must clear a high bar, not a positive one.** A player who improves
   this week's lineup by two points is not worth surrendering the ability to
   claim the next league-winner. The threshold is deliberately high and lives in
   `config: waivers.priority_threshold`.

2. **Streaming is free; claiming is not.** Most weeks the useful move is a
   kicker, defense or bye-week fill from the *free-agent* pool, which costs no
   priority at all. Those are surfaced separately and never described as claims.

`src/features/lineup.py` decides value the same way start/sit and the draft do:
what does this player add to the best legal starting lineup? For a waiver
pickup that is usually zero, and saying zero is the entire point.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..features.lineup import LineupSlots, lineup_points, value_of_dropping

# A claim that costs your priority should buy a starter, not a bench body.
# Expressed in projected points added to *this week's* starting lineup.
DEFAULT_PRIORITY_THRESHOLD = 4.0

# Below this, an add is not worth a roster spot either.
DEFAULT_STREAM_THRESHOLD = 0.5


@dataclass
class Move:
    """A suggested add, and what it would cost."""

    add: str
    add_key: str
    position: str
    add_points: float
    gain: float
    drop: str | None
    drop_key: str | None
    drop_cost: float
    burns_priority: bool
    trending_adds: int = 0

    @property
    def net(self) -> float:
        """Lineup improvement after accounting for whoever gets cut."""
        return self.gain - self.drop_cost

    def describe(self) -> str:
        kind = "CLAIM (burns priority)" if self.burns_priority else "free agent"
        line = (f"  {self.position:<4}{self.add[:24]:<25}"
                f"proj {self.add_points:>5.1f}  lineup +{self.net:>5.1f}  {kind}")
        if self.drop:
            line += f"\n       drop {self.drop[:24]} (costs {self.drop_cost:.1f})"
        return line


def _droppable(roster: pd.DataFrame, slots: LineupSlots, *,
               points_col: str) -> pd.DataFrame:
    """Roster players ordered by how little dropping them would cost.

    Cost is measured against the starting lineup, not against the player's own
    projection: a good player at a position you are three deep in costs nothing
    to lose, and a mediocre one plugging a hole costs a lot.

    Every bench player costs exactly zero by that measure, so the tie-break
    matters: among zero-cost players, drop the lowest projection. Without it
    the suggestion was whichever bench player happened to sort first, which
    could be your best handcuff.
    """
    rows = []
    for idx in roster.index:
        rows.append({
            "index": idx,
            "player_key": roster.at[idx, "player_key"],
            "player_name": roster.at[idx, "player_name"],
            "position": roster.at[idx, "position"],
            "cost": value_of_dropping(roster, idx, slots, points_col=points_col),
            "projection": pd.to_numeric(
                pd.Series([roster.at[idx, points_col]]),
                errors="coerce").fillna(0.0).iloc[0],
        })
    return pd.DataFrame(rows).sort_values(["cost", "projection"],
                                          kind="stable")


def evaluate(roster: pd.DataFrame, candidates: pd.DataFrame,
             slots: LineupSlots, *,
             points_col: str = "projection",
             priority_threshold: float = DEFAULT_PRIORITY_THRESHOLD,
             stream_threshold: float = DEFAULT_STREAM_THRESHOLD,
             roster_size: int | None = None,
             trending: dict[str, int] | None = None,
             top_n: int = 8) -> list[Move]:
    """Rank available players by what they would actually add.

    Each candidate is paired with the cheapest player to drop for him, and
    scored on the *net* change to the starting lineup. A move only earns the
    "claim" label — i.e. is worth burning priority — if that net clears
    `priority_threshold`.
    """
    if roster.empty or candidates.empty:
        return []

    trending = trending or {}
    base = lineup_points(roster, slots, points_col=points_col)
    drops = _droppable(roster, slots, points_col=points_col)
    roster_full = roster_size is not None and len(roster) >= roster_size

    moves: list[Move] = []
    for _, cand in candidates.iterrows():
        proj = pd.to_numeric(pd.Series([cand.get(points_col)]),
                             errors="coerce").fillna(0.0).iloc[0]
        with_add = pd.concat([roster, cand.to_frame().T], ignore_index=True)
        gain = lineup_points(with_add, slots, points_col=points_col) - base

        drop_row = None
        drop_cost = 0.0
        if roster_full and not drops.empty:
            # Never cut someone who costs more than the addition is worth.
            drop_row = drops.iloc[0]
            drop_cost = float(drop_row["cost"])

        net = gain - drop_cost
        if net < stream_threshold:
            continue

        moves.append(Move(
            add=str(cand.get("player_name")),
            add_key=str(cand.get("player_key")),
            position=str(cand.get("position")),
            add_points=float(proj),
            gain=float(gain),
            drop=str(drop_row["player_name"]) if drop_row is not None else None,
            drop_key=str(drop_row["player_key"]) if drop_row is not None else None,
            drop_cost=drop_cost,
            burns_priority=net >= priority_threshold,
            trending_adds=int(trending.get(str(cand.get("player_key")), 0)),
        ))

    moves.sort(key=lambda m: -m.net)
    return moves[:top_n]


def summarize(moves: list[Move], *, threshold: float) -> str:
    """A verdict, in the language the league actually works in."""
    if not moves:
        return ("Nothing on the wire improves the lineup. Keep your priority — "
                "that is the correct move most weeks.")

    claims = [m for m in moves if m.burns_priority]
    if not claims:
        best = moves[0]
        return (f"No claim worth your priority (best is {best.add} at "
                f"+{best.net:.1f}, threshold {threshold:.1f}). "
                f"Take the free-agent adds instead — they cost nothing.")

    best = claims[0]
    return (f"One claim clears the bar: {best.add} at +{best.net:.1f} to the "
            f"starting lineup. Claiming drops you to last in the queue, so this "
            f"is the week to spend it only if you agree he is that good.")
