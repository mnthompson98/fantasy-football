"""Best legal starting lineup, and what a roster change does to it.

Every in-season decision reduces to one question: does this move my starting
lineup? Start/sit asks it about players I already own, waivers asks it about a
player I could add, and a trade asks it about a swap. They share this module so
they cannot drift apart, the same way the board and the backtest share
`features/pipeline.py`.

The optimizer is greedy — fill each dedicated slot with the best players at that
position, then hand the flex to the best remaining eligible player — and greedy
is *optimal* for this slot structure, not merely convenient. Every player is
eligible for exactly one dedicated position plus possibly the flex, so there is
no exchange that improves the total: taking the best N at each position leaves
the best possible remainder for the flex.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FLEX_ELIGIBLE = ("RB", "WR", "TE")


@dataclass(frozen=True)
class LineupSlots:
    """The shape of a legal starting lineup."""

    starters: dict[str, int]
    flex_slots: int = 1
    flex_eligible: tuple[str, ...] = FLEX_ELIGIBLE

    @classmethod
    def from_config(cls, cfg: dict) -> "LineupSlots":
        counts: dict[str, int] = {}
        flex = 0
        for slot in cfg["verified"]["starters"]:
            if slot == "FLEX":
                flex += 1
            else:
                counts[slot] = counts.get(slot, 0) + 1
        return cls(
            starters=counts,
            flex_slots=flex,
            flex_eligible=tuple(cfg["verified"].get("flex_eligible", FLEX_ELIGIBLE)),
        )

    @property
    def size(self) -> int:
        return sum(self.starters.values()) + self.flex_slots

    def labels(self) -> list[str]:
        """Human slot names, in lineup order: QB, RB1, RB2, WR1, ... FLEX."""
        out: list[str] = []
        for pos, n in self.starters.items():
            out.extend([pos] if n == 1 else [f"{pos}{i + 1}" for i in range(n)])
        out.extend(["FLEX"] * self.flex_slots)
        return out


@dataclass
class Lineup:
    """A chosen lineup: who starts, who sits, and what it is projected to score."""

    starters: pd.DataFrame
    bench: pd.DataFrame
    points: float

    def __len__(self) -> int:
        return len(self.starters)


def best_lineup(players: pd.DataFrame, slots: LineupSlots, *,
                points_col: str = "projection",
                position_col: str = "position") -> Lineup:
    """Pick the highest-scoring legal lineup from `players`.

    Missing projections are treated as zero rather than dropped: a player with
    no projection is usually one who is not playing, and starting him really
    does score nothing. Unfilled slots simply do not appear in `starters` — the
    caller decides whether that is a hole to fill off waivers or a bye to
    absorb.
    """
    if players.empty:
        empty = players.copy()
        return Lineup(starters=empty, bench=empty, points=0.0)

    df = players.copy()
    df[points_col] = pd.to_numeric(df[points_col], errors="coerce").fillna(0.0)
    df = df.sort_values(points_col, ascending=False)

    chosen: list[tuple[str, object]] = []   # (slot label, index)
    used: set = set()

    for pos, n in slots.starters.items():
        pool = df[(df[position_col] == pos) & (~df.index.isin(used))].head(n)
        for i, idx in enumerate(pool.index):
            label = pos if n == 1 else f"{pos}{i + 1}"
            chosen.append((label, idx))
            used.add(idx)

    if slots.flex_slots:
        pool = df[
            df[position_col].isin(slots.flex_eligible) & (~df.index.isin(used))
        ].head(slots.flex_slots)
        for idx in pool.index:
            chosen.append(("FLEX", idx))
            used.add(idx)

    starters = df.loc[[idx for _, idx in chosen]].copy()
    starters.insert(0, "slot", [label for label, _ in chosen])
    bench = df[~df.index.isin(used)].copy()

    return Lineup(starters=starters, bench=bench,
                  points=float(starters[points_col].sum()))


def lineup_points(players: pd.DataFrame, slots: LineupSlots, *,
                  points_col: str = "projection") -> float:
    return best_lineup(players, slots, points_col=points_col).points


def value_of_adding(roster: pd.DataFrame, candidate: pd.Series,
                    slots: LineupSlots, *,
                    points_col: str = "projection") -> float:
    """How much a player would add to the starting lineup, this week.

    Zero means he cannot crack the lineup — which is the honest answer for most
    of the waiver wire, and the reason a rolling waiver priority should almost
    never be spent.
    """
    before = lineup_points(roster, slots, points_col=points_col)
    after = lineup_points(
        pd.concat([roster, candidate.to_frame().T], ignore_index=True),
        slots, points_col=points_col)
    return after - before


def value_of_dropping(roster: pd.DataFrame, index, slots: LineupSlots, *,
                      points_col: str = "projection") -> float:
    """What dropping this player would cost the starting lineup, this week.

    The mirror of `value_of_adding`, and the other half of any add/drop.
    """
    before = lineup_points(roster, slots, points_col=points_col)
    after = lineup_points(roster.drop(index=index), slots, points_col=points_col)
    return before - after
