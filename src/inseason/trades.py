"""Evaluating a trade by what it does to your starting lineup.

The common way to value a trade is to add up player rankings on each side. That
is wrong in a way that costs real points: it ignores your roster. Trading your
third running back for someone's second receiver is a downgrade on a ranking
sheet and an upgrade in your lineup if you start two receivers and are thin
there.

So a trade is scored the same way everything else in-season is — the change in
the best legal starting lineup you can field, before and after. Two horizons,
because they disagree and the disagreement is the interesting part:

- **Rest of season**, which is what most trades are about.
- **Playoff weeks 15-17**, which is what actually decides the league and which
  CLAUDE.md names as the primary metric. A player on bye in week 16, or on a
  team that rests starters, is worth less than his season line suggests.

Depth is charged for honestly. Giving up your only backup at a position you
start two of is a real cost even when he never starts, because one injury turns
a bench body into a starter. `depth_penalty` prices that rather than pretending
it is free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..features.lineup import LineupSlots, lineup_points

# Weeks that decide the league. From `verified.playoff_weeks`.
DEFAULT_PLAYOFF_WEEKS = (15, 16, 17)

# What a spare body at a started position is worth, as a fraction of his own
# projection, purely as insurance. Not a measured number — see the module note
# in HANDOFF.md about in-season work being unvalidated.
DEPTH_INSURANCE = 0.15


@dataclass
class TradeSide:
    """One direction of a trade."""

    players: pd.DataFrame
    label: str = ""


@dataclass
class TradeVerdict:
    weekly_before: float
    weekly_after: float
    depth_cost_out: float
    depth_cost_in: float
    positions_thinned: list[str] = field(default_factory=list)

    @property
    def lineup_delta(self) -> float:
        return self.weekly_after - self.weekly_before

    @property
    def net(self) -> float:
        return self.lineup_delta - self.depth_cost_out + self.depth_cost_in

    def describe(self) -> str:
        verdict = "ACCEPT" if self.net > 0 else "DECLINE"
        lines = [
            f"  {verdict}   net {self.net:+.1f} projected points per week",
            f"    starting lineup  {self.weekly_before:.1f} -> "
            f"{self.weekly_after:.1f}  ({self.lineup_delta:+.1f})",
            f"    depth given up   -{self.depth_cost_out:.1f}   "
            f"depth received +{self.depth_cost_in:.1f}",
        ]
        if self.positions_thinned:
            lines.append(f"    leaves you thin at: "
                         f"{', '.join(self.positions_thinned)}")
        return "\n".join(lines)


def _depth_cost(roster: pd.DataFrame, moving: pd.DataFrame,
                slots: LineupSlots, *, points_col: str) -> tuple[float, list[str]]:
    """What losing these players costs in insurance, and where it leaves holes.

    A player only carries insurance value if the roster still starts someone at
    his position; beyond the point where you have no spare, the next departure
    is charged as thinness rather than as depth.
    """
    if moving.empty:
        return 0.0, []

    cost = 0.0
    thin: list[str] = []
    for pos, group in moving.groupby("position"):
        started = slots.starters.get(str(pos), 0)
        if str(pos) in slots.flex_eligible:
            started += slots.flex_slots
        if started == 0:
            continue
        have = int((roster["position"] == pos).sum())
        spare_after = have - len(group) - started
        cost += DEPTH_INSURANCE * float(
            pd.to_numeric(group[points_col], errors="coerce").fillna(0.0).sum())
        if spare_after < 0:
            thin.append(str(pos))
    return cost, thin


def evaluate(roster: pd.DataFrame, giving: pd.DataFrame, getting: pd.DataFrame,
             slots: LineupSlots, *, points_col: str = "projection"
             ) -> TradeVerdict:
    """Score a proposed trade against your actual roster.

    `giving` and `getting` are frames of the same shape as `roster`, carrying at
    least `player_key`, `position` and the projection column.
    """
    before = lineup_points(roster, slots, points_col=points_col)

    keep = roster[~roster["player_key"].isin(set(giving.get("player_key", [])))]
    after_roster = pd.concat([keep, getting], ignore_index=True)
    after = lineup_points(after_roster, slots, points_col=points_col)

    cost_out, thin = _depth_cost(roster, giving, slots, points_col=points_col)
    cost_in, _ = _depth_cost(after_roster, getting, slots, points_col=points_col)

    return TradeVerdict(
        weekly_before=before, weekly_after=after,
        depth_cost_out=cost_out, depth_cost_in=cost_in,
        positions_thinned=thin,
    )


def find_surplus(roster: pd.DataFrame, slots: LineupSlots, *,
                 points_col: str = "projection") -> pd.DataFrame:
    """Players whose absence would not change your starting lineup.

    These are what you have to trade *from*: real value to another roster,
    none to yours. Sorted by projection, so the most attractive trade bait
    comes first.
    """
    rows = []
    base = lineup_points(roster, slots, points_col=points_col)
    for idx in roster.index:
        without = lineup_points(roster.drop(index=idx), slots,
                                points_col=points_col)
        if base - without <= 1e-9:
            rows.append(idx)
    out = roster.loc[rows].copy()
    if out.empty:
        return out
    return out.sort_values(points_col, ascending=False)
