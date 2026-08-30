"""Pricing a position at what the market pays for it.

Value-based drafting has a well-known failure at kicker and defense. It says the
best defense out-scores the waiver defense by ~12 points over a season, which is
true, and concludes you should spend a sixth-round pick to get him, which is
not. The market disagrees for reasons VORP does not model: those points are
unpredictable in advance, and the position is streamable all year, so the *pick*
buys far less than the *points* suggest.

Rather than invent a discount, an anchored position is moved toward where the
room actually drafts it:

    target_rank = (1 - w) * vorp_rank + w * adp_rank

Weight 1.0 puts a position exactly at its ADP. Weight 0.0 is pure VORP, i.e. no
anchoring at all. In between, a player whose VORP rank is far above his ADP can
still climb — the blend compresses how far the group floats without flattening
the ordering inside it.

**This blends ranks, not values, and that is the whole trick.** The obvious
implementation — replace an anchored player's VORP with the VORP the market
implies for his draft slot — does not work. That value is a conditional *mean*,
so every player scattered above the mean sorts past him and the anchored
position lands systematically below its own ADP rather than at it. Ranks compose
the way the intuition expects; values do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_REFERENCE_POSITIONS = ("QB", "RB", "WR", "TE")

# Anchor weight is capped just below 1.0 so a sliver of the position's own value
# always survives. At exactly 1.0 the ordering *within* an anchored position
# collapses to whatever the ADP list says, which is not what we want: ADP orders
# defenses by market noise, and we would rather break ties by our own valuation.
# At 0.999 the placement is indistinguishable from ADP and Houston still comes
# off the board before Denver.
MAX_ANCHOR = 0.999

# Below this many reference players there is no ladder to interpolate against.
MIN_REFERENCE_PLAYERS = 30


def merge_at_target(free_order: list, anchored: list[tuple[float, object]]
                    ) -> list:
    """Interleave anchored players into the free ordering at their target ranks.

    `free_order` is the un-anchored players, best first. `anchored` is
    `(target_rank, label)` pairs. Walking board positions in order and placing an
    anchored player as soon as the position reaches his target puts him *at*
    that rank in the finished board.

    Doing it by merge rather than by assigning him the value the market implies
    is what makes the placement actually land. Anchoring 67 kickers and defenses
    downward vacates 67 slots near the top, everything below shifts up, and a
    one-shot value assignment lands the whole group ~55 spots high. The merge
    has no such feedback: positions are allocated once, in order.
    """
    out: list = []
    queue = sorted(anchored, key=lambda t: t[0])
    free = list(free_order)
    fi = qi = 0

    while fi < len(free) or qi < len(queue):
        position = len(out) + 1
        take_anchored = qi < len(queue) and (
            queue[qi][0] <= position or fi >= len(free))
        if take_anchored:
            out.append(queue[qi][1])
            qi += 1
        else:
            out.append(free[fi])
            fi += 1
    return out


def apply_market_anchor(board: pd.DataFrame, weights: dict[str, float], *,
                        reference_positions=DEFAULT_REFERENCE_POSITIONS,
                        adp_col: str = "adp_rank",
                        value_col: str = "vorp",
                        raw_col: str = "vorp_unanchored",
                        ) -> pd.DataFrame:
    """Move each anchored position toward where the market drafts it.

    `weights` maps position -> anchor strength in [0, 1]. Positions absent from
    the dict are untouched. The pre-anchor value is preserved in `raw_col` so
    the board stays auditable: you can always see what pure VORP thought before
    the market got a vote.

    `reference_positions` are the ones whose VORP we trust; they define the
    value ladder that anchored players are placed into. Letting an anchored
    position help define its own ladder would be circular.
    """
    out = board.copy()
    out[raw_col] = out[value_col]
    out["market_anchor"] = 0.0

    weights = {p: min(float(w), MAX_ANCHOR)
               for p, w in (weights or {}).items() if float(w) > 0}
    if not weights:
        return out

    ref = out[out["position"].isin(reference_positions)]
    if len(ref) < MIN_REFERENCE_PLAYERS:
        return out

    # Provisional board rank from pure VORP, over everyone.
    vorp_rank = out[value_col].rank(ascending=False, method="first")
    adp_rank = pd.to_numeric(out[adp_col], errors="coerce")
    w = out["position"].map(weights).astype(float).fillna(0.0).clip(0.0, MAX_ANCHOR)

    # A player with no ADP has no market to be anchored to.
    w = w.where(adp_rank.notna(), 0.0)
    target = (1.0 - w) * vorp_rank + w * adp_rank.fillna(vorp_rank)

    out["market_anchor"] = w
    out["target_rank"] = target

    free = out[w <= 0].sort_values(value_col, ascending=False)
    moving = out[w > 0].sort_values("target_rank")
    if free.empty or moving.empty:
        return out

    order = merge_at_target(
        list(free.index), list(zip(moving["target_rank"], moving.index)))
    final_rank = {label: i + 1 for i, label in enumerate(order)}

    # Give each anchored player a value that sorts him where the merge put him.
    # The free players' values are left exactly as they are: their VORP is the
    # real thing, and the board should keep showing it.
    free_positions = np.array([final_rank[i] for i in free.index], dtype=float)
    free_values = free[value_col].to_numpy(dtype=float)

    moved_positions = np.array([final_rank[i] for i in moving.index], dtype=float)
    values = np.interp(moved_positions, free_positions, free_values,
                       left=free_values[0], right=free_values[-1])

    # Consecutive anchored players can interpolate to the same value where the
    # ladder is flat. Nudge by position so the ordering the merge decided is not
    # undone by a tie-break somewhere downstream.
    values = values - 1e-6 * moved_positions

    out.loc[moving.index, value_col] = values
    return out
