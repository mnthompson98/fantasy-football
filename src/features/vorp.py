"""Value over replacement (VORP) for value-based drafting.

The replacement level for a position is the projected points of the player who
would still be freely available if every team filled that position's starting
slots. That baseline is what makes cross-position comparison meaningful: a 260
point QB in a 1-QB league is worth far less than a 260 point RB, because the
QB you'd otherwise stream is much closer to 260 than the RB you'd otherwise start.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FLEX_ELIGIBLE = ("RB", "WR", "TE")


@dataclass(frozen=True)
class LeagueShape:
    teams: int
    starters: dict[str, int]          # e.g. {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
    flex_slots: int = 1
    flex_eligible: tuple[str, ...] = FLEX_ELIGIBLE

    @classmethod
    def from_config(cls, cfg: dict) -> "LeagueShape":
        slots = cfg["verified"]["starters"]
        counts: dict[str, int] = {}
        flex = 0
        for s in slots:
            if s == "FLEX":
                flex += 1
            else:
                counts[s] = counts.get(s, 0) + 1
        teams = cfg.get("assumed_2026", {}).get("teams") or cfg["verified"].get("teams", 10)
        return cls(
            teams=teams,
            starters=counts,
            flex_slots=flex,
            flex_eligible=tuple(cfg["verified"].get("flex_eligible", FLEX_ELIGIBLE)),
        )


def replacement_ranks(shape: LeagueShape, projections: pd.DataFrame,
                      points_col: str = "projection") -> dict[str, int]:
    """How many players at each position are 'starter-worthy' league-wide.

    Dedicated slots are trivial: teams * slots. Flex is not — flex demand is
    allocated to whichever eligible positions actually supply the best remaining
    players, which we resolve by walking the merged pool rather than splitting
    the flex evenly. Splitting evenly is the common shortcut and it misprices
    RB in PPR, where the flex is overwhelmingly RB/WR.
    """
    base = {pos: shape.teams * n for pos, n in shape.starters.items()}

    total_flex = shape.teams * shape.flex_slots
    if total_flex <= 0:
        return base

    # Walk the pool of flex-eligible players below their dedicated cutoffs and
    # award flex demand to whoever is actually next-best.
    pool = []
    for pos in shape.flex_eligible:
        cutoff = base.get(pos, 0)
        sub = projections[projections["position"] == pos]
        sub = sub.sort_values(points_col, ascending=False).iloc[cutoff:]
        for pts in sub[points_col].tolist():
            pool.append((pts, pos))

    pool.sort(reverse=True)
    for _, pos in pool[:total_flex]:
        base[pos] = base.get(pos, 0) + 1

    return base


def compute_vorp(projections: pd.DataFrame, shape: LeagueShape,
                 points_col: str = "projection") -> pd.DataFrame:
    """Attach `replacement_points` and `vorp` columns.

    Expects columns: `position`, `points_col`. Returns a copy sorted by VORP.
    """
    required = {"position", points_col}
    missing = required - set(projections.columns)
    if missing:
        raise ValueError(f"projections missing required columns: {sorted(missing)}")

    df = projections.copy()
    ranks = replacement_ranks(shape, df, points_col)

    repl: dict[str, float] = {}
    for pos, cutoff in ranks.items():
        sub = df[df["position"] == pos].sort_values(points_col, ascending=False)
        if sub.empty:
            repl[pos] = 0.0
        elif len(sub) > cutoff:
            repl[pos] = float(sub.iloc[cutoff][points_col])
        else:
            # Thinner pool than starting demand — use the worst rostered player
            # rather than 0, which would wildly inflate VORP for that position.
            repl[pos] = float(sub.iloc[-1][points_col])

    df["replacement_points"] = df["position"].map(repl).fillna(0.0)
    df["vorp"] = df[points_col] - df["replacement_points"]
    return df.sort_values("vorp", ascending=False).reset_index(drop=True)
