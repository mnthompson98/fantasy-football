"""Positional rank -> fantasy points curves.

The expert consensus (ECR) is a *ranking*. VORP needs *points*. This module
fits the bridge: for each position, what did the player who finished Nth
actually score, averaged over recent seasons?

    WR1 ~ 350 pts, WR12 ~ 250, WR36 ~ 165, WR60 ~ 110

Two things about this curve are deliberate and easy to get wrong:

**It is fit on finish rank, not draft rank.** The player who finishes RB1 scores
much more than the player drafted RB1 does, because finish rank is chosen with
hindsight. Mapping a *projected* rank onto a *finish*-rank curve therefore
produces spread that is systematically too wide — which is precisely the bias
`src/features/calibration.py` exists to shrink. The two are designed to be used
together; using this curve without the calibration step will overvalue the top
of every position.

**It is monotone by construction.** Raw averages at adjacent ranks cross each
other from sample noise, and a curve where RB18 outscores RB17 makes the board
non-monotone in a way that is confusing rather than informative. Isotonic
regression enforces the ordering without imposing a functional form.

Seasons can be downweighted — the 2024 kickoff rule made that year's K and DEF
distributions non-transferable (CLAUDE.md), so `config/league.yaml` gives them
a weight of 0.3 rather than dropping them entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Fantasy seasons run weeks 1-17. Week 18 is real football and irrelevant here.
FANTASY_WEEKS = tuple(range(1, 18))

# Below this many observed seasons a position's curve is noise; the caller is
# warned rather than silently handed a curve fit on one year.
MIN_SEASONS = 2


@dataclass
class RankCurve:
    """Fitted rank -> points curves, one per position."""

    # position -> (ranks array, points array), both ascending in rank
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    seasons_used: dict[str, int] = field(default_factory=dict)

    def points_for_rank(self, position: str, rank: float) -> float:
        """Expected season points for the player projected to finish `rank`.

        Ranks below 1 clamp to the top of the curve. Ranks past the fitted tail
        extrapolate along the slope of the last stretch and floor at zero —
        deep bench players are worth little, but never a negative amount.
        """
        curve = self.curves.get(position)
        if curve is None:
            return float("nan")
        ranks, pts = curve
        if not len(ranks):
            return float("nan")

        r = float(rank)
        if r <= ranks[0]:
            return float(pts[0])
        if r <= ranks[-1]:
            return float(np.interp(r, ranks, pts))

        # Extrapolate the tail. Use the last stretch's slope, not the whole
        # curve's — the top of a positional curve is far steeper than the tail,
        # and borrowing that slope would send deep players sharply negative.
        tail = min(10, len(ranks) - 1)
        if tail <= 0:
            return float(pts[-1])
        slope = (pts[-1] - pts[-1 - tail]) / (ranks[-1] - ranks[-1 - tail])
        return float(max(0.0, pts[-1] + slope * (r - ranks[-1])))

    def points_for_ranks(self, positions: pd.Series, ranks: pd.Series) -> pd.Series:
        vals = [
            self.points_for_rank(p, r) if pd.notna(p) and pd.notna(r) else np.nan
            for p, r in zip(positions, ranks)
        ]
        return pd.Series(vals, index=positions.index, dtype=float)

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for pos, (ranks, pts) in sorted(self.curves.items()):
            for r, p in zip(ranks, pts):
                rows.append({"position": pos, "rank": int(r), "points": float(p)})
        return pd.DataFrame(rows)


def season_totals(weekly: pd.DataFrame, *,
                  player_col: str = "player_key",
                  points_col: str = "fantasy_points",
                  weeks: tuple[int, ...] = FANTASY_WEEKS) -> pd.DataFrame:
    """Collapse scored weekly rows to one row per player-season.

    `games` counts weeks the player actually appears in the box score, which is
    what makes a per-game rate meaningful for someone who missed half a year.
    """
    df = weekly[weekly["week"].isin(weeks)]
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    grouped = df.groupby([player_col, "position", "season"], dropna=True)
    out = grouped.agg(
        points=(points_col, "sum"),
        games=(points_col, "size"),
    ).reset_index()
    out["points_per_game"] = out["points"] / out["games"].clip(lower=1)
    return out


def _season_weight(season: int, position: str,
                   downweight: list[dict] | None) -> float:
    for rule in downweight or []:
        if int(rule.get("season", -1)) != int(season):
            continue
        positions = rule.get("positions")
        if positions and position not in positions:
            continue
        return float(rule.get("weight", 1.0))
    return 1.0


def fit_rank_curve(totals: pd.DataFrame, *,
                   downweight: list[dict] | None = None,
                   max_rank: int = 120,
                   positions: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF"),
                   ) -> RankCurve:
    """Fit one monotone rank -> points curve per position.

    `totals` is the output of `season_totals`: one row per player-season with a
    `points` column. Within each season players are ranked inside their own
    position, and the points at each rank are averaged across seasons with the
    configured weights.
    """
    from sklearn.isotonic import IsotonicRegression

    curve = RankCurve()

    for pos in positions:
        sub = totals[totals["position"] == pos]
        if sub.empty:
            continue

        # rank -> [(points, weight)] across seasons
        acc: dict[int, list[tuple[float, float]]] = {}
        seasons = sorted(sub["season"].unique())
        for season in seasons:
            year = sub[sub["season"] == season].sort_values("points", ascending=False)
            w = _season_weight(int(season), pos, downweight)
            if w <= 0:
                continue
            for i, pts in enumerate(year["points"].tolist()[:max_rank], start=1):
                acc.setdefault(i, []).append((float(pts), w))

        if not acc:
            continue

        ranks = np.array(sorted(acc), dtype=float)
        means = np.array([
            float(np.average([p for p, _ in acc[int(r)]],
                             weights=[w for _, w in acc[int(r)]]))
            for r in ranks
        ])

        # Sample size falls off at the tail (not every season has 120 kickers),
        # so weight the isotonic fit by how many seasons contributed each rank.
        counts = np.array([len(acc[int(r)]) for r in ranks], dtype=float)
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        smoothed = iso.fit_transform(ranks, means, sample_weight=counts)

        curve.curves[pos] = (ranks, smoothed)
        curve.seasons_used[pos] = len(seasons)

    return curve


def thin_positions(curve: RankCurve, min_seasons: int = MIN_SEASONS) -> list[str]:
    """Positions whose curve rests on too few seasons to be trusted."""
    return sorted(
        pos for pos, n in curve.seasons_used.items() if n < min_seasons
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def measure_persistence(totals: pd.DataFrame, *,
                        top_n: int = 48,
                        positions: tuple[str, ...] = ("QB", "RB", "WR", "TE",
                                                      "K", "DEF"),
                        ) -> dict[str, float]:
    """How much of a position's ordering survives into the next season.

    Spearman correlation between a player's rank in season Y-1 and his points in
    season Y, sign-flipped so higher means more persistent. Restricted to the
    top `top_n` of each position, because the correlation across the whole pool
    is dominated by "starters outscore backups", which is not the question.

    Measured on 2020-2025: QB 0.62, TE 0.54, RB 0.46, WR 0.45, K 0.34, DEF 0.26.
    """
    df = totals.copy()
    df["rank_in_season"] = df.groupby(["season", "position"])["points"].rank(
        ascending=False, method="first")

    out: dict[str, float] = {}
    for pos in positions:
        sub = df[df["position"] == pos]
        pairs = []
        for season in sorted(sub["season"].unique())[:-1]:
            prev = sub[sub["season"] == season][["player_key", "rank_in_season"]]
            nxt = sub[sub["season"] == season + 1][["player_key", "points"]]
            merged = prev.merge(nxt, on="player_key")
            pairs.append(merged[merged["rank_in_season"] <= top_n])
        if not pairs:
            continue
        joined = pd.concat(pairs, ignore_index=True)
        if len(joined) < 20:
            continue
        rho = joined["rank_in_season"].corr(joined["points"], method="spearman")
        if pd.notna(rho):
            out[pos] = float(-rho)
    return out


def persistence_shrink(totals: pd.DataFrame, positions: list[str], *,
                       reference: str = "QB") -> dict[str, float]:
    """Extra spread shrink for positions whose rank ordering barely persists.

    The rank curve is fit on *finish* rank, so mapping a projected rank through
    it assumes the projection is as good as hindsight. How wrong that assumption
    is depends on the position: the consensus really can tell QB1 from QB12, and
    really cannot tell DEF1 from DEF12. The factor is a position's measured
    persistence divided by the reference position's, so the most predictable
    position is left alone and the rest shrink in proportion.

    **Deliberately scoped to the positions the caller names, which is K and DEF.**
    For QB/RB/WR/TE this measurement is a *lower bound* on how good a projection
    can be — it uses last season's rank as the projection, and the expert
    consensus is strictly better than that, because it has seen the offseason.
    Applying the factor there would shrink good projections for a weakness they
    do not have. For K and DEF the measurement and the folklore agree, and the
    correction is what keeps a streamable defense out of the fourth round.

    Revisit this once the walk-forward backtest can score it directly; it is the
    one place in the board where a measured number stands in for a validated one.
    """
    measured = measure_persistence(totals)
    ref = measured.get(reference)
    if not ref or ref <= 0:
        return {}
    return {
        pos: float(min(1.0, measured[pos] / ref))
        for pos in positions
        if pos in measured
    }


def apply_spread_shrink(df: pd.DataFrame, factors: dict[str, float], *,
                        points_col: str = "projection_calibrated",
                        position_col: str = "position") -> pd.DataFrame:
    """Shrink listed positions toward their own mean. Positions absent from
    `factors` are untouched, which is the point."""
    if not factors:
        return df
    out = df.copy()
    means = out.groupby(position_col)[points_col].transform("mean")
    factor = out[position_col].map(factors).astype(float).fillna(1.0)
    out[points_col] = (out[points_col] - means) * factor + means
    out["persistence_shrink"] = factor
    return out
