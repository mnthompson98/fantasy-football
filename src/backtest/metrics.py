"""Backtest scoring.

Metric priority, in order. This ordering is a design decision, not a default —
see CLAUDE.md. Do not reorder without a reason written down.

    1. playoff_points   — weeks 15-17 starting-lineup points. The thing that
                          actually decides a championship.
    2. season_points    — weeks 1-17 starting-lineup points.
    3. win_rate         — record against the simulated league. Noisy; sensitive
                          to schedule luck, which is why it sits below points.
    4. projection_mae   — diagnostic only. A model can win this and still lose
                          the league by drafting players who miss December.

A change is adopted only if it improves (1) without materially hurting (2).
Improving (4) alone is not evidence of anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

PLAYOFF_WEEKS = (15, 16, 17)
REGULAR_WEEKS = tuple(range(1, 15))


@dataclass
class BacktestResult:
    season: int
    seed: int
    my_slot: int
    playoff_points: float
    season_points: float
    regular_season_points: float
    win_rate: float
    wins: int
    losses: int
    league_rank_by_points: int
    projection_mae: float | None = None
    projection_rmse: float | None = None
    projection_spearman: float | None = None

    def to_row(self) -> dict:
        return asdict(self)


def optimal_lineup_points(week_points: pd.DataFrame, starters: dict[str, int],
                          flex_slots: int, flex_eligible: tuple[str, ...]) -> float:
    """Best legal starting lineup for one team in one week.

    `week_points` needs `position` and `points`. Players who did not play score
    0 rather than being dropped — that is the real cost of starting an inactive
    player, and hiding it would flatter every roster equally.

    This is the *hindsight-optimal* lineup, which slightly overstates what a
    manager would actually have started. It is applied identically to every
    team in the sim, so the comparison stays fair; treat the absolute number as
    an upper bound and the relative ranking as the signal.
    """
    if week_points.empty:
        return 0.0

    df = week_points.copy()
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    df = df.sort_values("points", ascending=False)

    used: set = set()
    total = 0.0

    for pos, n in starters.items():
        pool = df[(df["position"] == pos) & (~df.index.isin(used))].head(n)
        total += float(pool["points"].sum())
        used |= set(pool.index)

    if flex_slots:
        pool = df[
            df["position"].isin(flex_eligible) & (~df.index.isin(used))
        ].head(flex_slots)
        total += float(pool["points"].sum())

    return total


def score_roster(roster_player_ids: list, weekly_actuals: pd.DataFrame,
                 starters: dict[str, int], flex_slots: int,
                 flex_eligible: tuple[str, ...],
                 weeks: tuple[int, ...]) -> float:
    """Sum of best-legal-lineup points across `weeks` for one roster."""
    sub = weekly_actuals[weekly_actuals["player_id"].isin(roster_player_ids)]
    if sub.empty:
        return 0.0

    total = 0.0
    for week in weeks:
        wk = sub[sub["week"] == week]
        total += optimal_lineup_points(wk, starters, flex_slots, flex_eligible)
    return total


def head_to_head_record(my_weekly: dict[int, float],
                        opponent_weekly: dict[int, dict[int, float]],
                        schedule: dict[int, int]) -> tuple[int, int]:
    """Record against a fixed schedule. `schedule` maps week -> opponent team id."""
    wins = losses = 0
    for week, opp in schedule.items():
        mine = my_weekly.get(week, 0.0)
        theirs = opponent_weekly.get(opp, {}).get(week, 0.0)
        if mine > theirs:
            wins += 1
        elif mine < theirs:
            losses += 1
    return wins, losses


def round_robin_schedule(teams: int, weeks: tuple[int, ...],
                         my_team: int, seed: int | None = None) -> dict[int, int]:
    """A simple rotating schedule so win-rate is not an artifact of one pairing."""
    rng = np.random.default_rng(seed)
    others = [t for t in range(teams) if t != my_team]
    rng.shuffle(others)
    return {wk: others[i % len(others)] for i, wk in enumerate(weeks)}


def projection_accuracy(projected: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Diagnostic accuracy metrics. Reported, never optimized against directly."""
    df = pd.DataFrame({"p": projected, "a": actual}).dropna()
    if df.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}

    err = df["p"] - df["a"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman": float(df["p"].corr(df["a"], method="spearman")),
    }


def rank_teams_by_points(team_points: dict[int, float]) -> dict[int, int]:
    """1 = most points."""
    ordered = sorted(team_points.items(), key=lambda kv: -kv[1])
    return {team: i for i, (team, _) in enumerate(ordered, start=1)}


def summarize(results: list[BacktestResult]) -> pd.DataFrame:
    """One row per fold, plus an aggregate row. Aggregate is the headline."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_row() for r in results])
    agg = {
        "season": "ALL",
        "seed": df["seed"].iloc[0],
        "my_slot": "mixed" if df["my_slot"].nunique() > 1 else df["my_slot"].iloc[0],
        "playoff_points": df["playoff_points"].mean(),
        "season_points": df["season_points"].mean(),
        "regular_season_points": df["regular_season_points"].mean(),
        "win_rate": df["win_rate"].mean(),
        "wins": df["wins"].sum(),
        "losses": df["losses"].sum(),
        "league_rank_by_points": df["league_rank_by_points"].mean(),
        "projection_mae": df["projection_mae"].mean(),
        "projection_rmse": df["projection_rmse"].mean(),
        "projection_spearman": df["projection_spearman"].mean(),
    }
    return pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
