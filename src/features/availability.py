"""Durability: how much of the season a player is likely to be available for.

The primary metric is playoff-week points, and a player who misses weeks 15-17
scores zero however good the projection was. So availability sits directly on
the causal path — but only to the extent it is *predictable* and *not already
priced by the market*. Both were measured before this module was written.

**It is predictable, modestly.** Over 2012-2025, availability persists year to
year at ρ≈0.36 (QB 0.55, TE 0.38, RB 0.32, WR 0.29 — quarterbacks are the most
durable and the most predictable, running backs the least). A three-season
history beats one season, 0.39 vs 0.35. That rests on ~4,800 player-seasons,
not the four folds the draft board is stuck with, so unlike everything else in
this project it can actually be validated.

**The market already knows, where it is paying attention.** Inside ECR rank
buckets, comparing players with fragile against durable histories:

    ECR 1-24     0.863 vs 0.880   +0.017
    ECR 25-60    0.759 vs 0.781   +0.022
    ECR 61-120   0.726 vs 0.777   +0.051
    ECR 121-250  0.663 vs 0.769   +0.106      <- the signal

(probability of playing at least two of weeks 15-17). The raw correlation
between prior durability and realised availability is 0.380; controlling for ECR
bucket it falls to 0.148. Almost all of it is already in the consensus.

**Therefore this does not adjust projections.** Scaling a top-60 player's
projection by his durability would double-count what ECR has already priced —
the same error as adjusting projections for weather when the experts have
already seen the forecast. It is surfaced as information, and it earns its place
in the late rounds and on the waiver wire, where the consensus is thin and a
10-point gap in playoff availability is the difference between a useful bench
stash and an empty roster spot in December.
"""

from __future__ import annotations

import pandas as pd

# A team plays 16 games inside the 17 fantasy weeks, so 16 is the denominator.
# Using 17 would score a perfectly healthy season as 94% available and make
# every player look mildly fragile.
GAMES_IN_FANTASY_SEASON = 16.0

FANTASY_WEEKS = tuple(range(1, 18))
PLAYOFF_WEEKS = (15, 16, 17)

# How many seasons back to look. Three measurably beats one; beyond that the
# player is a different athlete.
LOOKBACK_SEASONS = 3

# Empirical-Bayes shrink toward the positional mean: weight n/(n+K). At K=1 a
# single season carries half its own weight, three seasons three quarters. A
# player with no history is simply the positional average, which is the honest
# prior for a rookie.
SHRINK_K = 1.0

# Below this, availability is worth mentioning on the board.
FRAGILE_BELOW = 0.70

# The residual signal is concentrated beyond this ECR rank; inside it, the
# consensus has already done this work.
SIGNAL_BEYOND_RANK = 120


def availability_history(weekly: pd.DataFrame, *,
                         positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
                         min_games: int = 4) -> pd.DataFrame:
    """One row per player-season: games played and the share of the season.

    `min_games` drops cameo seasons. A practice-squad callup who played twice is
    not a durability observation about a starter; he is noise that would drag
    every positional mean down.
    """
    df = weekly
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df[df["week"].isin(FANTASY_WEEKS) & df["position"].isin(positions)]

    out = (df.groupby(["player_key", "position", "season"])["week"]
           .agg(games="nunique",
                playoff_games=lambda w: len(set(w) & set(PLAYOFF_WEEKS)))
           .reset_index())
    out["availability"] = out["games"] / GAMES_IN_FANTASY_SEASON
    return out[out["games"] >= min_games].reset_index(drop=True)


def durability(history: pd.DataFrame, as_of_season: int, *,
               lookback: int = LOOKBACK_SEASONS,
               shrink_k: float = SHRINK_K) -> pd.DataFrame:
    """Each player's durability entering `as_of_season`.

    Uses only seasons strictly before `as_of_season`, so this is safe to call
    inside a backtest fold. Returns `player_key`, `durability`,
    `durability_seasons`, `durability_raw`.
    """
    past = history[(history["season"] < as_of_season)
                   & (history["season"] >= as_of_season - lookback)]
    if past.empty:
        return pd.DataFrame(columns=["player_key", "durability",
                                     "durability_seasons", "durability_raw"])

    agg = (past.groupby(["player_key", "position"])["availability"]
           .agg(durability_raw="mean", durability_seasons="size").reset_index())

    means = agg.groupby("position")["durability_raw"].transform("mean")
    w = agg["durability_seasons"] / (agg["durability_seasons"] + shrink_k)
    agg["durability"] = w * agg["durability_raw"] + (1 - w) * means

    return agg[["player_key", "durability", "durability_seasons",
                "durability_raw"]]


def attach(board: pd.DataFrame, dur: pd.DataFrame, *,
           rank_col: str = "adp_rank") -> pd.DataFrame:
    """Add durability to a board, and flag only where it carries information.

    `durability_flag` is deliberately conditional on draft rank. A fragile
    player inside the top 120 is fragile in a way the consensus has already
    discounted, and flagging him would invite double-counting by a human reading
    the board. Past that, the flag is the only durability signal on the page.
    """
    out = board.merge(dur, on="player_key", how="left")
    beyond = pd.to_numeric(out.get(rank_col), errors="coerce") > SIGNAL_BEYOND_RANK
    fragile = out["durability"] < FRAGILE_BELOW
    out["durability_flag"] = (beyond & fragile).fillna(False)
    return out
