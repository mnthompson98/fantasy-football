"""The valuation pipeline, shared by the live board and the backtest.

This module exists so that the thing the backtest scores is *the same code* that
picks players on draft day. If the board and the backtest each had their own
copy of blend-calibrate-VORP, the backtest would be validating a program nobody
drafts with, and every conclusion drawn from it would be about the wrong system.

The pipeline takes a preseason ECR snapshot plus history strictly older than the
season being predicted, and returns a valued board:

    ECR rank
      -> points, through the historical rank curve      (rank_curve)
      -> blended with prior-season expected and actual  (blend)
      -> spread shrunk toward the positional mean       (calibration)
      -> shrunk again for low-persistence positions     (rank_curve)
      -> VORP against flex-aware replacement level      (vorp)
      -> anchored to market price at K and DEF          (market_anchor)

Every input is parameterized by season. Nothing in here reaches for "now".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .blend import blend_projections
from .calibration import apply_calibration
from .market_anchor import apply_market_anchor
from .rank_curve import (
    apply_spread_shrink,
    fit_rank_curve,
    persistence_shrink,
)
from .vorp import LeagueShape, compute_vorp

DRAFTABLE = ("QB", "RB", "WR", "TE", "K", "DEF")

# A per-game rate from three games is mostly noise. Shrink toward the positional
# mean with weight g / (g + K); at K = 6, a full season keeps ~74% of its own
# rate and a three-game cameo keeps 33%.
GAMES_SHRINK_K = 6.0

# Rates are scaled to a nominal full season for readability. The blend z-scores
# within position, so this constant cannot change any ranking — it only decides
# whether the column reads as points-per-game or points-per-season.
NOMINAL_GAMES = 17.0


@dataclass
class ValuationInputs:
    """Everything the pipeline needs, all of it as-of before `target_season`."""

    ecr: pd.DataFrame          # player_key, position, ecr, adp_rank, pos_rank
    totals: pd.DataFrame       # player_key, position, season, points, games
    target_season: int
    train_seasons: list[int]


def shrink_rate(values: pd.Series, games: pd.Series,
                positions: pd.Series, *, k: float = GAMES_SHRINK_K) -> pd.Series:
    """Empirical-Bayes shrink of a per-game rate toward the positional mean.

    Per-game rather than per-season on purpose: a player who missed half of last
    year on an unrelated injury is not half as good this year, and a season
    total says he is. Games played is the sample size, and it decides how far
    the rate is pulled toward the mean.
    """
    frame = pd.DataFrame({"v": values, "g": games, "pos": positions})
    means = frame.groupby("pos")["v"].transform("mean")
    w = frame["g"] / (frame["g"] + k)
    return w * frame["v"] + (1 - w) * means


def prior_season_rates(totals: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    """Last season's actual production, as a shrunk per-game rate."""
    prior = totals[totals["season"] == prior_season].copy()
    if prior.empty:
        return pd.DataFrame(columns=["player_key", "prior_points"])

    prior["prior_ppg_shrunk"] = shrink_rate(
        prior["points_per_game"], prior["games"], prior["position"])
    prior["prior_points"] = prior["prior_ppg_shrunk"] * NOMINAL_GAMES
    return prior[["player_key", "games", "points_per_game", "prior_points"]]


def expected_points_rates(opportunity: pd.DataFrame) -> pd.DataFrame:
    """Last season's *expected* fantasy points, as a shrunk per-game rate.

    Expected points strip out touchdown luck: a receiver who scored on 14% of
    his red-zone targets got lucky, and his expected points say so. This is the
    most useful regression signal available before a snap is played, and it is
    why the blend is not just the consensus with extra steps.

    ffopportunity scores receptions at exactly 1.0, so its points are already on
    this league's full-PPR scale.
    """
    if opportunity.empty:
        return pd.DataFrame(columns=["player_key", "xfp_points"])

    agg = opportunity.groupby(["player_key", "position"], dropna=True).agg(
        xfp=("total_fantasy_points_exp", "sum"),
        xfp_games=("total_fantasy_points_exp", "size"),
    ).reset_index()
    agg["xfp_per_game"] = agg["xfp"] / agg["xfp_games"].clip(lower=1)
    agg["xfp_ppg_shrunk"] = shrink_rate(
        agg["xfp_per_game"], agg["xfp_games"], agg["position"])
    agg["xfp_points"] = agg["xfp_ppg_shrunk"] * NOMINAL_GAMES
    return agg[["player_key", "xfp_games", "xfp_per_game", "xfp_points"]]


def value_board(pool: pd.DataFrame, totals: pd.DataFrame, cfg: dict, *,
                train_seasons: list[int],
                shape: LeagueShape | None = None,
                verbose: bool = False) -> pd.DataFrame:
    """Blend, calibrate, shrink, value and anchor. The heart of the system.

    `pool` carries one row per draftable player with the component columns
    already attached. `totals` is history restricted to `train_seasons` — it is
    used to measure positional persistence, and nothing else.
    """
    train = totals[totals["season"].isin(train_seasons)]

    weights = {k: float(v) for k, v in cfg["blend"]["components"].items()}
    blended = blend_projections(pool, weights=weights)

    valued = apply_calibration(
        blended, slopes=cfg["calibration"]["slopes"],
        out_col="projection_calibrated",
    )

    shrink_cfg = cfg["calibration"].get("persistence_shrink") or {}
    factors = persistence_shrink(
        train, list(shrink_cfg.get("positions") or []),
        reference=shrink_cfg.get("reference", "QB"),
    )
    if factors:
        if verbose:
            print("  persistence shrink: "
                  + ", ".join(f"{p} x{f:.2f}" for p, f in sorted(factors.items())))
        valued = apply_spread_shrink(valued, factors)

    shape = shape or LeagueShape.from_config(cfg)
    valued = compute_vorp(valued, shape, points_col="projection_calibrated")

    anchor_cfg = cfg.get("market_anchor") or {}
    anchor_weights = anchor_cfg.get("positions") or {}
    if anchor_weights:
        valued = apply_market_anchor(
            valued, anchor_weights,
            reference_positions=tuple(
                anchor_cfg.get("reference_positions")
                or ("QB", "RB", "WR", "TE")),
        )
        if verbose:
            moved = valued[valued["market_anchor"] > 0]
            for pos in sorted(anchor_weights):
                sub = moved[moved["position"] == pos]
                if not sub.empty:
                    top = sub.nlargest(1, "vorp_unanchored").iloc[0]
                    print(f"  market anchor {pos} x{anchor_weights[pos]}: "
                          f"top {pos} VORP {top['vorp_unanchored']:.1f} "
                          f"-> {top['vorp']:.1f}")
        # Anchoring changes the ordering, so the board has to be re-sorted.
        valued = valued.sort_values("vorp", ascending=False).reset_index(drop=True)

    return valued


def ecr_to_pool(ecr: pd.DataFrame, totals: pd.DataFrame,
                opportunity: pd.DataFrame, *,
                prior_season: int, train_seasons: list[int],
                downweight: list[dict] | None = None,
                verbose: bool = False) -> pd.DataFrame:
    """Attach the three projection components to a reconciled ECR snapshot.

    The rank curve is fit on `train_seasons` only. For the live board that is
    every completed season; for a backtest fold it is everything up to the purge
    gap, which is what keeps the fold honest.
    """
    curve = fit_rank_curve(
        totals[totals["season"].isin(train_seasons)], downweight=downweight)

    pool = ecr.copy()
    pool["ecr_points"] = curve.points_for_ranks(pool["position"], pool["pos_rank"])

    prior = prior_season_rates(totals, prior_season)
    pool = pool.merge(prior, on="player_key", how="left")

    xfp = expected_points_rates(opportunity)
    pool = pool.merge(xfp, on="player_key", how="left")

    if verbose:
        for pos in DRAFTABLE:
            if pos in curve.curves:
                print(f"  {pos:<3} {pos}1={curve.points_for_rank(pos, 1):6.1f}  "
                      f"{pos}12={curve.points_for_rank(pos, 12):6.1f}  "
                      f"{pos}36={curve.points_for_rank(pos, 36):6.1f}")
    return pool
