"""Walk-forward multi-year backtest orchestration.

For each target season Y in the configured window:
    - build the training window (seasons up to Y - purge_gap)
    - assert the temporal boundary holds, loudly
    - build the as-of-draft-day board for Y
    - simulate N drafts from varied draft slots with varied seeds
    - score each resulting roster against Y's realized weekly points
    - log everything, reproducibly

Every run writes a row to `outputs/backtests/runs.parquet` carrying the git
commit, the config hash, the seed, and the metrics. A result that cannot be
reproduced from its logged row is a bug.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .draft_sim import simulate_draft
from .leakage_guard import (
    TemporalBoundary,
    assert_no_future_seasons,
    assert_preseason_only,
    assert_purge_gap,
    flag_suspicious_stability,
)


@dataclass
class FoldSpec:
    target_season: int
    train_seasons: list[int]
    boundary: TemporalBoundary


@dataclass
class BacktestConfig:
    seasons: list[int]
    purge_gap: int = 1
    seed: int = 20260830
    drafts_per_season: int = 40
    teams: int = 10
    rounds: int = 16
    starters: dict[str, int] = field(default_factory=lambda: {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1
    })
    flex_slots: int = 1
    flex_eligible: tuple[str, ...] = ("RB", "WR", "TE")
    sigma: dict[str, float] = field(default_factory=lambda: {
        "QB": 12.0, "RB": 10.0, "WR": 8.0, "TE": 11.0, "K": 6.0, "DEF": 6.0
    })
    need_boost: float = 1.6
    need_penalty: float = 0.4
    value_col: str = "vorp"
    min_train_seasons: int = 2
    # How many of each position is worth owning. None keeps ValueDrafter's
    # defaults; config/league.yaml `draft_policy.position_caps` overrides.
    caps: dict[str, int] | None = None
    # Credit an unfilled starting slot with what a streamer scored that week.
    # Off, the scorer punishes thin rosters for the absence of a waiver wire.
    model_waivers: bool = True

    def hash(self) -> str:
        payload = json.dumps(
            {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in self.__dict__.items()},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_folds(cfg: BacktestConfig) -> list[FoldSpec]:
    """Expanding-window folds with a purge gap.

    The earliest seasons are consumed as training only — a fold needs at least
    `min_train_seasons` behind it to be worth scoring.
    """
    seasons = sorted(cfg.seasons)
    folds: list[FoldSpec] = []

    for target in seasons:
        train = [s for s in seasons if s <= target - cfg.purge_gap]
        if len(train) < cfg.min_train_seasons:
            continue
        assert_purge_gap(train, target, cfg.purge_gap)
        folds.append(FoldSpec(
            target_season=target,
            train_seasons=train,
            boundary=TemporalBoundary(target_season=target),
        ))
    return folds


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_fold(fold: FoldSpec, cfg: BacktestConfig, *,
             board_builder, weekly_actuals: pd.DataFrame,
             historical_features: pd.DataFrame | None = None
             ) -> list[M.BacktestResult]:
    """Simulate and score one target season.

    `board_builder(fold, cfg) -> DataFrame` must return the as-of-draft-day
    board with columns `player_id, player_name, position, adp_rank, projection,
    vorp, season`. It is called inside the guard so a leaky board fails here
    rather than silently inflating results.
    """
    if historical_features is not None:
        assert_no_future_seasons(
            historical_features, fold.boundary, label="historical_features"
        )

    board = board_builder(fold, cfg)
    assert_preseason_only(board, fold.boundary, label="draft board")

    season_actuals = weekly_actuals[weekly_actuals["season"] == fold.target_season]
    if season_actuals.empty:
        raise ValueError(f"no realized weekly points for season {fold.target_season}")

    # What a streamer was worth each week, so an unfilled starting slot is
    # charged the cost of streaming rather than the cost of fielding nobody.
    waiver_by_week = {
        wk: M.waiver_levels(season_actuals[season_actuals["week"] == wk],
                            cfg.starters, cfg.teams)
        for wk in M.REGULAR_WEEKS + M.PLAYOFF_WEEKS
    } if cfg.model_waivers else {}

    rng = np.random.default_rng(cfg.seed + fold.target_season)
    results: list[M.BacktestResult] = []

    for i in range(cfg.drafts_per_season):
        # Vary the draft slot so results are not an artifact of one position.
        my_slot = int(rng.integers(1, cfg.teams + 1))
        seed = int(rng.integers(0, 2**31 - 1))

        rosters = simulate_draft(
            board,
            teams=cfg.teams, rounds=cfg.rounds, my_slot=my_slot,
            starters=cfg.starters, flex_slots=cfg.flex_slots,
            flex_eligible=cfg.flex_eligible, sigma=cfg.sigma,
            need_boost=cfg.need_boost, need_penalty=cfg.need_penalty,
            value_col=cfg.value_col, seed=seed, caps=cfg.caps,
        )

        my_team = my_slot - 1
        team_ids = {t: [p["player_id"] for p in r.picks] for t, r in rosters.items()}

        weekly_by_team: dict[int, dict[int, float]] = {}
        for t, ids in team_ids.items():
            sub = season_actuals[season_actuals["player_id"].isin(ids)]
            weekly_by_team[t] = {
                wk: M.optimal_lineup_points(
                    sub[sub["week"] == wk], cfg.starters,
                    cfg.flex_slots, cfg.flex_eligible,
                    waiver=waiver_by_week.get(wk),
                )
                for wk in M.REGULAR_WEEKS + M.PLAYOFF_WEEKS
            }

        mine = weekly_by_team[my_team]
        playoff_pts = sum(mine.get(w, 0.0) for w in M.PLAYOFF_WEEKS)
        regular_pts = sum(mine.get(w, 0.0) for w in M.REGULAR_WEEKS)

        schedule = M.round_robin_schedule(
            cfg.teams, M.REGULAR_WEEKS, my_team, seed=seed
        )
        wins, losses = M.head_to_head_record(mine, weekly_by_team, schedule)

        season_totals = {
            t: sum(w.values()) for t, w in weekly_by_team.items()
        }
        ranks = M.rank_teams_by_points(season_totals)

        acc = {"mae": None, "rmse": None, "spearman": None}
        my_ids = team_ids[my_team]
        proj = board[board["player_id"].isin(my_ids)][["player_id", "projection"]]
        realized = (
            season_actuals[season_actuals["player_id"].isin(my_ids)]
            .groupby("player_id")["points"].sum().rename("actual").reset_index()
        )
        merged = proj.merge(realized, on="player_id", how="inner")
        if not merged.empty:
            acc = M.projection_accuracy(merged["projection"], merged["actual"])

        results.append(M.BacktestResult(
            season=fold.target_season,
            seed=seed,
            my_slot=my_slot,
            playoff_points=playoff_pts,
            season_points=regular_pts + playoff_pts,
            regular_season_points=regular_pts,
            win_rate=wins / max(1, wins + losses),
            wins=wins,
            losses=losses,
            league_rank_by_points=ranks[my_team],
            projection_mae=acc["mae"],
            projection_rmse=acc["rmse"],
            projection_spearman=acc["spearman"],
        ))

    return results


def run_backtest(cfg: BacktestConfig, *, board_builder,
                 weekly_actuals: pd.DataFrame,
                 historical_features: pd.DataFrame | None = None,
                 output_dir: Path | str = "outputs/backtests",
                 label: str = "") -> pd.DataFrame:
    """Full walk-forward run. Returns the per-fold summary and logs it."""
    folds = build_folds(cfg)
    if not folds:
        raise ValueError(
            f"no scorable folds from seasons {cfg.seasons} with "
            f"purge_gap={cfg.purge_gap} and min_train_seasons={cfg.min_train_seasons}"
        )

    all_results: list[M.BacktestResult] = []
    per_fold_mean: list[float] = []

    for fold in folds:
        fold_results = run_fold(
            fold, cfg,
            board_builder=board_builder,
            weekly_actuals=weekly_actuals,
            historical_features=historical_features,
        )
        all_results.extend(fold_results)
        per_fold_mean.append(
            float(np.mean([r.playoff_points for r in fold_results]))
        )

    warning = flag_suspicious_stability(per_fold_mean)
    if warning:
        print(f"\n[leakage-guard] {warning}\n")

    summary = M.summarize(all_results)
    summary["config_hash"] = cfg.hash()
    summary["git_commit"] = _git_commit()
    summary["run_at"] = datetime.now(timezone.utc).isoformat()
    summary["label"] = label
    summary["leakage_warning"] = warning or ""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = out / "runs.parquet"
    if runs.exists():
        prior = pd.read_parquet(runs)
        pd.concat([prior, summary], ignore_index=True).to_parquet(runs, index=False)
    else:
        summary.to_parquet(runs, index=False)

    return summary
