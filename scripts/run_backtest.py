"""Run the walk-forward backtest.

    python -m scripts.run_backtest
    python -m scripts.run_backtest --drafts 10          # quick pass
    python -m scripts.run_backtest --label no-anchor    # tag a comparison run

For each target season the simulator builds the board *as it would have looked
before that season's draft*, drafts against an ADP-following field from varied
slots, and scores the resulting roster against what actually happened.

**What makes this honest:** ffverse keeps every FantasyPros ECR snapshot back to
2019, so the market rank fed to a fold is the one the room genuinely held in
August of that year — not a hindsight ranking dressed up as a projection.
Preseason coverage starts in 2021, so 2020 is trainable but never a target.

The valuation is `src/features/pipeline.py`, the same code the live board runs.
That is deliberate: a backtest of a different program tells you nothing about
the one you draft with.

Results append to `outputs/backtests/runs.parquet` with the git commit, the
config hash and the seed. Read the metric priority in `src/backtest/metrics.py`
before interpreting anything: playoff points first, projection accuracy last.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.leakage_guard import (  # noqa: E402
    assert_no_future_seasons,
    assert_purge_gap,
)
from src.backtest.walkforward import BacktestConfig, build_folds, run_backtest  # noqa: E402
from src.features.pipeline import ecr_to_pool, value_board  # noqa: E402
from src.features.scoring import Scoring  # noqa: E402
from src.features.slopes import fitted_slopes  # noqa: E402
from src.features.vorp import LeagueShape  # noqa: E402
from src.ingest import history as H  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def backtest_config(cfg: dict, *, drafts: int | None = None,
                    seasons: list[int] | None = None) -> BacktestConfig:
    bt = cfg["backtest"]
    shape = LeagueShape.from_config(cfg)
    opp = bt.get("opponent_model", {})
    return BacktestConfig(
        seasons=seasons or [int(s) for s in bt["seasons"]],
        purge_gap=int(bt.get("purge_gap_seasons", 1)),
        max_train_seasons=bt.get("max_train_seasons") or None,
        seed=int(bt.get("seed", 20260830)),
        drafts_per_season=drafts if drafts is not None else 40,
        teams=shape.teams,
        rounds=int(cfg["verified"].get("roster_size", 16)),
        starters=dict(shape.starters),
        flex_slots=shape.flex_slots,
        flex_eligible=tuple(shape.flex_eligible),
        sigma={k: float(v)
               for k, v in (opp.get("adp_sigma_fallback") or {}).items()},
        need_boost=float(opp.get("positional_need_boost", 1.6)),
        need_penalty=float(opp.get("positional_need_penalty", 0.4)),
        caps=(cfg.get("draft_policy", {}).get("position_caps") or None),
        valuation={
            "blend": cfg.get("blend", {}).get("components"),
            "calibration": cfg.get("calibration"),
            "market_anchor": cfg.get("market_anchor"),
            "downweight": bt.get("downweight"),
        },
    )


def make_board_builder(cfg: dict, *, totals: pd.DataFrame,
                       rankings: pd.DataFrame, crosswalk,
                       verbose: bool = False):
    """Return a `board_builder(fold, bt_cfg) -> DataFrame` for the walk-forward.

    The builder is where the temporal boundary is actually enforced, because it
    is the only place that knows which fold it is in. It asserts before it uses
    a single historical row, and it never touches the Sleeper player map —
    Sleeper's injury status is *current*, so consulting it inside a fold would
    leak the present into the past in the most direct way available.
    """
    shape = LeagueShape.from_config(cfg)

    def build(fold, bt_cfg) -> pd.DataFrame:
        train = list(fold.train_seasons)
        assert_purge_gap(train, fold.target_season, bt_cfg.purge_gap)

        train_totals = totals[totals["season"].isin(train)]
        assert_no_future_seasons(
            train_totals, fold.boundary, label=f"{fold.target_season} history")

        prior_season = max(train)
        ecr, unmatched, scrape_date = H.preseason_ecr(
            rankings, fold.target_season, crosswalk)

        pool = ecr_to_pool(
            ecr, train_totals,
            H.opportunity_for(prior_season),
            prior_season=prior_season, train_seasons=train,
            downweight=cfg["backtest"].get("downweight"),
        )

        # Slopes are fit on seasons strictly before this fold's target, which is
        # what keeps them leakage-free. Early folds have only one season of
        # preseason ECR to fit on and fall back to the config priors per
        # position, which is the honest behaviour.
        slopes = fitted_slopes(cfg, rankings, train_totals, crosswalk, train)

        board = value_board(pool, train_totals, cfg, slopes=slopes,
                            train_seasons=train, shape=shape)

        # The simulator and the scorer both key on `player_id`. Inside a
        # backtest that is the canonical key: there is no live Sleeper draft to
        # reconcile against, and gsis is what the realized weekly points use.
        board["player_id"] = board["player_key"]
        board["season"] = fold.target_season

        if verbose:
            print(f"  {fold.target_season}: ECR {scrape_date} · "
                  f"{len(board)} players · train {train[0]}-{train[-1]} · "
                  f"{len(unmatched)} unmatched")
        return board

    return build


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest")
    ap.add_argument("--drafts", type=int, default=None,
                    help="drafts simulated per season (default 40)")
    ap.add_argument("--seasons", default=None,
                    help="comma-separated seasons (default: config)")
    ap.add_argument("--label", default="",
                    help="tag for this run in outputs/backtests/runs.parquet")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    seasons = ([int(s) for s in args.seasons.split(",")]
               if args.seasons else None)
    bt_cfg = backtest_config(cfg, drafts=args.drafts, seasons=seasons)

    folds = build_folds(bt_cfg)
    if not folds:
        print("no scorable folds; widen `backtest.seasons` in the config",
              file=sys.stderr)
        return 1

    print(f"walk-forward · seasons {bt_cfg.seasons} · purge gap "
          f"{bt_cfg.purge_gap} · {bt_cfg.drafts_per_season} drafts/season")
    print(f"folds: " + ", ".join(
        f"{f.target_season}(train {f.train_seasons[0]}-{f.train_seasons[-1]})"
        for f in folds))

    scoring = Scoring.from_league(None)
    print("\nscoring history and building the crosswalk...")
    crosswalk = build_crosswalk(nv.load_ff_playerids(refresh=args.refresh))
    weekly = H.scored_weekly(bt_cfg.seasons, scoring, refresh=args.refresh)
    totals = H.season_totals_for(bt_cfg.seasons, scoring)
    rankings = nv.load_ff_rankings_history(refresh=args.refresh)
    print(f"  {len(totals)} player-seasons · {len(weekly)} player-weeks · "
          f"{len(rankings)} archived ranking rows")

    # The scorer wants one row per player-week with `points`.
    weekly_actuals = weekly.rename(
        columns={"player_key": "player_id", "fantasy_points": "points"})
    weekly_actuals = weekly_actuals[weekly_actuals["season_type"] == "REG"]

    print("\nbuilding boards...")
    builder = make_board_builder(
        cfg, totals=totals, rankings=rankings, crosswalk=crosswalk, verbose=True)

    print("\nsimulating...")
    summary = run_backtest(
        bt_cfg, board_builder=builder, weekly_actuals=weekly_actuals,
        output_dir=ROOT / "outputs" / "backtests", label=args.label,
    )

    per_fold = summary[summary["season"] != "ALL"]
    agg = summary[summary["season"] == "ALL"].iloc[0]

    print("\nper-season (mean over simulated drafts):")
    print(f"  {'season':<8}{'playoff':>9}{'season':>9}{'win%':>8}"
          f"{'rank':>7}{'MAE':>8}")
    for season, grp in per_fold.groupby("season"):
        print(f"  {str(season):<8}{grp['playoff_points'].mean():>9.1f}"
              f"{grp['season_points'].mean():>9.1f}"
              f"{grp['win_rate'].mean():>8.3f}"
              f"{grp['league_rank_by_points'].mean():>7.2f}"
              f"{grp['projection_mae'].mean():>8.1f}")
    print(f"  {'ALL':<8}{agg['playoff_points']:>9.1f}{agg['season_points']:>9.1f}"
          f"{agg['win_rate']:>8.3f}{agg['league_rank_by_points']:>7.2f}"
          f"{agg['projection_mae']:>8.1f}")

    print(f"\nleague rank by points averages {agg['league_rank_by_points']:.2f} "
          f"of {bt_cfg.teams}; 5.50 is what a coin flip gets you.")
    if agg["leakage_warning"]:
        print(f"\n[leakage-guard] {agg['leakage_warning']}")

    print(f"\nlogged to outputs/backtests/runs.parquet "
          f"(config {agg['config_hash']}, commit {agg['git_commit']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
