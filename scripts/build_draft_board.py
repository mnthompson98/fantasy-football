"""Build the draft board.

    python -m scripts.build_draft_board
    python -m scripts.build_draft_board --refresh          # force fresh pulls
    python -m scripts.build_draft_board --league-id <id>   # live 2026 scoring

Outputs `outputs/projections/draft_board.{parquet,csv,html}`. The HTML file is
self-contained and works offline on a phone, which is where you will actually
be drafting from.

The valuation itself lives in `src/features/pipeline.py`, shared with the
backtest — so what gets scored over 2022-2025 is the same code that picks
players on draft day. This script is the *live* half: current ECR, current
injury status, current rosters, and the export.

Everything unmatched or dropped is reported. A board that quietly loses a
top-40 rookie is worse than no board.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.leakage_guard import (  # noqa: E402
    TemporalBoundary,
    assert_no_future_seasons,
)
from src.draft.board import export, finalize_board  # noqa: E402
from src.features.pipeline import ecr_to_pool, value_board  # noqa: E402
from src.features.scoring import Scoring  # noqa: E402
from src.features.slopes import fitted_slopes  # noqa: E402
from src.features.vorp import LeagueShape  # noqa: E402
from src.ingest import history as H  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402
from src.ingest.sleeper_api import (  # noqa: E402
    UNAVAILABLE_STATUSES,
    get_league,
    get_players,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def attach_availability(pool: pd.DataFrame, cw) -> pd.DataFrame:
    """Fold in Sleeper's live injury status and current team.

    Sleeper is hours fresher than the nightly ranking scrape, and in the last
    week of August that gap is where the season-ending news lives.

    This is the one step the backtest deliberately does not share: injury status
    from the Sleeper API is *current*, not historical, so using it inside a
    backtest fold would be leakage of the most direct kind.
    """
    meta = pool["player_key"].map(lambda k: cw.meta.get(k, {}))
    pool["injury_status"] = meta.map(lambda m: m.get("injury_status"))
    pool["sleeper_id"] = meta.map(lambda m: m.get("sleeper_id"))

    live_team = meta.map(lambda m: m.get("nfl_team"))
    pool["nfl_team"] = live_team.fillna(pool["nfl_team"])

    # The live monitor removes drafted players by matching Sleeper's `player_id`
    # against this column, so it has to *be* the Sleeper id. Falling back to the
    # canonical key keeps the row on the board; it just won't auto-strike when
    # someone drafts him, which is why the count gets reported.
    pool["player_id"] = pool["sleeper_id"].fillna(pool["player_key"])

    status = pool["injury_status"].fillna("").astype(str).str.strip()
    pool["available"] = ~status.isin(UNAVAILABLE_STATUSES)
    return pool


def assemble(cfg: dict, *, refresh: bool = False, league_id: str | None = None,
             seasons: list[int] | None = None, target_season: int | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("resolving player identities...")
    ff_ids = nv.load_ff_playerids(refresh=refresh)
    sleeper_players = get_players(cache_dir=nv.RAW_DIR)
    cw = build_crosswalk(ff_ids, sleeper_players)
    print(f"  {len(cw.meta)} players in crosswalk")

    league = get_league(league_id) if league_id else None
    scoring = Scoring.from_league(league)
    source = f"league {league_id}" if league else "verified 2025 settings"
    print(f"  scoring: {'full PPR' if scoring.is_full_ppr else 'NOT full PPR'} "
          f"({source})")

    hist_seasons = seasons or [int(s) for s in cfg["backtest"]["seasons"]]
    prior_season = max(hist_seasons)
    target = int(target_season) if target_season else prior_season + 1
    boundary = TemporalBoundary(target_season=target)

    print(f"scoring history {min(hist_seasons)}-{prior_season}...")
    totals = H.season_totals_for(hist_seasons, scoring, refresh=refresh)
    # The same guard the backtest uses. Preseason is exactly when it is cheapest
    # to accidentally reach into the season you are about to predict.
    assert_no_future_seasons(totals, boundary, label="history totals")
    print(f"  {len(totals)} player-seasons · boundary {boundary.describe()}")

    print("loading consensus rankings...")
    rankings = nv.load_ff_rankings(refresh=refresh)
    ecr, unmatched, scrape_date = H.preseason_ecr(rankings, target, cw)
    print(f"  ECR scraped {scrape_date} · {len(ecr)} players matched, "
          f"{len(unmatched)} unmatched")
    for row in unmatched.head(10).itertuples():
        print(f"    unmatched: {row.player} ({row.pos}, ECR {row.ecr:.0f})")

    pool = ecr_to_pool(
        ecr, totals,
        H.opportunity_for(prior_season, refresh=refresh),
        prior_season=prior_season, train_seasons=hist_seasons,
        downweight=cfg["backtest"].get("downweight"), verbose=True,
    )
    pool = attach_availability(pool, cw)

    no_sleeper = int(pool["sleeper_id"].isna().sum())
    if no_sleeper:
        print(f"  [warn] {no_sleeper} players have no Sleeper id; the live "
              f"monitor cannot auto-strike them when drafted")

    for comp in cfg["blend"]["components"]:
        n = int(pool[comp].notna().sum()) if comp in pool.columns else 0
        print(f"  component {comp:<14} {n:4d}/{len(pool)} players")

    print("blending, calibrating, computing VORP...")
    # Slopes are fit against *past* preseason ECR snapshots paired with what
    # actually happened, so this needs the dated archive, not the current scrape
    # that `rankings` holds. Handing it the current-only frame would find no
    # history, silently fall back to the priors, and print nothing about it.
    archive = nv.load_ff_rankings_history(refresh=refresh)
    slopes = fitted_slopes(cfg, archive, totals, cw, hist_seasons, verbose=True)

    shape = LeagueShape.from_config(cfg)
    excluded = pool[~pool["available"]].copy()
    valued = value_board(
        pool[pool["available"]], totals, cfg, slopes=slopes,
        train_seasons=hist_seasons, shape=shape, verbose=True,
    )

    board = finalize_board(valued, drop_unavailable=False)
    meta = {
        "ecr": scrape_date,
        "teams": shape.teams,
        "scoring": "PPR" if scoring.is_full_ppr else "custom",
        "players": len(board),
    }
    return board, excluded, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the value-based draft board")
    ap.add_argument("--refresh", action="store_true",
                    help="force fresh pulls, ignoring the data/raw cache")
    ap.add_argument("--league-id", default=None,
                    help="Sleeper league id to read live scoring settings from")
    ap.add_argument("--seasons", default=None,
                    help="comma-separated history seasons (default: config)")
    ap.add_argument("--target-season", type=int, default=None,
                    help="season being drafted for; sets the leakage boundary "
                         "(default: latest history season + 1)")
    ap.add_argument("--out", default="outputs/projections")
    ap.add_argument("--top", type=int, default=30,
                    help="how many rows to print when done")
    args = ap.parse_args()

    cfg = load_config()
    seasons = ([int(s) for s in args.seasons.split(",")]
               if args.seasons else None)

    board, excluded, meta = assemble(
        cfg, refresh=args.refresh, league_id=args.league_id, seasons=seasons,
        target_season=args.target_season,
    )

    out_dir = ROOT / args.out
    paths = export(board, out_dir, meta=meta)

    if len(excluded):
        excluded_path = out_dir / "excluded_players.csv"
        cols = [c for c in ("player_name", "position", "nfl_team",
                            "injury_status", "adp_rank") if c in excluded.columns]
        excluded.sort_values("adp_rank")[cols].to_csv(excluded_path, index=False)
        print(f"\n{len(excluded)} players gated out (injury/roster status) -> "
              f"{excluded_path.relative_to(ROOT)}")
        for row in excluded.sort_values("adp_rank").head(8).itertuples():
            print(f"    {row.position:<3} {str(row.player_name)[:24]:<24} "
                  f"{row.injury_status}  (ECR {row.adp_rank})")

    print(f"\ntop {args.top} by VORP:")
    print(f"  {'#':>3}  {'POS':<4}{'PLAYER':<24}{'TM':<5}"
          f"{'VORP':>7}{'PROJ':>8}{'ADP':>6}{'d':>6}")
    for row in board.head(args.top).itertuples():
        print(f"  {row.rank:>3}  {str(row.position):<4}"
              f"{str(row.player_name)[:23]:<24}{str(row.nfl_team or '-'):<5}"
              f"{row.vorp:>7.1f}{row.projection_calibrated:>8.1f}"
              f"{row.adp_rank:>6.0f}{row.adp_delta:>+6.0f}")

    print("\nfirst kicker and defense (market-anchored):")
    for pos in ("K", "DEF"):
        sub = board[board["position"] == pos].head(2)
        for row in sub.itertuples():
            rnd = (row.rank - 1) // int(meta["teams"]) + 1
            print(f"  {pos:<4}{str(row.player_name)[:23]:<24}"
                  f"board {row.rank:>4} (round {rnd:>2})   ADP {row.adp_rank:>4}")

    print("\nwrote:")
    for kind, path in paths.items():
        print(f"  {kind:<8} {path.relative_to(ROOT)}")
    print("\nlive monitor:")
    print("  python -m src.draft.monitor --draft-id <id> --my-slot <n>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
