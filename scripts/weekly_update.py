"""The weekly in-season run: lineup, waivers, and what you have to trade.

    python -m scripts.weekly_update
    python -m scripts.weekly_update --refresh          # force fresh pulls
    python -m scripts.weekly_update --check-only       # ingestion health only

Reads the live league from `config.current.league_id`, pulls this week's
FantasyPros projections, and answers the three questions that matter each week.

**Everything here is unvalidated.** The draft board was backtested over four
seasons; these recommendations cannot be, because the weekly ranking feed is a
live snapshot with no archive (see `src/inseason/projections.py`). Treat the
output as a well-sourced second opinion, not as a verified edge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lineup import LineupSlots  # noqa: E402
from src.inseason import start_sit, trades, waivers  # noqa: E402
from src.inseason.projections import (  # noqa: E402
    attach_projections,
    load_weekly_projections,
)
from src.inseason.roster import load_league_state  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402
from src.ingest.sleeper_api import get_players, get_trending  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def check_ingestion(cfg: dict, cw, *, refresh: bool = False) -> int:
    """Verify every weekly input is present, fresh and reconcilable.

    Run this before trusting anything downstream. Each check prints what it
    found rather than a bare pass/fail, because "0 players matched" and "480
    players matched" are both technically successes.
    """
    problems = 0
    print("weekly ingestion check\n")

    raw = nv.load_ff_rankings_weekly(refresh=refresh)
    scraped = str(raw["scrape_date"].max()) if "scrape_date" in raw else "unknown"
    pages = sorted(raw["page"].astype(str).unique())
    print(f"  weekly rankings   {len(raw):>5} rows · scraped {scraped}")
    print(f"    pages: {', '.join(pages)}")

    try:
        proj = load_weekly_projections(cw, refresh=refresh, verbose=True)
    except ValueError as exc:
        print(f"  [FAIL] {exc}")
        return 1

    have = proj["projection"].notna().sum()
    print(f"  projections       {len(proj):>5} players · {have} with points")
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        sub = proj[proj["position"] == pos]
        top = sub.nlargest(1, "projection")
        best = (f"{top.iloc[0]['player_name'][:20]} {top.iloc[0]['projection']:.1f}"
                if len(top) else "—")
        print(f"    {pos:<4}{len(sub):>4} ranked   best: {best}")
        if sub.empty:
            print(f"    [FAIL] no {pos} in the weekly feed — lineup cannot be filled")
            problems += 1

    missing = proj["projection"].isna().sum()
    if missing:
        print(f"  [warn] {missing} ranked players have no point projection; "
              f"they will be treated as zero")

    season = nv._current_season()
    stats = nv.load_player_stats([season], refresh=refresh)
    weeks = sorted(pd.to_numeric(stats["week"], errors="coerce").dropna().unique())
    print(f"  box scores        season {season} · "
          f"{'weeks ' + str(int(weeks[0])) + '-' + str(int(weeks[-1])) if weeks else 'none yet'}")
    if not weeks:
        print("    (expected before week 1 — nothing has been played)")

    print(f"\n{'ingestion OK' if not problems else str(problems) + ' problem(s)'}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly in-season update")
    ap.add_argument("--league-id", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="verify weekly ingestion and stop")
    ap.add_argument("--waiver-threshold", type=float, default=None,
                    help="points a claim must add before it is worth priority")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    current = cfg.get("current") or {}
    league_id = args.league_id or current.get("league_id")
    if not league_id:
        print("no league id: pass --league-id or set config.current.league_id",
              file=sys.stderr)
        return 1

    print("resolving player identities...")
    cw = build_crosswalk(nv.load_ff_playerids(refresh=args.refresh),
                         get_players(cache_dir=nv.RAW_DIR))
    print(f"  {len(cw.meta)} players in crosswalk\n")

    problems = check_ingestion(cfg, cw, refresh=args.refresh)
    if args.check_only:
        return 1 if problems else 0
    if problems:
        print("\nrefusing to advise on broken ingestion.", file=sys.stderr)
        return 1

    proj = load_weekly_projections(cw, refresh=args.refresh)
    slots = LineupSlots.from_config(cfg)
    state = load_league_state(league_id, cw,
                              user_id=cfg["user"]["sleeper_user_id"])

    if state.my_roster.empty:
        print(f"\nNo roster found for user {cfg['user']['sleeper_user_id']} in "
              f"league {league_id}. Before the draft this is expected.")
        return 0

    roster = attach_projections(state.my_roster, proj)
    print(f"\n{'=' * 62}\nSTART / SIT\n{'=' * 62}")
    lineup, calls, holes = start_sit.recommend(roster, slots)
    for row in lineup.starters.itertuples():
        proj_pts = getattr(row, "projection", 0.0) or 0.0
        print(f"  {row.slot:<5} {str(row.player_name)[:26]:<27}"
              f"{proj_pts:>6.1f}  {str(getattr(row, 'opponent', '') or '')}")
    print(f"  {'':<5} {'projected total':<27}{lineup.points:>6.1f}")

    if holes:
        print(f"\n  UNFILLED: {', '.join(holes)} — bye or injury. "
              f"Fill from free agency below.")

    tight = [c for c in calls if c.gap < 6][:5]
    if tight:
        print("\n  closest calls:")
        for call in tight:
            print(call.describe())

    print(f"\n{'=' * 62}\nWAIVERS  (rolling priority — a claim costs your place)\n{'=' * 62}")
    universe = proj[proj["position"].isin(("QB", "RB", "WR", "TE", "K", "DEF"))]
    available = state.free_agents(universe)
    trending = {}
    for entry in get_trending("add", limit=100):
        key = cw.resolve(sleeper_id=entry.get("player_id"))
        if key:
            trending[key] = int(entry.get("count") or 0)

    threshold = args.waiver_threshold
    if threshold is None:
        threshold = float((cfg.get("waivers") or {}).get(
            "priority_threshold", waivers.DEFAULT_PRIORITY_THRESHOLD))

    moves = waivers.evaluate(
        roster, available, slots,
        priority_threshold=threshold,
        roster_size=int(cfg["verified"].get("roster_size", 16)),
        trending=trending,
    )
    for move in moves:
        print(move.describe())
    print(f"\n  {waivers.summarize(moves, threshold=threshold)}")

    print(f"\n{'=' * 62}\nTRADE BAIT  (costs your lineup nothing)\n{'=' * 62}")
    surplus = trades.find_surplus(roster, slots)
    if surplus.empty:
        print("  Nothing spare — every player on the roster is doing work.")
    else:
        for row in surplus.head(6).itertuples():
            pts = getattr(row, "projection", None)
            # An unranked player is not a zero-point player — he is one
            # FantasyPros did not rank this week, usually because he is hurt or
            # buried. Printing "—" says that; printing 0.0 would not.
            shown = f"{pts:>5.1f}" if pts == pts and pts is not None else "    —"
            print(f"  {str(row.position):<4}{str(row.player_name)[:26]:<27}"
                  f"proj {shown}")
        print("\n  Evaluate a specific offer with src.inseason.trades.evaluate().")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
