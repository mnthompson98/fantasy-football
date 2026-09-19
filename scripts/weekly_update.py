"""The weekly in-season run: lineup, waivers, and what you have to trade.

    python -m scripts.weekly_update
    python -m scripts.weekly_update --refresh          # force fresh pulls
    python -m scripts.weekly_update --check-only       # ingestion health only

Reads the live league from `config.current.league_id`, pulls this week's
FantasyPros projections, and answers the three questions that matter each week.

Two files come out of a full run. `outputs/reports/weekNN.md` is the whole
report; `outputs/reports/weekNN.brief.md` (also copied to `latest.brief.md`)
is the short form with a STATUS line first — OK, DEGRADED when any input was
stale, FAILED when the run refused — followed by only the verdicts. The brief
is what a scheduled reader relays; it is printed last so it is also the last
thing on the terminal.

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
from src.inseason import matchup, report, start_sit, trades, waivers  # noqa: E402
from src.inseason.projections import (  # noqa: E402
    attach_projections,
    load_weekly_projections,
)
from src.inseason.roster import load_league_state  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.injuries import apply_to_roster, weekly_report  # noqa: E402
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


def _current_week(season: int) -> int | None:
    """The NFL week about to be played, or None before kickoff.

    nflverse keeps the calendar; asking it beats inferring the week from
    whichever data source happens to have updated last.
    """
    try:
        import nflreadpy as nfl
        if int(nfl.get_current_season()) != int(season):
            return None                       # preseason of a new year
        wk = int(nfl.get_current_week())
        return wk if 1 <= wk <= 18 else None
    except Exception as exc:
        print(f"  [warn] could not read the current week from nflverse: {exc}")
        return None


def _trending(cw, kind: str) -> dict[str, int]:
    """Sleeper's trending adds/drops on `player_key`. Empty on failure: this
    is colour, and a network blip here must not discard the lineup, waiver
    and trade work that has already been done."""
    try:
        entries = get_trending(kind, limit=100)
    except Exception as exc:
        print(f"  [warn] trending {kind}s unavailable: {exc}")
        return {}
    out: dict[str, int] = {}
    for entry in entries:
        key = cw.resolve(sleeper_id=entry.get("player_id"))
        if key:
            out[key] = int(entry.get("count") or 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly in-season update")
    ap.add_argument("--league-id", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="verify weekly ingestion and stop")
    ap.add_argument("--waiver-threshold", type=float, default=None,
                    help="points a claim must add before it is worth priority")
    ap.add_argument("--week", type=int, default=None,
                    help="NFL week being decided (default: nflverse's current "
                         "week, i.e. the upcoming one)")
    ap.add_argument("--out", default="outputs/reports",
                    help="where to write the markdown report")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    current = cfg.get("current") or {}
    league_id = args.league_id or current.get("league_id")
    if not league_id:
        print("no league id: pass --league-id or set config.current.league_id",
              file=sys.stderr)
        return 1

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    season = int(current.get("season") or nv._current_season())
    week = args.week if args.week is not None else _current_week(season)
    stem = f"week{week:02d}" if week else "preseason"

    def _write_brief(text: str) -> Path:
        path = out_dir / f"{stem}.brief.md"
        path.write_text(text, encoding="utf-8")
        (out_dir / "latest.brief.md").write_text(text, encoding="utf-8")
        return path

    nv.reset_fallbacks()
    print("resolving player identities...")
    cw = build_crosswalk(nv.load_ff_playerids(refresh=args.refresh),
                         get_players(cache_dir=nv.RAW_DIR))
    print(f"  {len(cw.meta)} players in crosswalk\n")

    problems = check_ingestion(cfg, cw, refresh=args.refresh)
    if args.check_only:
        return 1 if problems else 0
    if problems:
        print("\nrefusing to advise on broken ingestion.", file=sys.stderr)
        _write_brief(report.brief(
            week=week, season=season, issues=[],
            failed=f"{problems} ingestion problem(s); see the check above"))
        return 1

    proj = load_weekly_projections(cw, refresh=args.refresh)
    slots = LineupSlots.from_config(cfg)
    state = load_league_state(league_id, cw,
                              user_id=cfg["user"]["sleeper_user_id"])

    if state.my_roster.empty:
        print(f"\nNo roster found for user {cfg['user']['sleeper_user_id']} in "
              f"league {league_id}. Before the draft this is expected.")
        _write_brief(report.brief(
            week=week, season=season, issues=[],
            failed=f"no roster for user {cfg['user']['sleeper_user_id']} in "
                   f"league {league_id} (expected before the draft)"))
        return 0

    roster = attach_projections(state.my_roster, proj)

    # The week being *decided* is the upcoming one (computed above, before
    # ingestion, so a failed run can still name the week it failed for). It
    # used to be inferred from the latest injury report, which on a Tuesday
    # night is last week's: the report was titled with the week just played,
    # overwrote that week's file, and pulled game context for games already
    # over. nflverse's own calendar says which week is next; the season comes
    # from config so the 2025-vs-2026 flip in early September cannot move it.

    # The official weekly injury report is the in-season authority. Sleeper's
    # season-long designations (IR, PUP) still gate on top of it — the two
    # sources cover different things and neither replaces the other. The
    # report for the coming week fills in Wednesday-Friday; before that, the
    # latest one available is the best there is, and it is labelled as such.
    inj = weekly_report(season, week)
    if inj.empty and week:
        inj = weekly_report(season, None)
    rep_week = int(inj["report_week"].max()) if not inj.empty else None
    if not inj.empty:
        roster = apply_to_roster(roster, inj)
        flagged = int((roster["concern"] != "clear").sum())
        stale = f" (week {rep_week} report; week {week}'s not published yet)" \
            if rep_week != week else ""
        print(f"  injury report: week {week}{stale} · {len(inj)} listed · "
              f"{flagged} on your roster")
    else:
        print("  injury report: none published yet (preseason)")

    # Market and weather context. Flags only — see matchup.py for why these
    # must not be used to adjust the projections.
    ctx = matchup.week_context(season, week) if week else pd.DataFrame()
    if not ctx.empty:
        roster = matchup.attach(roster, ctx)

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
    trending = _trending(cw, "add")

    # Drops are the other half of the plan's "trending adds/drops". They are not
    # a waiver signal — a player being dropped everywhere is available, not
    # good. They are a *trade* signal: the leagues giving up on him are telling
    # you where the buy-low is, if you disagree with them.
    dropped = _trending(cw, "drop")

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

    # Buy-low candidates: rostered elsewhere in *this* league, and being dropped
    # in leagues at large. If you disagree with the panic, that is the trade.
    panic = [(k, n) for k, n in sorted(dropped.items(), key=lambda kv: -kv[1])
             if k in state.all_rostered and k not in set(roster["player_key"])]
    if panic:
        print(f"\n{'=' * 62}\nBUY-LOW  (owned here, being dropped elsewhere)\n{'=' * 62}")
        for key, count in panic[:6]:
            meta = cw.meta.get(key, {})
            print(f"  {str(meta.get('position') or '?'):<4}"
                  f"{str(meta.get('player_name') or key)[:26]:<27}"
                  f"dropped in {count:,} leagues · owned by "
                  f"{state.owner_of.get(key, '?')}")

    md = report.build(
        week=week, season=season, lineup=lineup, calls=calls, holes=holes,
        slots=slots, moves=moves, waiver_threshold=threshold,
        surplus=surplus, roster=roster,
        scrape_date=str(proj["scrape_date"].max()) if len(proj) else None,
        league_name=str((cfg.get("current") or {}).get("name") or ""),
    )
    path = out_dir / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    print(f"\nreport written: {path.relative_to(ROOT)}")

    # The brief goes last, status first. The weekly feed's kickoff column is
    # the one check that catches a fresh-looking file ranking a week already
    # played; it is not carried onto `proj`, so read it off the cached raw
    # pull (free — same TTL, same file).
    raw = nv.load_ff_rankings_weekly()
    last_kickoff = None
    if "player_game_kickoff_ts" in raw:
        ts = pd.to_numeric(raw["player_game_kickoff_ts"], errors="coerce").max()
        if ts == ts:
            last_kickoff = pd.to_datetime(ts, unit="s", utc=True).to_pydatetime()
    scraped = str(proj["scrape_date"].max()) if len(proj) else None
    issues = report.freshness_issues(
        scrape_date=scraped, last_kickoff_utc=last_kickoff,
        fallbacks=list(nv.FALLBACKS), week=week, report_week=rep_week,
    )
    text = report.brief(
        week=week, season=season, issues=issues, lineup=lineup, calls=calls,
        holes=holes, moves=moves, waiver_threshold=threshold, surplus=surplus,
        roster=roster, scrape_date=scraped,
        report_path=str(path.relative_to(ROOT)),
    )
    bpath = _write_brief(text)
    print(f"brief written:  {bpath.relative_to(ROOT)}\n")
    print(f"{'=' * 62}\nBRIEF\n{'=' * 62}")
    print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
