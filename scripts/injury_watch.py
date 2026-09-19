"""Daily injury check on your roster only, as a diff against yesterday.

    python -m scripts.injury_watch              # cached Sleeper map if < 12h old
    python -m scripts.injury_watch --refresh    # always re-pull (the scheduled run)

Writes `outputs/reports/injuries.brief.md` (STATUS first, then what changed
since the last run, then who is currently flagged) and keeps its own
snapshot in `outputs/reports/injury_state.json`. That is the whole state:
delete the JSON and the next run is a "first run" again.

This is the daily companion to `weekly_update`. It decides nothing — it
watches sixteen players and says when one of them moved. The lineup call is
Wednesday's brief; on Sunday morning it is you, with this in front of you.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import freshness  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.injuries import weekly_report  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402
from src.ingest.sleeper_api import get_players  # noqa: E402
from src.inseason import injury_watch as iw  # noqa: E402
from src.inseason.roster import load_league_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def _current_week(season: int) -> int | None:
    try:
        import nflreadpy as nfl
        if int(nfl.get_current_season()) != int(season):
            return None
        wk = int(nfl.get_current_week())
        return wk if 1 <= wk <= 18 else None
    except Exception as exc:
        print(f"  [warn] could not read the current week from nflverse: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily roster injury check")
    ap.add_argument("--league-id", default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull Sleeper's player map and the NFL report "
                         "regardless of cache age")
    ap.add_argument("--out", default="outputs/reports")
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
    brief_path = out_dir / "injuries.brief.md"
    state_path = out_dir / "injury_state.json"
    season = int(current.get("season") or nv._current_season())
    week = _current_week(season)
    now = datetime.now(timezone.utc)

    def fail(reason: str) -> int:
        print(reason, file=sys.stderr)
        brief_path.write_text(iw.brief(snapshot={}, changes=[],
                                       previous_taken=None, issues=[],
                                       week=week, season=season,
                                       report_week=None, failed=reason),
                              encoding="utf-8")
        return 1

    freshness.reset()
    print("resolving player identities...")
    # Sleeper asks for at most one pull a day; the scheduled run is exactly
    # one a day, so --refresh forces it and the default keeps a 12h cache
    # for ad-hoc runs in between.
    players = get_players(cache_dir=nv.RAW_DIR,
                          max_age_hours=0.0 if args.refresh else 12.0)
    cw = build_crosswalk(nv.load_ff_playerids(refresh=args.refresh), players)

    state = load_league_state(league_id, cw,
                              user_id=cfg["user"]["sleeper_user_id"])
    if state.my_roster.empty:
        return fail(f"no roster for user {cfg['user']['sleeper_user_id']} in "
                    f"league {league_id}")

    report = weekly_report(season, week, refresh=args.refresh)
    if report.empty and week:
        report = weekly_report(season, None, refresh=args.refresh)
    report_week = int(report["report_week"].max()) if not report.empty else None

    snapshot = iw.build_snapshot(state.my_roster, players, report)
    previous, previous_taken = iw.load_snapshot(state_path)
    changes = iw.diff(previous, snapshot)

    issues = [f"STALE CACHE: {fb['name']} could not be pulled ({fb['error']}); "
              f"used a cached copy aged {fb['age_hours']:.1f}h"
              for fb in freshness.FALLBACKS]
    if week and report_week and report_week != week:
        issues.append(f"NFL report is week {report_week}'s; week {week}'s is "
                      f"not published yet (normal before Wednesday afternoon)")

    text = iw.brief(snapshot=snapshot, changes=changes,
                    previous_taken=previous_taken, issues=issues,
                    week=week, season=season, report_week=report_week, now=now)
    brief_path.write_text(text, encoding="utf-8")
    iw.save_snapshot(state_path, snapshot, now)
    print(f"brief written:  {brief_path.relative_to(ROOT)}\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
