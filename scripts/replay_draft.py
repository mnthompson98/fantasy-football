"""Replay a completed Sleeper draft through the live monitor.

    python -m scripts.replay_draft --draft-id 1262443300887994368 --my-slot 4

The live draft path runs once a year with a clock ticking, which is the worst
possible time to find out it does not work. This drives `src.draft.monitor.run`
against a real, completed Sleeper draft — its genuine payloads, its player ids,
its team-defense keys — feeding the picks back one at a time as if they were
arriving live. Everything above the HTTP call is the production code path.

It is a rehearsal, not a test; `tests/test_draft_monitor.py` is the test. Use
this before draft day to watch the thing behave, and after a Sleeper API change
to confirm the payload shape still parses.

Two real drafts from this league's own history:

    1262443300887994368   2025, 10 teams, 16 rounds  (the current shape)
    736349931933196288    2021,  8 teams, 15 rounds  (a different one)

Cross-season replay puts a 2026 board against an older season's picks, so most
players will not match and the board will barely thin. That is expected and it
does not affect what is being rehearsed: the polling, the pick detection, the
roster reconstruction, the caps, the HTML rewrite and the end-of-draft exit.

The live board is written somewhere disposable by default. Replaying onto
`outputs/projections/draft_board.html` would leave the real phone board covered
in someone else's picks.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.draft import monitor as M  # noqa: E402
from src.features.vorp import LeagueShape  # noqa: E402
from src.ingest.sleeper_api import get_draft_picks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a Sleeper draft locally")
    ap.add_argument("--draft-id", default="1262443300887994368")
    ap.add_argument("--my-slot", type=int, default=4)
    ap.add_argument("--board", default="outputs/projections/draft_board.parquet")
    ap.add_argument("--html", default=None,
                    help="where to write the live board (default: a temp file)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="seconds between simulated picks; 0 runs flat out")
    args = ap.parse_args()

    board_path = Path(args.board)
    if not board_path.exists():
        print(f"error: no board at {board_path}\n"
              f"build one first: python -m scripts.build_draft_board",
              file=sys.stderr)
        return 1

    picks = sorted(get_draft_picks(args.draft_id),
                   key=lambda p: p.get("pick_no") or 0)
    if not picks:
        print(f"error: draft {args.draft_id} has no picks to replay",
              file=sys.stderr)
        return 1
    print(f"replaying {len(picks)} picks from draft {args.draft_id}\n")

    # Serve one more pick on each poll. The monitor cannot tell the difference,
    # which is the point: nothing below this line knows it is a replay.
    served = {"n": 0}

    def feed(_draft_id: str) -> list[dict]:
        n = served["n"]
        served["n"] = min(n + 1, len(picks))
        return picks[:n]

    real_picks, real_sleep = M.get_draft_picks, M.time.sleep
    M.get_draft_picks = feed
    if args.speed <= 0:
        M.time.sleep = lambda _s: None

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    html = Path(args.html) if args.html else (
        Path(tempfile.gettempdir()) / "draft_board_replay.html")

    try:
        M.run(args.draft_id, pd.read_parquet(board_path), args.my_slot,
              interval=args.speed, shape=LeagueShape.from_config(cfg),
              caps=(cfg.get("draft_policy", {}) or {}).get("position_caps"),
              html_path=html)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        M.get_draft_picks, M.time.sleep = real_picks, real_sleep

    print(f"\nserved {served['n']}/{len(picks)} picks · live board at {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
