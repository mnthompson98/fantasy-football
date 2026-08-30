"""Live draft monitor.

Polls the Sleeper draft endpoint, removes drafted players from the board, and
prints best-available plus any positional run in progress.

    python -m src.draft.monitor --draft-id 1262443300887994368 --my-slot 4

Poll interval defaults to 5s. Sleeper's limit is ~1000 req/min; one request
every five seconds is 12/min, which is nothing. Do not drop below 2s — there is
no benefit and it is rude.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.draft.board import best_available, positional_run  # noqa: E402
from src.ingest.sleeper_api import get_draft, get_draft_picks  # noqa: E402

MIN_POLL_S = 2.0


def picks_until_next_turn(pick_no: int, teams: int, my_slot: int) -> int:
    """Picks between now and your next turn in a snake draft.

    This is the number that decides whether to reach: if the tier you want has
    fewer players left than this, it will be gone.
    """
    nxt = pick_no + 1
    while True:
        rnd = (nxt - 1) // teams
        idx = (nxt - 1) % teams
        slot = idx + 1 if rnd % 2 == 0 else teams - idx
        if slot == my_slot:
            return nxt - pick_no - 1
        nxt += 1


def run(draft_id: str, board: pd.DataFrame, my_slot: int, *,
        interval: float = 5.0, top_n: int = 8) -> None:
    interval = max(MIN_POLL_S, interval)

    draft = get_draft(draft_id) or {}
    teams = int((draft.get("settings") or {}).get("teams") or 10)
    rounds = int((draft.get("settings") or {}).get("rounds") or 16)
    total = teams * rounds

    print(f"draft {draft_id} · {teams} teams · {rounds} rounds · your slot {my_slot}")
    print(f"polling every {interval:.0f}s. ctrl-c to stop.\n")

    seen = 0
    while True:
        try:
            picks = get_draft_picks(draft_id)
        except Exception as exc:  # keep polling through transient failures
            print(f"  [warn] poll failed: {exc}")
            time.sleep(interval)
            continue

        if len(picks) != seen:
            seen = len(picks)
            drafted = {p.get("player_id") for p in picks}
            avail = board[~board["player_id"].isin(drafted)]

            until = picks_until_next_turn(seen, teams, my_slot)
            on_clock = until == 0

            print("=" * 66)
            head = f"pick {seen}/{total}"
            print(f"{head} · {'YOU ARE ON THE CLOCK' if on_clock else f'{until} picks until your turn'}")

            recent = [
                {"position": (p.get("metadata") or {}).get("position")}
                for p in sorted(picks, key=lambda x: x.get("pick_no") or 0)
            ]
            runs = positional_run(recent)
            if runs:
                desc = ", ".join(f"{k} {v:.0%}" for k, v in sorted(runs.items()))
                print(f"  run in progress: {desc}")

            print("\n  best available:")
            for _, r in best_available(avail, drafted, n=top_n).iterrows():
                delta = r.get("adp_delta")
                tag = ""
                if pd.notna(delta):
                    tag = f"  (ADP {int(r['adp_rank'])}, {int(delta):+d})"
                print(
                    f"    {r['position']:<3} {str(r['player_name'])[:24]:<24} "
                    f"VORP {r['vorp']:6.1f}{tag}"
                )

            # Thin positions are the real scarcity warning.
            print("\n  top remaining by position:")
            for pos in ("QB", "RB", "WR", "TE"):
                sub = avail[avail["position"] == pos].head(1)
                if not sub.empty:
                    row = sub.iloc[0]
                    n = int((avail["position"] == pos).sum())
                    print(f"    {pos:<3} {str(row['player_name'])[:22]:<22} "
                          f"VORP {row['vorp']:6.1f}   ({n} left)")
            print()

            if seen >= total:
                print("draft complete.")
                return

        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Sleeper draft monitor")
    ap.add_argument("--draft-id", required=True)
    ap.add_argument("--my-slot", type=int, required=True)
    ap.add_argument("--board", default="outputs/projections/draft_board.parquet")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    path = Path(args.board)
    if not path.exists():
        print(
            f"error: no board at {path}\n"
            f"build one first: python -m scripts.build_draft_board",
            file=sys.stderr,
        )
        return 1

    board = pd.read_parquet(path)
    try:
        run(args.draft_id, board, args.my_slot, interval=args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
