"""Paired comparison of the two drop-off lookahead rules at one draft slot.

    python -m scripts.compare_lookahead --my-slot 10

`scripts.run_backtest --lookahead ...` run twice gives two *aggregate* numbers,
and over four folds the gap between them is well inside the noise floor
CLAUDE.md warns about. This does the comparison the data actually supports: the
two rules face the **same board, the same seed and the same opponents** in every
simulated draft, so the drafts pair up one-to-one and the statistic is the mean
of the per-draft *difference*. That kills the between-draft variance, which is
the variance that swamps an unpaired read.

It answers "is this rule different from that one", not "is the board any good".
It writes nothing to `outputs/backtests/runs.parquet` — the two rules are not
two runs of a shipping config, and logging them as if they were would put rows
in the record that no config hash distinguishes usefully.

Only slots 1 and `teams` can show any difference at all; see
`draft_sim.picks_until_board_moves`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_backtest import (  # noqa: E402
    backtest_config,
    make_board_builder,
    make_opponent_provider,
)
from src.backtest.draft_sim import LOOKAHEAD_RULES  # noqa: E402
from src.backtest.walkforward import build_folds, run_fold  # noqa: E402
from src.features.scoring import Scoring  # noqa: E402
from src.ingest import history as H  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"

# Higher is better for all but rank, where finishing 1st is the good end.
METRICS = [
    ("playoff_points", 1, "playoff pts (15-17)"),
    ("season_points", 1, "season pts"),
    ("win_rate", 1, "win rate"),
    ("league_rank_by_points", -1, "league rank"),
]


def _paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple:
    """(mean_a, mean_b, mean_diff, t) for `col`, paired on (season, seed)."""
    m = a.merge(b, on=["season", "seed"], suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("no drafts paired — the seed sequences diverged")
    d = m[f"{col}_b"] - m[f"{col}_a"]
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan")
    return float(m[f"{col}_a"].mean()), float(m[f"{col}_b"].mean()), \
        float(d.mean()), t, len(d)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired lookahead-rule comparison")
    ap.add_argument("--my-slot", type=int, default=None,
                    help="default: config current.my_slot")
    ap.add_argument("--drafts", type=int, default=40)
    ap.add_argument("--opponent", default=None, choices=("gaussian", "league"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    slot = args.my_slot if args.my_slot is not None else cfg["current"].get("my_slot")
    if slot is None:
        print("error: no --my-slot and config current.my_slot is null",
              file=sys.stderr)
        return 1
    slot = int(slot)

    base = backtest_config(cfg, drafts=args.drafts, opponent=args.opponent,
                           my_slot=slot)
    print("loading history (once, shared by both rules)...")
    scoring = Scoring.from_league(None)
    crosswalk = build_crosswalk(nv.load_ff_playerids(refresh=args.refresh))
    weekly = H.scored_weekly(base.seasons, scoring, refresh=args.refresh)
    totals = H.season_totals_for(base.seasons, scoring)
    rankings = nv.load_ff_rankings_history(refresh=args.refresh)
    weekly_actuals = weekly.rename(
        columns={"player_key": "player_id", "fantasy_points": "points"})
    weekly_actuals = weekly_actuals[weekly_actuals["season_type"] == "REG"]

    provider = make_opponent_provider(cfg, rankings=rankings,
                                      crosswalk=crosswalk,
                                      opponent=base.opponent)
    builder = make_board_builder(cfg, totals=totals, rankings=rankings,
                                 crosswalk=crosswalk)

    turn = slot in (1, base.teams)
    print(f"slot {slot} of {base.teams}"
          f"{' (turn slot)' if turn else ' — NOT a turn slot; expect no difference'}"
          f" · {args.drafts} drafts/season · field {base.opponent}\n")

    per_rule: dict[str, pd.DataFrame] = {}
    for rule in LOOKAHEAD_RULES:
        rule_cfg = backtest_config(cfg, drafts=args.drafts,
                                   opponent=args.opponent, my_slot=slot,
                                   lookahead=rule)
        rows = []
        for fold in build_folds(rule_cfg):
            factory = provider(fold, rule_cfg)[0] if provider else None
            rows.extend(r.to_row() for r in run_fold(
                fold, rule_cfg, board_builder=builder,
                weekly_actuals=weekly_actuals, opponent_factory=factory))
        per_rule[rule] = pd.DataFrame(rows)
        print(f"  {rule}: {len(rows)} drafts simulated")

    a, b = per_rule["next_pick"], per_rule["next_exposed"]
    print(f"\npaired over the same board/seed/opponents "
          f"(next_exposed minus next_pick):")
    print(f"  {'metric':<22}{'next_pick':>11}{'next_exposed':>14}"
          f"{'diff':>9}{'t':>8}")
    for col, sign, label in METRICS:
        ma, mb, diff, t, n = _paired(a, b, col)
        better = "better" if diff * sign > 0 else "worse"
        note = "" if abs(t) >= 2 else "  (n.s.)"
        print(f"  {label:<22}{ma:>11.2f}{mb:>14.2f}{diff:>+9.2f}{t:>8.2f}"
              f"  {better}{note}")
    print(f"\n  n = {n} paired drafts. |t| >= 2 is roughly the 5% level; "
          f"anything below is noise.")

    print("\nper season (league rank, lower is better):")
    for season in sorted(a["season"].unique()):
        ma, mb, diff, t, n = _paired(a[a["season"] == season],
                                     b[b["season"] == season],
                                     "league_rank_by_points")
        print(f"  {season}  {ma:>6.2f} -> {mb:>6.2f}   {diff:>+6.2f}  t={t:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
