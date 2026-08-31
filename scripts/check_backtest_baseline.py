"""Guard board-vs-ADP parity against a checked-in baseline.

    python -m scripts.check_backtest_baseline            # compare, exit 1 on drift
    python -m scripts.check_backtest_baseline --write    # re-record the baseline

HANDOFF.md's central finding is that the board and raw ADP finish within 0.2
places of each other, and that ADP has the better playoff points. That is a
*result*, and results rot silently: someone changes a weight in
`features/pipeline.py`, the board still builds, the HTML still renders, every
unit test still passes, and the only thing that moved is the number nobody
re-measures until February.

So the number is checked in. `tests/fixtures/backtest_baseline.json` holds the
board and the pure-ADP baseline at the shipping config, and this script re-runs
both and diffs them.

**The guarded quantity is the gap between the two runs, not either level.**
Levels move when ffverse backfills a season; the gap is measured on the same
data, with the same seed, the same pick policy, the same caps and the same
scorer, changing only the player ordering. If the board's edge over the market
moves, the valuation moved.

Both runs go through `scripts.run_backtest.evaluate`, which is what `main` uses.
A parity check that scored the board through its own copy of the pipeline would
be comparing two programs rather than two orderings.

This is not a fast test. It re-runs the full walk-forward twice, and on a cold
`data/raw/` it pulls six seasons from nflverse first. Run it when the valuation
changes — which is exactly when the answer matters.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_backtest import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"
BASELINE = ROOT / "tests" / "fixtures" / "backtest_baseline.json"

# The two orderings under comparison: our valuation, and the market's.
VALUE_COLS = ("vorp", "adp_value")

# What counts as drift. Seed and inputs are fixed, so an unchanged pipeline on an
# unchanged cache reproduces exactly; these allow for an ffverse backfill moving
# a season underneath us without allowing a real regression through.
TOLERANCE = {
    "parity_gap": 0.35,        # board mean rank minus ADP mean rank
    "playoff_gap": 12.0,       # board playoff points minus ADP playoff points
    "aggregate_rank": 0.40,    # either run's own aggregate
    "season_rank": 0.90,       # a single fold, which is noisier by construction
}

# An absolute floor, independent of the recorded baseline. Even if the fixture
# is stale, the board losing to the raw market by half a draft slot means the
# valuation is no longer earning its place and should not ship unexamined.
MAX_LOSS_TO_ADP = 0.50


# The modules that decide the recorded numbers. The first block is the
# valuation the board and the backtest share; the last is the pick policy, which
# is held constant across the two orderings but moves both levels when it
# changes. An edit to any of them invalidates the baseline.
FINGERPRINTED = (
    "src/features/pipeline.py",
    "src/features/blend.py",
    "src/features/calibration.py",
    "src/features/rank_curve.py",
    "src/features/market_anchor.py",
    "src/features/slopes.py",
    "src/features/vorp.py",
    "src/backtest/draft_sim.py",
)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def valuation_fingerprint() -> dict[str, str]:
    """Semantic hash of the code behind the recorded numbers.

    Hashing the raw source would fire on a reflowed comment and train everyone
    to re-record without looking, which is worse than no check. So this hashes
    the AST with docstrings stripped: renaming a variable trips it, rewrapping a
    paragraph does not.

    It is a tripwire, not a measurement. It cannot tell you whether parity
    moved — only that it *might* have and nobody checked. That is the failure
    this is here to prevent, and unlike re-running the walk-forward it costs
    milliseconds, so CI can carry it.
    """
    out: dict[str, str] = {}
    for rel in FINGERPRINTED:
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = _strip_docstrings(ast.parse(src))
        out[rel] = hashlib.sha256(
            ast.dump(tree).encode("utf-8")).hexdigest()[:16]
    return out


def summarize(summary: pd.DataFrame) -> dict:
    """Reduce a walk-forward summary to the two metrics HANDOFF.md tabulates."""
    per_fold = summary[summary["season"] != "ALL"]
    agg = summary[summary["season"] == "ALL"].iloc[0]
    return {
        "seasons": {
            str(season): {
                "league_rank_by_points": round(
                    float(grp["league_rank_by_points"].mean()), 4),
                "playoff_points": round(float(grp["playoff_points"].mean()), 3),
            }
            for season, grp in per_fold.groupby("season")
        },
        "all": {
            "league_rank_by_points": round(
                float(agg["league_rank_by_points"]), 4),
            "playoff_points": round(float(agg["playoff_points"]), 3),
            "win_rate": round(float(agg["win_rate"]), 4),
        },
        "config_hash": str(agg["config_hash"]),
    }


def compare(actual: dict, baseline: dict, tol: dict | None = None) -> list[str]:
    """Return one line per drift. Empty means the baseline still holds.

    Pure, and separated from the expensive part on purpose: the comparison logic
    is what a unit test can actually cover.
    """
    tol = {**TOLERANCE, **(tol or {})}
    problems: list[str] = []

    board = actual["runs"]["vorp"]
    market = actual["runs"]["adp_value"]
    b0 = baseline["runs"]["vorp"]
    m0 = baseline["runs"]["adp_value"]

    gap = board["all"]["league_rank_by_points"] - market["all"]["league_rank_by_points"]
    gap0 = b0["all"]["league_rank_by_points"] - m0["all"]["league_rank_by_points"]
    if abs(gap - gap0) > tol["parity_gap"]:
        problems.append(
            f"board-vs-ADP rank gap moved {gap0:+.3f} -> {gap:+.3f} "
            f"(tolerance {tol['parity_gap']}). Lower is better; a rise means "
            f"the valuation lost ground to the raw market.")
    if gap > MAX_LOSS_TO_ADP:
        problems.append(
            f"board finishes {gap:+.3f} places behind pure ADP, past the "
            f"{MAX_LOSS_TO_ADP} floor. The valuation is not earning its place.")

    pgap = board["all"]["playoff_points"] - market["all"]["playoff_points"]
    pgap0 = b0["all"]["playoff_points"] - m0["all"]["playoff_points"]
    if abs(pgap - pgap0) > tol["playoff_gap"]:
        problems.append(
            f"board-vs-ADP playoff points moved {pgap0:+.1f} -> {pgap:+.1f} "
            f"(tolerance {tol['playoff_gap']}). Playoff points is the primary "
            f"metric.")

    for col in VALUE_COLS:
        got = actual["runs"][col]["all"]["league_rank_by_points"]
        want = baseline["runs"][col]["all"]["league_rank_by_points"]
        if abs(got - want) > tol["aggregate_rank"]:
            problems.append(
                f"{col}: aggregate rank {want:.3f} -> {got:.3f} "
                f"(tolerance {tol['aggregate_rank']})")

        for season, want_row in baseline["runs"][col]["seasons"].items():
            got_row = actual["runs"][col]["seasons"].get(season)
            if got_row is None:
                problems.append(f"{col}: season {season} missing from the run")
                continue
            d = abs(got_row["league_rank_by_points"]
                    - want_row["league_rank_by_points"])
            if d > tol["season_rank"]:
                problems.append(
                    f"{col} {season}: rank "
                    f"{want_row['league_rank_by_points']:.2f} -> "
                    f"{got_row['league_rank_by_points']:.2f} "
                    f"(tolerance {tol['season_rank']})")
    return problems


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def measure(drafts: int, *, verbose: bool = True) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summaries = evaluate(cfg, value_cols=VALUE_COLS, drafts=drafts,
                         label="baseline-check", verbose=verbose)
    return {
        "drafts_per_season": drafts,
        "seed": int(cfg["backtest"]["seed"]),
        "runs": {col: summarize(s) for col, s in summaries.items()},
    }


def render(actual: dict) -> str:
    board = actual["runs"]["vorp"]
    market = actual["runs"]["adp_value"]
    lines = [f"  {'season':<8}{'board':>9}{'ADP':>9}{'gap':>9}"]
    for season in sorted(board["seasons"]):
        b = board["seasons"][season]["league_rank_by_points"]
        m = market["seasons"][season]["league_rank_by_points"]
        lines.append(f"  {season:<8}{b:>9.2f}{m:>9.2f}{b - m:>+9.2f}")
    b = board["all"]["league_rank_by_points"]
    m = market["all"]["league_rank_by_points"]
    lines.append(f"  {'mean':<8}{b:>9.2f}{m:>9.2f}{b - m:>+9.2f}")
    lines.append(f"  {'plf pts':<8}{board['all']['playoff_points']:>9.1f}"
                 f"{market['all']['playoff_points']:>9.1f}"
                 f"{board['all']['playoff_points'] - market['all']['playoff_points']:>+9.1f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Board-vs-ADP parity guard")
    ap.add_argument("--drafts", type=int, default=40)
    ap.add_argument("--write", action="store_true",
                    help="re-record the baseline instead of checking it. Only "
                         "do this when you have decided the new numbers are "
                         "correct, and say why in the commit message.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    actual = measure(args.drafts, verbose=not args.quiet)
    print("\nboard vs pure ADP (league rank by points; lower is better):")
    print(render(actual))

    if args.write:
        payload = {
            "note": (
                "Board-vs-ADP parity baseline. Guarded quantity is the GAP "
                "between the two runs: same data, same seed, same pick policy, "
                "same caps, same scorer, only the player ordering differs. "
                "Regenerate with `python -m scripts.check_backtest_baseline "
                "--write` and justify the move in the commit message."
            ),
            "recorded": date.today().isoformat(),
            "git_commit": _git_commit(),
            "tolerance": TOLERANCE,
            "max_loss_to_adp": MAX_LOSS_TO_ADP,
            "valuation_fingerprint": valuation_fingerprint(),
            **actual,
        }
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(payload, indent=1) + "\n",
                            encoding="utf-8")
        print(f"\nrecorded {BASELINE.relative_to(ROOT)}")
        return 0

    if not BASELINE.exists():
        print(f"\nno baseline at {BASELINE.relative_to(ROOT)}; "
              f"record one with --write", file=sys.stderr)
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = compare(actual, baseline)
    if problems:
        print(f"\nPARITY DRIFT vs {baseline['recorded']} "
              f"({baseline['git_commit']}):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nIf this change is intended and understood, re-record with "
              "--write.", file=sys.stderr)
        return 1

    print(f"\nparity holds against {baseline['recorded']} "
          f"({baseline['git_commit']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
