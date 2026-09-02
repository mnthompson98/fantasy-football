"""Mock-draft this year's board from a fixed slot, against this league's field.

    python -m scripts.mock_draft                       # slot from config, league field
    python -m scripts.mock_draft --drafts 200          # position mix over many runs
    python -m scripts.mock_draft --lookahead next_exposed

This is the *offline* rehearsal: the live 2026 board, drafted from our real
slot, against the opponent model fitted to this league's own 2021 and 2025
drafts. It answers "what roster does this system actually build from here",
which `scripts/replay_draft.py` cannot — that one replays a finished draft's
real picks, so our own choices never change anything downstream.

Two things it is deliberately NOT:

  - Not a backtest. There is no 2026 outcome to score against, so nothing here
    says a roster is *good*. Use `scripts.run_backtest` for that; the metric
    priority in `src/backtest/metrics.py` is the one that counts.
  - Not a substitute for a live Sleeper mock. Everything above the HTTP call is
    exercised here, but the feed parsing, the id reconciliation against
    Sleeper's own player ids, and the HTML board are not. Do both.

The default field is `league`, not the config's shipping `gaussian`. The
shipping default exists so recorded numbers keep reproducing (CLAUDE.md); for
a rehearsal you want the field you will actually sit across from, which by the
measurement in HANDOFF.md is worth about 1.2 places.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.draft_sim import (  # noqa: E402
    LOOKAHEAD_RULES,
    league_opponent,
    simulate_draft,
)
from src.backtest.opponent_fit import TendencyFitter  # noqa: E402
from src.features.vorp import LeagueShape  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def _field(cfg: dict, season: int, *, opponent: str):
    """The opponent factory, plus a line describing where it came from."""
    if opponent == "gaussian":
        return None, "gaussian (ADP + noise)"
    opp = cfg["backtest"].get("opponent_model", {}) or {}
    fitter = TendencyFitter(
        nv.load_ff_rankings_history(), build_crosswalk(nv.load_ff_playerids()),
        # `all` rather than `prior`: for a 2026 rehearsal every recorded draft
        # is already in the past, so there is no fold boundary to respect and
        # nothing to leak. `prior` would mean the same thing here anyway.
        scope="all",
        over_ceiling_penalty=float(opp.get("over_ceiling_penalty", 8.0)),
    )
    tend = fitter.for_season(season)
    return league_opponent(tend), f"league-fitted ({tend.provenance})"


def _show_roster(roster, rounds: int, teams: int, value_col: str) -> None:
    print(f"  {'rd':>3} {'pick':>5}  {'pos':<4}{'player':<24}"
          f"{'VORP':>7}{'ADP':>6}")
    for p in roster.picks:
        print(f"  {p['round']:>3} {p['pick_no']:>5}  {p['position']:<4}"
              f"{str(p['player_name'])[:23]:<24}"
              f"{float(p[value_col]):>7.1f}{int(p['adp_rank']):>6}")
    counts = Counter(p["position"] for p in roster.picks)
    print("  shape: " + " ".join(f"{k}{counts[k]}" for k in sorted(counts)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Mock draft the live board")
    ap.add_argument("--board", default="outputs/projections/draft_board.parquet")
    ap.add_argument("--slot", type=int, default=None,
                    help="draft slot (default: config current.my_slot)")
    ap.add_argument("--drafts", type=int, default=1,
                    help="how many to run; >1 prints the position mix instead "
                         "of every roster")
    ap.add_argument("--seed", type=int, default=None,
                    help="default: config backtest.seed")
    ap.add_argument("--opponent", default="league",
                    choices=("league", "gaussian"))
    ap.add_argument("--lookahead", default=None,
                    choices=tuple(LOOKAHEAD_RULES),
                    help="default: run every rule, so they can be compared")
    ap.add_argument("--value-col", default="vorp")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    path = Path(args.board)
    if not path.exists():
        print(f"error: no board at {path}\n"
              f"build one first: python -m scripts.build_draft_board",
              file=sys.stderr)
        return 1
    board = pd.read_parquet(path)

    slot = args.slot if args.slot is not None else cfg["current"].get("my_slot")
    if slot is None:
        print("error: no --slot and config current.my_slot is null",
              file=sys.stderr)
        return 1
    slot = int(slot)

    shape = LeagueShape.from_config(cfg)
    rounds = int(cfg["verified"].get("roster_size", 16))
    season = int(cfg["current"]["season"])
    seed = args.seed if args.seed is not None else int(
        cfg["backtest"].get("seed", 20260830))
    caps = cfg.get("draft_policy", {}).get("position_caps") or None
    opp = cfg["backtest"].get("opponent_model", {}) or {}

    if not 1 <= slot <= shape.teams:
        print(f"error: slot {slot} outside 1-{shape.teams}", file=sys.stderr)
        return 1

    factory, provenance = _field(cfg, season, opponent=args.opponent)
    rules = (args.lookahead,) if args.lookahead else tuple(LOOKAHEAD_RULES)

    turn = slot in (1, shape.teams)
    print(f"{season} board · {len(board)} players · slot {slot} of "
          f"{shape.teams}{' (TURN SLOT — back-to-back picks)' if turn else ''}")
    print(f"  {rounds} rounds · caps {caps} · field: {provenance}")
    print(f"  seed {seed} · {args.drafts} draft(s) per lookahead rule\n")

    for rule in rules:
        print("=" * 68)
        print(f"lookahead rule: {rule}")
        print("=" * 68)
        mixes: list[Counter] = []
        for i in range(args.drafts):
            rosters = simulate_draft(
                board, teams=shape.teams, rounds=rounds, my_slot=slot,
                starters=dict(shape.starters), flex_slots=shape.flex_slots,
                flex_eligible=tuple(shape.flex_eligible),
                sigma={k: float(v) for k, v in
                       (opp.get("adp_sigma_fallback") or {}).items()},
                need_boost=float(opp.get("positional_need_boost", 1.6)),
                need_penalty=float(opp.get("positional_need_penalty", 0.4)),
                value_col=args.value_col, seed=seed + i, caps=caps,
                opponent_factory=factory, lookahead_rule=rule,
            )
            mine = rosters[slot - 1]
            mixes.append(Counter(p["position"] for p in mine.picks))
            if args.drafts == 1:
                _show_roster(mine, rounds, shape.teams, args.value_col)
        if args.drafts > 1:
            print(f"  position mix over {args.drafts} drafts (mean per roster):")
            for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
                vals = [m.get(pos, 0) for m in mixes]
                print(f"    {pos:<4}{sum(vals)/len(vals):>5.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
