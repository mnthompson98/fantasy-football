"""Check a Sleeper league against the settings this repo assumes.

    python -m scripts.verify_league_settings <league_id>

CLAUDE.md records a block of *verified* settings — full PPR, rolling waivers,
weeks 15-17 playoffs, a 9-slot starting lineup — that were confirmed against the
2021 and 2025 leagues. It also records a block of *assumed* 2026 settings that
nobody has confirmed, because the 2026 league did not exist when they were
written.

Run this the moment the 2026 league is created. Every downstream number depends
on these: replacement level is a function of league size and starting slots, the
whole board is priced in full-PPR points, and the waiver module assumes rolling
priority rather than FAAB. A silent change to any of them produces a board that
is confidently wrong.

Exit code is 1 if anything disagrees, so this is safe to gate a build on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.sleeper_api import get_league  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"

WAIVER_TYPES = {0: "rolling_priority", 1: "reverse_standings", 2: "faab"}


def _starters(roster_positions: list[str]) -> list[str]:
    """Sleeper lists bench and IR alongside the real starting slots."""
    return [p for p in roster_positions if p not in ("BN", "IR", "TAXI")]


def _bench(roster_positions: list[str]) -> int:
    return sum(1 for p in roster_positions if p == "BN")


def check(league: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """Compare a live league to the config. Returns (agreements, mismatches)."""
    verified = cfg["verified"]
    settings = league.get("settings") or {}
    scoring = league.get("scoring_settings") or {}
    roster_positions = league.get("roster_positions") or []

    ok: list[str] = []
    bad: list[str] = []

    def compare(label: str, expected, actual, *, note: str = "") -> None:
        suffix = f"  ({note})" if note else ""
        if expected == actual:
            ok.append(f"{label:<22} {actual}{suffix}")
        else:
            bad.append(f"{label:<22} expected {expected!r}, got {actual!r}{suffix}")

    waiver_type = settings.get("waiver_type")
    compare("waiver_type", verified["waiver_type"], waiver_type,
            note=WAIVER_TYPES.get(waiver_type, "unknown"))
    if waiver_type == 2:
        bad.append(
            "waiver_budget          league is on FAAB. Every waiver decision in "
            "src/inseason/ assumes rolling priority and is now wrong."
        )

    compare("playoff_week_start", verified["playoff_week_start"],
            settings.get("playoff_week_start"))
    compare("playoff_teams", verified["playoff_teams"],
            settings.get("playoff_teams"))
    compare("trade_deadline_week", verified["trade_deadline_week"],
            settings.get("trade_deadline"))

    # Full PPR is not a flag on Sleeper; it is the value of a reception.
    compare("points per reception", 1.0, float(scoring.get("rec", 0.0)),
            note="full PPR" if float(scoring.get("rec", 0)) == 1.0 else "NOT full PPR")
    compare("passing TD", 4.0, float(scoring.get("pass_td", 0.0)))

    compare("starters", list(verified["starters"]), _starters(roster_positions))

    # Roster shape drives replacement level, so a change here silently reprices
    # the whole board.
    compare("teams", verified.get("teams"),
            settings.get("num_teams") or league.get("total_rosters"))
    compare("bench_slots", verified.get("bench_slots"), _bench(roster_positions))
    compare("roster_size", verified.get("roster_size"), len(roster_positions))

    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify a Sleeper league against config/league.yaml")
    ap.add_argument("league_id")
    args = ap.parse_args()

    league = get_league(args.league_id)
    if not league:
        print(f"error: no league {args.league_id} (Sleeper returned null)",
              file=sys.stderr)
        return 1

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ok, bad = check(league, cfg)

    print(f"{league.get('name')} · season {league.get('season')} · "
          f"{league.get('total_rosters')} teams · status {league.get('status')}\n")

    for line in ok:
        print(f"  ok    {line}")
    for line in bad:
        print(f"  DIFF  {line}")

    if bad:
        print(f"\n{len(bad)} setting(s) disagree with config/league.yaml.")
        print("Update the config before building a board — replacement level, "
              "scoring and waiver logic all depend on these.")
        return 1

    print(f"\nAll {len(ok)} checked settings match.")
    current = cfg.get("current") or {}
    if current.get("league_id") and current["league_id"] != args.league_id:
        print(f"\nNote: config `current.league_id` is {current['league_id']}, "
              f"not the league just checked. Update it if this is the one you "
              f"are drafting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
