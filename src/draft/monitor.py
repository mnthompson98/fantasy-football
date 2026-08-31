"""Live draft monitor.

Polls the Sleeper draft endpoint, removes drafted players from the board, and
prints the pick the draft policy would make plus the state of the run.

    python -m src.draft.monitor --draft-id 1262443300887994368 --my-slot 4

Poll interval defaults to 5s. Sleeper's limit is ~1000 req/min; one request
every five seconds is 12/min, which is nothing. Do not drop below 2s — there is
no benefit and it is rude.

**This module runs the same pick policy the backtest scores.** It used to print
a raw best-available list ordered by VORP, which is precisely the policy
`ValueDrafter` was written to replace — see its docstring for the roster it
built. A monitor that recommends by raw VORP while the backtest validates
drop-off drafting is the pick-policy version of the board/backtest divergence
that `features/pipeline.py` exists to prevent: the number in HANDOFF.md would
describe a program nobody is drafting with. So `Roster`, `ValueDrafter` and
`picks_until_next_turn` are imported from the simulator, never reimplemented.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtest.draft_sim import (  # noqa: E402
    Roster,
    ValueDrafter,
    picks_until_next_turn,
)
from src.draft.board import best_available, positional_run, refresh_html  # noqa: E402
from src.features.vorp import LeagueShape  # noqa: E402
from src.ingest.sleeper_api import get_draft, get_draft_picks  # noqa: E402

MIN_POLL_S = 2.0

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "league.yaml"

# `picks_until_next_turn` is re-exported: it lived here first and callers import
# it from here. The implementation is the simulator's, so the live monitor and
# the backtest can never disagree about whose turn it is.
__all__ = ["picks_until_next_turn", "draft_state", "render", "run", "DraftState"]


@dataclass
class DraftState:
    """Everything one poll of the draft feed tells us.

    Split out from the print loop so the whole thing is testable without a
    network or a clock — the live draft path had no tests at all, and the two
    bugs that would have cost the most (a hang on a bad slot, a board that goes
    stale after an undone pick) are both invisible unless you can drive this.
    """

    made: int                       # picks completed
    on_the_clock: int               # pick number now on the clock (1-indexed)
    total: int
    until_my_turn: int
    mine: bool
    drafted: set[str]
    available: pd.DataFrame
    roster: Roster
    runs: dict[str, float] = field(default_factory=dict)
    recommendation: pd.Series | None = None
    complete: bool = False


def _roster_for(picks: list[dict], my_slot: int, board: pd.DataFrame,
                shape: LeagueShape, rounds: int) -> Roster:
    """Rebuild our roster from the feed.

    Sleeper stamps every pick with `draft_slot`, so our picks are identifiable
    without knowing our user id — which matters, because in a mock draft you are
    not the account that owns the league.

    A player we drafted who is not on the board (injury-gated, or one of the
    handful that never reconciled) still occupies a roster spot and still counts
    against the position cap, so he is carried at zero value using the position
    Sleeper reports. Dropping him would silently raise our own caps.
    """
    bench = max(0, rounds - sum(shape.starters.values()) - shape.flex_slots)
    roster = Roster(
        team_id=my_slot - 1,
        starters=dict(shape.starters),
        flex_slots=shape.flex_slots,
        flex_eligible=tuple(shape.flex_eligible),
        bench_slots=bench,
    )

    by_id = board.set_index("player_id") if not board.empty else None
    for p in sorted(picks, key=lambda x: x.get("pick_no") or 0):
        if int(p.get("draft_slot") or 0) != my_slot or not p.get("player_id"):
            continue
        pid = str(p["player_id"])
        meta = p.get("metadata") or {}
        entry = {
            "pick_no": p.get("pick_no"),
            "player_id": pid,
            "player_name": f"{meta.get('first_name', '')} "
                           f"{meta.get('last_name', '')}".strip(),
            "position": meta.get("position") or "",
            "vorp": 0.0,
            "replacement_points": 0.0,
        }
        if by_id is not None and pid in by_id.index:
            row = by_id.loc[pid]
            if isinstance(row, pd.DataFrame):   # duplicate id, take the best
                row = row.iloc[0]
            entry["position"] = str(row["position"]) or entry["position"]
            entry["player_name"] = str(row["player_name"])
            entry["vorp"] = float(row["vorp"])
            entry["replacement_points"] = float(row.get("replacement_points") or 0.0)
        roster.picks.append(entry)
    return roster


def draft_state(picks: list[dict], board: pd.DataFrame, *, my_slot: int,
                teams: int, rounds: int, shape: LeagueShape,
                caps: dict[str, int] | None = None) -> DraftState:
    """Fold the raw picks feed into everything the display needs."""
    if not 1 <= my_slot <= teams:
        raise ValueError(f"my_slot {my_slot} is outside this draft's 1-{teams}")
    made = len(picks)
    total = teams * rounds

    # Match on the *set* of drafted ids, not the count. Sleeper lets a
    # commissioner undo a pick; if the undo and its replacement land between two
    # polls the count is unchanged and a count-based monitor keeps the wrong
    # player struck off for the rest of the draft.
    drafted = {str(p["player_id"]) for p in picks if p.get("player_id")}
    available = board[~board["player_id"].astype(str).isin(drafted)]

    # `picks_until_next_turn` happily counts on past the end of the draft, so
    # after your last pick it reports a turn that will never come — the monitor
    # spent the whole final round telling you four picks were left.
    until = picks_until_next_turn(made, teams, my_slot)
    more_turns = made + 1 + until <= total
    roster = _roster_for(picks, my_slot, board, shape, rounds)

    runs = positional_run([
        {"position": (p.get("metadata") or {}).get("position")}
        for p in sorted(picks, key=lambda x: x.get("pick_no") or 0)
    ])

    state = DraftState(
        made=made,
        on_the_clock=made + 1,
        total=total,
        until_my_turn=until if more_turns else -1,
        mine=(until == 0 and more_turns),
        drafted=drafted,
        available=available,
        roster=roster,
        runs=runs,
        complete=made >= total,
    )

    picks_remaining = rounds - len(roster.picks)
    if (not state.complete and more_turns and picks_remaining > 0
            and not available.empty):
        drafter = ValueDrafter(caps=caps)
        try:
            idx = drafter.choose(available, roster, rounds, picks_remaining,
                                 until)
            state.recommendation = available.loc[idx]
        except (ValueError, KeyError):
            # A recommendation is a nicety; the board is the deliverable. Never
            # let a policy edge case take the monitor down mid-draft.
            state.recommendation = None
    return state


def _line(row: pd.Series) -> str:
    delta = row.get("adp_delta")
    tag = ""
    if pd.notna(delta) and pd.notna(row.get("adp_rank")):
        tag = f"  (ADP {int(row['adp_rank'])}, {int(delta):+d})"
    return (f"{row['position']:<3} {str(row['player_name'])[:24]:<24} "
            f"VORP {row['vorp']:6.1f}{tag}")


def render(state: DraftState, *, top_n: int = 8,
           caps: dict[str, int] | None = None) -> str:
    """The per-pick display, as a string so it can be asserted on."""
    out: list[str] = ["=" * 66]

    if state.complete:
        head = f"pick {state.made}/{state.total} · draft complete"
    elif state.mine:
        head = (f"pick {state.on_the_clock}/{state.total} · "
                f"YOU ARE ON THE CLOCK")
    elif state.until_my_turn < 0:
        head = (f"pick {state.on_the_clock}/{state.total} · "
                f"your picks are done")
    else:
        head = (f"pick {state.on_the_clock}/{state.total} · "
                f"{state.until_my_turn} picks until your turn")
    out.append(head)

    if state.runs:
        desc = ", ".join(f"{k} {v:.0%}" for k, v in sorted(state.runs.items()))
        out.append(f"  run in progress: {desc}")

    counts = state.roster.counts()
    if state.roster.picks:
        shape = " ".join(f"{p}{counts[p]}" for p in sorted(counts))
        need = [p for p, n in state.roster.unfilled_dedicated_slots().items()
                if n > 0]
        gap = f"   need: {', '.join(sorted(need))}" if need else "   starters set"
        out.append(f"  you: {shape} ({len(state.roster.picks)} picks){gap}")

    if state.recommendation is not None:
        out.append("")
        out.append(f"  >> TAKE: {_line(state.recommendation)}")

    out.append("")
    out.append("  best available (cap-adjusted):")
    top = best_available(state.available, set(), n=top_n,
                         roster_counts=counts, caps=caps)
    for _, r in top.iterrows():
        out.append(f"    {_line(r)}")

    # Thin positions are the real scarcity warning.
    out.append("")
    out.append("  top remaining by position:")
    for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
        sub = state.available[state.available["position"] == pos]
        if sub.empty:
            continue
        row = sub.iloc[0]
        n = len(sub)
        cap = (caps or {}).get(pos)
        # The whole point of showing this: as the board thins, a position you
        # have already filled to its cap is not a shortage, it is finished.
        mark = "  [capped]" if cap is not None and counts.get(pos, 0) >= cap else ""
        out.append(f"    {pos:<3} {str(row['player_name'])[:22]:<22} "
                   f"VORP {row['vorp']:6.1f}   ({n} left){mark}")
    out.append("")
    return "\n".join(out)


def run(draft_id: str, board: pd.DataFrame, my_slot: int, *,
        interval: float = 5.0, top_n: int = 8,
        shape: LeagueShape | None = None, caps: dict[str, int] | None = None,
        html_path: Path | None = None, expect_user_id: str | None = None
        ) -> None:
    interval = max(MIN_POLL_S, interval)

    draft = get_draft(draft_id) or {}
    settings = draft.get("settings") or {}
    teams = int(settings.get("teams") or 10)
    rounds = int(settings.get("rounds") or 16)

    if not 1 <= my_slot <= teams:
        raise ValueError(
            f"--my-slot {my_slot} is outside this draft's 1-{teams}. "
            f"The old code looped forever looking for a turn that never came.")

    # A wrong slot is the one input error the monitor cannot detect on its own:
    # it would build somebody else's roster, apply our caps to it, and give
    # confident advice all night. Sleeper publishes the mapping once the order
    # is drawn, so check it rather than trust the flag.
    order = draft.get("draft_order") or {}
    actual = order.get(str(expect_user_id)) if expect_user_id else None
    if actual is not None and int(actual) != my_slot:
        raise ValueError(
            f"--my-slot {my_slot} disagrees with Sleeper: this draft has you "
            f"at slot {int(actual)}. Re-run with --my-slot {int(actual)}.")

    if shape is None:
        shape = LeagueShape(teams=teams,
                            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1,
                                      "K": 1, "DEF": 1})

    print(f"draft {draft_id} · {teams} teams · {rounds} rounds · "
          f"your slot {my_slot}")
    if caps:
        print("  caps: " + ", ".join(f"{k} {v}" for k, v in sorted(caps.items())))
    if html_path:
        print(f"  live board: {html_path}")
    print(f"polling every {interval:.0f}s. ctrl-c to stop.\n")

    last: set[str] | None = None
    while True:
        try:
            picks = get_draft_picks(draft_id)
        except Exception as exc:  # keep polling through transient failures
            print(f"  [warn] poll failed: {exc}")
            time.sleep(interval)
            continue

        state = draft_state(picks, board, my_slot=my_slot, teams=teams,
                            rounds=rounds, shape=shape, caps=caps)

        # `last is None` is the first poll. Print then even at zero picks —
        # otherwise a monitor started before the draft shows nothing at all,
        # and from slot 1 you are on the clock for pick 1 staring at a blank
        # terminal.
        if last is None or state.drafted != last:
            last = state.drafted
            print(render(state, top_n=top_n, caps=caps))
            if html_path:
                try:
                    refresh_html(board, html_path, state.drafted,
                                 meta={"picks": f"{state.made}/{state.total}"})
                except OSError as exc:
                    print(f"  [warn] could not rewrite {html_path}: {exc}")
            if state.complete:
                print("draft complete.")
                return

        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Sleeper draft monitor")
    ap.add_argument("--draft-id", required=True)
    ap.add_argument("--my-slot", type=int, required=True)
    ap.add_argument("--board", default="outputs/projections/draft_board.parquet")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--no-html", action="store_true",
                    help="do not rewrite the phone board as picks come in")
    ap.add_argument("--html", default=None,
                    help="where to write the live phone board "
                         "(default: alongside --board). Point this somewhere "
                         "else when practising on a mock, so the real board "
                         "is not left covered in mock picks.")
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

    shape = None
    caps = None
    user_id = None
    if CONFIG.exists():
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        shape = LeagueShape.from_config(cfg)
        caps = (cfg.get("draft_policy", {}) or {}).get("position_caps") or None
        user_id = (cfg.get("user", {}) or {}).get("sleeper_user_id")

    html_path = None if args.no_html else Path(args.html or path.with_suffix(".html"))

    try:
        run(args.draft_id, board, args.my_slot, interval=args.interval,
            shape=shape, caps=caps, html_path=html_path,
            expect_user_id=user_id)
    except KeyboardInterrupt:
        print("\nstopped.")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
