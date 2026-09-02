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
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.backtest.draft_sim import (  # noqa: E402
    LOOKAHEAD_RULES,
    Candidate,
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
__all__ = ["picks_until_next_turn", "draft_state", "render", "run", "DraftState",
           "scoring_mismatch", "default_html_path", "explain", "resolve_my_slot"]

# Sleeper's own vocabulary for a draft's scoring format, from `draft.metadata`.
# This is frequently the *only* place a mock draft (`league_id: null`, no
# scoring_settings anywhere) records its scoring at all.
_SCORING_TYPE_REC = {"std": 0.0, "standard": 0.0, "half_ppr": 0.5, "ppr": 1.0}
_REC_LABEL = {0.0: "standard", 0.5: "half-PPR", 1.0: "full PPR"}


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
    # `candidates` is the full ranked list from `ValueDrafter.rank()` — the
    # pick plus a couple of runners-up, each with its own reasoning, not just
    # the winner. `recommendation`/`reason` are `candidates[0]`'s board row
    # and its `.reason`, kept as their own fields because that is what
    # existing callers (and the HTML/terminal renderers) already expect.
    candidates: list[Candidate] = field(default_factory=list)
    recommendation: pd.Series | None = None
    reason: str | None = None
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
                caps: dict[str, int] | None = None,
                lookahead_rule: str = "next_pick") -> DraftState:
    """Fold the raw picks feed into everything the display needs.

    `lookahead_rule` is `config draft_policy.lookahead_rule`, the same key
    `scripts/run_backtest.py` reads. It must not become a literal here: the
    monitor and the backtest running different pick policies is the failure
    this module's own docstring exists to prevent.
    """
    try:
        lookahead_for = LOOKAHEAD_RULES[lookahead_rule]
    except KeyError:
        raise ValueError(
            f"unknown lookahead_rule {lookahead_rule!r}; "
            f"expected one of {sorted(LOOKAHEAD_RULES)}") from None
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

    # The lookahead the *drafter* needs is not `until`. `until` answers "how
    # many picks before my turn", which is correctly 0 the instant you are on
    # the clock — exactly the moment `until` used to get reused as the gap fed
    # into `ValueDrafter.choose()`. `choose()` treats <= 0 as "nobody more is
    # coming off the board", so `survivors()` returned the whole pool
    # unfiltered, the drop-off between every position's best-now and
    # best-later collapsed to zero, and the tiebreak fell back to raw VORP —
    # reintroducing the exact "backup QB by raw VORP" failure the drop-off
    # policy exists to prevent, on every single pick that mattered, because it
    # only fired while a recommendation was actually live. `simulate_draft`
    # never has this bug: it always passes the gap from the pick *being
    # decided* (`pick_no`, i.e. `made + 1` here), never from picks already
    # completed. This computes the same thing the backtest does.
    lookahead = lookahead_for(made + 1, teams, my_slot)

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
            # Top 3: the pick, plus a couple of runners-up worth knowing about.
            # A forced (scarcity / end-of-draft) pick is always alone in the
            # list — see `Candidate.forced` — because there is no real
            # alternative to show at that point, only a reason there wasn't one.
            state.candidates = drafter.rank(
                available, roster, rounds, picks_remaining, lookahead, top_n=3)
            top = state.candidates[0]
            state.recommendation = available.loc[top.index]
            state.reason = top.reason
        except (ValueError, KeyError, IndexError):
            # A recommendation is a nicety; the board is the deliverable. Never
            # let a policy edge case take the monitor down mid-draft.
            state.candidates = []
            state.recommendation = None
    return state


def _draft_rec_value(draft: dict) -> float | None:
    """Points per reception this Sleeper draft is scored under, or None if the
    draft does not say (an unusual scoring_type string)."""
    scoring_type = str((draft.get("metadata") or {}).get("scoring_type") or "").lower()
    if not scoring_type:
        return None
    if "half" in scoring_type:
        return 0.5
    if "ppr" in scoring_type:
        return 1.0
    return _SCORING_TYPE_REC.get(scoring_type, 0.0)


def scoring_mismatch(board_meta: dict, draft: dict, *,
                     tol: float = 0.1) -> str | None:
    """None if the board and this draft room agree on points-per-reception,
    else a one-line explanation of the disagreement.

    A board built for full PPR and used in a standard room misprices every
    reception by a point — the WR the board loves at pick 16 is priced for a
    game that is not being played, VORP and every ADP delta included. Silent,
    and exactly the class of bug this project keeps finding (see the `PK`/`K`
    position alias in CLAUDE.md): a small mismatch between two sources' spelling
    of the same thing, invisible until you look for it, and expensive when you
    don't. This is what happened testing against a mock draft that turned out
    to be standard-scored: the board never noticed and neither did anyone
    reading it.

    `board_meta` is the sidecar `draft_board.meta.json` `export()` writes —
    empty or missing means "cannot check", not "no mismatch".
    """
    board_rec = board_meta.get("rec_value")
    draft_rec = _draft_rec_value(draft)
    if board_rec is None or draft_rec is None:
        return None
    if abs(float(board_rec) - draft_rec) <= tol:
        return None

    def fmt(v: float) -> str:
        return _REC_LABEL.get(v, f"{v:g} pt/reception")

    return (f"board is priced for {fmt(float(board_rec))} but this draft room "
           f"is {fmt(draft_rec)} — every reception is worth a different amount "
           f"than the board assumes. Treat VORP and ADP deltas as directional "
           f"only, especially at WR.")


def _line(row: pd.Series) -> str:
    delta = row.get("adp_delta")
    tag = ""
    if pd.notna(delta) and pd.notna(row.get("adp_rank")):
        tag = f"  (ADP {int(row['adp_rank'])}, {int(delta):+d})"
    return (f"{row['position']:<3} {str(row['player_name'])[:24]:<24} "
            f"VORP {row['vorp']:6.1f}{tag}")


def render(state: DraftState, *, top_n: int = 8,
           caps: dict[str, int] | None = None,
           scoring_warning: str | None = None) -> str:
    """The per-pick display, as a string so it can be asserted on."""
    out: list[str] = ["=" * 66]

    # Repeated on every poll, not just at startup — a warning that scrolled off
    # screen an hour ago might as well not exist during a live draft.
    if scoring_warning:
        out.append(f"  !! SCORING MISMATCH: {scoring_warning}")

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
        if state.reason:
            out.append(f"     why:  {state.reason}")
        # A forced pick (scarcity / end-of-draft) has no real alternative —
        # `rank()` returns it alone — so there is nothing to list below it.
        rest = state.candidates[1:] if state.candidates else []
        if rest:
            out.append("     also considered:")
            for alt in rest:
                out.append(f"       {alt.position:<4}{alt.player_name[:22]:<23}"
                           f"VORP {alt.raw_value:6.1f}")
                out.append(f"         {alt.reason}")

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
        html_path: Path | None = None, expect_user_id: str | None = None,
        board_meta: dict | None = None, slot_source: str = "flag",
        lookahead_rule: str = "next_pick") -> None:
    interval = max(MIN_POLL_S, interval)

    # Redirecting stdout to a file (or a task runner's log capture) switches
    # Python from line-buffering to full 4KB block-buffering. A monitor that
    # prints maybe 200 bytes a pick can then sit for fifteen picks with zero
    # bytes on disk, looking exactly like a hang. Line-buffer explicitly rather
    # than rely on the terminal to save us; `reconfigure` can fail on a stream
    # that isn't a real TextIOWrapper (e.g. under some test/capture setups),
    # which is not worth stopping the draft over.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

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

    # Say out loud where the slot came from and whether anyone checked it.
    #
    # This matters because `--my-slot` is no longer required: it falls back to
    # `config current.my_slot`, which is right for the one draft the config
    # describes and wrong for every mock. Sleeper's `draft_order` catches a
    # wrong slot above — but only once the order is drawn AND we appear in it,
    # and in a mock entered outside our own account we do not. That leaves a
    # path where a config value silently drafts somebody else's roster, which
    # is precisely the failure the required flag used to prevent by making you
    # type the number. It cannot be prevented here, so it is made loud.
    if actual is not None:
        verdict = f"confirmed by Sleeper's draft_order"
    elif order:
        verdict = ("NOT confirmed — the order is drawn but does not list "
                  f"user {expect_user_id or '?'}")
    else:
        verdict = "NOT confirmed — Sleeper has not drawn the order yet"
    origin = ("--my-slot" if slot_source == "flag"
              else "config current.my_slot")
    print(f"slot {my_slot} from {origin}; {verdict}.")
    if actual is None:
        print("  ^ check this against the draft room before trusting a "
              "recommendation. A wrong slot builds someone else's roster.")

    if shape is None:
        shape = LeagueShape(teams=teams,
                            starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1,
                                      "K": 1, "DEF": 1})

    warning = scoring_mismatch(board_meta or {}, draft)

    print(f"draft {draft_id} · {teams} teams · {rounds} rounds · "
          f"your slot {my_slot}")
    if my_slot in (1, teams):
        # Only a turn slot can tell the two rules apart, and this is one.
        print(f"  turn slot: back-to-back picks · lookahead rule "
              f"'{lookahead_rule}'")
    if caps:
        print("  caps: " + ", ".join(f"{k} {v}" for k, v in sorted(caps.items())))
    if html_path:
        print(f"  live board: {html_path}")
    if warning:
        # Loud and up front, not just folded into the per-pick render — this is
        # the moment someone is most likely to still be watching the terminal
        # rather than glancing at it between picks.
        print(f"\n  {'!' * 60}\n  SCORING MISMATCH\n  {warning}\n  {'!' * 60}\n")
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
                            rounds=rounds, shape=shape, caps=caps,
                            lookahead_rule=lookahead_rule)

        # `last is None` is the first poll. Print then even at zero picks —
        # otherwise a monitor started before the draft shows nothing at all,
        # and from slot 1 you are on the clock for pick 1 staring at a blank
        # terminal.
        if last is None or state.drafted != last:
            last = state.drafted
            print(render(state, top_n=top_n, caps=caps,
                        scoring_warning=warning))
            if html_path:
                try:
                    refresh_html(board, html_path, state.drafted,
                                 meta={"picks": f"{state.made}/{state.total}"},
                                 candidates=state.candidates)
                except OSError as exc:
                    print(f"  [warn] could not rewrite {html_path}: {exc}")
            if state.complete:
                print("draft complete.")
                return

        time.sleep(interval)


def default_html_path(board_path: Path, *, draft_id: str,
                      current_draft_id: str | None) -> tuple[Path, str | None]:
    """Where the live board goes when `--html` was not given.

    Returns `(path, note)`. `note` is non-None exactly when the default was
    steered away from the real board's HTML and should be printed — silently
    picking a safe path is barely better than not picking one, since the whole
    failure mode is someone not noticing which file is being written.

    Only `--draft-id` matching the *configured real draft* gets the real path.
    Everything else — a mock, a rehearsal, a typo, no config at all — gets a
    sibling `.mock.html` instead. This is the direction the default was wrong
    in: testing against a mock draft silently overwrote the actual draft-day
    artifact for fifteen picks before anyone noticed. Getting it backwards the
    other way just costs a `--html` flag on the one day it matters.
    """
    real_path = board_path.with_suffix(".html")
    is_real = (current_draft_id is not None
              and str(draft_id) == str(current_draft_id))
    if is_real:
        return real_path, None

    mock_path = board_path.with_name(board_path.stem + ".mock.html")
    note = (f"--draft-id {draft_id} does not match config current.draft_id "
           f"({current_draft_id or 'unset'}); writing the live board to "
           f"{mock_path.name} instead of {real_path.name} so a practice draft "
           f"cannot overwrite the real one. Pass --html to override.")
    return mock_path, note


def resolve_my_slot(flag: int | None, configured: object) -> int:
    """Our draft slot, from `--my-slot` if given and `config current.my_slot`
    otherwise.

    The flag used to be required, which made draft day depend on retyping a
    number correctly under time pressure. A wrong slot is the one input error
    the monitor cannot detect on its own until Sleeper publishes `draft_order`
    (see `run`) — it silently rebuilds somebody else's roster, applies our caps
    to it, and advises confidently all night. Reading the checked-in value by
    default means the number is fixed once, in a file, in advance.

    The flag still wins when passed, because mocks and rehearsals are drafted
    from whatever slot they hand you.
    """
    if flag is not None:
        return int(flag)
    if configured is None:
        raise ValueError(
            "no --my-slot given and config current.my_slot is null. Set the "
            "slot in config/league.yaml or pass --my-slot.")
    try:
        return int(configured)
    except (TypeError, ValueError):
        raise ValueError(
            f"config current.my_slot is {configured!r}, which is not a slot "
            f"number. Fix it or pass --my-slot.") from None


def main() -> int:
    ap = argparse.ArgumentParser(description="Live Sleeper draft monitor")
    ap.add_argument("--draft-id", required=True)
    ap.add_argument("--my-slot", type=int, default=None,
                    help="1-indexed draft slot. Default: config "
                         "current.my_slot.")
    ap.add_argument("--board", default="outputs/projections/draft_board.parquet")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--no-html", action="store_true",
                    help="do not rewrite the phone board as picks come in")
    ap.add_argument("--html", default=None,
                    help="where to write the live phone board. Default: the "
                         "real board's HTML only when --draft-id matches "
                         "config current.draft_id, otherwise a sibling "
                         "*.mock.html so a practice draft never overwrites "
                         "the real one.")
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

    board_meta: dict = {}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            board_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"  [warn] could not read {meta_path}: {exc}")

    shape = None
    caps = None
    user_id = None
    current_draft_id = None
    configured_slot = None
    lookahead_rule = "next_pick"
    if CONFIG.exists():
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        shape = LeagueShape.from_config(cfg)
        caps = (cfg.get("draft_policy", {}) or {}).get("position_caps") or None
        user_id = (cfg.get("user", {}) or {}).get("sleeper_user_id")
        current_draft_id = (cfg.get("current", {}) or {}).get("draft_id")
        configured_slot = (cfg.get("current", {}) or {}).get("my_slot")
        lookahead_rule = ((cfg.get("draft_policy", {}) or {})
                          .get("lookahead_rule") or "next_pick")

    try:
        my_slot = resolve_my_slot(args.my_slot, configured_slot)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.no_html:
        html_path = None
    elif args.html:
        html_path = Path(args.html)
    else:
        html_path, note = default_html_path(
            path, draft_id=args.draft_id, current_draft_id=current_draft_id)
        if note:
            print(f"note: {note}")

    try:
        run(args.draft_id, board, my_slot, interval=args.interval,
            shape=shape, caps=caps, html_path=html_path,
            expect_user_id=user_id, board_meta=board_meta,
            slot_source="flag" if args.my_slot is not None else "config",
            lookahead_rule=lookahead_rule)
    except KeyboardInterrupt:
        print("\nstopped.")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
