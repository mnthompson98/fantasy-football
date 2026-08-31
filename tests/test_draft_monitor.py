"""Tests for the live draft path.

This was the only critical path in the system with no coverage, and it is the
one whose failure costs the most: it runs once a year, for ninety minutes, with
a clock ticking. Every test here corresponds to something that actually broke
when the monitor was driven against a real Sleeper draft feed.

The feed in `fixtures/sleeper_draft_picks_2025.json` is the genuine 160-pick
payload from this league's 2025 draft, trimmed to the fields the monitor reads.
Synthetic picks would not have caught the parts that depend on Sleeper's own
shape — `draft_slot`, string player ids, team defenses keyed by team abbrev.

Run: .venv/Scripts/python.exe -m pytest tests/test_draft_monitor.py -q
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.draft.board import (  # noqa: E402
    best_available,
    positional_run,
    refresh_html,
)
from src.draft.monitor import (  # noqa: E402
    draft_state,
    picks_until_next_turn,
    render,
)
from src.features.vorp import LeagueShape  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sleeper_draft_picks_2025.json"

SHAPE = LeagueShape(
    teams=10,
    starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
    flex_slots=1,
    flex_eligible=("RB", "WR", "TE"),
)
CAPS = {"QB": 1, "TE": 2, "K": 1, "DEF": 1}


@pytest.fixture(scope="module")
def feed() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def board() -> pd.DataFrame:
    """A board covering every player in the fixture feed, plus depth.

    Built rather than loaded so the tests do not depend on `outputs/`, which is
    a build product and not in the repo.
    """
    rows = []
    seen: set[str] = set()
    picks = json.loads(FIXTURE.read_text(encoding="utf-8"))["picks"]
    for i, p in enumerate(picks):
        pid = str(p["player_id"])
        if pid in seen:
            continue
        seen.add(pid)
        meta = p["metadata"]
        rows.append({
            "player_id": pid,
            "player_key": f"K{pid}",
            "player_name": f"{meta['first_name']} {meta['last_name']}",
            "position": meta["position"],
            "vorp": 200.0 - i,
            "replacement_points": 100.0,
            "adp_rank": i + 1,
            "adp_delta": 0,
            "projection": 200.0 - i,
            "projection_calibrated": 200.0 - i,
            "bye_week": 7.0,
            "nfl_team": meta.get("team") or "FA",
            "injury_status": "",
            "durability": 0.9,
            "durability_flag": False,
            "ecr_best": i + 1,
            "ecr_worst": i + 1,
            "rank": i + 1,
        })
    # Depth past the drafted pool, so "everyone left" is never empty.
    n = len(rows)
    for j, pos in enumerate(["RB", "WR", "TE", "QB", "K", "DEF"] * 10):
        rows.append({**rows[0],
                     "player_id": f"depth{j}", "player_key": f"Kdepth{j}",
                     "player_name": f"Depth {pos} {j}", "position": pos,
                     "vorp": -10.0 - j, "adp_rank": n + j + 1,
                     "rank": n + j + 1})
    return pd.DataFrame(rows)


def _state(picks, board, slot=4, teams=10, rounds=16):
    return draft_state(picks, board, my_slot=slot, teams=teams, rounds=rounds,
                       shape=SHAPE, caps=CAPS)


# --------------------------------------------------------------------------
# The hang. This one had no output at all: the monitor just sat there.
# --------------------------------------------------------------------------

def test_picks_until_next_turn_terminates_on_impossible_slot():
    """A slot outside the draft used to spin forever in an unbounded `while`.

    The monitor's copy of this function was a duplicate of the simulator's with
    the bound dropped. It now *is* the simulator's, so the duplicate cannot rot
    again — but the termination guarantee is worth pinning either way.
    """
    done = threading.Event()

    def call():
        picks_until_next_turn(5, teams=10, my_slot=12)
        done.set()

    threading.Thread(target=call, daemon=True).start()
    assert done.wait(timeout=5), "picks_until_next_turn hung on an invalid slot"


def test_draft_state_rejects_slot_outside_the_draft(board):
    with pytest.raises(ValueError, match="outside this draft"):
        _state([], board, slot=12)
    with pytest.raises(ValueError, match="outside this draft"):
        _state([], board, slot=0)


# --------------------------------------------------------------------------
# Pick detection
# --------------------------------------------------------------------------

def test_zero_picks_still_produces_a_board(board):
    """The monitor printed nothing until the first pick was made.

    Started before a draft from slot 1 that meant an empty terminal at the exact
    moment you were on the clock for pick 1.
    """
    state = _state([], board, slot=1)
    assert state.made == 0
    assert state.on_the_clock == 1
    assert state.mine is True
    assert state.recommendation is not None
    assert "YOU ARE ON THE CLOCK" in render(state, caps=CAPS)


def test_pick_counter_is_the_pick_on_the_clock_not_the_last_one_made(feed, board):
    """The header said "pick N" when N picks were *complete* — one behind.

    At your turn it read "pick 3/160 · YOU ARE ON THE CLOCK" while you were in
    fact making pick 4, which is exactly the number you are counting down to.
    """
    picks = feed["picks"]
    assert _state(picks[:3], board, slot=4).on_the_clock == 4
    assert _state(picks[:3], board, slot=4).mine is True
    assert _state(picks[:0], board, slot=1).on_the_clock == 1


def test_undone_pick_is_detected_even_when_the_count_is_unchanged(board):
    """Detection keyed on `len(picks)`, so an undo+replace between two polls was
    invisible and the replaced player stayed struck off for the rest of the
    draft.
    """
    def pick(pid, pos):
        return [{"pick_no": 1, "draft_slot": 1, "player_id": pid,
                 "metadata": {"position": pos, "first_name": "A",
                              "last_name": "B"}}]

    ids = board["player_id"].tolist()
    a, b = ids[0], ids[1]

    sa = _state(pick(a, "RB"), board)
    sb = _state(pick(b, "WR"), board)

    assert len(pick(a, "RB")) == len(pick(b, "WR"))     # same count
    assert sa.drafted != sb.drafted                      # different state
    assert a in set(sb.available["player_id"])           # the undone pick is back
    assert b not in set(sb.available["player_id"])


def test_every_teams_picks_leave_the_board_not_just_mine(feed, board):
    """The board must thin by nine players between your turns, not zero."""
    picks = feed["picks"][:30]
    state = _state(picks, board, slot=4)

    mine = {str(p["player_id"]) for p in picks if p["draft_slot"] == 4}
    theirs = {str(p["player_id"]) for p in picks if p["draft_slot"] != 4}
    assert theirs, "fixture should contain other teams' picks"

    left = set(state.available["player_id"])
    assert not (mine & left)
    assert not (theirs & left)
    assert len(state.drafted) == len(picks)


# --------------------------------------------------------------------------
# Roster reconstruction and position caps
# --------------------------------------------------------------------------

def test_roster_is_built_from_draft_slot(feed, board):
    picks = feed["picks"]
    state = _state(picks[:60], board, slot=4)
    expected = [p for p in picks[:60] if p["draft_slot"] == 4]
    assert len(state.roster.picks) == len(expected)
    assert [p["player_id"] for p in state.roster.picks] == \
           [str(p["player_id"]) for p in expected]


def test_a_drafted_player_missing_from_the_board_still_counts_against_caps(board):
    """Injury-gated and unreconciled players hold a roster spot all the same.

    Dropping them would quietly raise our own caps — the monitor would offer a
    second quarterback because it had forgotten the first.
    """
    picks = [{"pick_no": 1, "draft_slot": 4, "player_id": "not-on-the-board",
              "metadata": {"position": "QB", "first_name": "Ghost",
                           "last_name": "Passer"}}]
    state = _state(picks, board, slot=4)
    assert state.roster.counts() == {"QB": 1}
    assert state.recommendation["position"] != "QB"


def test_best_available_drops_positions_the_roster_has_filled(board):
    counts = {"QB": 1, "TE": 2, "K": 1, "DEF": 1}
    top = best_available(board, set(), n=40, roster_counts=counts, caps=CAPS)
    assert set(top["position"]) <= {"RB", "WR"}

    # Uncapped, the same call is free to return anything.
    plain = best_available(board, set(), n=40)
    assert set(plain["position"]) - {"RB", "WR"}


def test_best_available_never_returns_nothing_when_caps_exclude_everyone():
    """A cap that empties the board is a bad cap, not an empty board."""
    df = pd.DataFrame([
        {"player_id": "1", "player_name": "Only K", "position": "K",
         "vorp": 1.0, "adp_rank": 1},
    ])
    out = best_available(df, set(), roster_counts={"K": 1}, caps={"K": 1})
    assert len(out) == 1


def test_caps_hold_across_a_whole_replayed_draft(feed, board):
    """The regression that matters most: as the board thins, the monitor must
    stop recommending positions we are done with.

    Before the fix it ranked by raw VORP and happily offered a backup
    quarterback in the last round — the exact roster `ValueDrafter` exists to
    stop us building.
    """
    picks = feed["picks"]
    offered = 0
    late_offers: set[str] = set()

    for n in range(len(picks)):
        state = _state(picks[:n], board, slot=4)
        counts = state.roster.counts()
        if state.recommendation is None:
            continue
        offered += 1
        pos = str(state.recommendation["position"])
        cap = CAPS.get(pos)
        assert cap is None or counts.get(pos, 0) < cap, (
            f"recommended {pos} at pick {n + 1} with "
            f"{counts.get(pos, 0)}/{cap} already rostered")
        if n > 100:
            late_offers.add(pos)

    assert offered > 100, "expected a recommendation for most of the draft"
    # The fixture's slot-4 manager really did take two quarterbacks in 2025, so
    # by the back half the cap is spent and QB must never be offered again.
    assert "QB" not in late_offers


def test_render_marks_capped_positions(board):
    picks = [{"pick_no": 1, "draft_slot": 4, "player_id": "ghost-k",
              "metadata": {"position": "K", "first_name": "A", "last_name": "B"}}]
    text = render(_state(picks, board, slot=4), caps=CAPS)
    k_line = next(ln for ln in text.splitlines() if ln.strip().startswith("K "))
    assert "[capped]" in k_line


# --------------------------------------------------------------------------
# End of the draft
# --------------------------------------------------------------------------

def test_no_phantom_turn_after_your_last_pick(feed, board):
    """`picks_until_next_turn` counts on past the end of the draft, so the
    monitor spent the whole final round promising a turn that never came.
    """
    picks = feed["picks"]
    # Slot 4's last turn in a 10x16 snake is pick 157.
    assert _state(picks[:156], board, slot=4).mine is True
    after = _state(picks[:158], board, slot=4)
    assert after.mine is False
    assert after.until_my_turn == -1
    assert "your picks are done" in render(after, caps=CAPS)
    assert after.recommendation is None


def test_full_draft_replays_to_completion(feed, board):
    """End to end over the real feed: no crash, no stall, and it finishes."""
    picks = feed["picks"]
    seen_sizes = []
    for n in range(0, len(picks) + 1):
        state = _state(picks[:n], board, slot=4)
        seen_sizes.append(len(state.available))
        render(state, caps=CAPS)          # rendering must not raise either

    # Availability only ever shrinks.
    assert seen_sizes == sorted(seen_sizes, reverse=True)
    final = _state(picks, board, slot=4)
    assert final.complete is True
    assert final.made == feed["settings"]["teams"] * feed["settings"]["rounds"]


# --------------------------------------------------------------------------
# Run detection
# --------------------------------------------------------------------------

def test_no_positional_run_declared_off_a_handful_of_picks():
    """"run in progress: WR 100%" after pick one is noise, and it fired every
    single draft.
    """
    assert positional_run([{"position": "WR"}]) == {}
    assert positional_run([{"position": "WR"}, {"position": "WR"}]) == {}
    six = [{"position": "WR"}] * 4 + [{"position": "RB"}] * 2
    assert positional_run(six) == {"WR": pytest.approx(4 / 6)}


# --------------------------------------------------------------------------
# The phone board
# --------------------------------------------------------------------------

def _baked_drafted(html: str) -> set[str]:
    """The drafted ids the page will strike out, read back out of the page."""
    body = html.split("const DRAFTED = new Set(", 1)[1]
    return set(json.loads(body[:body.index("]") + 1]))


def test_live_html_strikes_out_other_teams_picks(feed, board, tmp_path):
    """The HTML shipped as tap-to-strike over localStorage, so it only ever
    reflected picks you tapped yourself. Nine other managers were invisible.
    """
    picks = feed["picks"][:40]
    state = _state(picks, board, slot=4)
    path = refresh_html(board, tmp_path / "board.html", state.drafted)
    html = path.read_text(encoding="utf-8")

    baked = _baked_drafted(html)
    theirs = {str(p["player_id"]) for p in picks if p["draft_slot"] != 4}
    mine = {str(p["player_id"]) for p in picks if p["draft_slot"] == 4}
    assert theirs and mine

    assert theirs <= baked, "picks by other teams are missing from the board"
    assert mine <= baked
    # And nobody else. A board that strikes out a player still on the wire is
    # worse than one that strikes out nobody.
    assert baked == state.drafted
    assert not baked & (set(board["player_id"]) - state.drafted)


def test_live_html_survives_an_empty_draft(board, tmp_path):
    path = refresh_html(board, tmp_path / "board.html", set())
    assert "const DRAFTED = new Set([])" in path.read_text(encoding="utf-8")


def test_defenses_reconcile_by_team_abbreviation(feed, board):
    """Sleeper keys team defenses by team code (`SEA`), not a numeric id. If the
    board ever moved to `DEF_SEA` for `player_id`, defenses would never come off
    the board and the monitor would recommend one that is long gone.
    """
    defs = [p for p in feed["picks"] if p["metadata"]["position"] == "DEF"]
    assert defs
    for p in defs:
        assert str(p["player_id"]).isalpha(), "expected a team abbreviation"
    state = _state(feed["picks"], board, slot=4)
    for p in defs:
        assert str(p["player_id"]) not in set(state.available["player_id"])


# --------------------------------------------------------------------------
# The one input error the monitor cannot detect for itself
# --------------------------------------------------------------------------

def _stub_draft(monkeypatch, feed, *, draft_order=None):
    """Point the monitor at the fixture instead of Sleeper."""
    import src.draft.monitor as M

    monkeypatch.setattr(M, "get_draft", lambda _id: {
        "settings": feed["settings"], "draft_order": draft_order or {}})
    monkeypatch.setattr(M, "get_draft_picks", lambda _id: feed["picks"])
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)
    return M


def test_run_refuses_a_slot_sleeper_disagrees_with(monkeypatch, feed, board):
    """A wrong --my-slot builds somebody else's roster, applies our caps to it
    and advises confidently all night. Sleeper publishes the mapping; use it.
    """
    M = _stub_draft(monkeypatch, feed, draft_order={"me": 6})
    with pytest.raises(ValueError, match="slot 6"):
        M.run("d", board, 7, expect_user_id="me", html_path=None)


def test_run_accepts_the_slot_sleeper_confirms(monkeypatch, feed, board, capsys):
    M = _stub_draft(monkeypatch, feed, draft_order={"me": 6})
    M.run("d", board, 6, expect_user_id="me", html_path=None)
    assert "draft complete." in capsys.readouterr().out


def test_run_does_not_guess_when_the_order_is_not_drawn_yet(monkeypatch, feed,
                                                            board, capsys):
    """`draft_order` is null until the commissioner sets it — that is the state
    the 2026 league is in. The check must stay quiet, not block the monitor.
    """
    M = _stub_draft(monkeypatch, feed, draft_order=None)
    M.run("d", board, 3, expect_user_id="me", html_path=None)
    assert "draft complete." in capsys.readouterr().out


def test_run_survives_a_failing_poll(monkeypatch, feed, board, capsys):
    """Sleeper hiccuping mid-draft must not end the monitor."""
    import src.draft.monitor as M

    calls = {"n": 0}

    def flaky(_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("boom")
        return feed["picks"]

    monkeypatch.setattr(M, "get_draft", lambda _id: {"settings": feed["settings"]})
    monkeypatch.setattr(M, "get_draft_picks", flaky)
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)

    M.run("d", board, 4, html_path=None)
    out = capsys.readouterr().out
    assert "poll failed" in out
    assert "draft complete." in out


def test_run_writes_the_live_board_as_picks_arrive(monkeypatch, feed, board,
                                                   tmp_path):
    M = _stub_draft(monkeypatch, feed)
    html = tmp_path / "live.html"
    M.run("d", board, 4, html_path=html)
    assert _baked_drafted(html.read_text(encoding="utf-8")) == {
        str(p["player_id"]) for p in feed["picks"]}
