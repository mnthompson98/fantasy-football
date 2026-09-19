"""The two daily/weekly watchers: what the league just dropped, and what
changed on your own roster's injury report. Both are diffs over live
feeds with no archive, so these tests are the specification."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lineup import LineupSlots  # noqa: E402
from src.ingest.injuries import DNP, FULL  # noqa: E402
from src.inseason import injury_watch as iw  # noqa: E402
from src.inseason import report, transactions, waivers  # noqa: E402

SLOTS = LineupSlots(
    starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
    flex_slots=1, flex_eligible=("RB", "WR", "TE"),
)
NOW = datetime(2026, 9, 23, 11, 0, tzinfo=timezone.utc)      # Wed 06:00 CT
WED_0713 = int(datetime(2026, 9, 23, 7, 13, tzinfo=timezone.utc).timestamp() * 1000)
MON = int(datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc).timestamp() * 1000)
LAST_WEEK = int(datetime(2026, 9, 14, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _players(rows, sd=3.0):
    return pd.DataFrame([{"player_key": n, "player_name": n, "position": p,
                          "projection": v, "sd": sd} for n, p, v in rows])


def _roster():
    return _players([
        ("QB1", "QB", 20.0), ("RBa", "RB", 15.0), ("RBb", "RB", 12.0),
        ("RBc", "RB", 8.0), ("WRa", "WR", 18.0), ("WRb", "WR", 14.0),
        ("WRc", "WR", 11.0), ("TEa", "TE", 9.0), ("Ka", "K", 8.0),
        ("DEFa", "DEF", 7.0),
    ])


# --------------------------------------------------------------------------
# Drops — Sleeper's transaction log
# --------------------------------------------------------------------------

TXS = [
    # Wednesday processing: a completed claim that dropped 111.
    {"type": "waiver", "status": "complete", "leg": 3, "adds": {"999": 2},
     "drops": {"111": 2}, "created": MON, "status_updated": WED_0713},
    # A failed claim dropped nobody, whatever its drops field says.
    {"type": "waiver", "status": "failed", "leg": 3, "adds": {"998": 3},
     "drops": {"222": 3}, "created": MON, "status_updated": WED_0713},
    # A free-agent move on Monday.
    {"type": "free_agent", "status": "complete", "leg": 3, "adds": {"997": 4},
     "drops": {"333": 4}, "created": MON, "status_updated": MON},
    # Our own drop.
    {"type": "free_agent", "status": "complete", "leg": 3, "adds": None,
     "drops": {"444": 7}, "created": MON, "status_updated": MON},
    # Last week's, outside the window.
    {"type": "waiver", "status": "complete", "leg": 2, "adds": {"996": 5},
     "drops": {"555": 5}, "created": LAST_WEEK, "status_updated": LAST_WEEK},
]
OWNERS = {2: "alice", 3: "bob", 4: "carol", 5: "dave", 7: "me"}


def test_only_completed_drops_by_other_teams_inside_the_window_count():
    rows = transactions.drops_from(
        TXS, since=NOW - timedelta(hours=72), waiver_clear_days=2,
        owner_names=OWNERS, my_roster_id=7)
    got = {r["player_id"]: r for r in rows}
    assert set(got) == {"111", "333"}
    assert got["111"]["dropped_by"] == "alice" and got["111"]["via"] == "waiver"
    assert got["333"]["dropped_by"] == "carol" and got["333"]["via"] == "free_agent"


def test_a_dropped_player_is_on_waivers_for_clear_days_then_free():
    rows = transactions.drops_from(
        TXS, since=NOW - timedelta(hours=72), waiver_clear_days=2,
        owner_names=OWNERS)
    wed = next(r for r in rows if r["player_id"] == "111")
    assert wed["on_waivers_until"] == wed["dropped_at"] + timedelta(days=2)
    assert wed["on_waivers_until"] > NOW           # still a claim at 06:00 Wed
    mon = next(r for r in rows if r["player_id"] == "333")
    assert mon["on_waivers_until"] < NOW           # cleared: free agent


def _drops_frame():
    return pd.DataFrame([
        {"player_key": "Stud", "player_id": "111", "player_name": "Stud",
         "position": "RB", "nfl_team": "X", "dropped_by": "alice",
         "dropped_at": NOW - timedelta(hours=4), "via": "waiver",
         "on_waivers_until": NOW + timedelta(hours=44)},
        {"player_key": "Ok", "player_id": "333", "player_name": "Ok",
         "position": "WR", "nfl_team": "Y", "dropped_by": "carol",
         "dropped_at": NOW - timedelta(days=3), "via": "free_agent",
         "on_waivers_until": NOW - timedelta(days=1)},
        {"player_key": "Meh", "player_id": "555", "player_name": "Meh",
         "position": "WR", "nfl_team": "Z", "dropped_by": "dave",
         "dropped_at": NOW - timedelta(hours=4), "via": "waiver",
         "on_waivers_until": NOW + timedelta(hours=44)},
    ])


def _annotated():
    roster = _roster()
    cands = _players([("Stud", "RB", 24.0),      # +9 on the flex: a real claim
                      ("Ok", "WR", 12.0),        # +1: free agent, worth a spot
                      ("Meh", "WR", 12.0)])      # +1: still on waivers
    moves = waivers.evaluate(roster, cands, SLOTS, priority_threshold=4.0,
                             stream_threshold=-1e9, top_n=10)
    return transactions.annotate(
        _drops_frame(), moves, now=NOW,
        stream_threshold=waivers.DEFAULT_STREAM_THRESHOLD)


def test_a_dropped_player_still_on_waivers_is_held_to_the_priority_bar():
    """Guardrail #7 applies here exactly as on the main wire: the same
    +1.0 upgrade is worth a free-agent add and is NOT worth a claim."""
    df = _annotated().set_index("player_key")
    assert df.loc["Ok", "worth_it"] and not df.loc["Ok", "on_waivers"]
    assert not df.loc["Meh", "worth_it"] and df.loc["Meh", "on_waivers"]
    assert df.loc["Meh", "net"] == pytest.approx(df.loc["Ok", "net"])


def test_a_dropped_difference_maker_is_worth_the_claim():
    df = _annotated().set_index("player_key")
    assert df.loc["Stud", "worth_it"] and df.loc["Stud", "burns_priority"]
    assert df.loc["Stud", "on_waivers"]


def test_a_dropped_player_who_does_not_help_says_so_rather_than_vanishing():
    df = _annotated()
    row = next(df[df.player_key == "Meh"].itertuples(index=False))
    text = transactions.describe(row)
    assert "not worth priority" in text and "CLAIM" in text
    assert "Meh" in text


def test_the_brief_lists_dropped_players_separately_and_names_the_cost():
    df = _annotated()
    text = report.brief(week=3, season=2026, issues=[], drops=df)
    section = text[text.index("## Dropped this week"):text.index("## Trade flags")]
    assert "Stud" in section and "CLAIM" in section and "burns priority" in section
    assert "Ok" in section and "free agent now" in section
    assert "Not worth it" in section and "Meh" in section


def test_the_brief_says_when_drops_were_not_checked_or_empty():
    assert "Not checked" in report.brief(week=3, season=2026, issues=[])
    empty = pd.DataFrame(columns=transactions.COLUMNS)
    assert "Nobody dropped anyone" in report.brief(week=3, season=2026,
                                                   issues=[], drops=empty)


# --------------------------------------------------------------------------
# Injuries — a diff over your roster
# --------------------------------------------------------------------------

def _status(key, designation="", practice="", report_status="", body="",
            pos="RB"):
    p = iw.PlayerStatus(player_key=key, player_name=key, position=pos,
                        designation=designation, practice=practice,
                        report_status=report_status, body_part=body)
    p.concern = iw.concern(report_status or designation, practice)
    return p


def test_first_run_reports_nothing_and_says_so():
    now = {"A": _status("A", "Questionable")}
    assert iw.diff(None, now) == []
    text = iw.brief(snapshot=now, changes=[], previous_taken=None, issues=[],
                    week=3, season=2026, report_week=3)
    assert "First run" in text and "A (RB): Questionable" in text


def test_an_unchanged_roster_says_no_changes():
    prev = {"A": _status("A", "Questionable"), "B": _status("B")}
    now = {"A": _status("A", "Questionable"), "B": _status("B")}
    assert iw.diff(prev, now) == []
    text = iw.brief(snapshot=now, changes=[], previous_taken="2026-09-22T11:00",
                    issues=[], week=3, season=2026, report_week=3)
    assert "No changes" in text


def test_a_new_designation_is_reported_as_new_and_a_recovery_as_better():
    prev = {"A": _status("A"), "B": _status("B", "Out")}
    now = {"A": _status("A", "Questionable", body="hamstring"),
           "B": _status("B")}
    changes = {c.player.player_key: c for c in iw.diff(prev, now)}
    assert changes["A"].kind == "new"
    assert "clear → Questionable, hamstring" in changes["A"].describe()
    assert changes["B"].kind == "better"


def test_worsening_is_ordered_before_improving():
    prev = {"A": _status("A", "Questionable"), "B": _status("B", "Out")}
    now = {"A": _status("A", "Out"), "B": _status("B", "Questionable")}
    kinds = [c.kind for c in iw.diff(prev, now)]
    assert kinds == ["worse", "better"]


def test_a_practice_change_alone_is_a_change_not_a_verdict():
    prev = {"A": _status("A", "Questionable", practice=DNP)}
    now = {"A": _status("A", "Questionable", practice=FULL)}
    (c,) = iw.diff(prev, now)
    assert c.kind == "changed"
    assert "DNP" in c.before and "full practice" in c.after


def test_roster_moves_are_reported_only_when_the_player_is_flagged():
    prev = {"A": _status("A", "Out"), "B": _status("B")}
    now = {"C": _status("C", "Questionable"), "D": _status("D")}
    kinds = sorted((c.player.player_key, c.kind) for c in iw.diff(prev, now))
    assert kinds == [("A", "gone"), ("C", "added")]


def test_snapshot_round_trips_through_json(tmp_path):
    now = {"A": _status("A", "Questionable", body="knee")}
    path = tmp_path / "state.json"
    iw.save_snapshot(path, now, NOW)
    back, taken = iw.load_snapshot(path)
    assert taken == NOW.isoformat()
    assert back["A"].designation == "Questionable" and back["A"].body_part == "knee"
    assert iw.diff(back, now) == []


def test_build_snapshot_prefers_the_nfl_practice_report_over_sleeper():
    roster = pd.DataFrame([{"player_key": "A", "player_id": "1",
                            "player_name": "A", "position": "RB"}])
    sleeper = {"1": {"injury_status": "Questionable",
                     "injury_body_part": "Hamstring",
                     "practice_participation": "Limited"}}
    rep = pd.DataFrame([{"player_key": "A", "report_status": "Questionable",
                         "practice_status": DNP,
                         "injury": "Hamstring", "report_week": 3}])
    snap = iw.build_snapshot(roster, sleeper, rep)
    assert snap["A"].practice == DNP
    assert snap["A"].concern == "risky"
    assert snap["A"].summary() == "Questionable, hamstring, DNP"


def test_a_failed_injury_run_writes_a_failed_brief():
    text = iw.brief(snapshot={}, changes=[], previous_taken=None, issues=[],
                    week=3, season=2026, report_week=None, failed="no roster")
    assert text.startswith("# STATUS: FAILED") and "no roster" in text
