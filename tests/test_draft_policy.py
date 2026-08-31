"""Tests for the pick policy.

Every case here is a bug the walk-forward backtest actually caught, costing the
simulated team roughly 1.7 places in the standings between them. They are cheap
to reintroduce and invisible without a backtest, so they are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.draft_sim import (  # noqa: E402
    Roster,
    ValueDrafter,
    picks_until_next_turn,
    snake_order,
)
from src.features.slopes import fitted_slopes  # noqa: E402

STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
FLEX = ("RB", "WR", "TE")

# Roughly the real board's replacement levels, which is what puts the positions
# on a common points scale.
REPLACEMENT = {"QB": 218.0, "RB": 159.0, "WR": 167.0, "TE": 118.0,
               "K": 119.0, "DEF": 109.0}


def _roster(picks: list[tuple[str, float]]) -> Roster:
    r = Roster(team_id=0, starters=dict(STARTERS), flex_slots=1,
               flex_eligible=FLEX, bench_slots=7)
    for i, (pos, vorp) in enumerate(picks):
        r.picks.append({
            "pick_no": i + 1, "round": i + 1, "player_id": f"p{i}",
            "player_name": f"Player {i}", "position": pos,
            "adp_rank": float(i + 1), "vorp": vorp,
            "replacement_points": REPLACEMENT[pos],
        })
    return r


def _pool(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": name, "player_name": name, "position": pos,
         "vorp": vorp, "adp_rank": adp,
         "replacement_points": REPLACEMENT[pos]}
        for name, pos, vorp, adp in rows
    ])


def test_snake_turn_helper_agrees_with_the_draft_order():
    teams, rounds, slot = 10, 16, 4
    order = snake_order(teams, rounds)
    mine = [i + 1 for i, t in enumerate(order) if t == slot - 1]
    for a, b in zip(mine, mine[1:]):
        assert picks_until_next_turn(a, teams, slot) == b - a - 1


def _filler(n: int, start_adp: int = 300) -> list[tuple[str, str, float, float]]:
    """Deep, undraftable bodies.

    Without them `survivors` comes back empty, which correctly means "everyone
    at this position will be gone before my next pick" and correctly triggers
    grabbing one now. A real board always has depth behind the useful players,
    so a fixture without it tests a situation that cannot occur.
    """
    return [(f"Filler {i}", "WR", -60.0 - i, start_adp + i) for i in range(n)]


def test_a_backup_quarterback_never_beats_a_starting_running_back():
    """The original bug. Raw VORP took Josh Allen in round 2 and Lamar Jackson
    in round 3 of a 1-QB league, then reached round 13 with no running back."""
    # Every skill slot filled except the two running back slots.
    roster = _roster([("WR", 92.0), ("WR", 50.0), ("WR", 30.0),
                      ("TE", 30.0), ("QB", 85.0)])
    pool = _pool([
        ("Backup QB", "QB", 41.0, 74),    # highest VORP, cannot ever start
        ("Some RB", "RB", 2.0, 64),       # barely above replacement, but starts
        ("Bench WR", "WR", 3.0, 70),
    ] + _filler(12))
    drafter = ValueDrafter()
    pick = pool.loc[drafter.choose(pool, roster, 16, 11, picks_until_next=8)]
    assert pick["position"] == "RB", pick["player_name"]


def test_filling_an_empty_slot_beats_a_below_replacement_bench_body():
    """The second bug. On a VORP scale an empty slot and a replacement-level
    starter both score zero, so filling RB1 looked like a downgrade and the
    policy took a ninth receiver instead."""
    roster = _roster([("WR", 90.0), ("WR", 40.0), ("WR", 20.0), ("TE", 30.0),
                      ("QB", 60.0)])
    pool = _pool([
        ("Weak RB", "RB", -5.0, 100),     # below replacement, but RB1 is empty
        ("Another WR", "WR", 1.0, 85),    # above replacement, pure bench
    ])
    drafter = ValueDrafter()
    pick = pool.loc[drafter.choose(pool, roster, 16, 11, picks_until_next=8)]
    assert pick["position"] == "RB", pick["player_name"]


def test_a_kicker_does_not_jump_the_queue_once_the_skill_slots_are_set():
    """The third bug, introduced by the fix to the second. Rewarding any filled
    slot pulled K and DEF into rounds 8 and 9, undoing the market anchor. Their
    anchored VORP is deeply negative, and that has to keep them last."""
    roster = _roster([("WR", 90.0), ("WR", 40.0), ("RB", 50.0), ("RB", 30.0),
                      ("TE", 30.0), ("QB", 60.0), ("WR", 20.0)])
    # More than SCARCITY_FLOOR of each, or the scarcity backstop fires and
    # forces a kicker regardless — correctly, since a position down to its last
    # three bodies really is about to run out.
    pool = _pool(
        [("Best K", "K", -40.0, 182), ("Bench WR", "WR", -7.0, 83)]
        + [(f"K {i}", "K", -42.0 - i, 190 + i) for i in range(5)]
        + [(f"DEF {i}", "DEF", -46.0 - i, 185 + i) for i in range(6)]
        + _filler(12, start_adp=84)
    )
    drafter = ValueDrafter()
    # The kicker's ADP is far enough back that he is still there next turn, so
    # his drop-off is zero and the anchored VORP tiebreak keeps him waiting.
    pick = pool.loc[drafter.choose(pool, roster, 16, 9, picks_until_next=8)]
    assert pick["position"] == "WR", pick["player_name"]


def test_a_position_being_run_on_is_taken_before_one_that_is_not():
    """The drop-off is the whole point: two equally valuable players, but only
    one of them will still be there next turn."""
    roster = _roster([("WR", 90.0), ("QB", 60.0)])
    pool = _pool([
        # Scarce: the only good RB, and his ADP means he goes before our turn.
        ("Scarce RB", "RB", 40.0, 20),
        ("Weak RB", "RB", -30.0, 200),
        # Plentiful: an equally good TE, and another nearly as good behind him.
        ("Good TE", "TE", 40.0, 120),
        ("Also TE", "TE", 39.0, 130),
    ])
    drafter = ValueDrafter()
    pick = pool.loc[drafter.choose(pool, roster, 16, 14, picks_until_next=10)]
    assert pick["player_name"] == "Scarce RB"


def test_the_starter_backstop_still_wins_at_the_end_of_the_draft():
    """With as many picks left as empty starting slots, every pick must fill
    one. An unfilled starter is a zero every week."""
    roster = _roster([("WR", 90.0), ("WR", 40.0), ("RB", 50.0), ("RB", 30.0),
                      ("TE", 30.0), ("QB", 60.0), ("WR", 20.0)])
    pool = _pool([
        ("Bench WR", "WR", 5.0, 83),      # better raw value, fills nothing
        ("Only K", "K", -40.0, 182),
    ])
    drafter = ValueDrafter()
    # Two picks left, two empty starting slots (K and DEF).
    pick = pool.loc[drafter.choose(pool, roster, 16, 2, picks_until_next=8)]
    assert pick["position"] == "K"


def test_starting_value_ignores_players_who_cannot_crack_the_lineup():
    drafter = ValueDrafter()
    roster = _roster([("QB", 85.0)])
    with_backup = drafter.starting_value(roster, extra=("QB", 41.0 + REPLACEMENT["QB"]))
    assert with_backup == pytest.approx(drafter.starting_value(roster))


def test_starting_value_uses_the_flex_for_a_surplus_flex_eligible_player():
    drafter = ValueDrafter()
    roster = _roster([("WR", 90.0), ("WR", 40.0)])
    base = drafter.starting_value(roster)
    # A third receiver has nowhere to go but the flex, and should count there.
    with_third = drafter.starting_value(roster, extra=("WR", 20.0 + REPLACEMENT["WR"]))
    assert with_third > base


def test_survivors_drops_the_players_the_room_takes_first():
    drafter = ValueDrafter()
    pool = _pool([
        ("Early", "RB", 50.0, 1),
        ("Middle", "RB", 40.0, 2),
        ("Late", "RB", 30.0, 99),
    ])
    left = drafter.survivors(pool, 2)
    assert list(left["player_name"]) == ["Late"]
    # Nothing is taken when we are on the clock again immediately.
    assert len(drafter.survivors(pool, 0)) == 3


# --------------------------------------------------------------------------
# Fitted calibration slopes
# --------------------------------------------------------------------------

def test_fitted_slopes_returns_the_priors_when_fitting_is_disabled():
    cfg = {"calibration": {"slopes": {"RB": 0.79, "WR": 0.85},
                           "fit_from_history": {"enabled": False}}}
    got = fitted_slopes(cfg, pd.DataFrame(), pd.DataFrame(), None, [2024])
    assert got["RB"] == 0.79 and got["WR"] == 0.85


def test_fitted_slopes_never_returns_an_empty_mapping():
    """Falling back to nothing would silently calibrate at slope 1.0 — no
    shrink at all — which is worse than using a generic prior."""
    cfg = {"calibration": {"slopes": {"RB": 0.79},
                           "fit_from_history": {"enabled": True}}}
    got = fitted_slopes(cfg, pd.DataFrame(columns=["page_type", "scrape_date"]),
                        pd.DataFrame(columns=["season"]), None, [])
    assert got["RB"] == 0.79
    assert set(got) >= {"QB", "RB", "WR", "TE", "K", "DEF"}
