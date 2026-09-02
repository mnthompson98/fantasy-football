"""The optional pick-policy switches, and the monitor's between-turns preview.

Each case is a situation from the 2025 draft replay (`HANDOFF.md`,
"Draft-day stress test") reduced to a fixture. Both switches default off and
the default path must stay bit-identical to what the recorded baseline was
measured under; that is pinned first.
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
    policy_from_config,
    positional_run,
    simulate_draft,
)
from src.draft.monitor import draft_state, render  # noqa: E402
from src.features.calibration import fit_slopes  # noqa: E402
from src.features.vorp import LeagueShape  # noqa: E402

STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
FLEX = ("RB", "WR", "TE")
REPL = {"QB": 218.0, "RB": 159.0, "WR": 167.0, "TE": 118.0, "K": 119.0,
        "DEF": 109.0}
CAPS = {"QB": 1, "TE": 2, "K": 1, "DEF": 1}


def _roster(picks):
    r = Roster(team_id=0, starters=dict(STARTERS), flex_slots=1,
               flex_eligible=FLEX, bench_slots=7)
    for i, (pos, vorp) in enumerate(picks):
        r.picks.append({"pick_no": i + 1, "round": i + 1, "player_id": f"p{i}",
                        "player_name": f"P{i}", "position": pos,
                        "adp_rank": float(i + 1), "vorp": vorp,
                        "replacement_points": REPL[pos]})
    return r


def _pool(rows):
    return pd.DataFrame([
        {"player_id": n, "player_name": n, "position": p, "vorp": v,
         "adp_rank": a, "replacement_points": REPL[p]}
        for n, p, v, a in rows
    ])


def _filler(n, start_adp=300):
    return [(f"Filler {i}", "WR", -60.0 - i, start_adp + i) for i in range(n)]


# --------------------------------------------------------------------------
# Defaults are the recorded policy
# --------------------------------------------------------------------------

def test_switches_default_off_and_parse_from_config():
    d = ValueDrafter()
    assert d.run_aware is False and d.depth_tiebreak is False
    assert policy_from_config(None) is None
    assert policy_from_config({"position_caps": {"QB": 1}}) is None
    assert policy_from_config({"run_aware": True}) == {"run_aware": True}
    assert policy_from_config({"run_aware": False, "depth_tiebreak": True}) \
        == {"depth_tiebreak": True}


def _board():
    rows = []
    for pos, n, base, step in [("QB", 32, 330, 4.0), ("RB", 72, 270, 3.0),
                               ("WR", 84, 255, 2.4), ("TE", 32, 190, 4.0),
                               ("K", 24, 135, 1.2), ("DEF", 24, 128, 1.5)]:
        for i in range(n):
            rows.append({"player_id": f"{pos}{i}", "player_name": f"{pos} {i}",
                         "position": pos, "projection": base - i * step})
    from src.features.vorp import compute_vorp
    b = compute_vorp(pd.DataFrame(rows),
                     LeagueShape(teams=10, starters=STARTERS, flex_slots=1))
    b = b.sort_values("vorp", ascending=False).reset_index(drop=True)
    b["adp_rank"] = range(1, len(b) + 1)
    return b


def test_policy_none_and_all_off_are_the_same_draft():
    kw = dict(teams=10, rounds=16, my_slot=4, starters=STARTERS, flex_slots=1,
              flex_eligible=FLEX, sigma={p: 8.0 for p in STARTERS}, seed=21)
    a = simulate_draft(_board(), **kw)
    b = simulate_draft(_board(), **kw,
                       policy={"run_aware": False, "depth_tiebreak": False})
    for t in a:
        assert [p["player_id"] for p in a[t].picks] == \
               [p["player_id"] for p in b[t].picks]


# --------------------------------------------------------------------------
# run_aware: the RB run in round 2
# --------------------------------------------------------------------------

def test_positional_run_is_the_same_function_the_board_exports():
    from src.draft.board import positional_run as from_board
    assert from_board is positional_run


def test_run_aware_survivors_assume_the_run_continues():
    pool = _pool([("RB A", "RB", 50.0, 20), ("RB B", "RB", 45.0, 30),
                  ("RB C", "RB", 40.0, 40), ("WR A", "WR", 50.0, 10),
                  ("WR B", "WR", 45.0, 15), ("WR C", "WR", 40.0, 25)])
    runs = {"RB": 1.0}                       # six straight running backs
    plain = ValueDrafter().survivors(pool, 3, runs)
    aware = ValueDrafter(run_aware=True).survivors(pool, 3, runs)
    # ADP alone removes WR A, WR B and RB A. A run at 100% over three picks
    # means three RBs are gone: RB B and RB C leave as well.
    assert set(plain["player_name"]) == {"RB B", "RB C", "WR C"}
    assert set(aware["player_name"]) == {"WR C"}
    # Off, `runs` is ignored entirely.
    assert len(ValueDrafter().survivors(pool, 3, None)) == 3


def test_a_run_raises_the_urgency_of_the_position_being_run_on():
    """The replay case: picks 11-16 all RBs, our pick 17. ADP said the RB
    drop-off was negligible; the run said otherwise, and the RB the tool
    passed on outscored the one left at our next turn by 100 points."""
    roster = _roster([("WR", 92.0)])
    # Four picks until our turn. By ADP the room takes Best RB, Best WR,
    # Next WR and Next RB, leaving Deep RB as the RB survivor. A 100% RB run
    # over four picks says four RBs go, so Deep RB is gone too.
    # Four RBs, or the scarcity backstop (<= 3 left at a needed position)
    # forces the pick and there is no comparison to make.
    pool = _pool([
        ("Best RB", "RB", 30.0, 18), ("Next RB", "RB", 28.0, 40),
        ("Deep RB", "RB", 5.0, 90), ("Deeper RB", "RB", 4.0, 95),
        ("Best WR", "WR", 30.0, 25), ("Next WR", "WR", 29.0, 27),
        ("Deep WR", "WR", 6.0, 60),
    ] + _filler(12, 100))
    runs = {"RB": 1.0}
    off = ValueDrafter().rank(pool, roster, 16, 15, 4, runs=runs)
    on = ValueDrafter(run_aware=True).rank(pool, roster, 16, 15, 4, runs=runs)
    gain_off = {c.position: c.gain for c in off}
    gain_on = {c.position: c.gain for c in on}
    assert gain_on["RB"] > gain_off["RB"]
    assert on[0].position == "RB"
    assert "RB" in on[0].reason


# --------------------------------------------------------------------------
# depth_tiebreak: the second tight end at six consecutive turns
# --------------------------------------------------------------------------

def test_once_starters_are_set_a_bench_back_beats_a_second_tight_end():
    """Every drop-off is zero here, so the choice is the tie-break. Raw VORP
    took the TE2 (5.0 > 3.0); insurance-weighted VORP takes the RB3, who
    covers two starters, over a TE2 who covers one."""
    roster = _roster([("QB", 60.0), ("RB", 50.0), ("RB", 30.0), ("WR", 90.0),
                      ("WR", 40.0), ("WR", 35.0), ("TE", 60.0)])
    pool = _pool([("TE2", "TE", 5.0, 150), ("RB3", "RB", 3.0, 160),
                  ("TE3", "TE", 4.0, 170), ("RB4", "RB", 2.0, 180),
                  ("TE4", "TE", 3.5, 175), ("RB5", "RB", 1.5, 185)]
                 + _filler(12, 300))
    off = ValueDrafter(caps=CAPS).rank(pool, roster, 16, 9, 8)
    on = ValueDrafter(caps=CAPS, depth_tiebreak=True).rank(pool, roster, 16, 9, 8)
    assert all(c.gain == 0.0 for c in off)
    assert off[0].player_name == "TE2"
    assert on[0].player_name == "RB3"


def test_depth_tiebreak_never_lifts_a_market_anchored_kicker():
    d = ValueDrafter(depth_tiebreak=True)
    roster = _roster([("K", -40.0)])
    assert d._tiebreak(roster, "K", -42.0) == -42.0


def test_depth_tiebreak_leaves_a_starter_filling_pick_alone():
    d = ValueDrafter(depth_tiebreak=True)
    roster = _roster([("RB", 50.0)])            # one RB slot still open
    assert d._tiebreak(roster, "RB", 20.0) == 20.0


# --------------------------------------------------------------------------
# The between-turns preview
# --------------------------------------------------------------------------

SHAPE = LeagueShape(teams=10, starters=STARTERS, flex_slots=1, flex_eligible=FLEX)


def _feed_board():
    rows = []
    for i, pos in enumerate(["RB", "WR"] * 30 + ["QB"] * 12 + ["TE"] * 12
                            + ["K"] * 12 + ["DEF"] * 12):
        rows.append({"player_id": f"x{i}", "player_key": f"x{i}",
                     "player_name": f"{pos} {i}", "position": pos,
                     "vorp": 150.0 - i, "replacement_points": 100.0,
                     "adp_rank": i + 1, "adp_delta": 0, "projection": 150.0 - i,
                     "projection_calibrated": 150.0 - i, "bye_week": 7,
                     "nfl_team": "CIN", "injury_status": "", "rank": i + 1})
    return pd.DataFrame(rows)


def _picks(n):
    b = _feed_board()
    out = []
    for i in range(n):
        slot = (i % 10) + 1 if (i // 10) % 2 == 0 else 10 - (i % 10)
        r = b.iloc[i]
        out.append({"pick_no": i + 1, "draft_slot": slot,
                    "player_id": r["player_id"],
                    "metadata": {"position": r["position"], "first_name": "A",
                                 "last_name": "B"}})
    return out


def test_between_turns_the_recommendation_is_a_preview_for_the_next_pick():
    """Right after pick 4 (ours, slot 4) the next turn is pick 17; the rec
    should be computed on who survives picks 5-16, and say so."""
    st = draft_state(_picks(4), _feed_board(), my_slot=4, teams=10, rounds=16,
                     shape=SHAPE, caps=CAPS)
    assert st.mine is False
    assert st.preview is True
    assert st.my_next_pick == 17
    # The twelve best-by-ADP players are assumed gone: the preview cannot
    # recommend any of them.
    gone = set(_feed_board().sort_values("adp_rank").iloc[4:16]["player_id"])
    assert st.recommendation["player_id"] not in gone
    assert "LIKELY AT YOUR TURN (pick 17)" in render(st, caps=CAPS)


def test_on_the_clock_is_not_a_preview():
    st = draft_state(_picks(3), _feed_board(), my_slot=4, teams=10, rounds=16,
                     shape=SHAPE, caps=CAPS)
    assert st.mine is True and st.preview is False
    assert ">> TAKE:" in render(st, caps=CAPS)


def test_the_phone_board_labels_a_preview(tmp_path):
    from src.backtest.draft_sim import Candidate
    from src.draft.board import refresh_html
    c = Candidate(index=0, position="RB", player_name="RB 0", gain=1.0,
                  value=1.0, raw_value=1.0, reason="r")
    html = refresh_html(_feed_board(), tmp_path / "b.html", set(),
                        candidates=[c], preview_pick=17).read_text("utf-8")
    assert "Likely pick at your turn (#17)" in html
    html = refresh_html(_feed_board(), tmp_path / "b.html", set(),
                        candidates=[c]).read_text("utf-8")
    assert "Suggested pick" in html


# --------------------------------------------------------------------------
# Slope shrinkage toward the prior
# --------------------------------------------------------------------------

def test_fitted_slopes_are_shrunk_toward_the_prior_by_sample_size():
    n = 40
    proj = pd.Series(range(n), dtype=float)
    hist = pd.DataFrame({"position": ["RB"] * n, "projection": proj,
                         "actual": (proj - proj.mean()) * 1.1 + proj.mean()})
    raw = fit_slopes(hist, priors={"RB": 0.79}, min_obs=40)["RB"]
    shrunk = fit_slopes(hist, priors={"RB": 0.79}, min_obs=40,
                        prior_weight=80)["RB"]
    assert raw == pytest.approx(1.1, abs=0.01)
    assert shrunk == pytest.approx((40 * 1.1 + 80 * 0.79) / 120, abs=0.01)


# --------------------------------------------------------------------------
# upside_tiebreak: what a round-12 pick is actually buying
# --------------------------------------------------------------------------

def test_the_pipeline_derives_a_best_case_positional_rank_and_value():
    from src.features.pipeline import ecr_to_pool, value_board
    totals = pd.DataFrame([
        {"player_key": f"{p}{i}", "position": p, "season": 2024,
         "points": base - i * 5, "games": 17, "points_per_game": (base - i * 5) / 17}
        for p, base in (("WR", 300), ("RB", 280), ("QB", 350), ("TE", 200),
                        ("K", 140), ("DEF", 130))
        for i in range(30)
    ])
    rows = []
    for p in ("WR", "RB", "QB", "TE", "K", "DEF"):
        for i in range(12):
            rows.append({"player_key": f"{p}{i}", "position": p, "ecr": 10.0 * i + 1,
                         "pos_rank": i + 1, "adp_rank": 6 * i + 1,
                         # the last receiver is contested: best case WR3
                         "ecr_best": 21.0 if (p == "WR" and i == 11) else 10.0 * i + 1})
    ecr = pd.DataFrame(rows)
    pool = ecr_to_pool(ecr, totals, pd.DataFrame(), prior_season=2024,
                       train_seasons=[2024])
    contested = pool[(pool.position == "WR") & (pool.pos_rank == 12)].iloc[0]
    assert contested["pos_rank_best"] == 3          # two WRs have ecr < 21
    assert contested["upside_points"] > contested["ecr_points"]
    cfg = {"blend": {"components": {"ecr_points": 1.0}},
           "calibration": {"slopes": {"WR": 0.8}, "persistence_shrink": {}},
           "market_anchor": {"positions": {"K": 0.7}, "reference_positions": ["QB", "RB", "WR", "TE"]},
           "verified": {"starters": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"],
                        "teams": 2}}
    board = value_board(pool, totals, cfg, train_seasons=[2024])
    row = board[(board.position == "WR") & (board.pos_rank == 12)].iloc[0]
    assert row["upside_vorp"] > row["vorp"]
    # Never below the median, and anchored kickers keep their anchored value.
    assert (board["upside_vorp"] >= board["vorp"] - 1e-9).all()
    ks = board[board.position == "K"]
    assert (ks["upside_vorp"] == ks["vorp"]).all()


def test_a_bench_pick_goes_to_the_higher_upside_player():
    roster = _roster([("QB", 60.0), ("RB", 50.0), ("RB", 30.0), ("WR", 90.0),
                      ("WR", 40.0), ("WR", 35.0), ("TE", 60.0)])
    pool = _pool([("Safe WR", "WR", 4.0, 150), ("Lottery RB", "RB", 3.0, 160),
                  ("WR b", "WR", 2.0, 170), ("RB b", "RB", 2.5, 180),
                  ("WR c", "WR", 1.0, 190), ("RB c", "RB", 1.5, 200)]
                 + _filler(12, 300))
    pool["upside_vorp"] = pool["vorp"]
    pool.loc[pool.player_name == "Lottery RB", "upside_vorp"] = 40.0
    off = ValueDrafter(caps=CAPS).rank(pool, roster, 16, 9, 8)
    on = ValueDrafter(caps=CAPS, upside_tiebreak=True).rank(pool, roster, 16, 9, 8)
    assert off[0].player_name == "Safe WR"
    assert on[0].player_name == "Lottery RB"
    assert on[0].upside == 40.0


def test_upside_never_overrides_a_real_drop_off():
    d = ValueDrafter(upside_tiebreak=True)
    roster = _roster([])
    # gain > 0: the pick is about now, and upside stays out of it
    assert d._tiebreak(roster, "RB", 10.0, gain=5.0, upside=99.0) == 10.0
    assert d._tiebreak(roster, "RB", 10.0, gain=0.0, upside=99.0) == 99.0
    assert d._tiebreak(roster, "RB", 10.0, gain=0.0, upside=float("nan")) == 10.0


# --------------------------------------------------------------------------
# The noise draw must not depend on the board's row order
# --------------------------------------------------------------------------

def test_a_seeded_draft_is_the_same_whatever_order_the_board_arrives_in():
    """`OpponentModel.choose` draws one gaussian per row, in row order. Before
    `canonical_order`, sorting the same board by VORP instead of ADP handed
    every player a different draw, and the identical pure-ADP drafter scored
    4.35 in one run and 4.71 in another with nothing else changed."""
    b = _board()
    kw = dict(teams=10, rounds=16, my_slot=4, starters=STARTERS, flex_slots=1,
              flex_eligible=FLEX, sigma={p: 8.0 for p in STARTERS}, seed=21)
    a = simulate_draft(b, **kw)
    c = simulate_draft(b.sample(frac=1, random_state=3), **kw)
    d = simulate_draft(b.sort_values("vorp", ascending=False), **kw)
    for t in a:
        ids = [p["player_id"] for p in a[t].picks]
        assert ids == [p["player_id"] for p in c[t].picks]
        assert ids == [p["player_id"] for p in d[t].picks]
