"""Tests for the logic that is easy to get subtly wrong and hard to notice.

Run: .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.draft_sim import Roster, snake_order  # noqa: E402
from src.backtest.leakage_guard import (  # noqa: E402
    LeakageError,
    TemporalBoundary,
    assert_no_future_seasons,
    assert_preseason_only,
    assert_purge_gap,
    flag_suspicious_stability,
)
from src.backtest.metrics import optimal_lineup_points  # noqa: E402
from src.draft.monitor import picks_until_next_turn  # noqa: E402
from src.features.blend import blend_projections  # noqa: E402
from src.features.calibration import apply_calibration, fit_slopes  # noqa: E402
from src.features.vorp import LeagueShape, compute_vorp, replacement_ranks  # noqa: E402

STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}


# --------------------------------------------------------------------------
# Snake draft order
# --------------------------------------------------------------------------

def test_snake_order_reverses_each_round():
    order = snake_order(teams=4, rounds=3)
    assert order[:4] == [0, 1, 2, 3]
    assert order[4:8] == [3, 2, 1, 0]
    assert order[8:12] == [0, 1, 2, 3]


def test_picks_until_next_turn_matches_snake_turn_round():
    # 10-team, slot 1: picks 1 and 20 are yours, so 18 picks in between.
    assert picks_until_next_turn(1, teams=10, my_slot=1) == 18
    # Slot 10 picks at 10 and 11 — back-to-back turn.
    assert picks_until_next_turn(10, teams=10, my_slot=10) == 0


def test_picks_until_next_turn_agrees_with_snake_order():
    teams, rounds, slot = 10, 16, 4
    order = snake_order(teams, rounds)
    my_picks = [i + 1 for i, t in enumerate(order) if t == slot - 1]
    for a, b in zip(my_picks, my_picks[1:]):
        assert picks_until_next_turn(a, teams, slot) == b - a - 1


# --------------------------------------------------------------------------
# VORP
# --------------------------------------------------------------------------

def _pool():
    rows = []
    for pos, n, base in [("QB", 30, 300), ("RB", 60, 260), ("WR", 70, 250),
                         ("TE", 30, 180), ("K", 20, 130), ("DEF", 20, 120)]:
        for i in range(n):
            rows.append({"player_id": f"{pos}{i}", "player_name": f"{pos} {i}",
                         "position": pos, "projection": base - i * 3.0})
    return pd.DataFrame(rows)


def test_replacement_rank_accounts_for_flex():
    shape = LeagueShape(teams=10, starters=STARTERS, flex_slots=1)
    ranks = replacement_ranks(shape, _pool())
    # Dedicated slots only, for a position with no flex eligibility.
    assert ranks["QB"] == 10
    # Flex adds 10 more starters spread across RB/WR/TE.
    flex_total = ranks["RB"] + ranks["WR"] + ranks["TE"]
    assert flex_total == (10 * 2) + (10 * 2) + (10 * 1) + 10


def test_vorp_is_zero_at_replacement_level():
    shape = LeagueShape(teams=10, starters=STARTERS, flex_slots=1)
    out = compute_vorp(_pool(), shape)
    for pos in out["position"].unique():
        sub = out[out["position"] == pos].sort_values("projection", ascending=False)
        repl = sub["replacement_points"].iloc[0]
        # The player at the replacement cutoff has VORP ~0 by construction.
        at_repl = sub[sub["projection"] == repl]
        assert not at_repl.empty
        assert abs(float(at_repl["vorp"].iloc[0])) < 1e-9


def test_vorp_ranks_scarce_position_above_deep_one():
    """A QB and an RB with identical raw points must not have identical value."""
    shape = LeagueShape(teams=10, starters=STARTERS, flex_slots=1)
    out = compute_vorp(_pool(), shape).set_index("player_id")
    # QB pool is shallow above replacement; RB pool is deep. Same raw points
    # should therefore produce different VORP.
    assert out.loc["QB0", "vorp"] != out.loc["RB0", "vorp"]


def test_vorp_handles_position_thinner_than_demand():
    thin = pd.DataFrame([
        {"player_id": "K0", "position": "K", "projection": 130.0},
        {"player_id": "K1", "position": "K", "projection": 120.0},
    ])
    shape = LeagueShape(teams=10, starters={"K": 1}, flex_slots=0)
    out = compute_vorp(thin, shape)
    # Must not treat replacement as 0 and report absurd VORP.
    assert out["vorp"].max() < 20


# --------------------------------------------------------------------------
# Blend
# --------------------------------------------------------------------------

def test_blend_keeps_players_missing_a_component():
    df = pd.DataFrame([
        {"player_id": "a", "position": "WR", "xfp": 200.0, "fantasypros": 210.0},
        {"player_id": "b", "position": "WR", "xfp": 150.0, "fantasypros": None},
        {"player_id": "c", "position": "WR", "xfp": 100.0, "fantasypros": 90.0},
    ])
    out = blend_projections(df, weights={"xfp": 1.0, "fantasypros": 1.0})
    assert len(out) == 3
    assert out["projection"].notna().all()
    # b used one component, a and c used two.
    assert out.set_index("player_id").loc["b", "blend_components_used"] == 1
    assert out.set_index("player_id").loc["a", "blend_components_used"] == 2


def test_blend_preserves_ordering_when_components_agree():
    df = pd.DataFrame([
        {"player_id": "a", "position": "RB", "xfp": 300.0, "fantasypros": 300.0},
        {"player_id": "b", "position": "RB", "xfp": 200.0, "fantasypros": 200.0},
        {"player_id": "c", "position": "RB", "xfp": 100.0, "fantasypros": 100.0},
    ])
    out = blend_projections(df).sort_values("projection", ascending=False)
    assert list(out["player_id"]) == ["a", "b", "c"]


def test_blend_zscores_within_position_not_across():
    """QB and RB scales differ; the blend must not let one dominate."""
    df = pd.DataFrame([
        {"player_id": "qb1", "position": "QB", "xfp": 400.0, "fantasypros": 400.0},
        {"player_id": "qb2", "position": "QB", "xfp": 380.0, "fantasypros": 380.0},
        {"player_id": "rb1", "position": "RB", "xfp": 260.0, "fantasypros": 260.0},
        {"player_id": "rb2", "position": "RB", "xfp": 240.0, "fantasypros": 240.0},
    ])
    out = blend_projections(df).set_index("player_id")
    # Each position's top player stays on its own scale.
    assert out.loc["qb1", "projection"] > out.loc["qb2", "projection"]
    assert out.loc["rb1", "projection"] > out.loc["rb2", "projection"]
    assert out.loc["qb1", "projection"] > out.loc["rb1", "projection"]


def test_blend_raises_when_no_components_present():
    df = pd.DataFrame([{"player_id": "a", "position": "WR", "something": 1.0}])
    with pytest.raises(ValueError, match="none of the blend components"):
        blend_projections(df)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_calibration_shrinks_spread_toward_positional_mean():
    df = pd.DataFrame([
        {"position": "QB", "projection": 400.0},
        {"position": "QB", "projection": 300.0},
        {"position": "QB", "projection": 200.0},
    ])
    out = apply_calibration(df, slopes={"QB": 0.5})
    assert out["projection_calibrated"].mean() == pytest.approx(300.0)
    # Spread halved, mean preserved.
    raw_spread = df["projection"].max() - df["projection"].min()
    cal_spread = out["projection_calibrated"].max() - out["projection_calibrated"].min()
    assert cal_spread == pytest.approx(raw_spread * 0.5)


def test_fit_slopes_keeps_prior_when_sample_too_small():
    history = pd.DataFrame({
        "position": ["QB"] * 5,
        "projection": [300, 310, 320, 330, 340],
        "actual": [200, 400, 150, 500, 100],
    })
    slopes = fit_slopes(history, priors={"QB": 0.67})
    assert slopes["QB"] == 0.67


def test_fit_slopes_recovers_a_known_slope():
    n = 200
    proj = pd.Series(range(n), dtype=float)
    actual = (proj - proj.mean()) * 0.8 + proj.mean()
    history = pd.DataFrame({
        "position": ["RB"] * n, "projection": proj, "actual": actual,
    })
    slopes = fit_slopes(history)
    assert slopes["RB"] == pytest.approx(0.8, abs=0.02)


# --------------------------------------------------------------------------
# Optimal lineup
# --------------------------------------------------------------------------

def test_optimal_lineup_uses_flex_for_best_leftover():
    wk = pd.DataFrame([
        {"position": "QB", "points": 20.0},
        {"position": "RB", "points": 15.0}, {"position": "RB", "points": 12.0},
        {"position": "RB", "points": 11.0},          # flex candidate
        {"position": "WR", "points": 14.0}, {"position": "WR", "points": 13.0},
        {"position": "WR", "points": 5.0},           # flex candidate, worse
        {"position": "TE", "points": 8.0},
        {"position": "K", "points": 7.0},
        {"position": "DEF", "points": 6.0},
    ])
    total = optimal_lineup_points(wk, STARTERS, flex_slots=1,
                                  flex_eligible=("RB", "WR", "TE"))
    # 20+15+12+14+13+8+7+6 = 95, plus the best leftover flex (RB 11) = 106
    assert total == pytest.approx(106.0)


def test_optimal_lineup_does_not_double_start_a_player():
    wk = pd.DataFrame([
        {"position": "RB", "points": 30.0},
        {"position": "QB", "points": 10.0},
    ])
    total = optimal_lineup_points(wk, {"QB": 1, "RB": 2}, flex_slots=1,
                                  flex_eligible=("RB",))
    # Only one RB exists; he fills one RB slot and cannot also fill flex.
    assert total == pytest.approx(40.0)


def test_missing_player_scores_zero_not_dropped():
    wk = pd.DataFrame([{"position": "QB", "points": None}])
    assert optimal_lineup_points(wk, {"QB": 1}, 0, ()) == 0.0


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------

def test_future_season_raises():
    df = pd.DataFrame({"season": [2022, 2023, 2024]})
    with pytest.raises(LeakageError, match="2024"):
        assert_no_future_seasons(df, TemporalBoundary(target_season=2024))


def test_missing_season_column_raises_rather_than_skipping():
    with pytest.raises(LeakageError, match="cannot be verified"):
        assert_no_future_seasons(
            pd.DataFrame({"x": [1]}), TemporalBoundary(target_season=2024)
        )


def test_preseason_snapshot_of_target_season_is_allowed():
    df = pd.DataFrame({"season": [2024, 2024], "week": [0, 0]})
    assert_preseason_only(df, TemporalBoundary(target_season=2024))


def test_regular_season_rows_disguised_as_snapshot_raise():
    df = pd.DataFrame({"season": [2024, 2024], "week": [0, 3]})
    with pytest.raises(LeakageError, match="week > 0"):
        assert_preseason_only(df, TemporalBoundary(target_season=2024))


def test_purge_gap_enforced():
    # gap=1, target 2024: training may end at 2023.
    assert_purge_gap([2020, 2021, 2023], target_season=2024, purge_gap=1)

    # Training that includes the target season itself is a straight violation.
    with pytest.raises(LeakageError, match="purge_gap"):
        assert_purge_gap([2020, 2021, 2024], target_season=2024, purge_gap=1)

    # A wider gap pushes the required cutoff back.
    with pytest.raises(LeakageError, match="at or before 2022"):
        assert_purge_gap([2020, 2021, 2023], target_season=2024, purge_gap=2)


def test_suspicious_stability_flags_near_identical_folds():
    assert flag_suspicious_stability([100.0, 100.1, 99.9, 100.05]) is not None
    assert flag_suspicious_stability([100.0, 130.0, 85.0, 115.0]) is None


# --------------------------------------------------------------------------
# Roster need
# --------------------------------------------------------------------------

def _roster():
    return Roster(team_id=0, starters=dict(STARTERS), flex_slots=1,
                  flex_eligible=("RB", "WR", "TE"), bench_slots=7)


def test_unfilled_slots_on_empty_roster():
    need = _roster().unfilled_starting_slots()
    assert need["RB"] == 2 + 1  # dedicated + flex
    assert need["QB"] == 1


def test_flex_demand_clears_once_a_surplus_exists():
    r = _roster()
    for _ in range(3):
        r.picks.append({"position": "RB"})
    need = r.unfilled_starting_slots()
    # 3 RBs covers 2 dedicated + the flex, so RB need is gone.
    assert need["RB"] == 0
    assert need["WR"] == 2


def test_total_unfilled_does_not_triple_count_flex():
    """Regression: summing the per-position need dict counts flex 3x.

    That overcount made the end-of-draft starter backstop mis-fire and let a
    simulated roster finish with an empty QB slot.
    """
    r = _roster()
    per_position = r.unfilled_starting_slots()
    # Flex appears under RB, WR and TE, so the naive sum overcounts.
    assert sum(per_position.values()) > r.total_unfilled_starting_slots()
    # 1 QB + 2 RB + 2 WR + 1 TE + 1 K + 1 DEF + 1 FLEX = 9
    assert r.total_unfilled_starting_slots() == 9


# --------------------------------------------------------------------------
# Draft simulation — end-to-end invariants
# --------------------------------------------------------------------------

def _synthetic_board():
    """A board with *value-aware* ADP, which is what real ADP looks like.

    Ranking ADP by raw projected points instead would push every QB into the
    first three rounds and exhaust the position — an artifact of the fixture,
    not of the league.
    """
    rows = []
    for pos, n, base, step in [
        ("QB", 32, 330, 4.0), ("RB", 72, 270, 3.0), ("WR", 84, 255, 2.4),
        ("TE", 32, 190, 4.0), ("K", 24, 135, 1.2), ("DEF", 24, 128, 1.5),
    ]:
        for i in range(n):
            rows.append({
                "player_id": f"{pos}{i}", "player_name": f"{pos} {i}",
                "position": pos, "projection": base - i * step,
            })
    shape = LeagueShape(teams=10, starters=STARTERS, flex_slots=1)
    board = compute_vorp(pd.DataFrame(rows), shape)
    board = board.sort_values("vorp", ascending=False).reset_index(drop=True)
    board["adp_rank"] = range(1, len(board) + 1)
    return board


SIGMA = {"QB": 12.0, "RB": 10.0, "WR": 8.0, "TE": 11.0, "K": 6.0, "DEF": 6.0}


def _sim(slot: int, seed: int):
    from src.backtest.draft_sim import simulate_draft
    return simulate_draft(
        _synthetic_board(), teams=10, rounds=16, my_slot=slot,
        starters=STARTERS, flex_slots=1, flex_eligible=("RB", "WR", "TE"),
        sigma=SIGMA, seed=seed,
    )


@pytest.mark.parametrize("slot", [1, 4, 7, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_simulated_roster_always_fills_every_starting_slot(slot, seed):
    """Regression: an unfilled starter is a zero every week, never acceptable."""
    me = _sim(slot, seed)[slot - 1]
    assert me.total_unfilled_starting_slots() == 0, (
        f"slot {slot} seed {seed} finished with "
        f"{me.unfilled_dedicated_slots()} unfilled"
    )


@pytest.mark.parametrize("slot", [1, 5, 10])
def test_simulated_roster_never_finishes_without_a_quarterback(slot):
    """Regression: the scarcity backstop must grab a QB before the pool dries up."""
    me = _sim(slot, seed=3)[slot - 1]
    assert me.counts().get("QB", 0) >= 1


def test_simulation_never_drafts_the_same_player_twice():
    rosters = _sim(4, seed=11)
    ids = [p["player_id"] for r in rosters.values() for p in r.picks]
    assert len(ids) == len(set(ids))
    assert len(ids) == 10 * 16


def test_position_caps_prevent_hoarding():
    me = _sim(4, seed=5)[3]
    counts = me.counts()
    assert counts.get("TE", 0) <= 2
    assert counts.get("K", 0) <= 1
    assert counts.get("QB", 0) <= 2
