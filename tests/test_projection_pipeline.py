"""Tests for the board pipeline: identity, scoring, and the rank curve.

These cover the parts that fail *quietly*. A broken snake-draft order announces
itself; a player who silently fails to reconcile just stops appearing on the
board, and a scoring rule that is subtly wrong produces a plausible number.

Run: .venv/Scripts/python.exe -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.draft.board import finalize_board  # noqa: E402
from src.features.rank_curve import (  # noqa: E402
    apply_spread_shrink,
    fit_rank_curve,
    measure_persistence,
    persistence_shrink,
    season_totals,
)
from src.features.scoring import (  # noqa: E402
    DEFAULT_SCORING,
    Scoring,
    points_allowed_points,
    score_defense,
    score_kicker,
    score_offense,
    score_weekly,
)
from src.ingest.player_ids import (  # noqa: E402
    build_crosswalk,
    canonical_key,
    merge_name,
    normalize_position,
    normalize_team,
    reconcile,
)

SCORING = Scoring(dict(DEFAULT_SCORING))


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def test_team_aliases_collapse_to_one_code():
    # The three franchises the sources actually disagree about.
    assert normalize_team("LA") == normalize_team("LAR") == "LAR"
    assert normalize_team("JAC") == normalize_team("JAX") == "JAX"
    assert normalize_team("OAK") == normalize_team("LV") == "LV"
    assert normalize_team("FA") is None
    assert normalize_team(np.nan) is None


def test_dst_and_def_are_the_same_position():
    assert normalize_position("DST") == "DEF"
    assert normalize_position("D/ST") == "DEF"
    assert normalize_position("def") == "DEF"
    assert normalize_position("wr") == "WR"


def test_merge_name_strips_suffixes_and_accents():
    assert merge_name("Marvin Harrison Jr.") == merge_name("Marvin Harrison")
    assert merge_name("Amon-Ra St. Brown") == "amon ra st brown"
    assert merge_name("Ka'imi Fairbairn") == "kaimi fairbairn"
    assert merge_name(None) == ""


def test_canonical_key_prefers_gsis_and_keys_defenses_by_team():
    assert canonical_key(position="WR", team="CIN", gsis_id="00-0036900",
                         fantasypros_id="17083") == "00-0036900"
    # No gsis (an undrafted rookie) falls back to FantasyPros, not to a name.
    assert canonical_key(position="RB", team="LV",
                         fantasypros_id="26801") == "FP_26801"
    # Defenses are not players and have no gsis id.
    assert canonical_key(position="DST", team="JAC") == "DEF_JAX"


def test_canonical_key_survives_the_parquet_float_round_trip():
    """Ids come back from parquet as floats; "22968.0" is not an id."""
    assert canonical_key(position="RB", team="DET",
                         fantasypros_id=22968.0) == "FP_22968"
    assert canonical_key(position="RB", team="DET",
                         fantasypros_id=float("nan"),
                         sleeper_id="9221") == "SL_9221"


def _playerids() -> pd.DataFrame:
    return pd.DataFrame([
        {"mfl_id": "1", "name": "Ja'Marr Chase", "position": "WR", "team": "CIN",
         "gsis_id": "00-0036900", "fantasypros_id": "17083",
         "sleeper_id": "7564", "db_season": 2026},
        {"mfl_id": "2", "name": "Rookie Runner", "position": "RB", "team": "LV",
         "gsis_id": None, "fantasypros_id": "26801",
         "sleeper_id": "99999", "db_season": 2026},
        # An older row for the same player, on his previous team.
        {"mfl_id": "1", "name": "Ja'Marr Chase", "position": "WR", "team": "LAR",
         "gsis_id": "00-0036900", "fantasypros_id": "17083",
         "sleeper_id": "7564", "db_season": 2021},
    ])


def test_crosswalk_resolves_from_every_id_space():
    cw = build_crosswalk(_playerids())
    assert cw.resolve(gsis_id="00-0036900") == "00-0036900"
    assert cw.resolve(fantasypros_id="17083") == "00-0036900"
    assert cw.resolve(sleeper_id="7564") == "00-0036900"
    # Name matching is the last resort, and it works when nothing else is given.
    assert cw.resolve(name="Ja'Marr Chase", position="WR") == "00-0036900"


def test_crosswalk_keeps_the_most_recent_team():
    cw = build_crosswalk(_playerids())
    assert cw.meta["00-0036900"]["nfl_team"] == "CIN"


def test_sleeper_enrichment_adds_defenses_and_injury_status():
    sleeper = {
        "HOU": {"position": "DEF", "team": "HOU", "injury_status": None,
                "full_name": "Houston Texans"},
        "7564": {"position": "WR", "team": "CIN", "injury_status": "IR",
                 "full_name": "Ja'Marr Chase"},
    }
    cw = build_crosswalk(_playerids(), sleeper)
    assert cw.resolve(sleeper_id="HOU") == "DEF_HOU"
    # Sleeper is the authority on status, and it reaches the existing entry.
    assert cw.meta["00-0036900"]["injury_status"] == "IR"
    assert cw.meta["00-0036900"]["sleeper_id"] == "7564"


def test_reconcile_reports_rather_than_drops():
    cw = build_crosswalk(_playerids())
    pool = pd.DataFrame([
        {"player": "Ja'Marr Chase", "id": "17083", "position": "WR"},
        {"player": "Nobody At All", "id": "00000", "position": "WR"},
    ])
    matched, unmatched = reconcile(
        pool, cw, fantasypros_col="id", name_col="player",
        position_col="position")
    assert list(matched["player_key"]) == ["00-0036900"]
    assert list(unmatched["player"]) == ["Nobody At All"]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_offense_scoring_matches_a_hand_computed_ppr_line():
    week = pd.DataFrame([{
        "receptions": 8, "receiving_yards": 120, "receiving_tds": 1,
        "rushing_yards": 10, "rushing_tds": 0, "rushing_fumbles_lost": 1,
    }])
    # 8 rec + 12.0 rec yds + 6 TD + 1.0 rush yds - 2 fumble = 25.0
    assert score_offense(week, SCORING).iloc[0] == pytest.approx(25.0)


def test_missing_stat_columns_score_as_zero_not_as_a_crash():
    """nflverse renames columns between releases; the board must survive it."""
    week = pd.DataFrame([{"receptions": 5, "receiving_yards": 50}])
    assert score_offense(week, SCORING).iloc[0] == pytest.approx(10.0)


def test_kicker_scoring_respects_distance_buckets():
    week = pd.DataFrame([{
        "fg_made_30_39": 1, "fg_made_50_59": 1, "fg_missed": 1, "pat_made": 3,
    }])
    # 3 (short) + 5 (long) - 1 (miss) + 3 (PATs) = 10
    assert score_kicker(week, SCORING).iloc[0] == pytest.approx(10.0)


def test_a_long_field_goal_is_worth_more_than_a_short_one():
    long_fg = pd.DataFrame([{"fg_made_50_59": 1}])
    short_fg = pd.DataFrame([{"fg_made_20_29": 1}])
    assert (score_kicker(long_fg, SCORING).iloc[0]
            > score_kicker(short_fg, SCORING).iloc[0])


@pytest.mark.parametrize("allowed,expected", [
    (0, 10.0), (1, 7.0), (6, 7.0), (7, 4.0), (13, 4.0),
    (14, 1.0), (20, 1.0), (21, 0.0), (27, 0.0),
    (28, -1.0), (34, -1.0), (35, -4.0), (52, -4.0),
])
def test_points_allowed_tiers_land_on_the_right_side_of_every_boundary(
        allowed, expected):
    got = points_allowed_points(pd.Series([allowed]), SCORING).iloc[0]
    assert got == pytest.approx(expected)


def test_defense_is_charged_with_what_the_opponent_scored():
    team_weekly = pd.DataFrame([{
        "season": 2025, "week": 1, "team": "HOU", "opponent_team": "IND",
        "def_sacks": 3, "def_interceptions": 2, "def_tds": 1,
        "fumble_recovery_opp": 1, "def_safeties": 0,
    }])
    schedules = pd.DataFrame([{
        "season": 2025, "week": 1, "home_team": "HOU", "home_score": 27,
        "away_team": "IND", "away_score": 10,
    }])
    out = score_defense(team_weekly, schedules, SCORING)
    # IND scored 10, so HOU's defense is in the 7-13 tier (+4), not HOU's 27.
    assert out["points_allowed"].iloc[0] == 10
    # 3 sacks + 4 int + 6 TD + 2 fum rec (`fum_rec`, the defensive key, not
    # the special-teams one) + 4 tier = 19
    assert out["fantasy_points"].iloc[0] == pytest.approx(19.0)


def test_score_weekly_does_not_pay_a_kicker_for_receiving_yards():
    weekly = pd.DataFrame([
        {"position": "K", "receptions": 5, "receiving_yards": 60,
         "fg_made_30_39": 2},
        {"position": "WR", "receptions": 5, "receiving_yards": 60,
         "fg_made_30_39": 2},
    ])
    out = score_weekly(weekly, SCORING)
    assert out["fantasy_points"].iloc[0] == pytest.approx(6.0)   # K: 2 FGs only
    assert out["fantasy_points"].iloc[1] == pytest.approx(11.0)  # WR: 5 + 6.0


# --------------------------------------------------------------------------
# Rank curve
# --------------------------------------------------------------------------

def _totals(seasons=(2022, 2023, 2024), n=40, noise=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for s in seasons:
        for i in range(n):
            rows.append({
                "player_key": f"WR{i}", "position": "WR", "season": s,
                "points": 300 - i * 5 + rng.normal(0, noise),
                "games": 17,
            })
    return pd.DataFrame(rows)


def test_season_totals_separates_rate_from_volume():
    weekly = pd.DataFrame([
        {"player_key": "a", "position": "WR", "season": 2025, "week": w,
         "season_type": "REG", "fantasy_points": 20.0} for w in range(1, 9)
    ])
    out = season_totals(weekly)
    assert out["points"].iloc[0] == pytest.approx(160.0)
    assert out["games"].iloc[0] == 8
    # Eight big games is a 20 ppg player, not a 9.4 ppg player.
    assert out["points_per_game"].iloc[0] == pytest.approx(20.0)


def test_season_totals_ignores_week_18_and_the_postseason():
    weekly = pd.DataFrame([
        {"player_key": "a", "position": "WR", "season": 2025, "week": 1,
         "season_type": "REG", "fantasy_points": 10.0},
        {"player_key": "a", "position": "WR", "season": 2025, "week": 18,
         "season_type": "REG", "fantasy_points": 99.0},
        {"player_key": "a", "position": "WR", "season": 2025, "week": 19,
         "season_type": "POST", "fantasy_points": 99.0},
    ])
    assert season_totals(weekly)["points"].iloc[0] == pytest.approx(10.0)


def test_curve_is_monotone_non_increasing_even_with_noisy_input():
    curve = fit_rank_curve(_totals(noise=40.0), positions=("WR",))
    pts = [curve.points_for_rank("WR", r) for r in range(1, 41)]
    assert all(a >= b - 1e-9 for a, b in zip(pts, pts[1:])), pts


def test_curve_clamps_below_rank_one_and_never_extrapolates_negative():
    curve = fit_rank_curve(_totals(), positions=("WR",))
    assert curve.points_for_rank("WR", 0.5) == curve.points_for_rank("WR", 1)
    # Far past the fitted tail, the curve floors rather than going negative.
    assert curve.points_for_rank("WR", 500) >= 0.0


def test_curve_interpolates_between_fitted_ranks():
    curve = fit_rank_curve(_totals(), positions=("WR",))
    lo = curve.points_for_rank("WR", 10)
    hi = curve.points_for_rank("WR", 11)
    mid = curve.points_for_rank("WR", 10.5)
    assert hi <= mid <= lo


def test_downweighting_a_season_moves_the_curve_away_from_it():
    totals = _totals(seasons=(2023, 2024))
    # Make 2024 an outlier year, then discount it the way config does for K/DEF.
    totals.loc[totals["season"] == 2024, "points"] *= 3.0

    full = fit_rank_curve(totals, positions=("WR",))
    damped = fit_rank_curve(
        totals, positions=("WR",),
        downweight=[{"season": 2024, "positions": ["WR"], "weight": 0.1}],
    )
    assert damped.points_for_rank("WR", 1) < full.points_for_rank("WR", 1)


def test_unknown_position_returns_nan_rather_than_a_wrong_number():
    curve = fit_rank_curve(_totals(), positions=("WR",))
    assert np.isnan(curve.points_for_rank("QB", 1))


# --------------------------------------------------------------------------
# Persistence shrink
# --------------------------------------------------------------------------

def _two_season_totals(pos: str, order_second_year: list[int]) -> pd.DataFrame:
    rows = []
    for i in range(len(order_second_year)):
        rows.append({"player_key": f"{pos}{i}", "position": pos,
                     "season": 2024, "points": 300 - i * 5, "games": 17})
    for rank, i in enumerate(order_second_year):
        rows.append({"player_key": f"{pos}{i}", "position": pos,
                     "season": 2025, "points": 300 - rank * 5, "games": 17})
    return pd.DataFrame(rows)


def test_persistence_is_one_when_the_order_repeats_exactly():
    totals = _two_season_totals("QB", list(range(30)))
    assert measure_persistence(totals, positions=("QB",))["QB"] == pytest.approx(1.0)


def test_persistence_is_negative_one_when_the_order_fully_reverses():
    totals = _two_season_totals("QB", list(reversed(range(30))))
    assert measure_persistence(totals, positions=("QB",))["QB"] == pytest.approx(-1.0)


def test_shrink_factor_is_relative_to_the_reference_position():
    stable = _two_season_totals("QB", list(range(30)))
    # A position that half-reverses persists less than one that repeats.
    shuffled = list(range(30))
    shuffled[:20] = list(reversed(shuffled[:20]))
    churny = _two_season_totals("DEF", shuffled)
    totals = pd.concat([stable, churny], ignore_index=True)

    factors = persistence_shrink(totals, ["DEF"], reference="QB")
    assert 0.0 <= factors["DEF"] < 1.0
    # Scoped: the reference position is never itself shrunk.
    assert "QB" not in factors


def test_spread_shrink_only_touches_the_named_positions():
    df = pd.DataFrame([
        {"position": "DEF", "projection_calibrated": 160.0},
        {"position": "DEF", "projection_calibrated": 100.0},
        {"position": "WR", "projection_calibrated": 300.0},
        {"position": "WR", "projection_calibrated": 100.0},
    ])
    out = apply_spread_shrink(df, {"DEF": 0.5})

    wr = out[out["position"] == "WR"]["projection_calibrated"].tolist()
    assert wr == [300.0, 100.0]

    de = out[out["position"] == "DEF"]["projection_calibrated"].tolist()
    # Mean 130 is preserved; the 60-point spread halves to 30.
    assert de == pytest.approx([145.0, 115.0])
    assert np.mean(de) == pytest.approx(130.0)


def test_shrink_is_a_no_op_when_no_positions_are_configured():
    df = pd.DataFrame([{"position": "K", "projection_calibrated": 150.0}])
    pd.testing.assert_frame_equal(apply_spread_shrink(df, {}), df)


# --------------------------------------------------------------------------
# Board contract
# --------------------------------------------------------------------------

def test_board_carries_both_the_sleeper_id_and_the_canonical_key():
    """The monitor matches on `player_id` (Sleeper); everything else joins on
    `player_key`. Losing either one breaks a different half of the system."""
    df = pd.DataFrame([
        {"player_id": "7564", "player_key": "00-0036900",
         "player_name": "Ja'Marr Chase", "position": "WR", "nfl_team": "CIN",
         "vorp": 150.0, "adp_rank": 1},
        {"player_id": "HOU", "player_key": "DEF_HOU",
         "player_name": "Houston Texans", "position": "DEF", "nfl_team": "HOU",
         "vorp": 12.0, "adp_rank": 146},
    ])
    board = finalize_board(df)
    assert list(board["player_id"]) == ["7564", "HOU"]
    assert list(board["player_key"]) == ["00-0036900", "DEF_HOU"]
    # Positive delta = the board likes him more than the room does.
    assert board["adp_delta"].iloc[1] == 144
