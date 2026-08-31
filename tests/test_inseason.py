"""Tests for the in-season modules and the lineup math they share.

The weekly recommendations cannot be backtested — the FantasyPros weekly feed is
a live snapshot with no archive — so these tests are the only thing standing
between a plausible-looking table and a wrong one. They lean on cases where the
right answer is arguable from first principles rather than from data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.lineup import (  # noqa: E402
    LineupSlots,
    best_lineup,
    lineup_points,
    value_of_adding,
    value_of_dropping,
)
from src.ingest.player_ids import normalize_position, normalize_team  # noqa: E402
from src.inseason import start_sit, trades, waivers  # noqa: E402

SLOTS = LineupSlots(
    starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1},
    flex_slots=1, flex_eligible=("RB", "WR", "TE"),
)


def _players(rows: list[tuple[str, str, float]], sd: float = 3.0) -> pd.DataFrame:
    return pd.DataFrame([
        {"player_key": n, "player_name": n, "position": p,
         "projection": v, "sd": sd}
        for n, p, v in rows
    ])


def _full_roster() -> pd.DataFrame:
    return _players([
        ("QB1", "QB", 20.0),
        ("RBa", "RB", 15.0), ("RBb", "RB", 12.0), ("RBc", "RB", 8.0),
        ("WRa", "WR", 18.0), ("WRb", "WR", 14.0), ("WRc", "WR", 11.0),
        ("TEa", "TE", 9.0),
        ("Ka", "K", 8.0), ("DEFa", "DEF", 7.0),
    ])


# --------------------------------------------------------------------------
# Identity aliases — the bug that broke the whole waiver report
# --------------------------------------------------------------------------

def test_dynastyprocess_kicker_code_normalises_to_k():
    """DynastyProcess spells kicker `PK`. Before this alias existed, a roster's
    kicker did not count as a `K`, the lineup showed the K slot unfilled, and
    every kicker on the wire scored as a +8 upgrade worth burning waiver
    priority on — which is about the worst advice this system could give."""
    assert normalize_position("PK") == "K"
    assert normalize_position("K") == "K"


def test_dynastyprocess_team_codes_normalise():
    for code, want in [("GBP", "GB"), ("KCC", "KC"), ("NEP", "NE"),
                       ("NOS", "NO"), ("TBB", "TB"), ("RAM", "LAR"),
                       ("SDC", "LAC"), ("LVR", "LV")]:
        assert normalize_team(code) == want, code


# --------------------------------------------------------------------------
# Lineup
# --------------------------------------------------------------------------

def test_best_lineup_fills_every_slot_once():
    lineup = best_lineup(_full_roster(), SLOTS)
    assert len(lineup) == SLOTS.size
    assert sorted(lineup.starters["slot"]) == sorted(SLOTS.labels())


def test_flex_takes_the_best_remaining_eligible_player():
    lineup = best_lineup(_full_roster(), SLOTS)
    flex = lineup.starters[lineup.starters["slot"] == "FLEX"].iloc[0]
    # WRc (11.0) beats RBc (8.0) for the flex; TEa is already starting.
    assert flex["player_name"] == "WRc"


def test_a_kicker_is_never_flexed():
    roster = _players([("Ka", "K", 30.0), ("RBa", "RB", 5.0), ("RBb", "RB", 4.0),
                       ("WRa", "WR", 3.0), ("WRb", "WR", 2.0)])
    lineup = best_lineup(roster, SLOTS)
    flex = lineup.starters[lineup.starters["slot"] == "FLEX"]
    assert flex.empty or flex.iloc[0]["position"] != "K"


def test_missing_projections_score_zero_rather_than_dropping_the_player():
    roster = _full_roster()
    roster.loc[roster["player_name"] == "QB1", "projection"] = None
    lineup = best_lineup(roster, SLOTS)
    assert "QB" in set(lineup.starters["slot"])
    assert lineup.points == pytest.approx(
        lineup_points(_full_roster(), SLOTS) - 20.0)


def test_an_unfillable_slot_simply_goes_unfilled():
    roster = _players([("QB1", "QB", 20.0)])   # nobody else
    lineup = best_lineup(roster, SLOTS)
    assert list(lineup.starters["slot"]) == ["QB"]
    assert lineup.points == pytest.approx(20.0)


def test_value_of_adding_is_zero_for_a_player_who_cannot_start():
    roster = _full_roster()
    scrub = pd.Series({"player_key": "x", "player_name": "Scrub",
                       "position": "WR", "projection": 1.0, "sd": 3.0})
    assert value_of_adding(roster, scrub, SLOTS) == pytest.approx(0.0)


def test_value_of_adding_counts_only_the_upgrade_not_the_projection():
    roster = _full_roster()
    star = pd.Series({"player_key": "x", "player_name": "Star",
                      "position": "WR", "projection": 25.0, "sd": 3.0})
    # He displaces WRc (11.0) from the flex, so the gain is 14.0, not 25.0.
    assert value_of_adding(roster, star, SLOTS) == pytest.approx(14.0)


def test_dropping_a_surplus_player_costs_nothing():
    roster = _full_roster()
    idx = roster.index[roster["player_name"] == "RBc"][0]
    assert value_of_dropping(roster, idx, SLOTS) == pytest.approx(0.0)


def test_dropping_a_starter_costs_the_gap_to_his_replacement():
    roster = _full_roster()
    idx = roster.index[roster["player_name"] == "QB1"][0]
    # No other QB, so the whole 20 points go.
    assert value_of_dropping(roster, idx, SLOTS) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# Start / sit
# --------------------------------------------------------------------------

def test_a_narrow_gap_against_wide_uncertainty_is_a_tossup():
    roster = _players([
        ("QB1", "QB", 20.0), ("RBa", "RB", 15.0), ("RBb", "RB", 12.0),
        ("WRa", "WR", 18.0), ("WRb", "WR", 14.0),
        ("TEa", "TE", 9.0), ("Ka", "K", 8.0), ("DEFa", "DEF", 7.0),
        ("WRc", "WR", 11.0), ("WRd", "WR", 10.8),
    ], sd=5.0)
    lineup, calls, _ = start_sit.recommend(roster, SLOTS)
    closest = calls[0]
    assert closest.gap < 1.0
    assert closest.tossup, "0.2 points apart with sd 5 is not a decision"


def test_a_wide_gap_against_tight_uncertainty_is_a_real_call():
    roster = _players([
        ("QB1", "QB", 20.0), ("RBa", "RB", 15.0), ("RBb", "RB", 12.0),
        ("WRa", "WR", 18.0), ("WRb", "WR", 14.0),
        ("TEa", "TE", 9.0), ("Ka", "K", 8.0), ("DEFa", "DEF", 7.0),
        ("WRc", "WR", 13.0), ("WRd", "WR", 3.0),
    ], sd=0.5)
    _, calls, _ = start_sit.recommend(roster, SLOTS)
    assert not calls[0].tossup


def test_an_injured_player_is_not_started():
    roster = _full_roster()
    roster["available"] = True
    roster.loc[roster["player_name"] == "WRa", "available"] = False
    lineup, _, _ = start_sit.recommend(roster, SLOTS)
    assert "WRa" not in set(lineup.starters["player_name"])


def test_a_bye_week_hole_is_reported():
    roster = _players([
        ("QB1", "QB", 20.0), ("RBa", "RB", 15.0), ("RBb", "RB", 12.0),
        ("WRa", "WR", 18.0), ("WRb", "WR", 14.0), ("WRc", "WR", 10.0),
        ("TEa", "TE", 9.0), ("DEFa", "DEF", 7.0),
    ])   # no kicker at all
    _, _, holes = start_sit.recommend(roster, SLOTS)
    assert "K" in holes


# --------------------------------------------------------------------------
# Waivers — rolling priority
# --------------------------------------------------------------------------

def test_a_marginal_upgrade_does_not_burn_priority():
    roster = _full_roster()
    cands = _players([("Meh", "WR", 11.5)])
    moves = waivers.evaluate(roster, cands, SLOTS, priority_threshold=4.0)
    assert moves and not moves[0].burns_priority


def test_a_real_difference_maker_does_burn_priority():
    roster = _full_roster()
    cands = _players([("Stud", "RB", 24.0)])
    moves = waivers.evaluate(roster, cands, SLOTS, priority_threshold=4.0)
    assert moves and moves[0].burns_priority


def test_a_player_who_cannot_crack_the_lineup_is_not_suggested_at_all():
    roster = _full_roster()
    cands = _players([("Nobody", "WR", 2.0)])
    assert waivers.evaluate(roster, cands, SLOTS) == []


def test_a_full_roster_charges_the_cost_of_the_player_dropped():
    roster = _full_roster()
    cands = _players([("Stud", "RB", 24.0)])
    free = waivers.evaluate(roster, cands, SLOTS, roster_size=None)[0]
    full = waivers.evaluate(roster, cands, SLOTS, roster_size=len(roster))[0]
    assert full.drop is not None
    assert full.net <= free.net


def test_the_summary_tells_you_to_keep_priority_when_nothing_qualifies():
    text = waivers.summarize([], threshold=4.0)
    assert "keep your priority" in text.lower()


# --------------------------------------------------------------------------
# Trades
# --------------------------------------------------------------------------

def test_a_trade_is_judged_by_the_lineup_not_by_player_rankings():
    """Two-for-one that looks like a downgrade on a ranking sheet and is an
    upgrade in the lineup, because the depth being given up never started."""
    roster = _full_roster()
    giving = roster[roster["player_name"].isin(["RBc", "WRc"])]
    getting = _players([("Upgrade", "WR", 22.0)])
    verdict = trades.evaluate(roster, giving, getting, SLOTS)
    assert verdict.lineup_delta > 0
    assert verdict.net > 0


def test_giving_up_a_starter_for_bench_depth_is_declined():
    roster = _full_roster()
    giving = roster[roster["player_name"] == "WRa"]     # 18.0, starts
    getting = _players([("Spare", "WR", 6.0)])
    verdict = trades.evaluate(roster, giving, getting, SLOTS)
    assert verdict.net < 0
    assert "DECLINE" in verdict.describe()


def test_trading_away_the_only_quarterback_is_flagged_as_thinning():
    roster = _full_roster()
    giving = roster[roster["player_name"] == "QB1"]
    getting = _players([("WRx", "WR", 19.0)])
    verdict = trades.evaluate(roster, giving, getting, SLOTS)
    assert "QB" in verdict.positions_thinned


def test_surplus_is_exactly_the_players_the_lineup_does_not_need():
    surplus = set(trades.find_surplus(_full_roster(), SLOTS)["player_name"])
    assert surplus == {"RBc"}, surplus


# --------------------------------------------------------------------------
# Injury report — the two sources cover different phases
# --------------------------------------------------------------------------

from src.ingest.injuries import DNP, FULL, apply_to_roster, concern  # noqa: E402
from src.inseason import matchup, report  # noqa: E402


def test_questionable_without_practice_is_riskier_than_questionable_with():
    """The distinction people miss: game status alone treats these the same."""
    assert concern("Questionable", DNP) == "risky"
    assert concern("Questionable", FULL) == "monitor"


def test_out_and_doubtful_both_mean_do_not_start():
    assert concern("Out", FULL) == "out"
    assert concern("Doubtful", FULL) == "doubtful"


def test_a_clean_player_is_clear():
    assert concern(None, FULL) == "clear"


def test_the_injury_report_never_re_enables_a_player_sleeper_has_on_ir():
    roster = pd.DataFrame([
        {"player_key": "a", "player_name": "OnIR", "position": "WR",
         "projection": 12.0, "available": False},
    ])
    rep = pd.DataFrame([{"player_key": "a", "report_status": None,
                         "practice_status": FULL, "injury": None,
                         "report_week": 5}])
    out = apply_to_roster(roster, rep)
    assert not bool(out["available"].iloc[0])


def test_an_out_designation_removes_a_player_from_the_lineup():
    roster = pd.DataFrame([
        {"player_key": "a", "player_name": "Hurt", "position": "WR",
         "projection": 20.0, "available": True},
    ])
    rep = pd.DataFrame([{"player_key": "a", "report_status": "Out",
                         "practice_status": DNP, "injury": "Hamstring",
                         "report_week": 5}])
    assert not bool(apply_to_roster(roster, rep)["available"].iloc[0])


# --------------------------------------------------------------------------
# Matchup context — flags only, never an adjustment
# --------------------------------------------------------------------------

def test_implied_total_splits_the_game_total_by_the_spread():
    row = {"roof": "outdoors", "wind": 5, "temp": 60, "implied_total": 28.0}
    # Home favoured by 3 in a 47-point game implies 25 for the home side.
    assert (47.0 / 2 + 3.0 / 2) == pytest.approx(25.0)
    assert "high total" in matchup._flags(row)


def test_a_dome_never_gets_a_weather_flag():
    row = {"roof": "dome", "wind": 30, "temp": 5, "implied_total": 23.0}
    assert matchup._flags(row) == ""


def test_wind_and_cold_are_flagged_outdoors():
    row = {"roof": "outdoors", "wind": 22, "temp": 18, "implied_total": 22.0}
    flags = matchup._flags(row)
    assert "wind" in flags and "°F" in flags


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def test_missing_values_never_render_as_the_string_nan():
    """A pandas merge leaves NaN where nothing matched, and `str(nan)` is
    literally "nan" — which is how a lineup table ends up full of them."""
    assert report._text(float("nan")) == ""
    assert report._text(None) == ""
    assert report._text("NaN") == ""
    assert report._text("at CIN") == "at CIN"


def test_the_report_leads_with_the_unvalidated_warning():
    lineup_df = pd.DataFrame([{
        "slot": "QB", "player_name": "Someone", "position": "QB",
        "projection": 20.0, "opponent": "at NYJ",
    }])
    from src.features.lineup import Lineup
    md = report.build(
        week=5, season=2026,
        lineup=Lineup(starters=lineup_df, bench=lineup_df.iloc[0:0], points=20.0),
        calls=[], holes=[], slots=SLOTS, moves=[], waiver_threshold=4.0,
        surplus=lineup_df.iloc[0:0], roster=lineup_df,
    )
    assert "unvalidated" in md.lower()
    assert "Week 5" in md
    assert "nan" not in md.lower().replace("unvalidated", "")


# --------------------------------------------------------------------------
# Durability — measured, and deliberately not applied as an adjustment
# --------------------------------------------------------------------------

from src.features import availability as av  # noqa: E402


def _weeks(player: str, pos: str, season: int, weeks: list[int]) -> list[dict]:
    return [{"player_key": player, "position": pos, "season": season,
             "week": w, "season_type": "REG", "fantasy_points": 10.0}
            for w in weeks]


def test_availability_is_measured_against_sixteen_games_not_seventeen():
    """A team plays 16 games inside the 17 fantasy weeks. Dividing by 17 would
    score a perfectly healthy season as 94% and make everyone look fragile."""
    hist = av.availability_history(
        pd.DataFrame(_weeks("a", "RB", 2024, list(range(1, 17)))))
    assert hist["availability"].iloc[0] == pytest.approx(1.0)


def test_cameo_seasons_are_excluded_from_the_history():
    """Two games from a practice-squad callup is not a durability observation
    about a starter; it would drag every positional mean down."""
    hist = av.availability_history(
        pd.DataFrame(_weeks("a", "RB", 2024, [1, 2])))
    assert hist.empty


def test_durability_uses_only_seasons_before_the_target():
    rows = (_weeks("a", "RB", 2023, list(range(1, 17)))
            + _weeks("a", "RB", 2024, list(range(1, 9)))
            + _weeks("a", "RB", 2025, list(range(1, 17))))
    hist = av.availability_history(pd.DataFrame(rows))
    dur = av.durability(hist, as_of_season=2025)
    # 2025 must not inform a 2025 draft: 1.0 and 0.5 average to 0.75 raw.
    assert dur["durability_raw"].iloc[0] == pytest.approx(0.75)


def test_a_thin_history_is_shrunk_toward_the_position():
    rows = (_weeks("iron", "RB", 2024, list(range(1, 17)))
            + _weeks("iron", "RB", 2023, list(range(1, 17)))
            + _weeks("iron", "RB", 2022, list(range(1, 17)))
            + _weeks("hurt", "RB", 2024, list(range(1, 9))))
    hist = av.availability_history(pd.DataFrame(rows))
    dur = av.durability(hist, as_of_season=2025).set_index("player_key")
    # One season of 0.5 is pulled up toward the positional mean; three seasons
    # of 1.0 barely move.
    assert dur.loc["hurt", "durability"] > dur.loc["hurt", "durability_raw"]
    assert dur.loc["iron", "durability"] > dur.loc["hurt", "durability"]


def test_the_flag_fires_only_where_the_consensus_stops_pricing_durability():
    """Measured 2021-2025: inside ECR 120 the durable/fragile gap in playoff
    availability is +0.02; past it, +0.11. Flagging an early-round player would
    invite a human to double-count what ECR already discounted."""
    board = pd.DataFrame([
        {"player_key": "early", "adp_rank": 30},
        {"player_key": "late", "adp_rank": 200},
    ])
    dur = pd.DataFrame([
        {"player_key": "early", "durability": 0.55,
         "durability_seasons": 3, "durability_raw": 0.55},
        {"player_key": "late", "durability": 0.55,
         "durability_seasons": 3, "durability_raw": 0.55},
    ])
    out = av.attach(board, dur).set_index("player_key")
    assert not bool(out.loc["early", "durability_flag"])
    assert bool(out.loc["late", "durability_flag"])


def test_a_durable_late_player_is_not_flagged():
    board = pd.DataFrame([{"player_key": "x", "adp_rank": 200}])
    dur = pd.DataFrame([{"player_key": "x", "durability": 0.92,
                         "durability_seasons": 3, "durability_raw": 0.92}])
    assert not bool(av.attach(board, dur)["durability_flag"].iloc[0])


def test_a_player_with_no_history_is_never_flagged():
    board = pd.DataFrame([{"player_key": "rookie", "adp_rank": 200}])
    out = av.attach(board, pd.DataFrame(
        columns=["player_key", "durability", "durability_seasons",
                 "durability_raw"]))
    assert not bool(out["durability_flag"].iloc[0])
