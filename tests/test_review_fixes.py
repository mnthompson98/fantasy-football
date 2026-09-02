"""Regression tests for the 2026-09-01 review.

Each test pins one finding from that review. They are grouped the way the
findings were ranked: things that broke, things that were silently wrong,
then the walk-forward plumbing that had no coverage at all.

Run: .venv/Scripts/python.exe -m pytest tests/test_review_fixes.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.draft_sim import ValueDrafter  # noqa: E402
from src.backtest.walkforward import BacktestConfig, build_folds  # noqa: E402
from src.features import availability as av  # noqa: E402
from src.features.lineup import LineupSlots  # noqa: E402
from src.features.pipeline import ecr_to_pool  # noqa: E402
from src.features.scoring import DEFAULT_SCORING, Scoring, score_defense  # noqa: E402
from src.ingest import history as H  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402
from src.inseason import waivers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Broke outright
# --------------------------------------------------------------------------

def test_the_scheduled_task_changes_into_the_repo_before_running():
    """Task Scheduler starts tasks in System32. Without `cd /d` the job
    could not import `scripts.weekly_update` and failed every Tuesday."""
    from scripts.schedule_weekly import ROOT as R, _command
    cmd = _command()
    assert "cd /d" in cmd
    assert str(R) in cmd
    assert cmd.index("cd /d") < cmd.index("weekly_update")


def test_the_raw_cache_is_anchored_to_the_repo_not_the_cwd():
    assert nv.RAW_DIR.is_absolute()
    assert nv.RAW_DIR == ROOT / "data" / "raw"


def test_the_monitor_refuses_a_draft_sleeper_does_not_know(monkeypatch):
    """Sleeper returns null for an unknown id. Defaulting to 10x16 left a
    typo'd id polling an empty feed all night."""
    import src.draft.monitor as M
    monkeypatch.setattr(M, "get_draft", lambda _id: None)
    board = pd.DataFrame([{"player_id": "1", "player_name": "A",
                           "position": "RB", "vorp": 1.0, "adp_rank": 1}])
    with pytest.raises(ValueError, match="no draft"):
        M.run("bogus", board, 1, html_path=None)


def test_the_weekly_week_is_the_upcoming_one_not_the_last_injury_report(
        monkeypatch):
    from scripts import weekly_update as W
    import nflreadpy as nfl
    monkeypatch.setattr(nfl, "get_current_season", lambda: 2026)
    monkeypatch.setattr(nfl, "get_current_week", lambda: 7)
    assert W._current_week(2026) == 7
    # A different season means the configured one has not kicked off yet.
    assert W._current_week(2027) is None


def test_trending_failures_do_not_discard_the_report(monkeypatch):
    from scripts import weekly_update as W

    def boom(*a, **k):
        raise ConnectionError("sleeper down")
    monkeypatch.setattr(W, "get_trending", boom)

    class CW:
        def resolve(self, **kw):
            return None
    assert W._trending(CW(), "add") == {}


# --------------------------------------------------------------------------
# Silently wrong
# --------------------------------------------------------------------------

def test_defensive_fumble_recoveries_pay_the_defensive_rate():
    """The league pays `fum_rec` 2.0 for a defensive recovery; the scorer read
    the special-teams key `def_st_fum_rec` (1.0) for five seasons."""
    assert DEFAULT_SCORING["fum_rec"] == 2.0
    assert DEFAULT_SCORING["ff"] == 1.0
    team_weekly = pd.DataFrame([{
        "season": 2025, "week": 1, "team": "HOU", "opponent_team": "IND",
        "fumble_recovery_opp": 3, "def_fumbles_forced": 2,
    }])
    schedules = pd.DataFrame([{
        "season": 2025, "week": 1, "home_team": "HOU", "home_score": 27,
        "away_team": "IND", "away_score": 21,
    }])
    out = score_defense(team_weekly, schedules, Scoring(dict(DEFAULT_SCORING)))
    # 3 recoveries x 2 + 2 forced x 1 + 21 allowed -> 0-point tier = 8
    assert out["fantasy_points"].iloc[0] == pytest.approx(8.0)


def test_a_league_payload_overrides_the_defensive_fumble_keys():
    s = Scoring.from_league({"scoring_settings": {"fum_rec": 3.0}})
    assert s.get("fum_rec") == 3.0


def test_expected_points_are_restricted_to_the_fantasy_weeks(monkeypatch):
    """ffopportunity runs through week 22 with no season_type column, so a
    Super Bowl team's players carried up to 21 games into a rate that
    prior_points measures over weeks 1-17."""
    rows = [{"season": 2025, "week": w, "player_id": "00-1", "position": "WR",
             "total_fantasy_points_exp": 10.0} for w in range(1, 23)]
    monkeypatch.setattr(nv, "load_ff_opportunity",
                        lambda seasons, **k: pd.DataFrame(rows))
    opp = H.opportunity_for(2025)
    assert set(opp["week"]) == set(range(1, 18))


def test_the_pipeline_fails_loudly_on_a_position_with_no_curve():
    """NaN at a whole position used to walk through calibration, VORP and
    the anchor, then crash `ValueDrafter.rank()` mid-draft."""
    totals = pd.DataFrame([
        {"player_key": f"WR{i}", "position": "WR", "season": 2024,
         "points": 300 - i * 5, "games": 17, "points_per_game": (300 - i * 5) / 17}
        for i in range(30)
    ])
    ecr = pd.DataFrame([
        {"player_key": "WR0", "position": "WR", "pos_rank": 1, "adp_rank": 1},
        {"player_key": "DEF_HOU", "position": "DEF", "pos_rank": 1, "adp_rank": 2},
    ])
    with pytest.raises(ValueError, match="DEF"):
        ecr_to_pool(ecr, totals, pd.DataFrame(), prior_season=2024,
                    train_seasons=[2024])


def test_waiver_drop_suggestion_is_the_worst_zero_cost_player_not_the_first():
    slots = LineupSlots(starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1,
                                  "DEF": 1}, flex_slots=1)
    roster = pd.DataFrame([
        {"player_key": n, "player_name": n, "position": p, "projection": v}
        for n, p, v in [
            ("QB1", "QB", 20.0), ("RBa", "RB", 15.0), ("RBb", "RB", 12.0),
            ("WRa", "WR", 18.0), ("WRb", "WR", 14.0), ("TEa", "TE", 9.0),
            ("Ka", "K", 8.0), ("DEFa", "DEF", 7.0), ("WRc", "WR", 11.0),
            # Two zero-cost bench players; the good handcuff sorts first.
            ("GoodBench", "RB", 9.0), ("BadBench", "WR", 2.0),
        ]
    ])
    cands = pd.DataFrame([{"player_key": "Stud", "player_name": "Stud",
                           "position": "RB", "projection": 24.0}])
    move = waivers.evaluate(roster, cands, slots, roster_size=len(roster))[0]
    assert move.drop == "BadBench", move.drop


def test_established_players_keep_their_short_seasons():
    """Dropping every season under four games discarded exactly the fragile
    observations durability exists to measure."""
    def weeks(player, season, ws):
        return [{"player_key": player, "position": "RB", "season": season,
                 "week": w, "season_type": "REG", "fantasy_points": 1.0}
                for w in ws]
    hist = av.availability_history(pd.DataFrame(
        weeks("starter", 2023, range(1, 17)) + weeks("starter", 2024, [1, 2])
        + weeks("callup", 2024, [1, 2])))
    by = hist.set_index(["player_key", "season"])["games"]
    assert by.loc[("starter", 2024)] == 2          # the ACL season stays
    assert ("callup", 2024) not in by.index         # the cameo does not


def test_default_caps_match_the_configured_policy():
    caps = _cfg()["draft_policy"]["position_caps"]
    assert ValueDrafter.DEFAULT_CAPS == {k: int(v) for k, v in caps.items()}


def test_cache_names_do_not_collide_for_gapped_season_sets():
    assert nv._seasons_key([2020, 2025]) != nv._seasons_key(range(2020, 2026))
    assert nv._seasons_key(range(2020, 2026)) == "2020_2025"


# --------------------------------------------------------------------------
# Walk-forward plumbing
# --------------------------------------------------------------------------

def test_shipping_config_produces_the_documented_folds():
    """Training ends at Y-1 with no gap season. The 2025 fold must see 2024:
    last season's production is a blend component."""
    from scripts.run_backtest import backtest_config
    folds = build_folds(backtest_config(_cfg()))
    table = {f.target_season: f.train_seasons for f in folds}
    assert table == {
        2022: [2020, 2021],
        2023: [2020, 2021, 2022],
        2024: [2021, 2022, 2023],
        2025: [2022, 2023, 2024],
    }
    for target, train in table.items():
        assert max(train) == target - 1


def test_rolling_window_and_min_train_are_applied():
    cfg = BacktestConfig(seasons=[2018, 2019, 2020, 2021, 2022],
                         max_train_seasons=2, min_train_seasons=2)
    table = {f.target_season: f.train_seasons for f in build_folds(cfg)}
    assert table == {2020: [2018, 2019], 2021: [2019, 2020],
                     2022: [2020, 2021]}
    assert build_folds(BacktestConfig(seasons=[2024])) == []


def test_playoff_weeks_flow_from_config_into_the_scorer():
    from scripts.run_backtest import backtest_config
    bt = backtest_config(_cfg())
    assert bt.playoff_weeks == (15, 16, 17)
    assert bt.regular_weeks == tuple(range(1, 15))
    # Defaults are omitted from the hash so the recorded baseline survives
    # the field being added.
    assert (BacktestConfig(seasons=[2022]).hash()
            == BacktestConfig(seasons=[2022], playoff_weeks=(15, 16, 17)).hash())
    assert (BacktestConfig(seasons=[2022]).hash()
            != BacktestConfig(seasons=[2022], playoff_weeks=(14, 15, 16)).hash())


# --- preseason window and the slope fit --------------------------------

def _playerids() -> pd.DataFrame:
    return pd.DataFrame([
        {"mfl_id": str(i), "name": f"Player {i}", "position": "WR",
         "team": "CIN", "gsis_id": f"00-{i:07d}", "fantasypros_id": str(100 + i),
         "sleeper_id": str(1000 + i), "db_season": 2026}
        for i in range(60)
    ] + [
        {"mfl_id": "d", "name": "Houston Texans", "position": "DST",
         "team": "HOU", "gsis_id": None, "fantasypros_id": "9001",
         "sleeper_id": "HOU", "db_season": 2026},
    ])


def _rankings(season: int, dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        for i in range(60):
            rows.append({"page_type": "redraft-overall",
                         "scrape_date": f"{season}-{d}", "pos": "WR",
                         "team": "CIN", "id": str(100 + i),
                         "player": f"Player {i}", "ecr": float(i + 1),
                         "sd": 1.0, "best": i, "worst": i + 2, "bye": 7})
        for pos, fid, name in (("QB", "9101", "Some QB"), ("RB", "9102", "Some RB"),
                               ("TE", "9103", "Some TE"), ("K", "9104", "Some K")):
            rows.append({"page_type": "redraft-overall",
                         "scrape_date": f"{season}-{d}", "pos": pos,
                         "team": "CIN", "id": fid, "player": name,
                         "ecr": 200.0, "sd": 1.0, "best": 1, "worst": 2, "bye": 7})
        rows.append({"page_type": "redraft-overall", "scrape_date": f"{season}-{d}",
                     "pos": "DST", "team": "HOU", "id": "9001",
                     "player": "Houston Texans", "ecr": 150.0, "sd": 1.0,
                     "best": 1, "worst": 2, "bye": 7})
    return pd.DataFrame(rows)


def _extra_ids() -> pd.DataFrame:
    return pd.DataFrame([
        {"mfl_id": f"x{p}", "name": n, "position": p, "team": "CIN",
         "gsis_id": f"00-9{p}", "fantasypros_id": fid, "sleeper_id": f"s{p}",
         "db_season": 2026}
        for p, fid, n in (("QB", "9101", "Some QB"), ("RB", "9102", "Some RB"),
                          ("TE", "9103", "Some TE"), ("K", "9104", "Some K"))
    ])


def test_preseason_ecr_takes_the_last_snapshot_before_kickoff(monkeypatch):
    """The window rotted once already: a snapshot taken the day after the
    Thursday opener was being used as 'preseason'. Pin it against the
    schedule-derived opener."""
    monkeypatch.setattr(H, "season_opener", lambda season: "09-05")
    cw = build_crosswalk(pd.concat([_playerids(), _extra_ids()]))
    rk = _rankings(2025, ["08-22", "09-04", "09-05", "09-06"])
    _, _, scrape = H.preseason_ecr(rk, 2025, cw)
    assert scrape == "2025-09-04"
    with pytest.raises(ValueError, match="no preseason"):
        H.preseason_ecr(_rankings(2025, ["09-05", "09-12"]), 2025, cw)


def test_the_slope_fit_never_sees_the_target_season(monkeypatch):
    """Plant a pathological target season in the history frame and confirm
    the fitted slope for the training seasons is unmoved by it."""
    monkeypatch.setattr(H, "season_opener", lambda season: "09-05")
    cw = build_crosswalk(pd.concat([_playerids(), _extra_ids()]))
    rk = pd.concat([_rankings(s, ["08-25"]) for s in (2022, 2023, 2024)],
                   ignore_index=True)

    def totals_for(seasons, target_slope):
        rows = []
        for s in seasons:
            for i in range(60):
                pts = 300 - i * 4.0
                if s == 2024:                      # the target: inverted
                    pts = 60 + i * 4.0 * target_slope
                rows.append({"player_key": f"00-{i:07d}", "position": "WR",
                             "season": s, "points": pts, "games": 17,
                             "points_per_game": pts / 17})
        return pd.DataFrame(rows)

    a = H.projection_actual_pairs(rk, totals_for([2021, 2022, 2023], 1.0), cw,
                                  [2022, 2023])
    b = H.projection_actual_pairs(rk, totals_for([2021, 2022, 2023, 2024], -3.0),
                                  cw, [2022, 2023])
    assert set(a["season"]) == {2022, 2023}
    assert set(b["season"]) == {2022, 2023}
    pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                  b.reset_index(drop=True))


def test_pairs_are_built_through_the_blend_when_weights_are_given(monkeypatch):
    """The slope must be measured on the projection it is applied to."""
    monkeypatch.setattr(H, "season_opener", lambda season: "09-05")
    cw = build_crosswalk(pd.concat([_playerids(), _extra_ids()]))
    rk = _rankings(2023, ["08-25"])
    rows = []
    for s in (2021, 2022, 2023):
        for i in range(60):
            pts = 300 - i * 4.0 + (40.0 if (s == 2022 and i % 2) else 0.0)
            rows.append({"player_key": f"00-{i:07d}", "position": "WR",
                         "season": s, "points": pts, "games": 17,
                         "points_per_game": pts / 17})
        # The blend path values the whole snapshot, and the pipeline now
        # refuses a position with no curve, so every position needs history.
        for p, key in (("QB", "00-9QB"), ("RB", "00-9RB"), ("TE", "00-9TE"),
                       ("K", "00-9K"), ("DEF", "DEF_HOU")):
            rows.append({"player_key": key, "position": p, "season": s,
                         "points": 100.0, "games": 17,
                         "points_per_game": 100.0 / 17})
    totals = pd.DataFrame(rows)
    ecr_only = H.projection_actual_pairs(rk, totals, cw, [2023])
    blended = H.projection_actual_pairs(
        rk, totals, cw, [2023],
        blend_weights={"ecr_points": 2.0, "prior_points": 0.5},
        opportunity_for=lambda season: pd.DataFrame())
    blended = blended[blended["position"] == "WR"].sort_values("pos_rank")
    ecr_only = ecr_only[ecr_only["position"] == "WR"].sort_values("pos_rank")
    assert len(blended) == len(ecr_only) == 40
    # Prior-season production is in the blend and not in the curve, so the
    # two projections must differ where 2022 production disagreed with ECR.
    assert not np.allclose(blended["projection"].to_numpy(),
                           ecr_only["projection"].to_numpy())
    assert blended.attrs["zero_actual"] == 0
