"""Tests for market anchoring and the season-parameterized history helpers.

The anchoring tests pin behaviour that took two wrong implementations to get
right, and both wrong versions looked plausible in the output. Read the failure
notes in the test names before changing any of this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.metrics import BacktestResult, summarize  # noqa: E402
from src.features.market_anchor import (  # noqa: E402
    MAX_ANCHOR,
    apply_market_anchor,
    merge_at_target,
)


def _board(n_skill: int = 260, n_def: int = 32, n_k: int = 32) -> pd.DataFrame:
    """A board where K and DEF are absurdly overvalued by raw VORP.

    Their VORP puts them in the top 40 while the market drafts them near the
    end — which is exactly the situation anchoring exists to correct.

    `n_skill` has to comfortably exceed the deepest anchored ADP (212 here).
    A board that runs out of players before an anchored player's target rank
    can only append him to the end, and the placement assertions below would be
    measuring the fixture rather than the code.
    """
    rows = []
    for i in range(n_skill):
        rows.append({"player_name": f"Skill {i}", "position": "WR",
                     "vorp": 200.0 - i * 1.5, "adp_rank": i + 1})
    for i in range(n_def):
        rows.append({"player_name": f"Defense {i}", "position": "DEF",
                     "vorp": 60.0 - i * 1.0, "adp_rank": 140 + i * 2})
    for i in range(n_k):
        rows.append({"player_name": f"Kicker {i}", "position": "K",
                     "vorp": 55.0 - i * 0.8, "adp_rank": 150 + i * 2})
    return pd.DataFrame(rows)


def _ranked(board: pd.DataFrame) -> pd.DataFrame:
    out = board.sort_values("vorp", ascending=False).reset_index(drop=True)
    out["board_rank"] = range(1, len(out) + 1)
    return out


# --------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------

def test_merge_places_each_anchored_player_at_its_target_rank():
    free = ["a", "b", "c", "d", "e"]
    out = merge_at_target(free, [(2.0, "X"), (4.0, "Y")])
    assert out == ["a", "X", "b", "Y", "c", "d", "e"]


def test_merge_appends_anchored_players_past_the_end_of_the_free_list():
    out = merge_at_target(["a"], [(99.0, "X")])
    assert out == ["a", "X"]


def test_merge_keeps_anchored_players_in_target_order():
    out = merge_at_target(["a", "b", "c"], [(3.0, "late"), (1.0, "early")])
    assert out.index("early") < out.index("late")


# --------------------------------------------------------------------------
# Anchoring
# --------------------------------------------------------------------------

def test_anchoring_lands_a_fully_anchored_position_at_its_adp():
    """The bug this pins: assigning an anchored player the VORP the market
    implies for his slot lands the whole group ~55 spots *above* its ADP,
    because that value is a conditional mean and everyone above the mean sorts
    past him. Ranks compose; values do not."""
    board = _board()
    out = _ranked(apply_market_anchor(board, {"DEF": 1.0}))

    defenses = out[out["position"] == "DEF"]
    drift = (defenses["board_rank"] - defenses["adp_rank"]).abs()
    assert drift.max() <= 6, defenses[["player_name", "board_rank", "adp_rank"]]


def test_anchoring_preserves_ordering_within_the_anchored_position():
    """The other bug: at weight exactly 1.0 the value became a function of ADP
    alone, so wherever the curve was flat the ordering inside the position was
    decided by noise — the second-best defense came off the board first."""
    board = _board()
    out = _ranked(apply_market_anchor(board, {"DEF": 1.0}))

    defenses = out[out["position"] == "DEF"]
    # Best raw VORP must still be the first defense taken.
    assert defenses.iloc[0]["player_name"] == "Defense 0"
    assert defenses["vorp_unanchored"].is_monotonic_decreasing


def test_a_partial_anchor_lets_value_pull_a_player_above_his_adp():
    board = _board()
    out = _ranked(apply_market_anchor(board, {"K": 0.7}))

    top_k = out[out["position"] == "K"].iloc[0]
    # Pulled a long way down from where pure VORP had him...
    assert top_k["board_rank"] > 40
    # ...but his own value still buys him a meaningful climb above his ADP.
    assert top_k["board_rank"] < top_k["adp_rank"] - 10


def test_a_stronger_anchor_sits_closer_to_adp_than_a_weaker_one():
    board = _board()
    loose = _ranked(apply_market_anchor(board, {"K": 0.3}))
    tight = _ranked(apply_market_anchor(board, {"K": 0.9}))

    def gap(df):
        k = df[df["position"] == "K"]
        return float((k["adp_rank"] - k["board_rank"]).mean())

    assert gap(tight) < gap(loose)


def test_unanchored_positions_keep_their_exact_vorp():
    board = _board()
    out = apply_market_anchor(board, {"DEF": 1.0, "K": 0.7})

    skill = out[out["position"] == "WR"]
    assert (skill["vorp"] == skill["vorp_unanchored"]).all()
    assert (skill["market_anchor"] == 0.0).all()


def test_the_pre_anchor_value_is_always_preserved():
    board = _board()
    out = apply_market_anchor(board, {"DEF": 1.0})
    pd.testing.assert_series_equal(
        out["vorp_unanchored"], board["vorp"], check_names=False)


def test_anchor_weight_is_capped_below_one():
    board = _board()
    out = apply_market_anchor(board, {"DEF": 5.0})
    assert out["market_anchor"].max() == pytest.approx(MAX_ANCHOR)


def test_no_weights_is_a_no_op_on_value():
    board = _board()
    out = apply_market_anchor(board, {})
    assert (out["vorp"] == board["vorp"]).all()


def test_a_player_with_no_adp_is_never_anchored():
    board = _board()
    board.loc[board["position"] == "DEF", "adp_rank"] = np.nan
    out = apply_market_anchor(board, {"DEF": 1.0})

    defenses = out[out["position"] == "DEF"]
    assert (defenses["market_anchor"] == 0.0).all()
    assert (defenses["vorp"] == defenses["vorp_unanchored"]).all()


def test_anchoring_does_not_lose_or_duplicate_players():
    board = _board()
    out = apply_market_anchor(board, {"DEF": 1.0, "K": 0.7})
    assert len(out) == len(board)
    assert out["player_name"].nunique() == len(board)


# --------------------------------------------------------------------------
# Backtest summary
# --------------------------------------------------------------------------

def _result(season: int, slot: int) -> BacktestResult:
    return BacktestResult(
        season=season, seed=1, my_slot=slot, playoff_points=300.0,
        season_points=1800.0, regular_season_points=1500.0, win_rate=0.5,
        wins=7, losses=7, league_rank_by_points=5,
    )


def test_summary_columns_survive_a_parquet_round_trip(tmp_path):
    """The aggregate row labels `season` "ALL" and `my_slot` "mixed", so both
    columns have to be strings throughout. Parquet has one type per column: a
    column of ints with a single string in it cannot be written at all, and the
    whole backtest died on the last line after doing all the work."""
    summary = summarize([_result(2024, 3), _result(2025, 7)])
    path = tmp_path / "runs.parquet"
    summary.to_parquet(path, index=False)

    back = pd.read_parquet(path)
    assert set(back["season"]) == {"2024", "2025", "ALL"}
    assert "mixed" in set(back["my_slot"])


def test_summary_keeps_one_row_per_draft_plus_an_aggregate():
    summary = summarize([_result(2024, 3), _result(2025, 7)])
    assert len(summary) == 3
    assert (summary["season"] == "ALL").sum() == 1
