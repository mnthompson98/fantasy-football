"""Guard the board-vs-ADP parity baseline.

Two layers, for an honest reason.

The **cheap layer** always runs: the checked-in baseline parses, still describes
the four scorable folds, and still says what HANDOFF.md says it says. It also
exercises `compare()` on synthetic input, so the drift detector itself is tested
rather than assumed. This is what stops the fixture rotting into a decoration.

The **expensive layer** re-runs the walk-forward twice and diffs it, and it is
opt-in behind `RUN_BACKTEST_BASELINE=1` because it needs six seasons of nflverse
data and takes tens of minutes. It is not something a pre-commit hook can carry:

    RUN_BACKTEST_BASELINE=1 .venv/Scripts/python.exe -m pytest \\
        tests/test_backtest_baseline.py -q

or equivalently `python -m scripts.check_backtest_baseline`. Run it whenever
`features/pipeline.py` or the valuation config changes — which is the only time
the answer can move.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_backtest_baseline import (  # noqa: E402
    MAX_LOSS_TO_ADP,
    TOLERANCE,
    VALUE_COLS,
    compare,
)

BASELINE = Path(__file__).parent / "fixtures" / "backtest_baseline.json"

# ffverse preseason ECR starts in 2021 and the purge gap costs one more, so this
# is the entire scorable window and always will be. See HANDOFF.md.
SEASONS = {"2022", "2023", "2024", "2025"}


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE.exists(), (
        f"{BASELINE} is missing. Record it with "
        f"`python -m scripts.check_backtest_baseline --write`.")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The fixture itself
# --------------------------------------------------------------------------

def test_baseline_covers_both_orderings_and_every_fold(baseline):
    assert set(baseline["runs"]) == set(VALUE_COLS)
    for col in VALUE_COLS:
        assert set(baseline["runs"][col]["seasons"]) == SEASONS
        assert baseline["runs"][col]["all"]["league_rank_by_points"] > 0
        assert baseline["runs"][col]["all"]["playoff_points"] > 0


def test_baseline_records_what_produced_it(baseline):
    """A baseline you cannot reproduce is a number, not a baseline."""
    assert baseline["drafts_per_season"] >= 40
    assert baseline["seed"]
    assert baseline["git_commit"] != "unknown"
    assert baseline["recorded"]
    # The two runs must differ, and only in the ordering: `config_hash` covers
    # `value_col` along with everything else, so identical hashes would mean the
    # baseline recorded the same run twice.
    hashes = {baseline["runs"][c]["config_hash"] for c in VALUE_COLS}
    assert len(hashes) == len(VALUE_COLS), (
        f"both orderings logged the same config: {hashes}")


def test_the_baseline_matches_the_config_that_is_shipping_now(baseline):
    """The cheap half of the parity guard, and the half that runs in CI.

    `BacktestConfig.hash()` covers the valuation config — blend weights,
    calibration, market anchor, season downweighting — precisely so two runs
    that value players differently cannot log the same hash. So recomputing it
    from `config/league.yaml` and comparing against the fixture catches the
    change this whole exercise is about: someone edits the valuation, parity
    moves, and nobody re-measures until February. It costs milliseconds and
    needs no data, unlike actually re-running the walk-forward.

    If this fails, the config changed. Re-run
    `python -m scripts.check_backtest_baseline` and look at what the numbers
    did before re-recording.
    """
    import yaml

    from scripts.run_backtest import CONFIG, backtest_config

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for col in VALUE_COLS:
        bt = backtest_config(cfg, drafts=baseline["drafts_per_season"])
        bt.value_col = col
        assert bt.hash() == baseline["runs"][col]["config_hash"], (
            f"config/league.yaml no longer matches the recorded {col} "
            f"baseline. The parity numbers in the fixture were measured under "
            f"a different valuation.")
        assert bt.seed == baseline["seed"]


def test_the_valuation_code_has_not_changed_under_the_baseline(baseline):
    """The other half of the CI guard, and the one the config hash cannot cover.

    `config_hash` catches an edited *weight*. It does not notice an edited
    `features/pipeline.py`, which is the change most likely to move parity and
    least likely to be re-measured. So the fixture also carries a semantic hash
    of the valuation modules and the pick policy.

    Docstrings are stripped and the AST is hashed rather than the text, so
    rewrapping a comment does not fire this and renaming a variable does. It
    proves nothing about whether parity actually moved — only that the code
    behind the recorded numbers is the code that is shipping. When it fails, run
    `python -m scripts.check_backtest_baseline` and read the table before
    re-recording with `--write`.
    """
    from scripts.check_backtest_baseline import valuation_fingerprint

    recorded = baseline["valuation_fingerprint"]
    current = valuation_fingerprint()
    assert set(recorded) == set(current), (
        "the set of fingerprinted modules changed; update FINGERPRINTED and "
        "re-record the baseline")

    moved = sorted(k for k in current if current[k] != recorded[k])
    assert not moved, (
        f"the valuation changed under the recorded parity baseline: "
        f"{', '.join(moved)}. Re-run `python -m "
        f"scripts.check_backtest_baseline` and look at what the numbers did "
        f"before re-recording.")


def test_the_fingerprint_ignores_comments_but_not_code(tmp_path,
                                                       monkeypatch):
    """A tripwire that fires on reformatting teaches people to re-record blind."""
    import scripts.check_backtest_baseline as C

    src = "def f(x):\n    '''doc.'''\n    # a comment\n    return x * 2\n"
    cosmetic = "def f(x):\n    '''a totally rewritten docstring.'''\n\n    return x * 2\n"
    real = "def f(x):\n    '''doc.'''\n    # a comment\n    return x * 3\n"

    (tmp_path / "m.py").write_text(src, encoding="utf-8")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "FINGERPRINTED", ("m.py",))
    first = C.valuation_fingerprint()

    (tmp_path / "m.py").write_text(cosmetic, encoding="utf-8")
    assert C.valuation_fingerprint() == first, "reformatting must not fire"

    (tmp_path / "m.py").write_text(real, encoding="utf-8")
    assert C.valuation_fingerprint() != first, "a behaviour change must fire"


def test_baseline_still_says_what_handoff_says(baseline):
    """HANDOFF.md's finding, asserted rather than remembered.

    As recorded on 2026-09-02 under the shipping config (ECR-only blend,
    `run_aware` on for both drafters): the board and the raw consensus
    ordering finish within a tenth of a place of each other, the board
    wins 2023 and 2024, and 2025 is still the fold it loses — by about a
    place now rather than three. ADP still has the better playoff points.
    If a future edit makes this fixture disagree with that description, the
    description in HANDOFF.md needs rewriting too — this test is the
    reminder, in either direction.
    """
    board = baseline["runs"]["vorp"]
    market = baseline["runs"]["adp_value"]

    gap = (board["all"]["league_rank_by_points"]
           - market["all"]["league_rank_by_points"])
    assert abs(gap) < 0.5, (
        f"board-vs-ADP gap is {gap:+.3f}; HANDOFF.md describes a tie")

    per_season = {
        s: board["seasons"][s]["league_rank_by_points"]
           - market["seasons"][s]["league_rank_by_points"]
        for s in SEASONS
    }
    assert per_season["2025"] == max(per_season.values()), (
        f"2025 should be the fold the board loses worst: {per_season}")
    assert per_season["2025"] > 0.5, f"2025 is no longer a loss: {per_season}"
    assert board["all"]["playoff_points"] < market["all"]["playoff_points"], (
        "ADP has had the better playoff points in every measurement so far")


# --------------------------------------------------------------------------
# The drift detector
# --------------------------------------------------------------------------

def _shift(baseline: dict, col: str, delta: float, *,
           season: str | None = None, metric: str = "league_rank_by_points"):
    out = deepcopy(baseline)
    if season:
        out["runs"][col]["seasons"][season][metric] += delta
    else:
        out["runs"][col]["all"][metric] += delta
    return out


def _drift_only(problems: list[str]) -> list[str]:
    """`compare()` reports two different things: drift against the fixture,
    and the absolute floor against the market. An unchanged run has no
    drift by definition; whether it clears the floor is a separate fact,
    pinned by its own test below."""
    return [p for p in problems if "not earning its place" not in p]


def test_an_unchanged_run_reports_no_drift(baseline):
    assert _drift_only(compare(deepcopy(baseline), baseline)) == []


def test_the_recorded_board_clears_the_floor(baseline):
    """The full blend recorded on 2026-09-01 trailed pure consensus by 0.56,
    past `MAX_LOSS_TO_ADP`; the ECR-only blend recorded on 2026-09-02 does
    not. `check_backtest_baseline` must therefore exit 0 on an unchanged
    run, floor included."""
    assert compare(deepcopy(baseline), baseline) == []


def test_drift_in_the_parity_gap_is_caught(baseline):
    """The quantity that matters: the board losing ground to the market."""
    worse = _shift(baseline, "vorp", TOLERANCE["parity_gap"] + 0.2)
    problems = compare(worse, baseline)
    assert any("rank gap moved" in p for p in problems), problems


def test_a_move_inside_tolerance_is_allowed(baseline):
    ok = _shift(baseline, "vorp", TOLERANCE["parity_gap"] / 2)
    assert _drift_only(compare(ok, baseline)) == []


def test_the_absolute_floor_fires_even_against_a_stale_baseline(baseline):
    """If the fixture were re-recorded at a bad number, the gap check would go
    quiet. The floor is what still complains.
    """
    market = baseline["runs"]["adp_value"]["all"]["league_rank_by_points"]
    collapsed = deepcopy(baseline)
    collapsed["runs"]["vorp"]["all"]["league_rank_by_points"] = (
        market + MAX_LOSS_TO_ADP + 0.3)
    problems = compare(collapsed, collapsed)      # baseline == actual: stale
    assert any("not earning its place" in p for p in problems), problems


def test_playoff_points_drift_is_caught(baseline):
    """Playoff-week points is the primary metric and gets its own check."""
    worse = _shift(baseline, "vorp", -(TOLERANCE["playoff_gap"] + 5),
                   metric="playoff_points")
    problems = compare(worse, baseline)
    assert any("playoff points" in p for p in problems), problems


def test_a_single_fold_blowing_up_is_caught(baseline):
    worse = _shift(baseline, "vorp", TOLERANCE["season_rank"] + 0.5,
                   season="2023")
    problems = compare(worse, baseline)
    assert any("2023" in p for p in problems), problems


def test_a_missing_fold_is_caught(baseline):
    truncated = deepcopy(baseline)
    del truncated["runs"]["vorp"]["seasons"]["2024"]
    problems = compare(truncated, baseline)
    assert any("2024 missing" in p for p in problems), problems


# --------------------------------------------------------------------------
# The real thing
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_BACKTEST_BASELINE") != "1",
    reason="re-runs the full walk-forward twice; set RUN_BACKTEST_BASELINE=1 "
           "(or run `python -m scripts.check_backtest_baseline`)",
)
def test_the_backtest_still_matches_the_baseline(baseline):
    from scripts.check_backtest_baseline import measure

    actual = measure(baseline["drafts_per_season"], verbose=False)
    problems = compare(actual, baseline)
    assert not problems, "\n".join(problems)
