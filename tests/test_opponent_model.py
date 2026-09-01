"""Tests for the league-fitted opponent model.

Two things are being protected here, and they pull in opposite directions.

The first is that the **gaussian field must not move**. Every number in
HANDOFF.md and in `tests/fixtures/backtest_baseline.json` was measured against
it, and the whole value of that fixture is that it is comparable across
changes. So the default path has to construct the same object, from the same
generator, drawing in the same order, as it did before the model became
swappable.

The second is the set of defects the fitted model exists to fix, each pinned to
the specific thing that went wrong while building it:

    - the gaussian need factor is multiplicative, so its size depends on where
      you are on the board — which is why that field drafts every defense in
      round 8;
    - a flat curve tail makes every remaining quarterback tie at one round and
      the field takes six of them in round 14;
    - defenses resolve through the team code, not the crosswalk, or every
      defense in every draft silently leaves the fit;
    - a fold may only fit on drafts held before its own season.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.draft_sim import (  # noqa: E402
    TAIL_SLOPE_ROUNDS_PER_K,
    LeagueOpponentModel,
    LeagueTendencies,
    OpponentModel,
    Roster,
    gaussian_opponent,
    league_opponent,
    positional_queue_numbers,
    simulate_draft,
)
from src.backtest.opponent_fit import (  # noqa: E402
    LeagueDraft,
    fit_blend_weights,
    fit_curves,
    fit_seasons_for,
    load_league_drafts,
    roster_ceilings,
    run_points,
    simulate_field,
)
from src.backtest.walkforward import BacktestConfig  # noqa: E402

STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
FLEX = ("RB", "WR", "TE")
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


# Depth and consensus placement close to a real fold's board: ~500 ranked
# players, kickers buried past 200 and most defenses past 140. The depth
# matters — on a board only a little larger than the draft, every position runs
# out and the argmin is forced rather than chosen, so a shallow fixture tests
# the fixture instead of the model.
BOARD_DEPTH = {"QB": 35, "RB": 130, "WR": 180, "TE": 45, "K": 32, "DEF": 32}
BOARD_START = {"QB": 18, "RB": 1, "WR": 3, "TE": 30, "K": 205, "DEF": 145}
BOARD_SPACING = {"QB": 9, "RB": 3, "WR": 2, "TE": 8, "K": 5, "DEF": 5}


def _board(depth: dict[str, int] | None = None) -> pd.DataFrame:
    """A board shaped like a real one: skill players early, K and DEF buried.

    The burial is the point. Consensus overall ECR ranks kickers past 200 and
    most defenses past 140, and it is the gap between where they are ranked and
    when they are drafted that every model here has to handle.
    """
    n = {**BOARD_DEPTH, **(depth or {})}
    rows = [
        {"player_id": f"{pos}{i}", "player_name": f"{pos} {i}",
         "position": pos, "sort": BOARD_START[pos] + i * BOARD_SPACING[pos],
         "replacement_points": 100.0}
        for pos, count in n.items() for i in range(count)
    ]
    df = pd.DataFrame(rows).sort_values("sort").reset_index(drop=True)
    df["adp_rank"] = np.arange(1, len(df) + 1, dtype=float)
    df["vorp"] = -df["adp_rank"]
    return df.drop(columns="sort")


def _roster(picks: list[str]) -> Roster:
    r = Roster(team_id=0, starters=dict(STARTERS), flex_slots=1,
               flex_eligible=FLEX, bench_slots=7)
    for i, pos in enumerate(picks):
        r.picks.append({"pick_no": i + 1, "round": i + 1,
                        "player_id": f"p{i}", "player_name": f"P{i}",
                        "position": pos, "adp_rank": float(i + 1),
                        "vorp": 0.0, "replacement_points": 100.0})
    return r


GRID = [round(0.1 * i, 1) for i in range(1, 56)]

# Anchor points taken from the real fit over the 2021 and 2025 drafts
# (`python -m scripts.fit_opponent_model`), so these tests exercise the shape
# the backtest actually runs rather than a shape invented to pass them. Each
# position is linear between its anchors and climbs at TAIL_SLOPE_ROUNDS_PER_K
# past the last one — which is what the real curve does past the deepest k the
# room ever drafted to.
CURVE_ANCHORS = {
    "QB":  [(0.2, 2.2), (0.5, 3.8), (0.8, 6.2), (1.0, 9.1), (1.5, 12.5),
            (2.0, 12.9)],
    "RB":  [(0.2, 0.3), (0.5, 0.7), (1.0, 1.6), (2.0, 4.9), (3.0, 8.1),
            (4.0, 12.1), (4.7, 15.9)],
    "WR":  [(0.2, 0.8), (0.5, 1.4), (1.0, 2.3), (2.0, 5.2), (3.0, 7.2),
            (4.0, 9.2), (5.5, 15.3)],
    "TE":  [(0.2, 2.5), (0.5, 4.7), (1.0, 8.5), (1.5, 12.9), (2.0, 14.7)],
    "K":   [(0.2, 12.3), (0.5, 13.4), (0.8, 14.4), (1.0, 15.0)],
    "DEF": [(0.2, 11.4), (0.5, 12.1), (0.8, 13.0), (1.0, 15.2)],
}


def _curve(anchors: list[tuple[float, float]]) -> list[float]:
    xs = [a for a, _ in anchors]
    ys = [b for _, b in anchors]
    out = []
    for k in GRID:
        if k <= xs[-1]:
            out.append(float(np.interp(k, xs, ys)))
        else:
            out.append(ys[-1] + TAIL_SLOPE_ROUNDS_PER_K * (k - xs[-1]))
    return out


def _tendencies(**over) -> LeagueTendencies:
    """A fitted field shaped like the real one: K and DEF on a hard schedule."""
    kwargs = dict(
        k_per_team=list(GRID),
        rounds={p: _curve(a) for p, a in CURVE_ANCHORS.items()},
        curve_weight={"QB": 1.0, "RB": 0.78, "WR": 0.94, "TE": 1.0,
                      "K": 0.99, "DEF": 0.97},
        sigma_rounds={"QB": 0.61, "RB": 0.60, "WR": 0.37, "TE": 0.95,
                      "K": 0.81, "DEF": 0.77},
        max_per_team={"QB": 1.9, "RB": 4.54, "WR": 5.38, "TE": 1.69,
                      "K": 1.0, "DEF": 1.0},
    )
    kwargs.update(over)
    return LeagueTendencies(**kwargs)


# --------------------------------------------------------------------------
# The gaussian field must not move
# --------------------------------------------------------------------------

def test_the_default_field_is_the_same_object_drawing_in_the_same_order():
    """`simulate_draft` with no factory must build exactly what it always built.

    Not "an equivalent model" — the same construction, consuming the generator
    identically. Anything else silently invalidates every recorded number,
    because the seed no longer maps to the same sequence of picks.
    """
    board = _board()
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)

    direct = OpponentModel({"QB": 12.0, "RB": 10.0}, 1.6, 0.4, rng_a)
    made = gaussian_opponent({"QB": 12.0, "RB": 10.0}, 1.6, 0.4)(
        rng_b, board, 10)

    assert type(made) is OpponentModel
    assert made.sigma == direct.sigma
    assert made.need_boost == direct.need_boost
    assert made.need_penalty == direct.need_penalty

    roster = _roster([])
    for _ in range(25):
        assert made.choose(board, roster) == direct.choose(board, roster)


def test_an_explicit_gaussian_factory_matches_passing_none():
    """The flag's 'gaussian' branch and the untouched default are one path."""
    board = _board()
    sigma = {p: 8.0 for p in POSITIONS}
    kw = dict(teams=10, rounds=16, my_slot=4, starters=STARTERS, flex_slots=1,
              flex_eligible=FLEX, sigma=sigma, seed=99)

    default = simulate_draft(board, **kw)
    explicit = simulate_draft(board, **kw,
                              opponent_factory=gaussian_opponent(sigma))
    for t in default:
        assert ([p["player_id"] for p in default[t].picks]
                == [p["player_id"] for p in explicit[t].picks])


def test_the_config_hash_ignores_the_opponent_field_at_its_default():
    """A field added after the fixture was recorded must not invalidate it.

    The recorded baseline was measured under the gaussian field. A config that
    still says gaussian describes that same run, so it has to hash to the same
    value — otherwise the parity guard fails on a change that did not move a
    single number, and the fix everyone reaches for is to re-record it.
    """
    a = BacktestConfig(seasons=[2022, 2023])
    b = BacktestConfig(seasons=[2022, 2023], opponent="gaussian")
    c = BacktestConfig(seasons=[2022, 2023], opponent="league")

    assert a.hash() == b.hash()
    assert c.hash() != a.hash(), (
        "a run against a different field must be distinguishable in the log")


# --------------------------------------------------------------------------
# What the fitted field fixes
# --------------------------------------------------------------------------

def test_the_gaussian_need_factor_scales_with_adp_rank():
    """The defect, stated as a number rather than as prose.

    The gaussian model multiplies the key by 1/need_boost, so the adjustment is
    worth 1/1.6 of wherever you already are: four picks at the top of the board
    and ninety-four at the bottom. That is why its field drafts every defense in
    round 8, and it is the reason the fitted model works additively in rounds.
    """
    early, late = 10.0, 250.0
    shift_early = early - early / 1.6
    shift_late = late - late / 1.6
    assert shift_late > 20 * shift_early
    assert shift_late > 90


def test_the_fitted_field_leaves_k_and_def_alone_until_the_schedule_says_so():
    board = _board()
    picks = simulate_field(
        board, teams=10, rounds=16, starters=STARTERS, flex_slots=1,
        flex_eligible=FLEX, opponent_factory=league_opponent(_tendencies()),
        seed=3)
    for pos, floor in (("K", 10), ("DEF", 10)):
        first = int(picks[picks["position"] == pos]["round"].min())
        assert first >= floor, (
            f"first {pos} came off in round {first}; this league does not "
            f"touch one before round {floor}")


def test_the_gaussian_field_takes_k_and_def_much_earlier_on_the_same_board():
    """The two fields disagree, and in the documented direction.

    This is the comparison the whole change rests on, so it is asserted rather
    than left to the report script.
    """
    board = _board()
    kw = dict(teams=10, rounds=16, starters=STARTERS, flex_slots=1,
              flex_eligible=FLEX, seed=5)
    gauss = simulate_field(board, **kw, opponent_factory=gaussian_opponent(
        {p: 8.0 for p in POSITIONS}, 1.6, 0.4))
    fitted = simulate_field(board, **kw,
                            opponent_factory=league_opponent(_tendencies()))

    for pos in ("K", "DEF"):
        g = float(gauss[gauss["position"] == pos]["round"].min())
        f = float(fitted[fitted["position"] == pos]["round"].min())
        assert g < f - 1, (
            f"expected the gaussian field to reach for {pos} at least a round "
            f"earlier than the fitted one; got {g} vs {f}")


def test_a_flat_curve_tail_would_pile_a_position_into_one_round():
    """The bug the rising tail exists to prevent, reproduced from first
    principles.

    Carrying the last fitted value forward past the deepest k ever observed is
    the obvious way to extend a curve, and it says every remaining quarterback
    is due at the same round. They all tie there and the field drafts a heap of
    them in one round. The room did not slow down past its last quarterback, it
    stopped.
    """
    grid = [round(0.1 * i, 1) for i in range(1, 31)]
    schedule = [2.0 + 5.0 * k for k in grid]
    kmax = 10                                     # flat past k/team = 1.0

    flat = _tendencies(
        k_per_team=grid,
        rounds={**{p: [0.5 + 2.5 * k for k in grid] for p in POSITIONS},
                "QB": schedule[:kmax] + [schedule[kmax - 1]] * (len(grid) - kmax)},
        curve_weight={p: 1.0 for p in POSITIONS},
    )
    rising = _tendencies(
        k_per_team=grid,
        rounds={**{p: [0.5 + 2.5 * k for k in grid] for p in POSITIONS},
                "QB": schedule[:kmax] + [
                    schedule[kmax - 1] + TAIL_SLOPE_ROUNDS_PER_K * (g - grid[kmax - 1])
                    for g in grid[kmax:]]},
        curve_weight={p: 1.0 for p in POSITIONS},
    )

    kw = dict(teams=10, rounds=16, starters=STARTERS, flex_slots=1,
              flex_eligible=FLEX, seed=11)
    board = _board({"QB": 40, "RB": 60, "WR": 80, "TE": 20, "K": 20, "DEF": 20})

    def worst_round(t):
        picks = simulate_field(board, **kw, opponent_factory=league_opponent(t))
        qb = picks[picks["position"] == "QB"]
        return int(qb.groupby("round").size().max()) if len(qb) else 0

    assert worst_round(flat) > worst_round(rising), (
        "a flat tail should pile quarterbacks into one round and a rising one "
        "should not; if this stops holding the tail slope is not doing its job")


def test_the_roster_ceiling_stops_one_team_hoarding_a_position():
    """The curve says *when* kickers go; it does not say one each."""
    board = _board()
    picks = simulate_field(
        board, teams=10, rounds=16, starters=STARTERS, flex_slots=1,
        flex_eligible=FLEX, opponent_factory=league_opponent(_tendencies()),
        seed=17)
    per_team = picks[picks["position"] == "K"].groupby("team").size()
    assert per_team.max() <= 2, f"a team drafted {per_team.max()} kickers"


def test_an_unfitted_position_falls_back_to_the_consensus_order():
    """Silence is not an option, but neither is raising: a board carrying a
    position the fit never saw still has to draft."""
    board = _board()
    tend = _tendencies(rounds={}, curve_weight={},
                       sigma_rounds={p: 0.0 for p in POSITIONS})
    model = LeagueOpponentModel(tend, board, 10, np.random.default_rng(0))
    chosen = [model.choose(board.drop(index=range(i)), _roster([]))
              for i in range(5)]
    assert chosen == list(range(5)), (
        "with nothing fitted the field should walk the consensus board")


# --------------------------------------------------------------------------
# The queue number
# --------------------------------------------------------------------------

def test_the_queue_number_counts_gone_plus_place_in_line():
    positions = np.array(["RB", "WR", "RB", "WR", "RB"])
    adp = np.array([10.0, 5.0, 3.0, 20.0, 40.0])
    k = positional_queue_numbers(positions, adp, {"RB": 4, "WR": 1})
    # RBs by ADP: 3 -> 5th, 10 -> 6th, 40 -> 7th. WRs: 5 -> 2nd, 20 -> 3rd.
    assert list(k) == [6.0, 2.0, 5.0, 3.0, 7.0]


def test_the_queue_number_matches_the_obvious_loop():
    rng = np.random.default_rng(4)
    positions = rng.choice(np.array(POSITIONS), size=400)
    adp = rng.permutation(np.arange(1.0, 401.0))
    taken = {"QB": 3, "RB": 11, "WR": 9, "TE": 2, "K": 0, "DEF": 1}

    want = np.empty(len(adp))
    seen: dict[str, int] = {}
    for i in np.argsort(adp, kind="stable"):
        p = str(positions[i])
        seen[p] = seen.get(p, 0) + 1
        want[i] = taken.get(p, 0) + seen[p]
    assert np.array_equal(positional_queue_numbers(positions, adp, taken), want)


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------

def _deltas(rows: list[tuple[int, int, str, int, float]]) -> pd.DataFrame:
    """(season, teams, position, pick_no, adp_rank) -> the fit's input frame."""
    df = pd.DataFrame(rows, columns=["season", "teams", "position", "pick_no",
                                     "adp_rank"])
    df["delta"] = df["pick_no"] - df["adp_rank"]
    df["round"] = (df["pick_no"] - 1) // df["teams"] + 1
    return df


def test_run_points_normalizes_both_axes_by_league_size():
    """An 8-team draft and a 10-team draft have to describe the same thing, or
    they cannot be pooled and neither can drive a simulation of the other."""
    d = _deltas([(2021, 8, "K", 96, 250.0), (2025, 10, "K", 120, 250.0)])
    pts = run_points(d).set_index("season")
    # The 12th round of both drafts, whatever the pick number says.
    assert pts.loc[2021, "round"] == pytest.approx(12.0)
    assert pts.loc[2025, "round"] == pytest.approx(12.0)
    assert pts.loc[2021, "k_per_team"] == pytest.approx(1 / 8)
    assert pts.loc[2025, "k_per_team"] == pytest.approx(1 / 10)


def test_the_curve_rises_past_the_deepest_k_ever_drafted():
    rows = [(2025, 10, "K", 110 + 2 * i, 250.0 + i) for i in range(10)]
    grid, curves = fit_curves(_deltas(rows))
    k = np.asarray(grid)
    curve = np.asarray(curves["K"])
    observed = k <= 1.0 + 1e-9
    if (~observed).any():
        tail = np.diff(curve[k >= 1.0]) / np.diff(k[k >= 1.0])
        assert np.all(tail >= TAIL_SLOPE_ROUNDS_PER_K - 1e-6)
    assert np.all(np.diff(curve) >= -1e-9), "the curve must not go backwards"


def test_the_blend_weight_separates_a_board_position_from_a_scheduled_one():
    """The fitted weight has to recover which clock a position is on.

    A position drafted straight off the board should land near 0, and one taken
    on a schedule the board has no say in should land near 1. This is the
    parameter that stops the model averaging 2021's eight-running-back first
    round with 2025's five-receiver one.
    """
    # RB: taken exactly where the board ranks him.
    board_rows = [(2025, 10, "RB", i + 1, float(i + 1)) for i in range(40)]
    # K: buried on the board, taken one per team in rounds 12-14 regardless.
    sched_rows = [(2025, 10, "K", 115 + 3 * i, 250.0 + 7 * i) for i in range(10)]
    d = _deltas(board_rows + sched_rows)
    grid, curves = fit_curves(d)
    weights, sigmas = fit_blend_weights(d, grid, curves)

    assert weights["RB"] < 0.25, weights
    assert weights["K"] > 0.75, weights
    assert all(0.0 <= w <= 1.0 for w in weights.values())
    assert all(s > 0 for s in sigmas.values())


def test_ceilings_are_per_team_so_league_sizes_pool():
    drafts = [
        LeagueDraft(2021, "eight", 8, 15,
                    pd.DataFrame({"position": ["K"] * 8 + ["RB"] * 32})),
        LeagueDraft(2025, "ten", 10, 16,
                    pd.DataFrame({"position": ["K"] * 10 + ["RB"] * 50})),
    ]
    c = roster_ceilings(drafts)
    assert c["K"] == pytest.approx(1.0)
    assert c["RB"] == pytest.approx((4.0 + 5.0) / 2)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------

def test_a_fold_only_fits_on_drafts_held_before_its_own_season():
    drafts = [LeagueDraft(s, str(s), 10, 16, pd.DataFrame({"position": ["RB"]}))
              for s in (2021, 2025)]
    assert [d.season for d in fit_seasons_for(drafts, 2025)] == [2021]
    assert [d.season for d in fit_seasons_for(drafts, 2022)] == [2021]
    assert fit_seasons_for(drafts, 2021) == []
    assert [d.season for d in fit_seasons_for(drafts, 2025, scope="all")] \
        == [2021, 2025]
    with pytest.raises(ValueError):
        fit_seasons_for(drafts, 2025, scope="whenever")


def test_the_real_league_history_loads_and_carries_defenses(tmp_path):
    """The checked-in drafts are the fit's only input; if they stop parsing, or
    if defenses stop coming through, the fit degrades silently."""
    drafts = load_league_drafts()
    assert drafts, "no drafts under data/league_history/"
    for d in drafts:
        assert d.teams > 0 and d.rounds > 0
        assert len(d.picks) == d.teams * d.rounds
        counts = d.picks["position"].value_counts()
        assert counts.get("DEF", 0) >= 1, f"{d.season} has no defenses"
        assert counts.get("K", 0) >= 1, f"{d.season} has no kickers"
        assert set(counts.index) <= set(POSITIONS), (
            f"{d.season} has unnormalized positions: {sorted(counts.index)}")


def test_defenses_resolve_by_team_code_not_through_the_crosswalk():
    """Team defenses carry no gsis id, so a crosswalk built from nflverse ids
    contains none of them. Before this, every defense in every draft dropped out
    of the fit — and defenses are half of the effect being fitted."""
    from src.backtest.opponent_fit import _pick_keys

    class NeverResolves:
        def resolve(self, **kw):
            return None

    picks = pd.DataFrame([
        {"position": "DEF", "nfl_team": "PHI", "sleeper_player_id": "PHI",
         "player_name": "Philadelphia Eagles"},
        {"position": "DEF", "nfl_team": None, "sleeper_player_id": "GB",
         "player_name": "Green Bay Packers"},
        {"position": "RB", "nfl_team": "SF", "sleeper_player_id": "4034",
         "player_name": "Someone"},
    ])
    assert _pick_keys(picks, NeverResolves()) == ["DEF_PHI", "DEF_GB", None]


def test_the_history_files_are_the_ones_the_config_names():
    """The fit reads a directory; the config lists the leagues. If they drift,
    the fit is quietly running on a different set of drafts than the one
    HANDOFF.md describes."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "league.yaml")
        .read_text(encoding="utf-8"))
    configured = {int(h["season"]) for h in cfg["history"]}
    loaded = {d.season for d in load_league_drafts()}
    assert loaded == configured, (
        f"config/league.yaml lists {sorted(configured)} but "
        f"data/league_history/ holds {sorted(loaded)}; re-run "
        f"`python -m scripts.build_league_history`")


def test_tendencies_round_trip_through_json():
    """They are logged and can be dumped with `--json`; a lossy round trip would
    make a recorded field unreproducible."""
    tend = _tendencies(provenance="two drafts")
    back = LeagueTendencies.from_dict(json.loads(json.dumps(tend.to_dict())))
    assert back.curve_weight == tend.curve_weight
    assert back.max_per_team == tend.max_per_team
    assert back.provenance == tend.provenance
    assert np.allclose(back.k_per_team, tend.k_per_team)
    for pos, vals in tend.rounds.items():
        assert np.allclose(back.rounds[pos], vals, atol=1e-3)
