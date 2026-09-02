"""Fit the opponent model to this league's own drafts.

HANDOFF.md listed "opponents draft ADP with gaussian noise" under Known
Limitations, with the right reason attached: beating that field is not the same
as beating your league. This module replaces the guess with a measurement.

**What is measured.** Every pick in `data/league_history/*.json` is joined to
the preseason ECR snapshot for that season — the same `adp_rank` the simulator
itself drafts off, produced by the same `history.preseason_ecr` the backtest
uses, so the deviation is measured against the exact ordering the model
consumes rather than against some other year's ADP list.

Two views come out of that join, and the model uses both:

    delta = pick_no - adp_rank      per pick, how far off consensus it went
    curve[pos](k / teams) -> round  when the k-th player at a position goes

**Why both.** `delta` says a kicker goes ~81 picks earlier than his consensus
rank and a defense ~47. That is the headline number and it is real, but it is
not a model: fitting `pick_no ~ adp_rank` per position gives a slope of 0.97
for RB and 0.86 for WR against 0.25 for K and 0.11 for DEF. The room does not
*shift* kickers up the board, it ignores the consensus's ordering of them
almost entirely and takes one per team in rounds 12-14. No constant offset can
express that.

The run curve can, but on its own it is board-blind in the other direction:
this league opened 2021 with eight running backs and 2025 with five receivers,
and it did so because those were the two boards. One pooled schedule describes
neither draft. So `fit_blend_weights` fits, per position and in closed form,
the weight `w` that best reconciles the two clocks — and it lands where the
regression slopes say it should, near 0 for RB and WR and near 1 for K and DEF.
The fitted field is checked by re-simulating it over the same boards, which is
what `scripts/fit_opponent_model.py` prints.

**Normalization.** Both axes are divided by the number of teams — k becomes
"players at this position per team" and pick number becomes "round". Without
it, this league's 8-team 2021 draft and its 10-team 2025 draft could not be
pooled, and neither could drive a simulation of the other size.

**Leakage.** `fit_seasons_for` returns only drafts held strictly before the
target season, so a fold's field is built from drafts that had already happened
when that draft board was current. The opponent model feeds no information to
`ValueDrafter` either way — it shapes the board our side faces, not our side's
beliefs about it — but the boundary is free to hold here, so it holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import history as H
from ..ingest.player_ids import normalize_position, normalize_team
from .draft_sim import (
    TAIL_SLOPE_ROUNDS_PER_K,
    LeagueTendencies,
    Roster,
    canonical_order,
    snake_order,
)

HISTORY_DIR = Path("data/league_history")

# Resolution of the run curve, in players-per-team. 0.1 is one player in a
# 10-team league — the finest grid the data can support without interpolating
# between a point and itself.
K_STEP = 0.1

# Below this many picks at a position, the measurement is not a measurement and
# the position keeps its consensus ordering. Every position clears it on a
# single 8-team draft (the thinnest cell is 8: one kicker and one defense per
# team), so this fires only on a position the league genuinely never drafted.
MIN_OBS = 5

# The ADP-rank tiers the report groups by. Same buckets the durability analysis
# uses (HANDOFF.md), so the two tables can be read side by side.
TIERS = ((1, 24), (25, 60), (61, 120), (121, 10**6))


@dataclass(frozen=True)
class LeagueDraft:
    """One historical draft, reduced to what the fit needs."""

    season: int
    name: str
    teams: int
    rounds: int
    picks: pd.DataFrame       # pick_no, round, position, player_name, ...


def load_league_drafts(history_dir: Path | str = HISTORY_DIR
                       ) -> list[LeagueDraft]:
    """Every draft under `data/league_history/`, oldest first.

    Those files are build artifacts of `scripts/build_league_history.py`, not
    hand-edited sources; re-run it to refresh them. Re-running needs the Sleeper
    API, and the fit does not — which is the point of them being checked in.
    """
    out: list[LeagueDraft] = []
    for path in sorted(Path(history_dir).glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        league = d.get("league") or {}
        picks = pd.DataFrame(d.get("picks") or [])
        if picks.empty:
            continue
        picks["position"] = picks["position"].map(normalize_position)
        teams = int(league.get("num_teams") or 0)
        if teams <= 0:
            continue
        out.append(LeagueDraft(
            season=int(league["season"]),
            name=str(league.get("name") or path.stem),
            teams=teams,
            rounds=int(league.get("num_rounds") or 0),
            picks=picks,
        ))
    return sorted(out, key=lambda d: d.season)


def _pick_keys(picks: pd.DataFrame, crosswalk) -> list[str | None]:
    """Canonical key per pick.

    Team defenses are resolved from the team code rather than the crosswalk:
    they carry no gsis id, and a crosswalk built from nflverse ids alone
    contains none of them (`player_ids.canonical_key`). Without this every
    defense in every draft drops out of the fit silently — and defenses are half
    of the effect the fit exists to capture.
    """
    keys: list[str | None] = []
    for row in picks.itertuples(index=False):
        pos = getattr(row, "position", None)
        if pos == "DEF":
            team = (normalize_team(getattr(row, "nfl_team", None))
                    or normalize_team(getattr(row, "sleeper_player_id", None)))
            keys.append(f"DEF_{team}" if team else None)
            continue
        keys.append(crosswalk.resolve(
            sleeper_id=getattr(row, "sleeper_player_id", None),
            name=getattr(row, "player_name", None),
            position=pos,
        ))
    return keys


def pick_deltas(drafts: list[LeagueDraft], rankings: pd.DataFrame,
                crosswalk) -> pd.DataFrame:
    """Every real pick with the consensus rank it went against.

    Returns one row per pick: season, league, teams, pick_no, round, position,
    player_name, player_key, adp_rank, ecr_scrape_date, delta. Picks that cannot
    be reconciled to the ECR snapshot are dropped and *counted* —
    `attrs["unmatched"]` carries them, so a caller can print what it lost rather
    than discovering later that a position quietly vanished.
    """
    frames, unmatched = [], []
    for d in drafts:
        try:
            ecr, _, scrape_date = H.preseason_ecr(rankings, d.season, crosswalk)
        except ValueError:
            # No preseason snapshot that year (ffverse coverage starts 2021).
            continue
        p = d.picks.copy()
        p["player_key"] = _pick_keys(p, crosswalk)
        p["season"] = d.season
        p["teams"] = d.teams
        p["league"] = d.name
        p["ecr_scrape_date"] = scrape_date
        p = p.merge(ecr[["player_key", "adp_rank"]], on="player_key", how="left")
        unmatched.append(p[p["adp_rank"].isna()])
        frames.append(p[p["adp_rank"].notna()])

    cols = ["season", "league", "teams", "pick_no", "round", "position",
            "player_name", "player_key", "adp_rank", "ecr_scrape_date"]
    if not frames:
        out = pd.DataFrame(columns=cols + ["delta"])
        out.attrs["unmatched"] = pd.DataFrame(columns=cols)
        return out

    out = pd.concat(frames, ignore_index=True)[cols].copy()
    out["delta"] = out["pick_no"].astype(float) - out["adp_rank"].astype(float)
    miss = pd.concat(unmatched, ignore_index=True) if unmatched else pd.DataFrame()
    out.attrs["unmatched"] = (miss[[c for c in cols if c in miss.columns]]
                              if len(miss) else pd.DataFrame(columns=cols))
    return out


def run_points(deltas: pd.DataFrame) -> pd.DataFrame:
    """(k per team, round) for every pick, per position and season.

    The k-th player taken at a position, in the order the room took them —
    which is what the curve is a curve of. Both axes normalized by league size
    so an 8-team draft and a 10-team draft describe the same thing.
    """
    frames = []
    for (season, pos), g in deltas.groupby(["season", "position"], sort=True):
        g = g.sort_values("pick_no")
        teams = float(g["teams"].iloc[0])
        frames.append(pd.DataFrame({
            "season": season,
            "position": str(pos),
            "k_per_team": np.arange(1, len(g) + 1) / teams,
            "round": g["pick_no"].to_numpy(dtype=float) / teams,
        }))
    if not frames:
        return pd.DataFrame(columns=["season", "position", "k_per_team",
                                     "round"])
    return pd.concat(frames, ignore_index=True)


def fit_curves(deltas: pd.DataFrame, *, k_step: float = K_STEP,
               min_obs: int = MIN_OBS
               ) -> tuple[list[float], dict[str, list[float]]]:
    """The run curve per position, over a grid shared by every position.

    Each season's own curve is interpolated onto the shared grid *first* and the
    seasons averaged afterwards, rather than interpolating through the pooled
    scatter. Pooling first looks equivalent and is not: two seasons' points
    interleave along k, so a single `np.interp` through them produces a
    staircase that alternates between the two drafts instead of the average of
    both, and the staircase's flat stretches become ties in the simulator.

    A season contributes to a grid point only where it has data, so the 8-team
    2021 draft (which reaches k/teams = 2.0 at QB) does not have to be
    extrapolated to cover the 10-team 2025 draft's range, or vice versa.

    **Past the deepest k a position was ever drafted to, the curve climbs at
    `TAIL_SLOPE_ROUNDS_PER_K` rather than flattening.** Carrying the last fitted
    value forward is the obvious thing and it is wrong in a way that is
    invisible until you look at round 14: it says every remaining quarterback is
    due at round 12.9, they all tie there, and the field drafts six of them in a
    round. The room did not slow down past its eighteenth quarterback, it
    stopped, and the curve has to say so.

    The curve is forced non-decreasing. Averaging two monotone curves is
    monotone, but the per-season interpolation can leave rounding wobble, and a
    curve that dips would say the room takes its ninth receiver before its
    eighth.
    """
    pts = run_points(deltas)
    if pts.empty:
        return [], {}

    grid = np.arange(k_step, float(pts["k_per_team"].max()) + k_step / 2,
                     k_step)
    curves: dict[str, list[float]] = {}

    for pos, g in pts.groupby("position"):
        if int(len(g)) < min_obs:
            continue
        stacked = []
        for _, s in g.groupby("season"):
            s = s.sort_values("k_per_team")
            x = s["k_per_team"].to_numpy(dtype=float)
            y = s["round"].to_numpy(dtype=float)
            col = np.interp(grid, x, y)
            col[(grid < x[0]) | (grid > x[-1])] = np.nan
            stacked.append(col)
        # Column mean over the seasons that cover each grid point. Done by hand
        # rather than with `np.nanmean` because a grid point covered by no
        # season is expected — that is what the tail and back-fill below are
        # for — and `nanmean` warns on it, training everyone to ignore warnings
        # from this module.
        block = np.vstack(stacked)
        covered = ~np.isnan(block)
        n_cov = covered.sum(axis=0)
        mean = np.where(n_cov > 0,
                        np.where(covered, block, 0.0).sum(axis=0)
                        / np.maximum(n_cov, 1),
                        np.nan)

        good = ~np.isnan(mean)
        if not good.any():
            continue
        first, last = int(np.argmax(good)), int(len(good) - 1 - np.argmax(good[::-1]))
        # Below the shallowest observed k, back-fill: the curve there is about
        # the very first player at the position and the grid only misses it by
        # rounding.
        mean[:first] = mean[first]
        # Above the deepest observed k, climb. See the docstring.
        if last < len(mean) - 1:
            mean[last + 1:] = mean[last] + TAIL_SLOPE_ROUNDS_PER_K * (
                grid[last + 1:] - grid[last])
        # Interior gaps (one season covers k, the other does not) interpolate.
        good = ~np.isnan(mean)
        idx = np.arange(len(mean))
        mean = np.interp(idx, idx[good], mean[good])
        curves[str(pos)] = [float(v) for v in np.maximum.accumulate(mean)]

    return [float(k) for k in grid], curves


def fit_blend_weights(deltas: pd.DataFrame, grid: list[float],
                      curves: dict[str, list[float]], *, min_obs: int = MIN_OBS
                      ) -> tuple[dict[str, float], dict[str, float]]:
    """Per position, how much of its timing is schedule rather than board.

    Two predictors of the round a pick actually happened in — the consensus
    board (`adp_rank / teams`) and the room's schedule (`curve(k / teams)`) —
    and one weight reconciling them. Constraining them to sum to one turns the
    least-squares problem into a projection with a closed form:

        w = <y - consensus, schedule - consensus> / ||schedule - consensus||^2

    clipped to [0, 1] because a negative weight, or one above one, would be
    extrapolating outside the two explanations rather than choosing between
    them, on 280 picks.

    Returns `(weights, sigma_rounds)`, the spread being the residual of the
    blended predictor at the fitted weight — the noise the model should inject,
    measured rather than assumed.
    """
    weights: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    g_arr = np.asarray(grid, dtype=float)

    for pos, grp in deltas.groupby("position"):
        pos = str(pos)
        curve = curves.get(pos)
        if curve is None or len(grp) < min_obs or not g_arr.size:
            continue

        # k is the order the room took them in, which is what the curve indexes.
        frames = []
        for _, s in grp.groupby("season"):
            s = s.sort_values("pick_no")
            teams = float(s["teams"].iloc[0])
            frames.append(pd.DataFrame({
                "y": s["pick_no"].to_numpy(dtype=float) / teams,
                "consensus": s["adp_rank"].to_numpy(dtype=float) / teams,
                "k": np.arange(1, len(s) + 1) / teams,
            }))
        f = pd.concat(frames, ignore_index=True)
        schedule = np.interp(f["k"].to_numpy(dtype=float), g_arr,
                             np.asarray(curve, dtype=float))

        y = f["y"].to_numpy(dtype=float)
        c = f["consensus"].to_numpy(dtype=float)
        d = schedule - c
        denom = float(d @ d)
        w = float(np.clip((y - c) @ d / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0
        weights[pos] = w

        resid = y - ((1.0 - w) * c + w * schedule)
        sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        sigmas[pos] = sd if np.isfinite(sd) and sd > 0 else 0.5

    return weights, sigmas


def roster_ceilings(drafts: list[LeagueDraft]) -> dict[str, float]:
    """Mean players per team at each position, over the fit drafts.

    What the league's managers actually end up holding, and what stops a fitted
    opponent taking a third kicker because kickers are cheap in round 13. Kept
    fractional (1.9 QB, not 2) because the mean is the measurement; the model
    floors it when it applies the ceiling.
    """
    rows = []
    for d in drafts:
        for pos, n in d.picks["position"].value_counts().items():
            if pos:
                rows.append({"position": str(pos),
                             "per_team": float(n) / max(1, d.teams)})
    if not rows:
        return {}
    return (pd.DataFrame(rows).groupby("position")["per_team"].mean()
            .round(3).to_dict())


def fit_tendencies(deltas: pd.DataFrame, drafts: list[LeagueDraft], *,
                   over_ceiling_penalty: float = 8.0,
                   k_step: float = K_STEP, min_obs: int = MIN_OBS
                   ) -> LeagueTendencies:
    """The whole fitted field, from the joined picks.

    There is no iterative step and nothing to converge: every parameter is a
    direct measurement or a closed form over one. `scripts/fit_opponent_model.py`
    re-simulates the fitted field over the same boards and prints the comparison
    rather than asserting that it reproduces them.
    """
    grid, curves = fit_curves(deltas, k_step=k_step, min_obs=min_obs)
    weights, sigmas = fit_blend_weights(deltas, grid, curves, min_obs=min_obs)
    ceilings = roster_ceilings(drafts)

    if not curves:
        return LeagueTendencies(
            k_per_team=[], rounds={}, curve_weight={}, sigma_rounds={},
            max_per_team=ceilings,
            over_ceiling_penalty=over_ceiling_penalty,
            provenance="no drafts joined to an ECR snapshot; consensus order")

    seasons = sorted(set(int(s) for s in deltas["season"].unique()))
    names = ", ".join(
        f"{d.season} {d.name} ({d.teams}x{d.rounds})"
        for d in drafts if d.season in seasons)
    prov = (f"{int(len(deltas))} picks from {names}; "
            f"{len(curves)} positions fitted")
    return LeagueTendencies(
        k_per_team=grid, rounds=curves, curve_weight=weights,
        sigma_rounds=sigmas, max_per_team=ceilings,
        over_ceiling_penalty=over_ceiling_penalty, provenance=prov)


def simulate_field(board: pd.DataFrame, *, teams: int, rounds: int,
                   starters: dict[str, int], flex_slots: int,
                   flex_eligible: tuple[str, ...], opponent_factory,
                   seed: int) -> pd.DataFrame:
    """Draft a whole league with every seat run by the opponent model.

    Deliberately *not* `simulate_draft`: that reserves one seat for
    `ValueDrafter`, and what a fitted field is being checked against is a room
    of eight or ten real managers with no seat playing our policy.
    """
    rng = np.random.default_rng(seed)
    opponent = opponent_factory(rng, board, teams)
    bench = max(0, rounds - sum(starters.values()) - flex_slots)
    rosters = {
        t: Roster(team_id=t, starters=dict(starters), flex_slots=flex_slots,
                  flex_eligible=flex_eligible, bench_slots=bench)
        for t in range(teams)
    }

    available = canonical_order(board)
    picked: list[dict] = []
    for pick_no, team in enumerate(snake_order(teams, rounds), start=1):
        if available.empty:
            break
        idx = opponent.choose(available, rosters[team])
        row = available.loc[idx]
        rec = {"pick_no": pick_no, "round": (pick_no - 1) // teams + 1,
               "team": team, "position": str(row["position"]),
               "adp_rank": float(row["adp_rank"])}
        rosters[team].picks.append(rec)
        picked.append(rec)
        available = available.drop(index=idx)

    out = pd.DataFrame(picked)
    out["delta"] = out["pick_no"].astype(float) - out["adp_rank"]
    return out


def fit_seasons_for(drafts: list[LeagueDraft], target_season: int | None,
                    *, scope: str = "prior") -> list[LeagueDraft]:
    """Which drafts a fold is allowed to fit its field on.

    `scope="prior"` (the default, and what the backtest runs) keeps only drafts
    held strictly before the target season. `scope="all"` uses every draft, for
    the offline report where there is no fold and nothing to leak into.
    """
    if scope == "all" or target_season is None:
        return list(drafts)
    if scope != "prior":
        raise ValueError(f"unknown opponent fit scope {scope!r}")
    return [d for d in drafts if d.season < int(target_season)]


class TendencyFitter:
    """Fits per fold, memoized on the set of drafts that fold is allowed to see.

    Every scorable fold in the configured window is allowed the same single
    prior draft, so without memoization the identical fit would run once per
    fold. It is also what keeps a run reproducible: the same fold always gets
    the same field, whatever order the folds run in.
    """

    def __init__(self, rankings: pd.DataFrame, crosswalk, *,
                 scope: str = "prior", over_ceiling_penalty: float = 8.0,
                 history_dir: Path | str = HISTORY_DIR):
        self.rankings = rankings
        self.crosswalk = crosswalk
        self.scope = scope
        self.over_ceiling_penalty = over_ceiling_penalty
        self.drafts = load_league_drafts(history_dir)
        self._cache: dict[tuple[int, ...], LeagueTendencies] = {}

    def for_season(self, target_season: int | None) -> LeagueTendencies:
        drafts = fit_seasons_for(self.drafts, target_season, scope=self.scope)
        key = tuple(d.season for d in drafts)
        if key in self._cache:
            return self._cache[key]

        deltas = (pick_deltas(drafts, self.rankings, self.crosswalk)
                  if drafts else pd.DataFrame())
        if drafts and not deltas.empty:
            tend = fit_tendencies(
                deltas, drafts,
                over_ceiling_penalty=self.over_ceiling_penalty)
        else:
            # Nothing to learn from. Falling back silently would be a
            # downgrade to the consensus ordering wearing the fitted model's
            # name, so it goes in the provenance string, which gets printed and
            # logged with the run.
            tend = LeagueTendencies(
                k_per_team=[], rounds={}, curve_weight={}, sigma_rounds={},
                max_per_team={}, over_ceiling_penalty=self.over_ceiling_penalty,
                provenance=(f"no usable league draft before {target_season}; "
                            f"field falls back to consensus order"))
        self._cache[key] = tend
        return tend
