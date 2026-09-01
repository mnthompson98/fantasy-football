"""Measure how this league drafts against national consensus, and fit a field to it.

    python -m scripts.fit_opponent_model                   # the measurement + the fit
    python -m scripts.fit_opponent_model --scope prior --season 2025
    python -m scripts.fit_opponent_model --json opponent.json

HANDOFF.md carries "opponents draft ADP with gaussian noise" under Known
Limitations, and the reason it belongs there is that the whole measured edge
lives in the pick policy — drop-off, lookahead, caps — every part of which is a
bet on who survives to your next turn. If the simulated field takes players in
a different order than your actual league, that bet is being scored against the
wrong opposition.

This prints, in order:

    1. Per-round positional selection rates, league vs the ADP board taken
       straight down ("chalk"). The raw deviation, with no model in it.
    2. The same rates, league vs the gaussian field actually in use. This is
       the deviation that matters, because it is the error the backtest makes.
    3. delta = pick_no - adp_rank per position, with standard errors, per
       season and per ADP tier; and the per-position regression of pick number
       on consensus rank, which is what says the deviation is a compression
       rather than a shift.
    4. The fitted run curves, and a goodness-of-fit check: the fitted field
       re-simulated over the same boards, per position and per round.

Nothing here writes to `config/` or to the baseline fixture. The fit the
backtest runs happens inside `src/backtest/opponent_fit.py`, per fold, from the
same `data/league_history/*.json` files this reads — there is no artifact in
between to go stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.draft_sim import (  # noqa: E402
    gaussian_opponent,
    league_opponent,
)
from src.backtest.opponent_fit import (  # noqa: E402
    TIERS,
    fit_seasons_for,
    fit_tendencies,
    load_league_drafts,
    pick_deltas,
    simulate_field,
)
from src.features.vorp import LeagueShape  # noqa: E402
from src.ingest import history as H  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import build_crosswalk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _counts_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print("  " + f"{'rd':<4}" + "".join(f"{p:>12}" for p in POSITIONS))
    for row in rows:
        print("  " + f"{row['round']:<4}"
              + "".join(f"{row[p]:>12}" for p in POSITIONS))


def _simulate(board: pd.DataFrame, draft, shape, factory, *, sims: int,
              seed: int) -> pd.DataFrame:
    """`sims` full drafts of one historical league, tagged with `sim` so
    per-draft statistics (when the first kicker goes) stay per-draft."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(sims):
        f = simulate_field(
            board, teams=draft.teams, rounds=draft.rounds,
            starters=dict(shape.starters), flex_slots=shape.flex_slots,
            flex_eligible=tuple(shape.flex_eligible),
            opponent_factory=factory, seed=int(rng.integers(0, 2**31 - 1)))
        f["sim"] = i
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _round_table(draft, sim: pd.DataFrame, sims: int, title: str) -> None:
    rows = []
    for r in range(1, draft.rounds + 1):
        act = draft.picks[draft.picks["round"] == r]["position"].value_counts()
        mod = sim[sim["round"] == r]["position"].value_counts() / sims
        rows.append({"round": r, **{
            p: f"{int(act.get(p, 0))}/{mod.get(p, 0.0):.1f}" for p in POSITIONS}})
    _counts_table(title, rows)


def _first_pick_table(draft, sim: pd.DataFrame) -> None:
    """When each position first comes off the board: league vs the model.

    The single clearest way to see a field that is wrong — this league does not
    touch a kicker or a defense until round 10 at the earliest.
    """
    print("\n  first pick spent on each position (league vs model mean):")
    for p in POSITIONS:
        a = draft.picks[draft.picks["position"] == p]
        firsts = sim[sim["position"] == p].groupby("sim")["pick_no"].min()
        lg = (f"#{int(a['pick_no'].min()):<4} r{int(a['round'].min()):<3}"
              if len(a) else "never       ")
        md = (f"#{firsts.mean():<6.0f} r{(firsts.mean() - 1) // draft.teams + 1:<3.0f}"
              if len(firsts) else "never")
        print(f"    {p:<4} league {lg}   model {md}")


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def report_vs_chalk(drafts, boards) -> None:
    print("\n" + "=" * 78)
    print("1. PER-ROUND POSITIONAL SELECTION, LEAGUE vs ADP CHALK")
    print("=" * 78)
    print("   chalk = the consensus board taken straight down: pick k is ECR "
          "rank k.\n   Cells are league/chalk.")
    for d in drafts:
        board = boards.get(d.season)
        if board is None:
            continue
        ecr = board.sort_values("adp_rank")
        rows = []
        for r in range(1, d.rounds + 1):
            chalk = ecr.iloc[(r - 1) * d.teams: r * d.teams]["position"] \
                .value_counts()
            act = d.picks[d.picks["round"] == r]["position"].value_counts()
            rows.append({"round": r, **{
                p: f"{int(act.get(p, 0))}/{int(chalk.get(p, 0))}"
                for p in POSITIONS}})
        _counts_table(f"--- {d.season} {d.name} ({d.teams} teams, "
                      f"{d.rounds} rounds) ---", rows)

        print("\n  cumulative through round N (league minus chalk):")
        print("  " + f"{'thru':<6}" + "".join(f"{p:>7}" for p in POSITIONS))
        for r in (3, 6, 9, 12, d.rounds):
            n = r * d.teams
            chalk = ecr.iloc[:n]["position"].value_counts()
            act = d.picks[d.picks["pick_no"] <= n]["position"].value_counts()
            print("  " + f"{r:<6}" + "".join(
                f"{int(act.get(p, 0)) - int(chalk.get(p, 0)):>+7}"
                for p in POSITIONS))


def report_vs_gaussian(cfg: dict, drafts, boards, *, sims: int, seed: int
                       ) -> None:
    print("\n" + "=" * 78)
    print("2. PER-ROUND POSITIONAL SELECTION, LEAGUE vs THE GAUSSIAN FIELD")
    print("=" * 78)
    print("   The model currently in use, simulated over the same boards.")
    print("   Cells are league/model. This is the error the backtest is making.")

    shape = LeagueShape.from_config(cfg)
    opp = cfg["backtest"]["opponent_model"]
    sigma = {k: float(v) for k, v in opp["adp_sigma_fallback"].items()}
    factory = gaussian_opponent(sigma,
                                float(opp.get("positional_need_boost", 1.6)),
                                float(opp.get("positional_need_penalty", 0.4)))
    for d in drafts:
        board = boards.get(d.season)
        if board is None:
            continue
        sim = _simulate(board, d, shape, factory, sims=sims,
                        seed=seed + d.season)
        _round_table(d, sim, sims, f"--- {d.season} {d.name} ---")
        _first_pick_table(d, sim)


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares, then one refit without points beyond 3 robust sigma.

    One pass, not iterated. Iterating trims genuine spread — at 2.5 sigma over
    three passes it discards thirteen of ninety-seven receivers and reports a
    residual sd that is an artifact of the trimming. One pass at 3 sigma removes
    the cases that are obviously not draft-board behaviour (Deshaun Watson,
    consensus rank 341 in 2021, stashed at pick 94 while sitting out the season)
    and leaves the rest.
    """
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x)
    mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
    keep = np.abs(r - np.median(r)) <= 3.0 * mad if mad > 0 \
        else np.ones_like(r, dtype=bool)
    if keep.sum() >= max(4, int(0.6 * len(x))):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
    return float(a), float(b), float(np.std(r[keep], ddof=1))


def report_deltas(deltas: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("3. delta = pick_no - adp_rank   (negative = taken EARLIER than "
          "consensus)")
    print("=" * 78)
    overall = float(deltas["delta"].mean())
    print(f"   Pooled mean {overall:+.2f} picks. Only the RELATIVE differences "
          f"below are\n   identifiable — a common offset does not change an "
          f"argmin — and the raw mean\n   is structurally negative besides "
          f"(consensus ranks run past 500, picks stop\n   at 160).\n")

    g = deltas.groupby("position")["delta"].agg(["count", "mean", "std"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    g["centered"] = g["mean"] - overall
    g["t"] = g["centered"] / g["se"]
    print("  pooled over every draft:")
    print(g.round(2).to_string())

    print("\n  by draft — does the tendency replicate?")
    for season, sub in deltas.groupby("season"):
        c = sub.groupby("position")["delta"].mean() - sub["delta"].mean()
        print(f"    {season}: " + "  ".join(
            f"{p} {c.get(p, float('nan')):+6.1f}" for p in POSITIONS))

    print("\n  by ADP tier (centered on the pooled mean):")
    d = deltas.copy()
    d["tier"] = pd.cut(d["adp_rank"], [0] + [hi for _, hi in TIERS],
                       labels=[f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
                               for lo, hi in TIERS])
    piv = d.pivot_table(index="position", columns="tier", values="delta",
                        aggfunc="mean", observed=False) - overall
    n = d.pivot_table(index="position", columns="tier", values="delta",
                      aggfunc="count", observed=False)
    print(piv.round(1).to_string())
    print("\n  n per cell:")
    print(n.fillna(0).astype(int).to_string())
    print("\n  The 121+ column is censored by construction: a player ranked "
          "past the end of\n  the draft only appears here if somebody reached "
          "for him. K and DEF live\n  entirely in that column and are the "
          "exception that matters — every team takes\n  exactly one of each, "
          "every year, so nothing is being selected on there.")

    print("\n  pick_no ~ a + b * adp_rank, per position (robust, one 3-sigma "
          "refit).")
    print("  b near 1 = that position goes at consensus; b near 0 = the room "
          "ignores the\n  consensus ordering of it entirely. This is why the "
          "model is a run curve and\n  not a constant offset.")
    print("  " + f"{'pos':<5}{'n':>5}{'a':>9}{'b':>8}{'resid sd':>10}")
    for pos, grp in deltas.groupby("position"):
        a, b, sd = _robust_line(grp["adp_rank"].to_numpy(dtype=float),
                                grp["pick_no"].to_numpy(dtype=float))
        print("  " + f"{str(pos):<5}{len(grp):>5}{a:>9.1f}{b:>8.3f}{sd:>10.1f}")


def report_fit(tend, deltas, drafts, boards, cfg, *, sims: int, seed: int
               ) -> None:
    print("\n" + "=" * 78)
    print("4. FITTED FIELD")
    print("=" * 78)
    print(f"   {tend.provenance}\n")

    grid = np.asarray(tend.k_per_team, dtype=float)
    show = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    show = [k for k in show if grid.size and k <= grid[-1] + 1e-9]
    print("  run curve — the round the k-th player per team comes off the "
          "board:")
    print("  " + f"{'k/team':<8}" + "".join(f"{k:>8.1f}" for k in show))
    for p in POSITIONS:
        curve = tend.curve_for(p)
        if curve is None:
            print("  " + f"{p:<8}" + "   (not fitted; consensus order)")
            continue
        print("  " + f"{p:<8}" + "".join(
            f"{np.interp(k, grid, curve):>8.1f}"
            if k <= grid[-1] else f"{'-':>8}" for k in show))
    print("\n  " + f"{'pos':<6}{'curve weight':>14}{'sigma (rounds)':>16}"
          f"{'max per team':>15}")
    print("  " + " " * 6 + f"{'0 = board, 1 = schedule':>14}")
    for p in POSITIONS:
        print("  " + f"{p:<6}{tend.curve_weight.get(p, 0.0):>14.2f}"
              f"{tend.sigma_rounds.get(p, float('nan')):>16.2f}"
              f"{tend.max_per_team.get(p, float('nan')):>15.2f}")
    print(f"\n  over_ceiling_penalty {tend.over_ceiling_penalty} rounds")

    shape = LeagueShape.from_config(cfg)
    factory = league_opponent(tend)
    per_draft = {d.season: _simulate(boards[d.season], d, shape, factory,
                                     sims=sims, seed=seed + 7 + d.season)
                 for d in drafts if d.season in boards}
    sim = pd.concat(per_draft.values(), ignore_index=True)

    target = deltas.groupby("position")["delta"].mean() - deltas["delta"].mean()
    target_sd = deltas.groupby("position")["delta"].std(ddof=1)
    got = sim.groupby("position")["delta"].mean() - sim["delta"].mean()
    got_sd = sim.groupby("position")["delta"].std(ddof=1)

    print("\n  goodness of fit — centered mean delta (picks), league vs the "
          "fitted field:")
    print("  " + f"{'pos':<5}{'league':>9}{'fitted':>9}{'resid':>8}"
          f"{'sd lg':>9}{'sd fit':>9}")
    for p in POSITIONS:
        if p not in target.index:
            continue
        print("  " + f"{p:<5}{target[p]:>+9.1f}{got.get(p, np.nan):>+9.1f}"
              f"{target[p] - got.get(p, np.nan):>+8.1f}"
              f"{target_sd.get(p, np.nan):>9.1f}{got_sd.get(p, np.nan):>9.1f}")

    print("\n  per-round positional counts, league vs the fitted field:")
    for d in drafts:
        s = per_draft.get(d.season)
        if s is None:
            continue
        _round_table(d, s, sims, f"--- {d.season} {d.name} ---")
        _first_pick_table(d, s)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure this league's draft tendencies and fit a field")
    ap.add_argument("--scope", default="all", choices=("all", "prior"),
                    help="'all' uses every draft (the report); 'prior' shows "
                         "what a fold targeting --season is allowed to see")
    ap.add_argument("--season", type=int, default=None,
                    help="target season, for --scope prior")
    ap.add_argument("--sims", type=int, default=40,
                    help="simulated drafts per historical draft")
    ap.add_argument("--json", default=None,
                    help="write the fitted parameters to this path")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    opp = cfg["backtest"]["opponent_model"]
    seed = int(cfg["backtest"].get("seed", 20260830))

    print("loading rankings and the crosswalk...")
    crosswalk = build_crosswalk(nv.load_ff_playerids(refresh=args.refresh))
    rankings = nv.load_ff_rankings_history(refresh=args.refresh)

    drafts = fit_seasons_for(load_league_drafts(), args.season,
                             scope=args.scope)
    if not drafts:
        print("no league drafts under data/league_history/; run "
              "`python -m scripts.build_league_history` first", file=sys.stderr)
        return 1

    deltas = pick_deltas(drafts, rankings, crosswalk)
    if deltas.empty:
        print("no picks could be joined to a preseason ECR snapshot. ffverse "
              "preseason coverage starts in 2021.", file=sys.stderr)
        return 1

    boards: dict[int, pd.DataFrame] = {}
    for d in drafts:
        try:
            ecr, _, scrape = H.preseason_ecr(rankings, d.season, crosswalk)
        except ValueError as exc:
            print(f"  {d.season}: skipped — {exc}")
            continue
        boards[d.season] = ecr[["player_key", "position", "adp_rank"]].copy()
        n = int((deltas["season"] == d.season).sum())
        print(f"  {d.season} {d.name}: {len(d.picks)} picks, {n} joined to the "
              f"{scrape} ECR snapshot ({len(ecr)} ranked)")

    unmatched = deltas.attrs.get("unmatched")
    if unmatched is not None and len(unmatched):
        print(f"\n  {len(unmatched)} picks did not reconcile and are excluded:")
        print(unmatched[["season", "pick_no", "player_name", "position"]]
              .to_string(index=False))

    report_vs_chalk(drafts, boards)
    report_vs_gaussian(cfg, drafts, boards, sims=args.sims, seed=seed)
    report_deltas(deltas)

    tend = fit_tendencies(
        deltas, drafts,
        over_ceiling_penalty=float(opp.get("over_ceiling_penalty", 8.0)))
    report_fit(tend, deltas, drafts, boards, cfg, sims=args.sims, seed=seed)

    if args.json:
        Path(args.json).write_text(
            json.dumps(tend.to_dict(), indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    print("\nNothing was written to config/ or to the baseline fixture. The "
          "backtest\nre-fits this per fold from the same drafts; run it with "
          "`--opponent league`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
