"""Build the draft board.

    python -m scripts.build_draft_board
    python -m scripts.build_draft_board --refresh          # force fresh pulls
    python -m scripts.build_draft_board --league-id <id>   # use live 2026 scoring

Outputs `outputs/projections/draft_board.{parquet,csv,html}`. The HTML file is
self-contained and works offline on a phone, which is where you will actually
be drafting from.

The pipeline, in order:

  1. **Identity.** Build the gsis-anchored crosswalk so FantasyPros, Sleeper and
     nflverse rows can be joined without ever touching a player name.
  2. **History.** Score six seasons of nflverse box scores under *this league's*
     scoring settings, then fit a monotone rank -> points curve per position.
  3. **Components.** Three projections per player: the expert consensus mapped
     through the curve, and last season's expected and actual production as
     per-game rates shrunk toward the positional mean.
  4. **Blend.** z-score within position, weighted mean, back to points.
  5. **Calibrate.** Shrink the spread toward the positional mean, because a
     finish-rank curve is systematically too wide (see rank_curve.py).
  6. **VORP.** Flex-aware replacement level for this league's exact roster.
  7. **Gate.** Players who cannot play are removed outright, not discounted,
     and written to a separate file so nothing disappears silently.

Every unmatched or dropped player is reported. A board that quietly loses a
top-40 rookie is worse than no board.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.leakage_guard import (  # noqa: E402
    TemporalBoundary,
    assert_no_future_seasons,
)
from src.draft.board import export, finalize_board  # noqa: E402
from src.features.blend import blend_projections  # noqa: E402
from src.features.calibration import apply_calibration  # noqa: E402
from src.features.rank_curve import (  # noqa: E402
    apply_spread_shrink,
    fit_rank_curve,
    persistence_shrink,
    season_totals,
    thin_positions,
)
from src.features.scoring import Scoring, score_defense, score_weekly  # noqa: E402
from src.features.vorp import LeagueShape, compute_vorp  # noqa: E402
from src.ingest import nflverse as nv  # noqa: E402
from src.ingest.player_ids import (  # noqa: E402
    build_crosswalk,
    normalize_position,
    normalize_team,
    reconcile,
)
from src.ingest.sleeper_api import (  # noqa: E402
    UNAVAILABLE_STATUSES,
    get_league,
    get_players,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"

DRAFTABLE = ("QB", "RB", "WR", "TE", "K", "DEF")

# The ffverse ranking product we draft off. `redraft-overall` is the single-QB
# redraft consensus across every position — the same list the rest of the league
# is looking at, which is exactly what makes disagreeing with it worth points.
ECR_PAGE = "redraft-overall"

# A per-game rate from three games is mostly noise. Shrink toward the positional
# mean with weight g / (g + K); at K = 6, a full season keeps ~74% of its own
# rate and a three-game cameo keeps 33%.
GAMES_SHRINK_K = 6.0

# Rates are scaled to a nominal full season for readability. The blend z-scores
# within position, so this constant cannot change any ranking — it only decides
# whether the column reads as points-per-game or points-per-season.
NOMINAL_GAMES = 17.0


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# History: score six seasons under this league's rules, fit the curves
# --------------------------------------------------------------------------

def build_history(seasons: list[int], scoring: Scoring, *,
                  refresh: bool = False) -> pd.DataFrame:
    """One row per player-season with points, games, and points per game.

    Offense and kickers come from weekly player stats; team defenses are built
    separately from team stats plus the schedule, and joined in under the same
    synthetic `DEF_<team>` key the crosswalk uses.
    """
    weekly = nv.load_player_stats(seasons, refresh=refresh)
    weekly = score_weekly(weekly, scoring)
    weekly["position"] = weekly["position"].map(normalize_position)
    weekly = weekly[weekly["position"].isin(DRAFTABLE)].copy()
    # nflverse `player_id` is the gsis_id, which is the canonical key already.
    weekly["player_key"] = weekly["player_id"].astype("string")

    totals = season_totals(weekly)

    team_weekly = nv.load_team_stats(seasons, refresh=refresh)
    schedules = nv.load_schedules(seasons, refresh=refresh)
    defense = score_defense(team_weekly, schedules, scoring)
    defense["position"] = "DEF"
    defense["player_key"] = "DEF_" + defense["team"].map(normalize_team).fillna("")
    defense = defense[defense["player_key"] != "DEF_"]

    def_totals = season_totals(defense)
    return pd.concat([totals, def_totals], ignore_index=True)


def prior_season_rates(totals: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    """Last season's production, as a shrunk per-game rate.

    Per-game rather than per-season on purpose: a player who missed half of last
    year on an unrelated injury is not half as good this year, and a season
    total says he is. Games played is the sample size, and it drives how far the
    rate is pulled toward the positional mean.
    """
    prior = totals[totals["season"] == prior_season].copy()
    if prior.empty:
        return prior.assign(prior_points=pd.Series(dtype=float))

    means = prior.groupby("position")["points_per_game"].transform("mean")
    w = prior["games"] / (prior["games"] + GAMES_SHRINK_K)
    prior["prior_ppg_shrunk"] = w * prior["points_per_game"] + (1 - w) * means
    prior["prior_points"] = prior["prior_ppg_shrunk"] * NOMINAL_GAMES
    return prior[["player_key", "position", "games", "points_per_game",
                  "prior_ppg_shrunk", "prior_points"]]


def prior_season_xfp(prior_season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Last season's *expected* fantasy points, as a shrunk per-game rate.

    Expected points strip out touchdown luck: a receiver who scored on 14% of
    his red-zone targets got lucky, and his expected points say so. This is the
    single most useful regression signal available before a snap is played, and
    it is why the blend is not just the consensus with extra steps.

    ffopportunity scores receptions at exactly 1.0, so its points are already on
    this league's full-PPR scale.
    """
    opp = nv.load_ff_opportunity([prior_season], refresh=refresh)
    opp["position"] = opp["position"].map(normalize_position)
    opp = opp[opp["position"].isin(("QB", "RB", "WR", "TE"))].copy()
    opp["player_key"] = opp["player_id"].astype("string")

    grouped = opp.groupby(["player_key", "position"], dropna=True)
    agg = grouped.agg(
        xfp=("total_fantasy_points_exp", "sum"),
        xfp_games=("total_fantasy_points_exp", "size"),
    ).reset_index()
    agg["xfp_per_game"] = agg["xfp"] / agg["xfp_games"].clip(lower=1)

    means = agg.groupby("position")["xfp_per_game"].transform("mean")
    w = agg["xfp_games"] / (agg["xfp_games"] + GAMES_SHRINK_K)
    agg["xfp_ppg_shrunk"] = w * agg["xfp_per_game"] + (1 - w) * means
    agg["xfp_points"] = agg["xfp_ppg_shrunk"] * NOMINAL_GAMES
    return agg[["player_key", "position", "xfp_games", "xfp_per_game",
                "xfp_points"]]


# --------------------------------------------------------------------------
# The draftable pool
# --------------------------------------------------------------------------

def build_pool(rankings: pd.DataFrame, cw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The consensus board, reconciled to canonical keys.

    Positional rank is derived from the overall ECR rather than read from the
    per-position ranking pages. Those are separate scrapes and can disagree with
    the overall list; deriving keeps overall and positional rank consistent with
    each other, which matters because VORP compares across positions.
    """
    pool = rankings[rankings["page_type"] == ECR_PAGE].copy()
    pool["position"] = pool["pos"].map(normalize_position)
    pool["nfl_team"] = pool["team"].map(normalize_team)
    pool = pool[pool["position"].isin(DRAFTABLE)]
    pool = pool.dropna(subset=["ecr"]).sort_values("ecr")

    matched, unmatched = reconcile(
        pool, cw,
        fantasypros_col="id", name_col="player", position_col="position",
    )
    matched = matched.drop_duplicates("player_key", keep="first").copy()

    matched["adp_rank"] = range(1, len(matched) + 1)
    matched["pos_rank"] = matched.groupby("position")["ecr"].rank(method="first")
    matched = matched.rename(columns={"player": "player_name", "bye": "bye_week"})
    return matched, unmatched


def attach_availability(pool: pd.DataFrame, cw) -> pd.DataFrame:
    """Fold in Sleeper's live injury status and current team.

    Sleeper is hours fresher than the nightly ranking scrape, and in the last
    week of August that gap is where the season-ending news lives.
    """
    meta = pool["player_key"].map(lambda k: cw.meta.get(k, {}))
    pool["injury_status"] = meta.map(lambda m: m.get("injury_status"))
    pool["sleeper_id"] = meta.map(lambda m: m.get("sleeper_id"))
    pool["sleeper_status"] = meta.map(lambda m: m.get("sleeper_status"))

    live_team = meta.map(lambda m: m.get("nfl_team"))
    pool["nfl_team"] = live_team.fillna(pool["nfl_team"])

    # The live monitor removes drafted players by matching Sleeper's `player_id`
    # against this column, so it has to *be* the Sleeper id. Falling back to the
    # canonical key keeps the row on the board; it just won't auto-strike when
    # someone drafts him, which is why the count gets reported.
    pool["player_id"] = pool["sleeper_id"].fillna(pool["player_key"])

    status = pool["injury_status"].fillna("").astype(str).str.strip()
    pool["available"] = ~status.isin(UNAVAILABLE_STATUSES)
    return pool


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def assemble(cfg: dict, *, refresh: bool = False, league_id: str | None = None,
             seasons: list[int] | None = None, target_season: int | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("resolving player identities...")
    ff_ids = nv.load_ff_playerids(refresh=refresh)
    sleeper_players = get_players(cache_dir=nv.RAW_DIR)
    cw = build_crosswalk(ff_ids, sleeper_players)
    print(f"  {len(cw.meta)} players in crosswalk")

    league = get_league(league_id) if league_id else None
    scoring = Scoring.from_league(league)
    source = f"league {league_id}" if league else "verified 2025 settings"
    print(f"  scoring: {'full PPR' if scoring.is_full_ppr else 'NOT full PPR'} "
          f"({source})")

    hist_seasons = seasons or [int(s) for s in cfg["backtest"]["seasons"]]
    prior_season = max(hist_seasons)
    target = int(target_season) if target_season else prior_season + 1
    boundary = TemporalBoundary(target_season=target)

    print(f"scoring history {min(hist_seasons)}-{prior_season}...")
    totals = build_history(hist_seasons, scoring, refresh=refresh)
    # The same guard the backtest uses. Preseason is exactly when it is cheapest
    # to accidentally reach into the season you are about to predict — rebuild
    # the board in week 3 with the current season in `--seasons` and every
    # component silently becomes a report card instead of a projection.
    assert_no_future_seasons(totals, boundary, label="history totals")
    print(f"  {len(totals)} player-seasons · boundary {boundary.describe()}")

    curve = fit_rank_curve(
        totals, downweight=cfg["backtest"].get("downweight"),
    )
    thin = thin_positions(curve)
    if thin:
        print(f"  [warn] curve fit on <2 seasons for: {', '.join(thin)}")
    for pos in DRAFTABLE:
        if pos in curve.curves:
            print(f"  {pos:<3} {pos}1={curve.points_for_rank(pos, 1):6.1f}  "
                  f"{pos}12={curve.points_for_rank(pos, 12):6.1f}  "
                  f"{pos}36={curve.points_for_rank(pos, 36):6.1f}")

    print("loading consensus rankings...")
    rankings = nv.load_ff_rankings(refresh=refresh)
    scrape_date = str(rankings["scrape_date"].max())
    pool, unmatched = build_pool(rankings, cw)
    print(f"  ECR scraped {scrape_date} · {len(pool)} players matched, "
          f"{len(unmatched)} unmatched")
    if len(unmatched):
        for row in unmatched.head(10).itertuples():
            print(f"    unmatched: {row.player} ({row.pos}, ECR {row.ecr:.0f})")

    pool = attach_availability(pool, cw)
    no_sleeper = int(pool["sleeper_id"].isna().sum())
    if no_sleeper:
        print(f"  [warn] {no_sleeper} players have no Sleeper id; the live "
              f"monitor cannot auto-strike them when drafted")

    # --- components -------------------------------------------------------
    pool["ecr_points"] = curve.points_for_ranks(pool["position"], pool["pos_rank"])

    prior = prior_season_rates(totals, prior_season)
    pool = pool.merge(prior.drop(columns=["position"]), on="player_key", how="left")

    xfp = prior_season_xfp(prior_season, refresh=refresh)
    pool = pool.merge(xfp.drop(columns=["position"]), on="player_key", how="left")

    weights = {k: float(v) for k, v in cfg["blend"]["components"].items()}
    for comp in weights:
        n = int(pool[comp].notna().sum()) if comp in pool.columns else 0
        print(f"  component {comp:<14} {n:4d}/{len(pool)} players "
              f"(weight {weights[comp]})")

    print("blending, calibrating, computing VORP...")
    blended = blend_projections(pool, weights=weights)
    calibrated = apply_calibration(
        blended, slopes=cfg["calibration"]["slopes"],
        out_col="projection_calibrated",
    )

    shrink_cfg = cfg["calibration"].get("persistence_shrink") or {}
    factors = persistence_shrink(
        totals, list(shrink_cfg.get("positions") or []),
        reference=shrink_cfg.get("reference", "QB"),
    )
    if factors:
        print("  persistence shrink: "
              + ", ".join(f"{p} x{f:.2f}" for p, f in sorted(factors.items())))
        calibrated = apply_spread_shrink(calibrated, factors)

    shape = LeagueShape.from_config(cfg)
    valued = compute_vorp(
        calibrated[calibrated["available"]], shape,
        points_col="projection_calibrated",
    )
    excluded = calibrated[~calibrated["available"]].copy()

    board = finalize_board(valued, drop_unavailable=False)
    meta = {
        "ecr": scrape_date,
        "teams": shape.teams,
        "scoring": "PPR" if scoring.is_full_ppr else "custom",
        "players": len(board),
    }
    return board, excluded, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the value-based draft board")
    ap.add_argument("--refresh", action="store_true",
                    help="force fresh pulls, ignoring the data/raw cache")
    ap.add_argument("--league-id", default=None,
                    help="Sleeper league id to read live scoring settings from")
    ap.add_argument("--seasons", default=None,
                    help="comma-separated history seasons (default: config)")
    ap.add_argument("--target-season", type=int, default=None,
                    help="season being drafted for; sets the leakage boundary "
                         "(default: latest history season + 1)")
    ap.add_argument("--out", default="outputs/projections")
    ap.add_argument("--top", type=int, default=30,
                    help="how many rows to print when done")
    args = ap.parse_args()

    cfg = load_config()
    seasons = ([int(s) for s in args.seasons.split(",")]
               if args.seasons else None)

    board, excluded, meta = assemble(
        cfg, refresh=args.refresh, league_id=args.league_id, seasons=seasons,
        target_season=args.target_season,
    )

    out_dir = ROOT / args.out
    paths = export(board, out_dir, meta=meta)

    if len(excluded):
        excluded_path = out_dir / "excluded_players.csv"
        cols = ["player_name", "position", "nfl_team", "injury_status",
                "adp_rank", "projection_calibrated"]
        excluded.sort_values("adp_rank")[cols].to_csv(excluded_path, index=False)
        print(f"\n{len(excluded)} players gated out (injury/roster status) -> "
              f"{excluded_path.relative_to(ROOT)}")
        for row in excluded.sort_values("adp_rank").head(8).itertuples():
            print(f"    {row.position:<3} {str(row.player_name)[:24]:<24} "
                  f"{row.injury_status}  (ECR {row.adp_rank})")

    print(f"\ntop {args.top} by VORP:")
    print(f"  {'#':>3}  {'POS':<4}{'PLAYER':<24}{'TM':<5}"
          f"{'VORP':>7}{'PROJ':>8}{'ADP':>6}{'d':>6}")
    for row in board.head(args.top).itertuples():
        delta = row.adp_delta
        print(f"  {row.rank:>3}  {str(row.position):<4}"
              f"{str(row.player_name)[:23]:<24}{str(row.nfl_team or '-'):<5}"
              f"{row.vorp:>7.1f}{row.projection_calibrated:>8.1f}"
              f"{row.adp_rank:>6.0f}{delta:>+6.0f}")

    print("\nwrote:")
    for kind, path in paths.items():
        print(f"  {kind:<8} {path.relative_to(ROOT)}")
    print("\nlive monitor:")
    print("  python -m src.draft.monitor --draft-id <id> --my-slot <n>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
