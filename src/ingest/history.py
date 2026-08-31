"""Season-parameterized history, for the board and the backtest alike.

The live board and a backtest fold want the same three things, differing only in
which season they are standing in: box scores scored under league rules, an
expected-points table, and a preseason ECR snapshot. Building them here, once,
keeps the backtest honest — it cannot accidentally see a season the live board
would not have had.
"""

from __future__ import annotations

import pandas as pd

from ..features.scoring import Scoring, score_defense, score_weekly
from ..features.rank_curve import season_totals
from . import nflverse as nv
from .player_ids import normalize_position, normalize_team, reconcile

DRAFTABLE = ("QB", "RB", "WR", "TE", "K", "DEF")

# The ffverse ranking product we draft off: the single-QB redraft consensus
# across every position — the same list the rest of the league is looking at.
ECR_PAGE = "redraft-overall"

# A preseason snapshot is one taken in this window. Late enough to have the
# depth charts, early enough that no regular-season game has been played.
PRESEASON_START = "07-01"
PRESEASON_END = "09-05"


def scored_weekly(seasons: list[int], scoring: Scoring, *,
                  refresh: bool = False) -> pd.DataFrame:
    """Weekly fantasy points for every draftable position, on `player_key`.

    Offense and kickers come from weekly player stats; team defenses are built
    separately from team stats plus the schedule, because nflverse does not
    score them at all.
    """
    weekly = nv.load_player_stats(seasons, refresh=refresh)
    weekly = score_weekly(weekly, scoring)
    weekly["position"] = weekly["position"].map(normalize_position)
    weekly = weekly[weekly["position"].isin(DRAFTABLE)].copy()
    # nflverse `player_id` is the gsis_id, which is the canonical key already.
    weekly["player_key"] = weekly["player_id"].astype("string")

    team_weekly = nv.load_team_stats(seasons, refresh=refresh)
    schedules = nv.load_schedules(seasons, refresh=refresh)
    defense = score_defense(team_weekly, schedules, scoring)
    defense["position"] = "DEF"
    defense["player_key"] = "DEF_" + defense["team"].map(normalize_team).fillna("")
    defense = defense[defense["player_key"] != "DEF_"]

    cols = ["season", "week", "season_type", "position", "player_key",
            "fantasy_points"]
    return pd.concat([weekly[cols], defense[cols]], ignore_index=True)


def season_totals_for(seasons: list[int], scoring: Scoring, *,
                      refresh: bool = False) -> pd.DataFrame:
    """One row per player-season: points, games, and points per game."""
    return season_totals(scored_weekly(seasons, scoring, refresh=refresh))


def opportunity_for(season: int, cw=None, *, refresh: bool = False
                    ) -> pd.DataFrame:
    """ffopportunity expected points for one season, keyed canonically."""
    opp = nv.load_ff_opportunity([season], refresh=refresh)
    opp["position"] = opp["position"].map(normalize_position)
    opp = opp[opp["position"].isin(("QB", "RB", "WR", "TE"))].copy()
    opp["player_key"] = opp["player_id"].astype("string")
    return opp


def projection_actual_pairs(rankings: pd.DataFrame, totals: pd.DataFrame,
                            cw, seasons: list[int], *,
                            downweight: list[dict] | None = None,
                            top_n: int = 40) -> pd.DataFrame:
    """Preseason projections paired with what actually happened.

    For each season, fit the rank curve on *earlier* seasons only, project the
    preseason ECR snapshot through it, and attach realized points. That gives
    the (projection, actual) pairs `calibration.fit_slopes` needs to measure how
    much of a position's projected spread actually materializes — replacing the
    generic priors with numbers from this league's own scoring.

    `top_n` restricts each position to the players you would plausibly draft.
    Including the whole pool measures something else: the deep tail is mostly
    "starters outscore backups", the pools differ in depth by position, and the
    slopes it produces differ from position to position in ways that vanish once
    you look only at draftable players.

    Seasons without a preseason ECR snapshot are skipped, so this returns pairs
    for 2021 onward regardless of what is asked for.
    """
    frames = []
    for season in sorted(seasons):
        train = [s for s in totals["season"].unique() if s < season]
        if not train:
            continue
        try:
            ecr, _, _ = preseason_ecr(rankings, season, cw)
        except ValueError:
            continue  # no snapshot that year; 2020 and earlier

        from ..features.rank_curve import fit_rank_curve
        curve = fit_rank_curve(
            totals[totals["season"].isin(train)], downweight=downweight)

        sub = ecr[ecr["pos_rank"] <= top_n].copy()
        sub["projection"] = curve.points_for_ranks(sub["position"], sub["pos_rank"])

        realized = totals[totals["season"] == season].set_index("player_key")["points"]
        sub["actual"] = sub["player_key"].map(realized).fillna(0.0)
        sub["season"] = season
        frames.append(sub[["season", "position", "player_key", "pos_rank",
                           "projection", "actual"]])

    if not frames:
        return pd.DataFrame(
            columns=["season", "position", "player_key", "pos_rank",
                     "projection", "actual"])
    return pd.concat(frames, ignore_index=True)


def preseason_ecr(rankings: pd.DataFrame, season: int, cw, *,
                  page: str = ECR_PAGE) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The last ECR snapshot taken before season `season` kicked off.

    Returns `(matched, unmatched, scrape_date)`. Positional rank is derived from
    the overall ECR rather than read from the per-position ranking pages: those
    are separate scrapes and can disagree with the overall list, and deriving
    keeps overall and positional rank consistent with each other, which matters
    because VORP compares across positions.
    """
    df = rankings[rankings["page_type"] == page].copy()
    dates = pd.to_datetime(df["scrape_date"], errors="coerce")
    window = (dates >= f"{season}-{PRESEASON_START}") & (dates <= f"{season}-{PRESEASON_END}")
    df = df[window]
    if df.empty:
        raise ValueError(
            f"no preseason '{page}' ECR snapshot for {season} between "
            f"{season}-{PRESEASON_START} and {season}-{PRESEASON_END}. "
            f"ffverse coverage starts in 2021; earlier seasons can be trained "
            f"on but not targeted."
        )

    latest = df["scrape_date"].max()
    snap = df[df["scrape_date"] == latest].copy()

    snap["position"] = snap["pos"].map(normalize_position)
    snap["nfl_team"] = snap["team"].map(normalize_team)
    snap = snap[snap["position"].isin(DRAFTABLE)]
    snap = snap.dropna(subset=["ecr"]).sort_values("ecr")

    matched, unmatched = reconcile(
        snap, cw, fantasypros_col="id", name_col="player",
        position_col="position",
    )

    # Team defenses are not players and carry no gsis id, so a crosswalk built
    # from nflverse ids alone contains none of them. Their canonical key is
    # derivable from the team code by itself, which is the whole reason
    # `DEF_<team>` exists — so recover them rather than dropping them.
    #
    # This is not cosmetic. A backtest board with no defenses cannot fill a DEF
    # starting slot for any team in any fold, and every score it produces is
    # quietly wrong.
    if len(unmatched):
        rescued = unmatched[
            (unmatched["position"] == "DEF") & unmatched["nfl_team"].notna()
        ].copy()
        if len(rescued):
            rescued["player_key"] = "DEF_" + rescued["nfl_team"]
            matched = pd.concat([matched, rescued], ignore_index=True)
            unmatched = unmatched.drop(index=rescued.index)

    matched = matched.drop_duplicates("player_key", keep="first").copy()
    matched = matched.sort_values("ecr")
    matched["adp_rank"] = range(1, len(matched) + 1)
    matched["pos_rank"] = matched.groupby("position")["ecr"].rank(method="first")
    matched = matched.rename(columns={"player": "player_name"})
    if "bye" in matched.columns:
        matched = matched.rename(columns={"bye": "bye_week"})

    missing = [p for p in DRAFTABLE if not (matched["position"] == p).any()]
    if missing:
        raise ValueError(
            f"{season} ECR snapshot ({latest}) has no {', '.join(missing)} after "
            f"reconciliation. A board that cannot fill a starting slot produces "
            f"silently wrong rosters — fix the identity join, do not proceed."
        )

    return matched, unmatched, str(latest)
