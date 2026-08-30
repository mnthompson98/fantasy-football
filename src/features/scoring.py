"""League scoring, applied to nflverse box scores.

nflverse ships a `fantasy_points_ppr` column, and for QB/RB/WR/TE under this
league's settings it happens to agree with us exactly. We compute from the
league's own `scoring_settings` payload anyway, for two reasons:

  - `fantasy_points_ppr` is zero for kickers and does not exist at all for team
    defenses, and the whole point of the historical curve is that it covers
    every position we actually draft;
  - if the commissioner changes a setting for 2026, the curve should follow it
    rather than silently keep scoring last year's rules.

Everything here operates on *weekly* rows and sums to seasons, because a season
total is the wrong unit for a player who missed six games — per-game rate and
games played are separate signals and get separated at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")

# Sleeper's default full-PPR payload, used when a live league is not reachable.
# Verified against league 1262443299793276928 on 2026-08-30.
DEFAULT_SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -1.0, "pass_2pt": 2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec": 1.0, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum_lost": -2.0,
    "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0,
    "fgm_50_59": 5.0, "fgm_60p": 6.0, "fgmiss": -1.0,
    "xpm": 1.0, "xpmiss": -1.0,
    "sack": 1.0, "int": 2.0, "safe": 2.0, "blk_kick": 2.0,
    "def_td": 6.0, "def_st_td": 6.0, "st_td": 6.0,
    "def_st_ff": 1.0, "def_st_fum_rec": 1.0,
    "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "pts_allow_7_13": 4.0,
    "pts_allow_14_20": 1.0, "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0,
    "pts_allow_35p": -4.0,
}

# (upper bound inclusive, settings key). Walked in order; the last entry catches
# everything above 34.
_PTS_ALLOWED_TIERS = (
    (0, "pts_allow_0"),
    (6, "pts_allow_1_6"),
    (13, "pts_allow_7_13"),
    (20, "pts_allow_14_20"),
    (27, "pts_allow_21_27"),
    (34, "pts_allow_28_34"),
)


@dataclass(frozen=True)
class Scoring:
    settings: dict[str, float]

    @classmethod
    def from_league(cls, league: dict | None) -> "Scoring":
        """Build from a Sleeper `get_league()` payload, falling back to default."""
        raw = (league or {}).get("scoring_settings") or {}
        merged = dict(DEFAULT_SCORING)
        merged.update({k: float(v) for k, v in raw.items()})
        return cls(merged)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.settings.get(key, default))

    @property
    def is_full_ppr(self) -> bool:
        return abs(self.get("rec") - 1.0) < 1e-9


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """A missing stat column means zero of that stat, not a crash.

    nflverse adds and renames columns between releases; a board that dies
    because `passing_2pt_conversions` got renamed is worse than one that scores
    two-point conversions as zero and says so.
    """
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def score_offense(weekly: pd.DataFrame, scoring: Scoring) -> pd.Series:
    """Fantasy points for QB/RB/WR/TE from weekly player stats."""
    s = scoring
    fumbles_lost = (_col(weekly, "sack_fumbles_lost")
                    + _col(weekly, "rushing_fumbles_lost")
                    + _col(weekly, "receiving_fumbles_lost"))
    return (
        _col(weekly, "passing_yards") * s.get("pass_yd")
        + _col(weekly, "passing_tds") * s.get("pass_td")
        + _col(weekly, "passing_interceptions") * s.get("pass_int")
        + _col(weekly, "passing_2pt_conversions") * s.get("pass_2pt")
        + _col(weekly, "rushing_yards") * s.get("rush_yd")
        + _col(weekly, "rushing_tds") * s.get("rush_td")
        + _col(weekly, "rushing_2pt_conversions") * s.get("rush_2pt")
        + _col(weekly, "receptions") * s.get("rec")
        + _col(weekly, "receiving_yards") * s.get("rec_yd")
        + _col(weekly, "receiving_tds") * s.get("rec_td")
        + _col(weekly, "receiving_2pt_conversions") * s.get("rec_2pt")
        + fumbles_lost * s.get("fum_lost")
        + _col(weekly, "special_teams_tds") * s.get("st_td")
    )


def score_kicker(weekly: pd.DataFrame, scoring: Scoring) -> pd.Series:
    """Fantasy points for K.

    Distance buckets matter: a 50-yarder is worth 5 and a 30-yarder 3, so a
    kicker's leg strength is a real (if small) edge that a plain fg_made count
    would throw away.
    """
    s = scoring
    return (
        _col(weekly, "fg_made_0_19") * s.get("fgm_0_19")
        + _col(weekly, "fg_made_20_29") * s.get("fgm_20_29")
        + _col(weekly, "fg_made_30_39") * s.get("fgm_30_39")
        + _col(weekly, "fg_made_40_49") * s.get("fgm_40_49")
        + _col(weekly, "fg_made_50_59") * s.get("fgm_50_59")
        + _col(weekly, "fg_made_60_") * s.get("fgm_60p")
        + _col(weekly, "fg_missed") * s.get("fgmiss")
        + _col(weekly, "pat_made") * s.get("xpm")
        + _col(weekly, "pat_missed") * s.get("xpmiss")
    )


def points_allowed_points(points_allowed: pd.Series, scoring: Scoring) -> pd.Series:
    """Map opponent score to the DEF points-allowed tier bonus."""
    pa = pd.to_numeric(points_allowed, errors="coerce")
    out = pd.Series(scoring.get("pts_allow_35p"), index=pa.index, dtype=float)
    # Walk from the highest tier down so the tightest bound wins.
    for upper, key in reversed(_PTS_ALLOWED_TIERS):
        out = out.mask(pa <= upper, scoring.get(key))
    return out.mask(pa.isna(), np.nan)


def score_defense(team_weekly: pd.DataFrame, schedules: pd.DataFrame,
                  scoring: Scoring) -> pd.DataFrame:
    """Fantasy points for team defenses.

    Returns one row per team-week with `fantasy_points`. Points allowed comes
    from the schedule rather than the team stat frame — the defense is charged
    with what the *opponent* scored, including offensive and special-teams
    touchdowns the defense had nothing to do with, which is how the league
    scores it.
    """
    s = scoring
    df = team_weekly.copy()

    pts_for = _long_scores(schedules)
    df = df.merge(
        pts_for.rename(columns={"team": "opponent_team", "points": "points_allowed"}),
        on=["season", "week", "opponent_team"], how="left",
    )

    blocks = (_col(df, "def_punt_blocks") + _col(df, "def_pat_blocks")
              + _col(df, "def_fg_blocks"))
    df["fantasy_points"] = (
        _col(df, "def_sacks") * s.get("sack")
        + _col(df, "def_interceptions") * s.get("int")
        + _col(df, "fumble_recovery_opp") * s.get("def_st_fum_rec")
        + _col(df, "def_fumbles_forced") * s.get("def_st_ff")
        + _col(df, "def_tds") * s.get("def_td")
        + _col(df, "special_teams_tds") * s.get("st_td")
        + _col(df, "def_safeties") * s.get("safe")
        + blocks * s.get("blk_kick")
        + points_allowed_points(df["points_allowed"], s).fillna(0.0)
    )
    return df


def _long_scores(schedules: pd.DataFrame) -> pd.DataFrame:
    """Schedules are one row per game; DEF scoring wants one row per team-game."""
    home = schedules[["season", "week", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "points"})
    away = schedules[["season", "week", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points"})
    out = pd.concat([home, away], ignore_index=True)
    return out.dropna(subset=["points"])


def score_weekly(player_weekly: pd.DataFrame, scoring: Scoring) -> pd.DataFrame:
    """Attach `fantasy_points` to weekly player rows, per position.

    Offense and kickers score on disjoint stat sets, so a single pass over the
    frame with a position mask is both correct and cheaper than splitting.
    """
    df = player_weekly.copy()
    pos = df["position"].astype("string").str.upper()

    pts = pd.Series(0.0, index=df.index)
    off = pos.isin(OFFENSE_POSITIONS).fillna(False)
    kick = (pos == "K").fillna(False)

    pts = pts.mask(off, score_offense(df, scoring))
    pts = pts.mask(kick, score_kicker(df, scoring))

    df["fantasy_points"] = pts
    return df
