"""Runtime enforcement of temporal boundaries in the backtest.

This module raises. It does not warn. A leaky backtest produces confident,
wrong answers that look like success, and a warning in a log scroll is not
protection against that.

Rules, for a simulation of season Y:
  - features may use data through season Y-1
  - plus season Y's *preseason* snapshot (ADP, opening rosters, preseason
    injury designations) captured before the draft
  - nothing from season Y's regular season, and nothing from Y+1 or later
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a frame carries information the simulation must not see."""


@dataclass(frozen=True)
class TemporalBoundary:
    """The information horizon for a single backtest fold."""

    target_season: int
    allow_preseason_snapshot: bool = True

    @property
    def max_completed_season(self) -> int:
        return self.target_season - 1

    def describe(self) -> str:
        tail = " + preseason snapshot" if self.allow_preseason_snapshot else ""
        return (
            f"target={self.target_season}, "
            f"visible<=season {self.max_completed_season}{tail}"
        )


def assert_no_future_seasons(df: pd.DataFrame, boundary: TemporalBoundary,
                             season_col: str = "season",
                             label: str = "frame") -> None:
    """Every row must come from a season the simulation is allowed to see."""
    if season_col not in df.columns:
        raise LeakageError(
            f"{label}: no '{season_col}' column, so the temporal boundary "
            f"cannot be verified. Add one rather than skipping the check."
        )

    seasons = pd.to_numeric(df[season_col], errors="coerce")
    if seasons.isna().any():
        n = int(seasons.isna().sum())
        raise LeakageError(f"{label}: {n} rows have an unparseable {season_col}.")

    offending = sorted(set(seasons[seasons > boundary.max_completed_season].astype(int)))
    if offending:
        raise LeakageError(
            f"{label}: contains seasons {offending} but the boundary is "
            f"{boundary.describe()}. Filter to <= {boundary.max_completed_season} "
            f"before use, or route it through `assert_preseason_only` if it is a "
            f"legitimate preseason snapshot."
        )


def assert_preseason_only(df: pd.DataFrame, boundary: TemporalBoundary,
                          season_col: str = "season",
                          week_col: str | None = "week",
                          label: str = "snapshot") -> None:
    """A target-season frame is admissible only if it is genuinely preseason.

    Used for ADP boards and opening-day injury/roster status. If the frame
    carries week numbers, they must all be <= 0 (nflverse encodes preseason as
    week 0 or negative); a week 1+ row is regular-season data wearing a
    preseason label.
    """
    if season_col not in df.columns:
        raise LeakageError(f"{label}: no '{season_col}' column to verify.")

    if not boundary.allow_preseason_snapshot:
        assert_no_future_seasons(df, boundary, season_col, label)
        return

    seasons = pd.to_numeric(df[season_col], errors="coerce")
    too_new = sorted(set(seasons[seasons > boundary.target_season].dropna().astype(int)))
    if too_new:
        raise LeakageError(
            f"{label}: contains seasons {too_new}, beyond target season "
            f"{boundary.target_season}."
        )

    if week_col and week_col in df.columns:
        target_rows = df[seasons == boundary.target_season]
        weeks = pd.to_numeric(target_rows[week_col], errors="coerce")
        in_season = weeks[weeks > 0]
        if not in_season.empty:
            raise LeakageError(
                f"{label}: {len(in_season)} rows from season "
                f"{boundary.target_season} have week > 0 (weeks "
                f"{sorted(set(in_season.astype(int)))[:5]}...). Those are "
                f"regular-season results, not a preseason snapshot."
            )


def assert_purge_gap(train_seasons: list[int], target_season: int,
                     purge_gap: int = 1) -> None:
    """Enforce a gap between the training window and the target season.

    Without a gap, information from the very end of training bleeds into the
    test season through carryover features (rolling averages, depth-chart state,
    contract status) that straddle the boundary.
    """
    if not train_seasons:
        raise LeakageError("empty training window.")

    latest = max(train_seasons)
    required = target_season - purge_gap
    if latest > required:
        raise LeakageError(
            f"training window ends at {latest} but target is {target_season} "
            f"with purge_gap={purge_gap}; training must end at or before "
            f"{required}."
        )


def flag_suspicious_stability(fold_scores: list[float],
                              cv_threshold: float = 0.02) -> str | None:
    """Return a warning string if walk-forward scores are implausibly stable.

    Genuine year-over-year fantasy prediction is noisy. Near-identical scores
    across folds is the classic signature of leakage, not of a good model.
    This one returns rather than raises — it is a heuristic, and the caller
    decides. But do not ignore it.
    """
    if len(fold_scores) < 3:
        return None

    import statistics

    mean = statistics.fmean(fold_scores)
    if mean == 0:
        return None
    cv = statistics.pstdev(fold_scores) / abs(mean)
    if cv < cv_threshold:
        return (
            f"Fold scores are suspiciously stable (CV={cv:.4f} across "
            f"{len(fold_scores)} folds). Fantasy outcomes are noisy; this "
            f"pattern usually means leakage. Audit feature construction before "
            f"trusting these results."
        )
    return None
