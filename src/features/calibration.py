"""Post-hoc calibration of projection spread.

Projections systematically exaggerate spread: the top-projected players tend to
underperform their projection and the bottom-projected tend to beat it. The fix
is a shrink toward the positional mean, with a slope below 1.0 at every position.

    calibrated = (raw - position_mean) * slope + position_mean

Applied *after* projection, never learned inside the model. Two reasons:
it stays interpretable and auditable, and it lets us re-estimate the slope from
live results in-season without retraining anything.

Preseason uses the configured priors. In-season, `fit_slopes` re-estimates from
accumulated projection/actual pairs once there is enough data to be worth it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Priors. Source: Fantasy Football Analytics calibration study (2026).
DEFAULT_SLOPES = {
    "QB": 0.67,
    "RB": 0.79,
    "WR": 0.85,
    "TE": 0.72,
    "K": 0.60,
    "DEF": 0.60,
}

# Below this many observations a fitted slope is noisier than the prior.
MIN_OBS_FOR_FIT = 40


def apply_calibration(df: pd.DataFrame,
                      slopes: dict[str, float] | None = None,
                      points_col: str = "projection",
                      position_col: str = "position",
                      out_col: str = "projection_calibrated") -> pd.DataFrame:
    """Shrink projection spread toward each position's mean."""
    slopes = slopes or DEFAULT_SLOPES
    out = df.copy()
    means = out.groupby(position_col)[points_col].transform("mean")
    slope = out[position_col].map(slopes).astype(float).fillna(1.0)
    out[out_col] = (out[points_col] - means) * slope + means
    out["calibration_slope"] = slope
    return out


def fit_slopes(history: pd.DataFrame,
               projection_col: str = "projection",
               actual_col: str = "actual",
               position_col: str = "position",
               priors: dict[str, float] | None = None,
               min_obs: int = MIN_OBS_FOR_FIT) -> dict[str, float]:
    """Re-estimate calibration slopes from realized projection/actual pairs.

    Regresses actual on projection within position; the fitted coefficient is
    the calibration slope. Positions with too few observations keep their prior
    rather than adopting a slope fit on noise.

    `history` should be as-of-that-week projections paired with realized points —
    never end-of-season projections, which would be leakage.
    """
    priors = priors or DEFAULT_SLOPES
    fitted: dict[str, float] = dict(priors)

    for pos, grp in history.groupby(position_col):
        grp = grp[[projection_col, actual_col]].dropna()
        if len(grp) < min_obs:
            continue
        x = grp[projection_col].to_numpy(dtype=float)
        y = grp[actual_col].to_numpy(dtype=float)
        xc = x - x.mean()
        denom = float((xc * xc).sum())
        if denom <= 0:
            continue
        slope = float((xc * (y - y.mean())).sum() / denom)
        # Clamp: a slope outside this range means something upstream is broken,
        # not that the projections are that miscalibrated.
        fitted[pos] = float(np.clip(slope, 0.3, 1.2))

    return fitted
