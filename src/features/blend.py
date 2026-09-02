"""Consensus projection blend.

Aggregation is the highest-ROI component in this whole system. Across twelve
seasons, aggregate projections beat every individual source, and source accuracy
does not persist year to year — which is why the default weights are equal
rather than fit. Resist the urge to tune them on a small sample; that is
overfitting with extra steps.

Blend procedure:
    1. z-score each component *within position* (positions have different scales
       and different variances; a raw average would let QB dominate)
    2. weighted mean of the z-scores
    3. map back to points using the position's own mean and standard deviation

Step 3 matters: VORP needs points, not z-scores, and mapping back through the
position's own moments keeps the output on a real fantasy-points scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"xfp": 1.0, "fantasypros": 1.0, "adp_vorp": 1.0}


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def blend_projections(df: pd.DataFrame,
                      weights: dict[str, float] | None = None,
                      position_col: str = "position",
                      out_col: str = "projection") -> pd.DataFrame:
    """Blend component projection columns into a single points projection.

    `weights` keys must be columns in `df`. Rows missing a component are blended
    from whatever components they do have — a player with no FantasyPros entry
    still gets a projection from xFP alone rather than being dropped, which
    matters for rookies and late-breaking role changes.
    """
    weights = weights or DEFAULT_WEIGHTS
    components = [c for c in weights if c in df.columns]
    if not components:
        raise ValueError(
            f"none of the blend components {sorted(weights)} are present; "
            f"available columns: {sorted(df.columns)}"
        )

    out = df.copy()
    z_cols = []

    for comp in components:
        zc = f"_z_{comp}"
        out[zc] = out.groupby(position_col)[comp].transform(_zscore)
        # Preserve NaN so missing components are excluded from that row's mean
        out.loc[out[comp].isna(), zc] = np.nan
        z_cols.append(zc)

    w = np.array([weights[c] for c in components], dtype=float)
    z = out[z_cols].to_numpy(dtype=float)

    mask = ~np.isnan(z)
    weighted = np.nansum(z * w, axis=1)
    wsum = (mask * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["_z_blend"] = np.where(wsum > 0, weighted / wsum, np.nan)

    # Map z back to points using each position's own moments, taken from the
    # component with the widest coverage so the scale is stable.
    anchor = max(components, key=lambda c: df[c].notna().sum())
    # ddof=0 to match `_zscore`; the groupby default of ddof=1 re-inflated the
    # spread by sqrt(n / (n - 1)), ~1.6% at K/DEF pool sizes.
    stats = out.groupby(position_col)[anchor].agg(
        mean="mean", std=lambda s: s.std(ddof=0))
    out = out.join(stats, on=position_col, rsuffix="_anchor")
    out[out_col] = out["_z_blend"] * out["std"].fillna(0.0) + out["mean"].fillna(0.0)

    # Fall back to the anchor component where the blend is undefined.
    out[out_col] = out[out_col].fillna(out[anchor])

    out = out.drop(columns=z_cols + ["_z_blend", "mean", "std"], errors="ignore")
    out["blend_components_used"] = mask.sum(axis=1)
    return out
