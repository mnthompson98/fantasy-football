"""Measuring calibration slopes from this league's own history.

`config/league.yaml` ships prior slopes from a published study. They are a
reasonable starting point and they were wrong here in a way that mattered: on
this league's scoring, measured against 2021-2025 preseason ECR, RB fits 0.85
and WR 0.77 — an exact swap of the priors' 0.79 and 0.85.

The consequence was not subtle. Calibration shrinks a position's projected
spread toward its own mean, so a lower slope flattens the top of that position.
Shrinking RB harder than WR pushed receivers above backs everywhere on the
board, and the simulated drafter dutifully built rosters of eight receivers and
two running backs, reaching round 13 before taking its first RB.

So measure it instead. Each fold fits on seasons strictly earlier than the one
it is predicting, which keeps this leakage-free, and any position without enough
history keeps its prior rather than adopting a slope fit on noise.
"""

from __future__ import annotations

import pandas as pd

from .calibration import DEFAULT_SLOPES, fit_slopes


def fitted_slopes(cfg: dict, rankings: pd.DataFrame, totals: pd.DataFrame,
                  crosswalk, seasons: list[int], *,
                  opportunity_for=None,
                  verbose: bool = False) -> dict[str, float]:
    """Per-position calibration slopes, measured where possible.

    Returns the configured priors unchanged when `fit_from_history` is off or
    when there is not enough history to fit — never a silently empty dict.

    The pairs are built through the same blend the slope is applied to (see
    `history.projection_actual_pairs`); `opportunity_for(season)` supplies the
    expected-points table for a prior season, and is the only reason this
    function needs to know anything about ingest.
    """
    priors = {**DEFAULT_SLOPES, **(cfg.get("calibration", {}).get("slopes") or {})}
    fit_cfg = (cfg.get("calibration", {}).get("fit_from_history") or {})
    if not fit_cfg.get("enabled"):
        return priors

    # Imported here so the features layer does not depend on ingest at import
    # time; this is the one place the two need to meet.
    from ..ingest.history import projection_actual_pairs

    weights = {k: float(v)
               for k, v in (cfg.get("blend", {}).get("components") or {}).items()}
    pairs = projection_actual_pairs(
        rankings, totals, crosswalk, list(seasons),
        downweight=cfg.get("backtest", {}).get("downweight"),
        top_n=int(fit_cfg.get("top_n", 40)),
        blend_weights=weights or None,
        opportunity_for=opportunity_for,
    )
    if pairs.empty:
        return priors

    slopes = fit_slopes(
        pairs, projection_col="projection", actual_col="actual",
        priors=priors, min_obs=int(fit_cfg.get("min_obs", 40)),
        prior_weight=int(fit_cfg.get("prior_weight", 0)),
    )

    if verbose:
        seasons_fit = sorted(pairs["season"].unique())
        print(f"  calibration slopes fit on {seasons_fit} "
              f"({len(pairs)} projection/actual pairs, "
              f"{pairs.attrs.get('zero_actual', 0)} paired at zero because the "
              f"player recorded no stats that season):")
        for pos in ("QB", "RB", "WR", "TE", "K", "DEF"):
            n = int((pairs["position"] == pos).sum())
            mark = "prior" if n < int(fit_cfg.get("min_obs", 40)) else "fit"
            print(f"    {pos:<4} {priors.get(pos, 1.0):.2f} -> "
                  f"{slopes.get(pos, 1.0):.2f}  ({mark}, n={n})")

    return slopes
