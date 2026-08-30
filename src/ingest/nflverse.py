"""nflverse data access, cached to disk.

`nflreadpy` returns Polars frames; everything downstream of `src/ingest/` is
pandas, so the conversion happens here and only here (see docs/environment.md).

Every pull lands in `data/raw/` as parquet. The cache is what makes the board
buildable on draft day without a network round trip, and what keeps a backtest
reproducible: the same cache file gives the same answer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

RAW_DIR = Path("data/raw")

# Rankings are re-scraped daily by ffverse; anything older than this is stale
# for draft purposes. Season stats for completed seasons never change.
FRESH_HOURS_RANKINGS = 12.0
FRESH_HOURS_STATIC = 24.0 * 7


def _cache_path(name: str, cache_dir: Path | str | None = None) -> Path:
    d = Path(cache_dir) if cache_dir is not None else RAW_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.parquet"


def _cached(name: str, loader: Callable[[], "object"], *,
            max_age_hours: float,
            cache_dir: Path | str | None = None,
            refresh: bool = False) -> pd.DataFrame:
    """Return `loader()` as pandas, memoized on disk.

    A stale cache is preferred to a failed pull: if the network call raises and
    a cache exists at any age, the cache is used and the failure is reported.
    Losing the board on draft morning because ffverse is briefly down is a worse
    outcome than a board built on yesterday's rankings.
    """
    path = _cache_path(name, cache_dir)

    if path.exists() and not refresh:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            return pd.read_parquet(path)

    try:
        frame = loader()
    except Exception as exc:
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600
            print(f"  [warn] {name}: pull failed ({exc}); "
                  f"using cache aged {age_h:.1f}h")
            return pd.read_parquet(path)
        raise

    df = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    df.to_parquet(path, index=False)
    return df


def _seasons_key(seasons: Sequence[int]) -> str:
    s = sorted(set(int(x) for x in seasons))
    return f"{s[0]}_{s[-1]}" if s else "none"


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_ff_rankings(*, cache_dir: Path | str | None = None,
                     refresh: bool = False) -> pd.DataFrame:
    """FantasyPros expert consensus rankings, as redistributed by ffverse.

    This is the market/expert component of the blend. The frame carries every
    ranking product FantasyPros publishes; callers filter by `page_type`.
    Note `scrape_date` — an ECR snapshot more than a day or two old has missed
    preseason news, which is most of what moves a ranking in late August.
    """
    import nflreadpy as nfl
    return _cached("ff_rankings", nfl.load_ff_rankings,
                   max_age_hours=FRESH_HOURS_RANKINGS,
                   cache_dir=cache_dir, refresh=refresh)


def load_ff_rankings_history(*, cache_dir: Path | str | None = None,
                             refresh: bool = False) -> pd.DataFrame:
    """Every ECR snapshot ffverse has kept, dated by `scrape_date`.

    ~1.8M rows going back to 2019-12-27. This is what makes an honest backtest
    possible: for a target season it gives the ranking the market actually held
    *before* that season's draft, rather than a hindsight ranking dressed up as
    a projection. Preseason `redraft-overall` coverage starts in 2021 — 2020 has
    no usable snapshot, so 2020 can be trained on but never targeted.

    Completed snapshots never change, so this caches for a week.
    """
    import nflreadpy as nfl
    return _cached("ff_rankings_history",
                   lambda: nfl.load_ff_rankings(type="all"),
                   max_age_hours=FRESH_HOURS_STATIC,
                   cache_dir=cache_dir, refresh=refresh)


def load_ff_playerids(*, cache_dir: Path | str | None = None,
                      refresh: bool = False) -> pd.DataFrame:
    """DynastyProcess ID crosswalk: gsis, sleeper, fantasypros, pfr, and more."""
    import nflreadpy as nfl
    return _cached("ff_playerids", nfl.load_ff_playerids,
                   max_age_hours=FRESH_HOURS_RANKINGS,
                   cache_dir=cache_dir, refresh=refresh)


def load_ff_opportunity(seasons: Sequence[int], *,
                        cache_dir: Path | str | None = None,
                        refresh: bool = False) -> pd.DataFrame:
    """ffopportunity expected fantasy points (weekly).

    `total_fantasy_points_exp` is computed under **full PPR** (verified: the
    implied value of a reception is exactly 1.0), which matches this league.
    """
    import nflreadpy as nfl
    seasons = [int(s) for s in seasons]
    return _cached(
        f"ff_opportunity_{_seasons_key(seasons)}",
        lambda: nfl.load_ff_opportunity(seasons=seasons, stat_type="weekly"),
        max_age_hours=FRESH_HOURS_STATIC, cache_dir=cache_dir, refresh=refresh,
    )


def load_player_stats(seasons: Sequence[int], *,
                      cache_dir: Path | str | None = None,
                      refresh: bool = False) -> pd.DataFrame:
    """Weekly player stats, including the kicking detail K scoring needs."""
    import nflreadpy as nfl
    seasons = [int(s) for s in seasons]
    return _cached(
        f"player_stats_{_seasons_key(seasons)}",
        lambda: nfl.load_player_stats(seasons=seasons, summary_level="week"),
        max_age_hours=FRESH_HOURS_STATIC, cache_dir=cache_dir, refresh=refresh,
    )


def load_team_stats(seasons: Sequence[int], *,
                    cache_dir: Path | str | None = None,
                    refresh: bool = False) -> pd.DataFrame:
    """Weekly team stats — the defensive side of DEF scoring."""
    import nflreadpy as nfl
    seasons = [int(s) for s in seasons]
    return _cached(
        f"team_stats_{_seasons_key(seasons)}",
        lambda: nfl.load_team_stats(seasons=seasons, summary_level="week"),
        max_age_hours=FRESH_HOURS_STATIC, cache_dir=cache_dir, refresh=refresh,
    )


def load_schedules(seasons: Sequence[int], *,
                   cache_dir: Path | str | None = None,
                   refresh: bool = False) -> pd.DataFrame:
    """Game results and bye weeks. Points allowed for DEF scoring lives here."""
    import nflreadpy as nfl
    seasons = [int(s) for s in seasons]
    return _cached(
        f"schedules_{_seasons_key(seasons)}",
        lambda: nfl.load_schedules(seasons=seasons),
        max_age_hours=FRESH_HOURS_STATIC, cache_dir=cache_dir, refresh=refresh,
    )
