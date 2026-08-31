"""Game context for the week: opponent, betting market, and weather.

**This is context, not an adjustment, and the distinction is the whole point.**

The obvious move is to nudge projections up for players in high-total games and
down for players in the wind. Don't. The FantasyPros projections these modules
run on are made by people who have already seen the spread, the total and the
forecast — adjusting on top of them double-counts the matchup and makes the
number worse, not better. The market line is *already inside* the projection.

What the context is genuinely good for is explaining a number you are about to
act on, and flagging the cases where the projection may not have caught up:
a kicker in a 25 mph wind, or a game whose total moved after the rankings were
scraped. So this module surfaces flags and leaves the arithmetic alone.

Implied team total is the useful single number: half the game total, adjusted by
half the spread. It is what the market thinks a team will score, and it bounds
what its players can plausibly do.
"""

from __future__ import annotations

import pandas as pd

from ..ingest import nflverse as nv
from ..ingest.player_ids import normalize_team

# Above this, kicking and the deep passing game are measurably suppressed.
WINDY_MPH = 15.0

# Cold enough to matter for kickers and outdoor passing games.
COLD_F = 25.0

# A team the market expects to score this little is a bad place to own players.
LOW_IMPLIED_TOTAL = 18.0


def week_context(season: int, week: int, *, refresh: bool = False
                 ) -> pd.DataFrame:
    """One row per team for the given week.

    Columns: `nfl_team`, `opponent`, `home`, `spread`, `total_line`,
    `implied_total`, `temp`, `wind`, `roof`, `flags`.
    """
    sched = nv.load_schedules([int(season)], refresh=refresh)
    games = sched[(pd.to_numeric(sched["week"], errors="coerce") == int(week))]
    if games.empty:
        return pd.DataFrame()

    rows = []
    for g in games.itertuples():
        total = pd.to_numeric(pd.Series([getattr(g, "total_line", None)]),
                              errors="coerce").iloc[0]
        # nflverse states `spread_line` from the home team's perspective:
        # positive means the home team is favoured by that many points.
        spread = pd.to_numeric(pd.Series([getattr(g, "spread_line", None)]),
                               errors="coerce").iloc[0]

        for team, opp, is_home in (
            (g.home_team, g.away_team, True),
            (g.away_team, g.home_team, False),
        ):
            own_spread = spread if is_home else (-spread if spread == spread else spread)
            implied = None
            if total == total and own_spread == own_spread:
                implied = float(total) / 2.0 + float(own_spread) / 2.0

            rows.append({
                "nfl_team": normalize_team(team),
                "opponent": normalize_team(opp),
                "home": is_home,
                "spread": float(own_spread) if own_spread == own_spread else None,
                "total_line": float(total) if total == total else None,
                "implied_total": implied,
                "temp": getattr(g, "temp", None),
                "wind": getattr(g, "wind", None),
                "roof": getattr(g, "roof", None),
            })

    df = pd.DataFrame(rows)
    df["flags"] = [_flags(r) for _, r in df.iterrows()]
    return df


def _flags(row) -> str:
    """Short, human flags. Empty string when there is nothing worth saying."""
    out: list[str] = []
    roof = str(row.get("roof") or "").lower()
    outdoors = roof in ("outdoors", "open", "")

    wind = pd.to_numeric(pd.Series([row.get("wind")]), errors="coerce").iloc[0]
    temp = pd.to_numeric(pd.Series([row.get("temp")]), errors="coerce").iloc[0]

    if outdoors and wind == wind and wind >= WINDY_MPH:
        out.append(f"wind {wind:.0f}mph")
    if outdoors and temp == temp and temp <= COLD_F:
        out.append(f"{temp:.0f}°F")

    implied = row.get("implied_total")
    if implied is not None and implied == implied:
        if implied <= LOW_IMPLIED_TOTAL:
            out.append(f"low total {implied:.1f}")
        elif implied >= 27.0:
            out.append(f"high total {implied:.1f}")
    return ", ".join(out)


def attach(players: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Join week context onto a player frame by team."""
    if context.empty or "nfl_team" not in players.columns:
        return players
    cols = ["nfl_team", "opponent", "implied_total", "spread", "total_line",
            "temp", "wind", "roof", "flags"]
    return players.merge(context[cols], on="nfl_team", how="left",
                         suffixes=("", "_ctx"))
