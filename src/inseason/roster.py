"""Who owns whom, right now, from Sleeper.

Every in-season module needs the same three things: my roster, the rest of the
league's rosters (so "available" means genuinely available), and the free-agent
pool that is what remains. Built once here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..ingest.player_ids import normalize_position
from ..ingest.sleeper_api import (
    UNAVAILABLE_STATUSES,
    get_league,
    get_league_users,
    get_rosters,
)


@dataclass
class LeagueState:
    """A snapshot of who owns whom."""

    my_roster: pd.DataFrame
    all_rostered: set = field(default_factory=set)
    owner_of: dict = field(default_factory=dict)   # player_key -> display name
    my_roster_id: int | None = None
    teams: int = 10

    def free_agents(self, universe: pd.DataFrame) -> pd.DataFrame:
        """Everyone in `universe` that nobody in the league rosters."""
        return universe[~universe["player_key"].isin(self.all_rostered)].copy()


def _players_frame(keys: list[str], cw) -> pd.DataFrame:
    rows = []
    for key in keys:
        meta = cw.meta.get(key, {})
        rows.append({
            "player_key": key,
            "player_id": meta.get("sleeper_id"),
            "player_name": meta.get("player_name") or key,
            "position": normalize_position(meta.get("position")),
            "nfl_team": meta.get("nfl_team"),
            "injury_status": meta.get("injury_status"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        status = df["injury_status"].fillna("").astype(str).str.strip()
        df["available"] = ~status.isin(UNAVAILABLE_STATUSES)
    return df


def load_league_state(league_id: str, cw, *, user_id: str) -> LeagueState:
    """Fetch every roster and work out which one is ours.

    Sleeper keys rosters by `owner_id`, which is the stable user id — not the
    display name, which changes between seasons. `config.user.sleeper_user_id`
    is the join key for exactly that reason.
    """
    league = get_league(league_id) or {}
    rosters = get_rosters(league_id)
    users = {u["user_id"]: u.get("display_name") for u in get_league_users(league_id)}

    all_rostered: set = set()
    owner_of: dict = {}
    mine: list[str] = []
    my_roster_id = None

    for r in rosters:
        owner = r.get("owner_id")
        name = users.get(owner, owner)
        # Sleeper's `players` is every rostered player; `starters` is the
        # current lineup. We want the whole roster, bench included.
        sleeper_ids = r.get("players") or []
        keys = []
        for sid in sleeper_ids:
            key = cw.resolve(sleeper_id=sid)
            if key:
                keys.append(key)
                owner_of[key] = name
        all_rostered.update(keys)
        if owner == user_id:
            mine = keys
            my_roster_id = r.get("roster_id")

    return LeagueState(
        my_roster=_players_frame(mine, cw),
        all_rostered=all_rostered,
        owner_of=owner_of,
        my_roster_id=my_roster_id,
        teams=int(league.get("total_rosters") or len(rosters) or 10),
    )
