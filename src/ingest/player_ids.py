"""Player identity reconciliation.

Every source names players differently. `nflverse` gsis_id is the canonical key
(CLAUDE.md); FantasyPros ids and Sleeper ids are reconciled *to it* here, so no
downstream module ever joins on a name.

Three wrinkles that make this more than a dict lookup:

1. **Team defenses have no gsis_id.** They are not players. DEF gets a synthetic
   key, `DEF_<team>`, which is stable across every source.
2. **gsis_id is missing for many rookies** — nflverse assigns one after their
   first snap. A player without one keeps a synthetic `FP_<fantasypros_id>` key.
   That is still a real, stable identity; it just isn't a gsis one.
3. **Sleeper's own gsis_id field is mostly empty** (161 of 921 rostered skill
   players as of 2026-08-30), so the crosswalk is anchored on DynastyProcess's
   `ff_playerids`, which carries both, and Sleeper is joined via `sleeper_id`.

Name matching exists only as an explicitly-reported last resort for players the
crosswalk misses. It never runs silently — `reconcile` returns the unmatched set
so the caller can print it and you can eyeball whether it matters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

# Canonical team codes. Sources disagree on a handful of franchises; everything
# else is already consistent. nflverse uses LA/JAC in places, FantasyPros and
# Sleeper use LAR/JAX.
TEAM_ALIASES = {
    "LA": "LAR", "STL": "LAR", "SL": "LAR",
    "JAC": "JAX",
    "OAK": "LV", "LVR": "LV",
    "SD": "LAC",
    "WSH": "WAS", "WFT": "WAS",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",
}

# Sources spell the team-defense position three ways. Everything becomes DEF,
# which is what Sleeper's roster slots use.
DEF_ALIASES = {"DST", "D/ST", "DEF", "D"}

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NONALPHA = re.compile(r"[^a-z ]")


def normalize_team(team: object) -> str | None:
    if team is None or (isinstance(team, float) and pd.isna(team)):
        return None
    t = str(team).strip().upper()
    if not t or t in {"NAN", "NONE", "FA"}:
        return None
    return TEAM_ALIASES.get(t, t)


def normalize_position(pos: object) -> str | None:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)):
        return None
    p = str(pos).strip().upper()
    if not p:
        return None
    return "DEF" if p in DEF_ALIASES else p


def merge_name(name: object) -> str:
    """Aggressive name normalization, for last-resort matching only.

    Strips accents, punctuation, and generational suffixes: "Marvin Harrison Jr."
    and "Marvin Harrison" collapse to the same key. That collision is exactly
    why this is a fallback and not the join key.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace(".", " ").replace("'", "")
    s = _NONALPHA.sub(" ", s)
    s = _SUFFIXES.sub(" ", s)
    return " ".join(s.split())


def canonical_key(*, position: object, team: object,
                  gsis_id: object = None,
                  fantasypros_id: object = None,
                  sleeper_id: object = None) -> str | None:
    """The one identity every downstream table joins on.

    Priority: DEF team code > gsis_id > FantasyPros id > Sleeper id. The order is
    stability, not preference — a gsis_id never changes, a FantasyPros id
    effectively never changes, and a Sleeper id is only used when nothing else
    exists at all.
    """
    if normalize_position(position) == "DEF":
        t = normalize_team(team)
        return f"DEF_{t}" if t else None

    for prefix, value in (("", gsis_id), ("FP_", fantasypros_id),
                          ("SL_", sleeper_id)):
        s = _clean_id(value)
        if s:
            return f"{prefix}{s}"
    return None


def _clean_id(v: object) -> str | None:
    """Ids survive parquet round-trips as floats; "22968.0" is not an id."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if not s or s.lower() in {"nan", "none", "na", "<na>"}:
        return None
    return s


@dataclass
class Crosswalk:
    """Lookup tables from each source's id space into the canonical key."""

    by_fantasypros: dict[str, str] = field(default_factory=dict)
    by_sleeper: dict[str, str] = field(default_factory=dict)
    by_gsis: dict[str, str] = field(default_factory=dict)
    by_name_pos: dict[tuple[str, str], str] = field(default_factory=dict)
    # canonical key -> {gsis_id, sleeper_id, fantasypros_id, name, position, team}
    meta: dict[str, dict] = field(default_factory=dict)

    def resolve(self, *, fantasypros_id=None, sleeper_id=None, gsis_id=None,
                name=None, position=None) -> str | None:
        """First hit wins, in descending order of trustworthiness."""
        for table, value in (
            (self.by_gsis, gsis_id),
            (self.by_fantasypros, fantasypros_id),
            (self.by_sleeper, sleeper_id),
        ):
            s = _clean_id(value)
            if s is None:
                continue
            hit = table.get(s)
            if hit:
                return hit

        pos = normalize_position(position)
        if name is not None and pos:
            return self.by_name_pos.get((merge_name(name), pos))
        return None


def build_crosswalk(ff_playerids: pd.DataFrame,
                    sleeper_players: dict[str, dict] | None = None
                    ) -> Crosswalk:
    """Build the crosswalk from DynastyProcess ids, enriched with Sleeper.

    `ff_playerids` carries one row per player-season (`db_season`); we keep the
    most recent row per player so a traded player shows his current team.
    """
    df = ff_playerids.copy()
    if "db_season" in df.columns and "mfl_id" in df.columns:
        df = df.sort_values("db_season").drop_duplicates("mfl_id", keep="last")

    cw = Crosswalk()

    for row in df.itertuples(index=False):
        pos = normalize_position(getattr(row, "position", None))
        team = normalize_team(getattr(row, "team", None))
        gsis = _clean_id(getattr(row, "gsis_id", None))
        fp = _clean_id(getattr(row, "fantasypros_id", None))
        sl = _clean_id(getattr(row, "sleeper_id", None))
        name = getattr(row, "name", None)

        key = canonical_key(position=pos, team=team, gsis_id=gsis,
                            fantasypros_id=fp, sleeper_id=sl)
        if not key:
            continue

        if gsis:
            cw.by_gsis[gsis] = key
        if fp:
            cw.by_fantasypros[fp] = key
        if sl:
            cw.by_sleeper[sl] = key
        if name is not None and pos:
            cw.by_name_pos.setdefault((merge_name(name), pos), key)

        cw.meta[key] = {
            "player_key": key, "gsis_id": gsis, "sleeper_id": sl,
            "fantasypros_id": fp, "player_name": name,
            "position": pos, "nfl_team": team,
        }

    if sleeper_players:
        _enrich_from_sleeper(cw, sleeper_players)

    return cw


def _enrich_from_sleeper(cw: Crosswalk, sleeper_players: dict[str, dict]) -> None:
    """Add Sleeper-only entities (team defenses, unlisted rookies) to the map.

    Sleeper is also the authority on *current* team and injury status, both of
    which move after ffverse's nightly scrape and both of which matter more on
    draft day than anything else in the row.
    """
    for sleeper_id, p in sleeper_players.items():
        pos = normalize_position(p.get("position"))
        if pos is None:
            fps = p.get("fantasy_positions") or []
            pos = normalize_position(fps[0]) if fps else None
        if pos is None:
            continue

        team = normalize_team(p.get("team"))
        gsis = _clean_id(p.get("gsis_id"))
        name = p.get("full_name") or (
            f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip())

        key = cw.resolve(sleeper_id=sleeper_id, gsis_id=gsis)
        if key is None and pos == "DEF":
            key = canonical_key(position=pos, team=team or sleeper_id)
        if key is None:
            key = canonical_key(position=pos, team=team, gsis_id=gsis,
                                sleeper_id=sleeper_id)
        if not key:
            continue

        cw.by_sleeper[str(sleeper_id)] = key
        if gsis:
            cw.by_gsis[gsis] = key
        if name and pos:
            cw.by_name_pos.setdefault((merge_name(name), pos), key)

        meta = cw.meta.setdefault(
            key, {"player_key": key, "gsis_id": gsis, "fantasypros_id": None,
                  "position": pos})
        meta["sleeper_id"] = str(sleeper_id)
        if team:
            meta["nfl_team"] = team
        meta["injury_status"] = (p.get("injury_status") or "").strip() or None
        meta["sleeper_status"] = p.get("status")
        meta["years_exp"] = p.get("years_exp")
        if not meta.get("player_name"):
            meta["player_name"] = name or str(sleeper_id)


def crosswalk_frame(cw: Crosswalk) -> pd.DataFrame:
    """The crosswalk as a table, for joining and for eyeballing."""
    return pd.DataFrame(list(cw.meta.values()))


def reconcile(df: pd.DataFrame, cw: Crosswalk, *,
              fantasypros_col: str | None = None,
              sleeper_col: str | None = None,
              gsis_col: str | None = None,
              name_col: str | None = None,
              position_col: str | None = None,
              key_col: str = "player_key") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the canonical key to `df`.

    Returns `(matched, unmatched)`. The caller is expected to look at
    `unmatched` — a silent drop here is how a top-50 rookie quietly vanishes
    from the board.
    """
    out = df.copy()

    if not len(out):
        out[key_col] = pd.Series(dtype=object)
        return out, out.copy()

    def _key(row) -> str | None:
        return cw.resolve(
            fantasypros_id=row[fantasypros_col] if fantasypros_col else None,
            sleeper_id=row[sleeper_col] if sleeper_col else None,
            gsis_id=row[gsis_col] if gsis_col else None,
            name=row[name_col] if name_col else None,
            position=row[position_col] if position_col else None,
        )

    out[key_col] = out.apply(_key, axis=1)
    matched = out[out[key_col].notna()].copy()
    unmatched = out[out[key_col].isna()].copy()
    return matched, unmatched
