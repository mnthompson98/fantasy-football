"""The official injury report, keyed canonically.

Two sources, covering different phases, and neither replaces the other:

- **Sleeper** carries season-long designations — IR, PUP, NFI, suspended — and
  is the only one that exists *before* the season. It is what gates the draft
  board, and it is why `build_draft_board` reaches for Sleeper and not here.
- **nflverse** carries the weekly NFL injury report: Wednesday-through-Friday
  practice participation and the Friday game status. It starts at week 1 and is
  the authority once games are being played.

The practice column is the part people ignore and shouldn't. "Questionable" is
close to a coin flip on its own; "Questionable and did not practice all week" is
not, and the two are indistinguishable if you only read the game status.
"""

from __future__ import annotations

import pandas as pd

from . import nflverse as nv

# Game statuses that mean he should not be in your lineup. Doubtful is included
# deliberately: roughly a quarter of Doubtful players suit up, and the ones who
# do are usually limited enough to be a wasted slot.
OUT_STATUSES = {"Out", "Doubtful"}

# Practice participation, worst to best. A player who never practised is a
# different proposition from one who was limited on Wednesday and full by
# Friday, whatever the Friday status says.
DNP = "Did Not Participate In Practice"
LIMITED = "Limited Participation in Practice"
FULL = "Full Participation in Practice"


def weekly_report(season: int, week: int | None = None, *,
                  refresh: bool = False) -> pd.DataFrame:
    """The most recent injury report, one row per player.

    `week=None` takes the latest week present, which is what you want on a
    Tuesday or Wednesday when the current week's report is still filling in.
    Returns `player_key` (gsis id), `report_status`, `practice_status`,
    `injury`, `report_week`.
    """
    try:
        raw = nv.load_injuries([int(season)], refresh=refresh)
    except Exception as exc:
        # Before week 1 there is no report at all, and nflverse raises rather
        # than returning empty. That is not an error condition for us.
        return pd.DataFrame(columns=["player_key", "report_status",
                                     "practice_status", "injury", "report_week"])

    if raw.empty:
        return pd.DataFrame(columns=["player_key", "report_status",
                                     "practice_status", "injury", "report_week"])

    df = raw.copy()
    df["report_week"] = pd.to_numeric(df["week"], errors="coerce")
    if week is not None:
        df = df[df["report_week"] == int(week)]
    else:
        df = df[df["report_week"] == df["report_week"].max()]

    df["player_key"] = df["gsis_id"].astype("string")
    df["injury"] = df["report_primary_injury"].fillna(
        df.get("practice_primary_injury"))

    out = df[["player_key", "report_status", "practice_status", "injury",
              "report_week"]]
    return out.dropna(subset=["player_key"]).drop_duplicates(
        "player_key", keep="last").reset_index(drop=True)


def concern(report_status: object, practice_status: object) -> str:
    """A one-word read combining game status and practice participation.

    `out` he will not play. `doubtful` treat as out. `risky` he is questionable
    *and* did not practise, which is the case people misread most often.
    `monitor` questionable but practising. `clear` otherwise.
    """
    status = str(report_status or "").strip()
    practice = str(practice_status or "").strip()

    if status == "Out":
        return "out"
    if status == "Doubtful":
        return "doubtful"
    if status == "Questionable":
        return "risky" if practice == DNP else "monitor"
    if practice == DNP:
        # No game status yet — early in the week this is the only signal there
        # is, and a full week of missed practice usually becomes a designation.
        return "monitor"
    return "clear"


def apply_to_roster(roster: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """Attach the report and mark who cannot play.

    `available` is anded, never overwritten: a player Sleeper has on IR stays
    unavailable even if the weekly report has nothing to say about him.
    """
    out = roster.merge(report, on="player_key", how="left")
    out["concern"] = [
        concern(r, p) for r, p in zip(out.get("report_status"),
                                      out.get("practice_status"))
    ]
    playable = ~out["report_status"].fillna("").isin(OUT_STATUSES)
    if "available" in out.columns:
        out["available"] = out["available"].fillna(True) & playable
    else:
        out["available"] = playable
    return out
