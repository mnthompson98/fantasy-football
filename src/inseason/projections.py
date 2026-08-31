"""Weekly projections, reconciled to the canonical player key.

FantasyPros publishes a weekly ranking product that — unlike its draft
projections — *does* include point projections (`r2p_pts`), and ffverse
redistributes it. The skill pages are the PPR variants, which is this league's
scoring, so the numbers need no conversion.

What comes with them and is worth keeping:

- `sd` — how much the experts disagree about a player this week. A start/sit
  call between two players 0.4 points apart with an `sd` of 4 is not a call, and
  saying so is more useful than pretending to a decision.
- `player_opponent`, `player_bye_week` — a player on bye projects nothing, and
  the source is explicit about it rather than leaving a silent zero.
- `start_sit_grade`, `tag` — FantasyPros' own summary, carried through as
  context, never as the recommendation.

**The hard limitation:** this feed is a live snapshot of the current week only.
There is no archive, so nothing built on it can be backtested the way the draft
board was. Every in-season recommendation is therefore *unvalidated* — reasoned
from a credible source, but never scored against outcomes. Do not let a
confident-looking table obscure that.
"""

from __future__ import annotations

import pandas as pd

from ..ingest import nflverse as nv
from ..ingest.player_ids import normalize_position, normalize_team, reconcile

# The pages we draft and start from. FantasyPros splits by scoring for the
# skill positions; QB, K and DST have a single page each.
PPR_PAGES = {"ppr-rb": "RB", "ppr-wr": "WR", "ppr-te": "TE",
             "qb": "QB", "k": "K", "dst": "DEF"}

PROJECTION_COL = "projection"


def load_weekly_projections(cw, *, refresh: bool = False,
                            verbose: bool = False) -> pd.DataFrame:
    """This week's projected points per player, on `player_key`.

    Returns columns: `player_key`, `player_name`, `position`, `nfl_team`,
    `projection`, `sd`, `pos_rank`, `opponent`, `bye_week`, `grade`,
    `scrape_date`.
    """
    raw = nv.load_ff_rankings_weekly(refresh=refresh)

    df = raw[raw["page"].isin(PPR_PAGES)].copy()
    if df.empty:
        raise ValueError(
            f"no weekly rankings for pages {sorted(PPR_PAGES)}; got "
            f"{sorted(raw['page'].astype(str).unique())[:10]}"
        )

    df["position"] = df["page"].map(PPR_PAGES)
    df["nfl_team"] = df["team"].map(normalize_team)
    df[PROJECTION_COL] = pd.to_numeric(df.get("r2p_pts"), errors="coerce")
    df["sd"] = pd.to_numeric(df.get("sd"), errors="coerce")

    # DST rows name a team, not a person; their canonical key comes from the
    # team code, exactly as on the draft board.
    matched, unmatched = reconcile(
        df, cw, fantasypros_col="fantasypros_id", name_col="player_name",
        position_col="position",
    )
    if len(unmatched):
        rescued = unmatched[
            (unmatched["position"] == "DEF") & unmatched["nfl_team"].notna()
        ].copy()
        if len(rescued):
            rescued["player_key"] = "DEF_" + rescued["nfl_team"]
            matched = pd.concat([matched, rescued], ignore_index=True)
            unmatched = unmatched.drop(index=rescued.index)

    if verbose and len(unmatched):
        print(f"  [warn] {len(unmatched)} weekly-ranked players unreconciled; "
              f"they will look like zero-projection players")
        for row in unmatched.head(5).itertuples():
            print(f"    {row.player_name} ({row.position})")

    out = matched.rename(columns={
        "player_opponent": "opponent",
        "player_bye_week": "bye_week",
        "start_sit_grade": "grade",
    })
    keep = ["player_key", "player_name", "position", "nfl_team", PROJECTION_COL,
            "sd", "pos_rank", "opponent", "bye_week", "grade", "scrape_date"]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    out = out.drop_duplicates("player_key", keep="first")
    return out[keep].sort_values(PROJECTION_COL, ascending=False).reset_index(drop=True)


def attach_projections(roster: pd.DataFrame, projections: pd.DataFrame
                       ) -> pd.DataFrame:
    """Left-join projections onto a roster, keeping every rostered player.

    A player with no weekly ranking keeps a NaN projection rather than being
    dropped — he is on the roster whether or not FantasyPros ranked him, and
    the lineup optimizer treats NaN as zero, which is the right answer for
    someone who is not playing.
    """
    cols = [c for c in projections.columns if c != "player_name"]
    return roster.merge(projections[cols], on="player_key", how="left",
                        suffixes=("", "_proj"))
