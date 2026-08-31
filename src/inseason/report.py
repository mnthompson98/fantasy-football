"""The weekly markdown report.

The plan calls for a scheduled job that "emits a markdown report". This builds
it. Markdown rather than a terminal dump because the point is to read it on a
phone on Tuesday night, and to still have it on Sunday morning when you are
actually setting the lineup.

Structure follows what you act on, in the order you act on it: set the lineup,
decide the waiver claim, then everything else. Uncertainty and provenance are in
the report rather than in a footnote — a recommendation you cannot audit at 9am
Sunday is a recommendation you will ignore.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ..features.lineup import Lineup, LineupSlots
from ..ingest.injuries import concern
from . import waivers as waivers_mod


def _text(value) -> str:
    """Empty string for anything absent.

    A pandas merge leaves NaN where there was no match, and `str(nan)` is the
    four characters "nan" — which is how a lineup table ends up with a Notes
    column full of them.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else text


def _fmt(value, digits: int = 1, dash: str = "—") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    return dash if f != f else f"{f:.{digits}f}"


def build(*, week: int | None, season: int, lineup: Lineup,
          calls: list, holes: list[str], slots: LineupSlots,
          moves: list, waiver_threshold: float,
          surplus: pd.DataFrame, roster: pd.DataFrame,
          scrape_date: str | None = None,
          league_name: str = "") -> str:
    """Render the week's decisions as markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    wk = f"Week {week}" if week else "Preseason"
    lines: list[str] = [
        f"# {wk} — {league_name or 'fantasy'} ({season})",
        "",
        f"*Generated {now}. Projections scraped {scrape_date or 'unknown'}.*",
        "",
        "> These recommendations are **unvalidated**. The draft board was "
        "backtested over four seasons; the weekly FantasyPros feed is a live "
        "snapshot with no archive, so nothing below can be scored against "
        "history. Treat it as a well-sourced second opinion.",
        "",
        "## Lineup",
        "",
        "| Slot | Player | Proj | Opp | Notes |",
        "|---|---|---:|---|---|",
    ]

    for row in lineup.starters.itertuples():
        note_bits = []
        c = _text(getattr(row, "concern", None))
        if c and c != "clear":
            note_bits.append(f"**{c}**")
        flags = _text(getattr(row, "flags", None))
        if flags:
            note_bits.append(flags)
        lines.append(
            f"| {row.slot} | {row.player_name} | "
            f"{_fmt(getattr(row, 'projection', None))} | "
            f"{_text(getattr(row, 'opponent', None))} | "
            f"{', '.join(note_bits)} |"
        )
    lines += ["", f"**Projected total: {lineup.points:.1f}**", ""]

    if holes:
        lines += [
            f"> ⚠️ **Unfilled: {', '.join(holes)}.** Bye or injury — fill from "
            f"free agency before kickoff.", "",
        ]

    tight = [c for c in calls if c.gap < 6][:5]
    if tight:
        lines += ["### Close calls", "",
                  "| Slot | Start | Over | Gap | Verdict |",
                  "|---|---|---|---:|---|"]
        for call in tight:
            verdict = "toss-up" if call.tossup else "start"
            lines.append(
                f"| {call.slot} | {call.start} ({_fmt(call.start_points)}) | "
                f"{call.sit} ({_fmt(call.sit_points)}) | "
                f"+{call.gap:.1f} | {verdict} (sd {call.sigma:.1f}) |")
        lines.append("")

    lines += ["## Waivers", "",
              "*Rolling priority: a claim costs your place in the queue. "
              f"Threshold is {waiver_threshold:.1f} points added to the "
              "starting lineup.*", ""]
    if moves:
        lines += ["| Add | Pos | Proj | Lineup + | Cost | Trending |",
                  "|---|---|---:|---:|---|---:|"]
        for m in moves:
            cost = f"drop {m.drop}" if m.drop else "roster spot free"
            kind = "**CLAIM**" if m.burns_priority else "free agent"
            lines.append(
                f"| {m.add} | {m.position} | {_fmt(m.add_points)} | "
                f"{m.net:+.1f} | {cost} | {m.trending_adds or ''} |"
                f"".replace("| " + m.add + " |", f"| {m.add} ({kind}) |", 1))
        lines.append("")
    lines += [waivers_mod.summarize(moves, threshold=waiver_threshold), ""]

    if not surplus.empty:
        lines += ["## Trade bait", "",
                  "*Costs your starting lineup nothing this week.*", "",
                  "| Player | Pos | Proj |", "|---|---|---:|"]
        for row in surplus.head(6).itertuples():
            lines.append(f"| {row.player_name} | {row.position} | "
                         f"{_fmt(getattr(row, 'projection', None))} |")
        lines.append("")

    hurt = roster[roster.get("concern", pd.Series(dtype=object)).isin(
        ("out", "doubtful", "risky", "monitor"))] if "concern" in roster else None
    if hurt is not None and not hurt.empty:
        lines += ["## Injury report", "",
                  "| Player | Pos | Status | Practice | Injury |",
                  "|---|---|---|---|---|"]
        for row in hurt.itertuples():
            lines.append(
                f"| {row.player_name} | {row.position} | "
                f"{_text(getattr(row, 'report_status', None)) or '—'} | "
                f"{_text(getattr(row, 'practice_status', None)) or '—'} | "
                f"{_text(getattr(row, 'injury', None)) or '—'} |")
        lines.append("")

    return "\n".join(lines)
