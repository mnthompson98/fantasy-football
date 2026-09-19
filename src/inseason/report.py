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
          drops: pd.DataFrame | None = None,
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

    if drops is not None and not drops.empty:
        lines += ["### Dropped this week by other teams", "",
                  "*Already in the pool above; shown separately because a "
                  "dropped player is a claim until he clears waivers, and "
                  "free after.*", "",
                  "| Player | Pos | Proj | Lineup + | Dropped by | Status | Verdict |",
                  "|---|---|---:|---:|---|---|---|"]
        for r in drops.itertuples(index=False):
            status = (f"on waivers until {r.on_waivers_until:%a %H:%M} UTC"
                      if r.on_waivers else "free agent")
            verdict = ("**worth it**" if r.worth_it
                       else ("not worth priority" if r.on_waivers and r.net > 0
                             else "no"))
            lines.append(
                f"| {r.player_name} | {r.position} | {_fmt(r.add_points)} | "
                f"{r.net:+.1f} | {_text(r.dropped_by)} | {status} | {verdict} |")
        lines.append("")

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


# --------------------------------------------------------------------------
# The brief — what an automated reader relays, status first
# --------------------------------------------------------------------------

# Anything past this and the rankings are not "this week's" in any useful
# sense: ffverse re-scrapes daily, so two missed days means the feed is stuck.
STALE_RANKINGS_DAYS = 2


def freshness_issues(*, scrape_date: str | None, last_kickoff_utc,
                     fallbacks: list[dict], week: int | None,
                     report_week: int | None,
                     now: datetime | None = None) -> list[str]:
    """Reasons to distrust this run's inputs, in plain words, worst first.

    An empty list means every input was live. Each string is meant to be
    read verbatim by whoever (or whatever) relays the brief, so it names the
    input and the age rather than a code.

    Three distinct failure modes, because they hide differently:

    - A **pull that fell back to cache** printed one `[warn]` line hundreds of
      lines ago. `nflverse.FALLBACKS` remembers it.
    - A **feed that pulled fine but has not moved** looks perfectly healthy to
      the cache. `scrape_date` catches ffverse being stuck; the last kickoff
      catches the harder case where the file is fresh but ranks a week whose
      games have already been played.
    - An **injury report for the wrong week** is expected early in the week
      and is informational, not an error; it is listed last and without the
      "stale" label so a reader does not escalate it.
    """
    now = now or datetime.now(timezone.utc)
    issues: list[str] = []

    for fb in fallbacks:
        issues.append(
            f"STALE CACHE: {fb['name']} could not be pulled "
            f"({fb['error']}); used a cached copy aged {fb['age_hours']:.1f}h")

    if last_kickoff_utc is not None and last_kickoff_utc == last_kickoff_utc:
        if now > last_kickoff_utc:
            issues.append(
                f"STALE RANKINGS: every game in the weekly feed has already "
                f"kicked off (last kickoff {last_kickoff_utc:%Y-%m-%d %H:%M UTC}) "
                f"— these are last week's rankings")

    if scrape_date:
        try:
            scraped = datetime.strptime(str(scrape_date)[:10], "%Y-%m-%d")
            age_days = (now.date() - scraped.date()).days
            if age_days > STALE_RANKINGS_DAYS:
                issues.append(
                    f"STALE RANKINGS: weekly feed last scraped {scrape_date} "
                    f"({age_days} days ago)")
        except ValueError:
            issues.append(f"weekly feed scrape date unreadable: {scrape_date!r}")
    else:
        issues.append("weekly feed carries no scrape date")

    if week and report_week and report_week != week:
        issues.append(
            f"injury report is week {report_week}'s; week {week}'s is not "
            f"published yet (normal before Wednesday afternoon)")
    return issues


def brief(*, week: int | None, season: int, issues: list[str],
          lineup: Lineup | None = None, calls: list | None = None,
          holes: list[str] | None = None, moves: list | None = None,
          waiver_threshold: float | None = None,
          surplus: pd.DataFrame | None = None,
          roster: pd.DataFrame | None = None,
          drops: pd.DataFrame | None = None,
          scrape_date: str | None = None,
          report_path: str | None = None,
          failed: str | None = None) -> str:
    """The short form: status on line one, then only what changes a decision.

    Built for a reader that relays rather than reads — a scheduled agent, or
    you on a phone. The full report has every starter and every table; this
    has the verdicts. The status line is first and unconditional so "the
    inputs were stale" can never be buried under a tidy lineup.

    `failed` short-circuits everything: the brief then says the run refused
    and why, and nothing else, so a partial result cannot be mistaken for a
    recommendation.
    """
    wk = f"Week {week}" if week else "Preseason"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    if failed:
        lines += [f"# STATUS: FAILED — {wk} ({season})", "",
                  f"weekly_update refused to advise: {failed}", "",
                  f"*Generated {now}.* No recommendations were produced."]
        return "\n".join(lines)

    stale = [i for i in issues if i.startswith("STALE")]
    status = "DEGRADED" if stale else "OK"
    lines += [f"# STATUS: {status} — {wk} ({season})", ""]
    if issues:
        lines += [("**Read this first:**" if stale else "Notes:"), ""]
        lines += [f"- {i}" for i in issues]
        lines.append("")
    lines += [f"*Generated {now}. Rankings scraped {scrape_date or 'unknown'}."
              + (f" Full report: `{report_path}`." if report_path else "")
              + "*", ""]

    lines += ["## Start / sit", ""]
    if lineup is not None:
        lines.append(f"- Projected total **{lineup.points:.1f}**")
    if holes:
        lines.append(f"- **UNFILLED: {', '.join(holes)}** — bye or injury; "
                     f"fill from free agency before kickoff")
    tight = [c for c in (calls or []) if c.gap < 6][:5]
    if tight:
        for c in tight:
            verdict = "toss-up" if c.tossup else "start"
            lines.append(
                f"- {c.slot}: {c.start} ({_fmt(c.start_points)}) over "
                f"{c.sit} ({_fmt(c.sit_points)}), +{c.gap:.1f} — {verdict}")
    else:
        lines.append("- No close calls; the lineup sets itself this week.")
    hurt = None
    if roster is not None and "concern" in roster:
        hurt = roster[roster["concern"].isin(("out", "doubtful", "risky",
                                              "monitor"))]
    if hurt is not None and not hurt.empty:
        names = ", ".join(
            f"{r.player_name} ({_text(getattr(r, 'concern', ''))})"
            for r in hurt.itertuples())
        lines.append(f"- Injury flags on roster: {names}")
    lines.append("")

    lines += ["## Waivers (rolling priority — nothing is submitted for you)",
              ""]
    moves = moves or []
    claims = [m for m in moves if m.burns_priority]
    free = [m for m in moves if not m.burns_priority]
    for m in claims:
        cost = f", drop {m.drop}" if m.drop else ""
        lines.append(f"- **CLAIM candidate (burns priority):** {m.add} "
                     f"({m.position}) {m.net:+.1f} to the lineup{cost}")
    for m in free[:3]:
        cost = f", drop {m.drop}" if m.drop else ""
        lines.append(f"- Free add: {m.add} ({m.position}) "
                     f"{m.net:+.1f}{cost}")
    if waiver_threshold is not None:
        lines.append(f"- Verdict: "
                     f"{waivers_mod.summarize(moves, threshold=waiver_threshold)}")
    lines.append("")

    lines += ["## Dropped this week by other teams", ""]
    if drops is None:
        lines.append("- Not checked this run.")
    elif drops.empty:
        lines.append("- Nobody dropped anyone in the window.")
    else:
        worth = drops[drops["worth_it"]]
        for r in worth.itertuples(index=False):
            how = ("**CLAIM** (burns priority; on waivers until "
                   f"{r.on_waivers_until:%a %H:%M} UTC)" if r.on_waivers
                   else "free agent now")
            cost = f", drop {r.drop}" if r.drop else ""
            lines.append(f"- **Worth it:** {r.player_name} ({r.position}) "
                         f"{r.net:+.1f} to the lineup — {how}{cost}; "
                         f"dropped by {_text(r.dropped_by)}")
        rest = drops[~drops["worth_it"]]
        if not rest.empty:
            names = ", ".join(
                f"{r.player_name} ({r.position}, {r.net:+.1f})"
                for r in rest.head(8).itertuples(index=False))
            lines.append(f"- Not worth it: {names}"
                         + (f" and {len(rest) - 8} more" if len(rest) > 8 else ""))
    lines.append("")

    lines += ["## Trade flags", ""]
    if surplus is not None and not surplus.empty:
        names = ", ".join(
            f"{r.player_name} ({r.position} {_fmt(getattr(r, 'projection', None))})"
            for r in surplus.head(4).itertuples())
        lines.append(f"- Surplus the lineup does not need this week: {names}")
    else:
        lines.append("- Nothing spare; every rostered player is doing work.")
    lines.append("")

    lines += ["> Unvalidated, like every in-season output here: the weekly "
              "feed has no archive to score against. Second opinion, not an "
              "edge."]
    return "\n".join(lines)
