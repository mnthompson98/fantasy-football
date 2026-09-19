"""A daily injury snapshot of *your* roster, and what changed since yesterday.

Two sources, the same two `ingest.injuries` describes, read for the same
reason: Sleeper carries the designation (Questionable, Out, IR) every day of
the year and updates it as news breaks; the NFL's official report carries
practice participation Wednesday to Friday, which is the part that separates
a coin-flip Questionable from a probable scratch. Neither replaces the other,
so a snapshot holds both.

The output is a *diff*. A list of sixteen statuses, most of them fine, read
every morning trains the reader to stop reading; "Chase Brown went from
clear to Questionable (hamstring), DNP Wednesday" does not. The full current
picture follows the changes, for the mornings it is needed.

Snapshots are kept in one JSON file so the diff needs no history service:
each run reads the last, writes its own. The first run has nothing to diff
against and says so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..ingest.injuries import concern

# Sleeper's designations, worst last. Anything not listed is "clear" — a
# player with no designation, or one Sleeper spells in a way this list
# does not know, which is then reported verbatim rather than guessed at.
SEVERITY = ["", "Questionable", "Doubtful", "Out", "IR", "PUP", "NA", "Sus",
            "DNR", "COV"]

FIELDS = ("designation", "report_status", "practice", "body_part")


@dataclass
class PlayerStatus:
    player_key: str
    player_name: str
    position: str
    nfl_team: str | None = None
    designation: str = ""          # Sleeper injury_status
    body_part: str = ""            # Sleeper injury_body_part
    notes: str = ""                # Sleeper injury_notes
    report_status: str = ""        # NFL report game status
    practice: str = ""             # NFL report practice participation
    injury: str = ""               # NFL report primary injury
    news_updated: str | None = None
    concern: str = "clear"

    @property
    def severity(self) -> int:
        d = self.designation
        return SEVERITY.index(d) if d in SEVERITY else len(SEVERITY)

    @property
    def flagged(self) -> bool:
        return bool(self.designation) or self.concern != "clear"

    def summary(self) -> str:
        bits = []
        if self.designation:
            bits.append(self.designation)
        if self.body_part and self.body_part.lower() != "undisclosed":
            bits.append(self.body_part.lower())
        elif self.injury:
            bits.append(str(self.injury).lower())
        if self.practice:
            bits.append(_short_practice(self.practice))
        if self.report_status and self.report_status != self.designation:
            bits.append(f"report: {self.report_status}")
        return ", ".join(bits) if bits else "clear"


def _short_practice(text: str) -> str:
    t = str(text)
    if t.startswith("Did Not"):
        return "DNP"
    if t.startswith("Limited"):
        return "limited practice"
    if t.startswith("Full"):
        return "full practice"
    return t


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "<na>") else text


def build_snapshot(roster: pd.DataFrame, sleeper_players: dict,
                   report: pd.DataFrame) -> dict[str, PlayerStatus]:
    """One `PlayerStatus` per rostered player, from both sources.

    `roster` needs `player_key`, `player_id` (Sleeper), `player_name`,
    `position`; `sleeper_players` is the raw Sleeper player map; `report`
    is `ingest.injuries.weekly_report` (may be empty).
    """
    rep = report.set_index("player_key") if not report.empty else None
    out: dict[str, PlayerStatus] = {}
    for r in roster.itertuples(index=False):
        p = sleeper_players.get(str(getattr(r, "player_id", "") or ""), {}) or {}
        row = rep.loc[r.player_key] if rep is not None and r.player_key in rep.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        practice = _clean(row["practice_status"]) if row is not None else ""
        if not practice:
            practice = _clean(p.get("practice_participation"))
        report_status = _clean(row["report_status"]) if row is not None else ""
        news = p.get("news_updated")
        status = PlayerStatus(
            player_key=str(r.player_key),
            player_name=str(r.player_name),
            position=str(r.position),
            nfl_team=_clean(getattr(r, "nfl_team", None)) or None,
            designation=_clean(p.get("injury_status")),
            body_part=_clean(p.get("injury_body_part")),
            notes=_clean(p.get("injury_notes")),
            report_status=report_status,
            practice=practice,
            injury=_clean(row["injury"]) if row is not None else "",
            news_updated=(datetime.fromtimestamp(int(news) / 1000, tz=timezone.utc)
                          .isoformat() if news else None),
        )
        status.concern = concern(status.report_status or status.designation,
                                 status.practice)
        out[status.player_key] = status
    return out


@dataclass
class Change:
    player: PlayerStatus
    kind: str                       # new | worse | better | changed | added | gone
    before: str
    after: str

    def describe(self) -> str:
        p = self.player
        head = f"{p.player_name} ({p.position})"
        if self.kind == "added":
            return f"{head}: joined your roster — {self.after}"
        if self.kind == "gone":
            return f"{head}: no longer on your roster"
        arrow = {"new": "NEW", "worse": "WORSE", "better": "BETTER",
                 "changed": "CHANGED"}[self.kind]
        line = f"{head}: {arrow} — {self.before or 'clear'} → {self.after}"
        if p.notes:
            line += f" ({p.notes[:140]})"
        return line


def diff(previous: dict[str, PlayerStatus] | None,
         current: dict[str, PlayerStatus]) -> list[Change]:
    """What moved since the last snapshot, worst first.

    A change in any of `FIELDS` counts. Severity direction comes from the
    Sleeper designation; a practice-only change (DNP → limited) with the
    designation unchanged is reported as "changed" with both strings, so
    the reader sees the practice trend without a verdict being invented
    for it.
    """
    if previous is None:
        return []
    changes: list[Change] = []
    for key, now in current.items():
        was = previous.get(key)
        if was is None:
            if now.flagged:
                changes.append(Change(now, "added", "", now.summary()))
            continue
        if all(getattr(was, f) == getattr(now, f) for f in FIELDS):
            continue
        if now.severity > was.severity:
            kind = "new" if not was.designation else "worse"
        elif now.severity < was.severity:
            kind = "better"
        else:
            kind = "changed"
        changes.append(Change(now, kind, was.summary(), now.summary()))
    for key, was in previous.items():
        if key not in current and was.flagged:
            changes.append(Change(was, "gone", was.summary(), ""))
    order = {"new": 0, "worse": 0, "added": 1, "changed": 2, "better": 3,
             "gone": 4}
    changes.sort(key=lambda c: (order[c.kind], -c.player.severity,
                                c.player.player_name))
    return changes


def load_snapshot(path: Path) -> tuple[dict[str, PlayerStatus] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        players = {k: PlayerStatus(**v) for k, v in data["players"].items()}
        return players, data.get("taken")
    except (ValueError, KeyError, TypeError) as exc:
        print(f"  [warn] could not read {path}: {exc}; treating as first run")
        return None, None


def save_snapshot(path: Path, snapshot: dict[str, PlayerStatus],
                  taken: datetime) -> None:
    payload = {"taken": taken.isoformat(),
               "players": {k: asdict(v) for k, v in snapshot.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def brief(*, snapshot: dict[str, PlayerStatus], changes: list[Change],
          previous_taken: str | None, issues: list[str], week: int | None,
          season: int, report_week: int | None,
          now: datetime | None = None, failed: str | None = None) -> str:
    """Status first, then what changed, then who is currently flagged."""
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    wk = f"Week {week}" if week else "Preseason"
    lines: list[str] = []
    if failed:
        return "\n".join([f"# STATUS: FAILED — injuries, {wk} ({season})", "",
                          f"injury_watch could not run: {failed}", "",
                          f"*Generated {stamp}.*"])

    stale = [i for i in issues if i.startswith("STALE")]
    lines += [f"# STATUS: {'DEGRADED' if stale else 'OK'} — injuries, {wk} "
              f"({season})", ""]
    if issues:
        lines += ["**Read this first:**" if stale else "Notes:", ""]
        lines += [f"- {i}" for i in issues] + [""]
    src = (f"NFL report week {report_week}" if report_week
           else "no NFL report yet (Sleeper designations only)")
    lines += [f"*Generated {stamp}. {len(snapshot)} players on roster. "
              f"Sources: Sleeper designations, {src}.*", ""]

    lines += ["## Changes since last check", ""]
    if previous_taken is None:
        lines.append("- First run: no earlier snapshot to compare against. "
                     "Tomorrow's brief will show changes.")
    elif not changes:
        lines.append(f"- **No changes** since {previous_taken[:16].replace('T', ' ')} UTC.")
    else:
        lines += [f"- {c.describe()}" for c in changes]
    lines.append("")

    flagged = sorted((p for p in snapshot.values() if p.flagged),
                     key=lambda p: (-p.severity, p.player_name))
    lines += ["## Currently flagged", ""]
    if not flagged:
        lines.append("- Nobody. Every rostered player is clear.")
    for p in flagged:
        tag = {"out": "OUT", "doubtful": "DOUBTFUL", "risky": "RISKY",
               "monitor": "monitor"}.get(p.concern, p.concern)
        lines.append(f"- {p.player_name} ({p.position}"
                     f"{', ' + p.nfl_team if p.nfl_team else ''}): "
                     f"{p.summary()} — {tag}")
    lines.append("")
    lines.append("> Designations from Sleeper; practice data from the NFL's "
                 "official report, which fills in Wednesday–Friday. Nothing "
                 "here is a lineup decision — that is the Wednesday brief.")
    return "\n".join(lines)
