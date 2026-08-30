"""Regenerate the league history datasets from the Sleeper API.

The JSON files under `data/league_history/` are build artifacts, not hand-edited
sources. Re-run this to refresh them:

    python -m scripts.build_league_history

Each output carries the draft (order + every pick from every manager), the
regular-season standings, the resolved final standings, and the champion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.sleeper_api import (  # noqa: E402
    epoch_ms_to_iso,
    get_draft,
    get_draft_picks,
    get_league,
    get_league_users,
    get_rosters,
    get_standings,
    resolve_final_standings,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "league.yaml"
OUT_DIR = ROOT / "data" / "league_history"


def build_season(entry: dict) -> dict:
    league_id = entry["league_id"]
    draft_id = entry["draft_id"]

    league = get_league(league_id) or {}
    draft = get_draft(draft_id) or {}
    users = {u["user_id"]: u for u in get_league_users(league_id)}
    rosters = {r["roster_id"]: r for r in get_rosters(league_id)}
    picks = get_draft_picks(draft_id)

    # roster_id -> owner, so picks can be attributed to a human name.
    owner_of = {
        rid: users.get(r.get("owner_id"), {}).get("display_name")
        for rid, r in rosters.items()
    }

    settings = league.get("settings", {}) or {}
    slot_to_roster = draft.get("slot_to_roster_id") or {}

    standings = get_standings(league_id)
    finals = resolve_final_standings(league_id)
    champion = finals[0] if finals else None

    return {
        "league": {
            "name": league.get("name") or entry.get("name"),
            "league_id": league_id,
            "draft_id": draft_id,
            "season": int(entry["season"]),
            "scoring_type": (draft.get("metadata") or {}).get("scoring_type"),
            "draft_type": draft.get("type"),
            "status": league.get("status"),
            "num_teams": settings.get("num_teams") or entry.get("teams"),
            "num_rounds": (draft.get("settings") or {}).get("rounds"),
            "roster_positions": league.get("roster_positions"),
            "playoff_week_start": settings.get("playoff_week_start"),
            "playoff_teams": settings.get("playoff_teams"),
            "trade_deadline_week": settings.get("trade_deadline"),
            "waiver_type": settings.get("waiver_type"),
            "waiver_type_name": {
                0: "rolling_priority", 1: "reverse_standings", 2: "faab",
            }.get(settings.get("waiver_type"), "unknown"),
            "draft_started_at": epoch_ms_to_iso(draft.get("start_time")),
            "draft_last_pick_at": epoch_ms_to_iso(draft.get("last_picked")),
        },
        "draft_order": {
            "slot_to_roster_id": slot_to_roster,
            "slot_to_owner": {
                slot: owner_of.get(int(rid))
                for slot, rid in slot_to_roster.items()
            },
        },
        "teams": [
            {
                "roster_id": t.roster_id,
                "owner": t.display_name,
                "owner_id": t.owner_id,
                "wins": t.wins,
                "losses": t.losses,
                "ties": t.ties,
                "points_for": round(t.points_for, 2),
                "points_against": round(t.points_against, 2),
                "regular_season_seed": t.regular_season_seed,
            }
            for t in standings
        ],
        "final_standings": [
            {
                "final_rank": t.final_rank,
                "roster_id": t.roster_id,
                "owner": t.display_name,
                "record": f"{t.wins}-{t.losses}-{t.ties}",
                "points_for": round(t.points_for, 2),
                "made_playoffs": t.made_playoffs,
            }
            for t in finals
        ],
        "champion": {
            "roster_id": champion.roster_id,
            "owner": champion.display_name,
            "record": f"{champion.wins}-{champion.losses}-{champion.ties}",
            "points_for": round(champion.points_for, 2),
            "regular_season_seed": champion.regular_season_seed,
        } if champion else None,
        "picks": [
            {
                "pick_no": p.get("pick_no"),
                "round": p.get("round"),
                "draft_slot": p.get("draft_slot"),
                "roster_id": p.get("roster_id"),
                "owner": owner_of.get(p.get("roster_id")),
                "sleeper_player_id": p.get("player_id"),
                "player_name": " ".join(filter(None, [
                    (p.get("metadata") or {}).get("first_name"),
                    (p.get("metadata") or {}).get("last_name"),
                ])).strip(),
                "position": (p.get("metadata") or {}).get("position"),
                "nfl_team": (p.get("metadata") or {}).get("team"),
                "is_keeper": p.get("is_keeper"),
            }
            for p in sorted(picks, key=lambda x: x.get("pick_no") or 0)
        ],
    }


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in cfg["history"]:
        data = build_season(entry)
        slug = f"{entry['season']}_{entry['name']}".lower()
        slug = "".join(c if c.isalnum() else "_" for c in slug).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")

        path = OUT_DIR / f"{slug}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        champ = data["champion"]
        me = next(
            (t for t in data["final_standings"]
             if t["owner"] in cfg["user"]["known_display_names"]),
            None,
        )
        print(
            f"{path.relative_to(ROOT)}  "
            f"picks={len(data['picks'])}  "
            f"champion={champ['owner'] if champ else 'n/a'}  "
            f"my_finish={me['final_rank'] if me else 'n/a'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
