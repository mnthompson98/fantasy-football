"""Who to start this week, and how confident that call is.

The lineup itself is the easy part — `features/lineup.py` picks the best legal
one. The useful output is the *close calls*: which decisions actually mattered
and which were coin flips dressed up as decisions.

A recommendation that says "start Player A over Player B, projected 12.4 to
12.1" is worse than useless if the two projections carry standard deviations of
4 apiece, because it implies a distinction the data cannot support. So each
bench-versus-starter comparison is reported with the gap *and* the pooled
uncertainty, and anything inside the noise is labelled a toss-up rather than a
recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..features.lineup import Lineup, LineupSlots, best_lineup

# Below this many standard deviations of separation, the two players are not
# meaningfully different and the honest answer is "either".
TOSSUP_SIGMA = 0.5

# Fallback per-player weekly uncertainty when the source gives none. Weekly
# fantasy scoring is genuinely noisy; assuming certainty would be the error.
DEFAULT_SD = 4.0


@dataclass
class Call:
    """One start/sit decision worth looking at."""

    slot: str
    start: str
    sit: str
    start_points: float
    sit_points: float
    gap: float
    sigma: float
    position: str

    @property
    def tossup(self) -> bool:
        return self.gap < TOSSUP_SIGMA * self.sigma

    def describe(self) -> str:
        verdict = "TOSS-UP" if self.tossup else "start"
        return (f"  {self.slot:<5} {verdict:<8} {self.start[:22]:<23}"
                f"{self.start_points:>6.1f}  over {self.sit[:22]:<23}"
                f"{self.sit_points:>6.1f}   (+{self.gap:.1f}, sd {self.sigma:.1f})")


def _sd(row) -> float:
    sd = row.get("sd")
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        return DEFAULT_SD
    return sd if sd == sd and sd > 0 else DEFAULT_SD


def close_calls(lineup: Lineup, slots: LineupSlots, *,
                points_col: str = "projection") -> list[Call]:
    """Every bench player who could plausibly have started, and against whom.

    Compares each benched player to the weakest starter he is eligible to
    replace — the actual decision, rather than every pairing.
    """
    calls: list[Call] = []
    if lineup.starters.empty or lineup.bench.empty:
        return calls

    for _, cand in lineup.bench.iterrows():
        pos = cand.get("position")
        if pos is None:
            continue
        # A bench player can displace a starter at his own position, or the
        # flex if he is flex-eligible.
        same_pos = lineup.starters["position"] == pos
        if pos in slots.flex_eligible:
            same_pos = same_pos | (lineup.starters["slot"] == "FLEX")
        eligible = lineup.starters[same_pos]
        if eligible.empty:
            continue

        weakest = eligible.loc[eligible[points_col].idxmin()]
        gap = float(weakest[points_col]) - float(cand[points_col] or 0.0)
        sigma = math.hypot(_sd(weakest), _sd(cand))
        calls.append(Call(
            slot=str(weakest["slot"]),
            start=str(weakest.get("player_name")),
            sit=str(cand.get("player_name")),
            start_points=float(weakest[points_col]),
            sit_points=float(cand[points_col] or 0.0),
            gap=gap, sigma=sigma, position=str(pos),
        ))

    calls.sort(key=lambda c: c.gap)
    return calls


def unfilled_slots(lineup: Lineup, slots: LineupSlots) -> list[str]:
    """Starting slots with nobody in them — a bye or an injury hole.

    These are the only situations where a waiver claim is genuinely urgent, so
    they are surfaced separately rather than buried in the lineup table.
    """
    filled = set(lineup.starters["slot"]) if not lineup.starters.empty else set()
    return [label for label in slots.labels()
            if label not in filled or (
                label == "FLEX"
                and list(lineup.starters["slot"]).count("FLEX") < slots.flex_slots)]


def recommend(roster: pd.DataFrame, slots: LineupSlots, *,
              points_col: str = "projection") -> tuple[Lineup, list[Call], list[str]]:
    """The week's lineup, the calls that were close, and any hole in it."""
    playable = roster
    if "available" in roster.columns:
        # A player Sleeper lists as Out/IR cannot help this week. He stays on
        # the roster; he just cannot be started.
        playable = roster[roster["available"].fillna(True)]

    lineup = best_lineup(playable, slots, points_col=points_col)
    return lineup, close_calls(lineup, slots, points_col=points_col), \
        unfilled_slots(lineup, slots)
