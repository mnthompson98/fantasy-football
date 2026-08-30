"""Snake draft simulation with ADP-based opponent behavior.

The point of this module is to produce *realistic* opposition, not optimal
opposition. Opponents who draft perfectly leave a very different board than
real managers do, and a strategy tuned against perfect opponents will not
survive contact with a real league. Real managers reach, chase names, and
draft for need — so that is what we model.

Opponent pick model, per pick:
    1. score each available player by ADP rank with gaussian noise:
       key = adp_rank + N(0, sigma[position])
    2. multiply by a positional-need factor (unfilled starting slot -> more
       likely; already-full position -> much less likely)
    3. take the minimum key

Sigma is calibrated per position from historical ADP-vs-actual-draft-slot error
(`scripts/calibrate_adp_sigma.py`); config priors are the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Roster:
    """A single team's roster state during the draft."""

    team_id: int
    starters: dict[str, int]
    flex_slots: int
    flex_eligible: tuple[str, ...]
    bench_slots: int
    picks: list[dict] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for p in self.picks:
            c[p["position"]] = c.get(p["position"], 0) + 1
        return c

    def unfilled_dedicated_slots(self) -> dict[str, int]:
        """Dedicated (non-flex) starting slots still open, per position."""
        have = self.counts()
        return {
            pos: max(0, n - have.get(pos, 0))
            for pos, n in self.starters.items()
        }

    def flex_slots_open(self) -> int:
        """Flex slots not yet covered by surplus at a flex-eligible position."""
        have = self.counts()
        surplus = sum(
            max(0, have.get(pos, 0) - self.starters.get(pos, 0))
            for pos in self.flex_eligible
        )
        return max(0, self.flex_slots - surplus)

    def total_unfilled_starting_slots(self) -> int:
        """How many starting slots are still empty.

        Counted once each. `unfilled_starting_slots()` deliberately attributes
        the same flex opening to every eligible position, so summing *that*
        triple-counts the flex and must not be used as a slot count.
        """
        return sum(self.unfilled_dedicated_slots().values()) + self.flex_slots_open()

    def unfilled_starting_slots(self) -> dict[str, int]:
        """Per-position view of "would this pick fill a starting slot?".

        The same flex opening appears under every eligible position, because
        any of them could fill it. Useful for the opponent need-boost; not a
        slot count. Use `total_unfilled_starting_slots()` for that.
        """
        need = self.unfilled_dedicated_slots()
        flex_open = self.flex_slots_open()
        if flex_open:
            for pos in self.flex_eligible:
                need[pos] = need.get(pos, 0) + flex_open
        return need

    def is_full(self, roster_size: int) -> bool:
        return len(self.picks) >= roster_size


def snake_order(teams: int, rounds: int) -> list[int]:
    """Pick-by-pick team order for a snake draft. 0-indexed team ids."""
    order: list[int] = []
    for r in range(rounds):
        rng = range(teams) if r % 2 == 0 else range(teams - 1, -1, -1)
        order.extend(rng)
    return order


class OpponentModel:
    """Samples opponent picks around ADP with positional-need adjustment."""

    def __init__(self, sigma: dict[str, float], need_boost: float = 1.6,
                 need_penalty: float = 0.4, rng: np.random.Generator | None = None):
        self.sigma = sigma
        self.need_boost = need_boost
        self.need_penalty = need_penalty
        self.rng = rng or np.random.default_rng()

    def choose(self, available: pd.DataFrame, roster: Roster) -> int:
        """Return the index label of the chosen player."""
        if available.empty:
            raise ValueError("no players available")

        adp = available["adp_rank"].to_numpy(dtype=float)
        positions = available["position"].to_numpy()

        sigmas = np.array([self.sigma.get(p, 10.0) for p in positions], dtype=float)
        noise = self.rng.normal(0.0, sigmas)
        key = adp + noise

        need = roster.unfilled_starting_slots()
        factors = np.where(
            np.array([need.get(p, 0) > 0 for p in positions]),
            1.0 / self.need_boost,   # lower key == more likely to be picked
            1.0 / self.need_penalty,
        )
        key = key * factors

        return int(available.index[int(np.argmin(key))])


class ValueDrafter:
    """Our side: take the highest-VORP available, subject to roster legality.

    Deliberately simple. The board is where the intelligence lives; if a
    sophisticated pick policy is needed to beat ADP opponents, the board is
    not good enough yet.

    Two constraints keep naive VBD from producing an illegal roster:

    1. **Position caps.** Raw VORP is happy to take a fifth tight end. Bench
       depth has real value at RB/WR, near-none at QB/TE/K/DEF in a 1-slot
       league, so each position gets a sane ceiling.
    2. **Starter backstop.** Once the picks remaining equal the starting slots
       still empty, every remaining pick must fill one. An unfilled starter is
       a zero every single week — it costs far more than any bench upgrade.
    """

    # Beyond these, an extra body at the position cannot crack the lineup.
    # RB/WR are uncapped-ish because flex plus injury attrition makes depth real.
    DEFAULT_CAPS = {"QB": 2, "TE": 2, "K": 1, "DEF": 2}

    # If a position we still need a starter at drops to this few players, take
    # one now. Waiting for the end-of-draft backstop assumes the position will
    # still be there, and sometimes it isn't.
    SCARCITY_FLOOR = 3

    def __init__(self, value_col: str = "vorp",
                 caps: dict[str, int] | None = None,
                 scarcity_floor: int = SCARCITY_FLOOR):
        self.value_col = value_col
        self.caps = caps if caps is not None else dict(self.DEFAULT_CAPS)
        self.scarcity_floor = scarcity_floor

    def choose(self, available: pd.DataFrame, roster: Roster,
               roster_size: int, picks_remaining: int) -> int:
        df = available
        counts = roster.counts()

        # 1. Position caps, plus the hard legality ceiling.
        max_useful = {}
        for pos in set(df["position"]):
            ceiling = (
                roster.starters.get(pos, 0)
                + (roster.flex_slots if pos in roster.flex_eligible else 0)
                + roster.bench_slots
            )
            max_useful[pos] = min(ceiling, self.caps.get(pos, ceiling))

        legal = df[[
            counts.get(p, 0) < max_useful.get(p, 99) for p in df["position"]
        ]]
        if not legal.empty:
            df = legal

        # 2. Scarcity backstop. If a position we still need a starter at is
        #    nearly exhausted, take the best one now regardless of VORP —
        #    a replacement-level starter beats an empty slot by its whole score.
        dedicated_need = [
            p for p, n in roster.unfilled_dedicated_slots().items() if n > 0
        ]
        if dedicated_need:
            supply = df["position"].value_counts()
            scarce = [
                p for p in dedicated_need
                if supply.get(p, 0) <= self.scarcity_floor
            ]
            if scarce:
                forced = df[df["position"].isin(scarce)]
                if not forced.empty:
                    return int(forced[self.value_col].idxmax())

        # 3. End-of-draft backstop. Counted once per slot — never sum the
        #    per-position need dict, which triple-counts the flex.
        slots_open = roster.total_unfilled_starting_slots()
        if slots_open and picks_remaining <= slots_open:
            need = roster.unfilled_starting_slots()
            open_positions = [p for p, n in need.items() if n > 0]

            # Dedicated slots first: a missing QB cannot be covered by a flex
            # body, but a missing flex can be covered by several positions.
            dedicated = [
                p for p, n in roster.unfilled_dedicated_slots().items() if n > 0
            ]
            for candidates in (dedicated, open_positions):
                forced = df[df["position"].isin(candidates)]
                if not forced.empty:
                    df = forced
                    break

        return int(df[self.value_col].idxmax())


def simulate_draft(board: pd.DataFrame, *, teams: int, rounds: int,
                   my_slot: int, starters: dict[str, int], flex_slots: int,
                   flex_eligible: tuple[str, ...], sigma: dict[str, float],
                   need_boost: float = 1.6, need_penalty: float = 0.4,
                   value_col: str = "vorp", seed: int | None = None
                   ) -> dict[int, Roster]:
    """Run one full snake draft.

    `board` needs columns: `player_id`, `player_name`, `position`, `adp_rank`,
    and `value_col`. `my_slot` is 1-indexed.

    Returns every team's roster so the scorer can compare ours against the
    league, not just measure ours in isolation.
    """
    required = {"player_id", "position", "adp_rank", value_col}
    missing = required - set(board.columns)
    if missing:
        raise ValueError(f"board missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    roster_size = rounds
    bench_slots = max(0, rounds - sum(starters.values()) - flex_slots)

    rosters = {
        t: Roster(team_id=t, starters=dict(starters), flex_slots=flex_slots,
                  flex_eligible=flex_eligible, bench_slots=bench_slots)
        for t in range(teams)
    }

    opponent = OpponentModel(sigma, need_boost, need_penalty, rng)
    me = ValueDrafter(value_col)
    my_team = my_slot - 1

    available = board.copy()
    order = snake_order(teams, rounds)

    for pick_no, team in enumerate(order, start=1):
        if available.empty:
            break
        roster = rosters[team]
        if roster.is_full(roster_size):
            continue

        picks_remaining = roster_size - len(roster.picks)
        if team == my_team:
            idx = me.choose(available, roster, roster_size, picks_remaining)
        else:
            idx = opponent.choose(available, roster)

        row = available.loc[idx]
        roster.picks.append({
            "pick_no": pick_no,
            "round": (pick_no - 1) // teams + 1,
            "player_id": row["player_id"],
            "player_name": row.get("player_name"),
            "position": row["position"],
            "adp_rank": float(row["adp_rank"]),
            value_col: float(row[value_col]),
        })
        available = available.drop(index=idx)

    return rosters
