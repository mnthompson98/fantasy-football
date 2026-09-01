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


def picks_until_next_turn(pick_no: int, teams: int, my_slot: int) -> int:
    """Picks between `pick_no` and this manager's next turn in a snake draft.

    This is the number that decides whether to reach: if the tier you want has
    fewer players left than this, it will be gone. Mirrors the identical
    function in `src/draft/monitor.py`, which the live board uses.
    """
    nxt = pick_no + 1
    while nxt <= teams * 100:
        rnd = (nxt - 1) // teams
        idx = (nxt - 1) % teams
        slot = idx + 1 if rnd % 2 == 0 else teams - idx
        if slot == my_slot:
            return nxt - pick_no - 1
        nxt += 1
    return 0


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


# Above this, a position is genuinely being raided and waiting costs real
# points, worth calling out loudly. Below it, the wording should read as
# "fine, no rush" rather than manufacturing urgency out of a rounding error.
_URGENT_GAIN = 1.0
_BIG_GAIN = 15.0


def _drop_off_reason(position: str, gain: float, later_name: str | None,
                     picks_until_next: int) -> str:
    """Plain English for one candidate's `gain`, specific enough to actually
    differ pick to pick — the name, the point gap and the turn count are
    real numbers pulled from this decision, not a fixed two-template phrase.
    """
    if gain > _URGENT_GAIN:
        tier = "Big drop-off" if gain > _BIG_GAIN else "Real gap"
        survivor = f"{later_name} looks like the best {position} still around" \
            if later_name else f"nothing else at {position} is left"
        return (f"{tier}: {survivor} in {picks_until_next} picks — "
               f"{gain:.0f} fewer points to your lineup if you wait.")
    if later_name:
        return (f"No rush — {later_name} projects to survive to your next "
               f"turn nearly as well, so this is about who's best, not a "
               f"race against the clock.")
    return (f"No rush — {position} depth holds up through your next turn, "
           f"so this is about who's best, not a race against the clock.")


@dataclass
class Candidate:
    """One position's best available player, and why it ranked where it did.

    `gain` and `raw_value` are exactly what `ValueDrafter.rank()` sorts by —
    carried through so a caller can see the numbers behind the reasoning,
    not just the sentence.
    """

    index: int
    position: str
    player_name: str
    gain: float          # drop-off vs. the likely survivor; inf = forced pick
    value: float          # points-scale (VORP + replacement_points)
    raw_value: float      # the board's own value_col (e.g. anchored VORP)
    reason: str
    forced: bool = False  # scarcity / end-of-draft backstop, not a real choice


class ValueDrafter:
    """Our side: take the pick that most improves our *starting lineup*.

    The board is still where the intelligence lives — this policy adds no
    opinion about players, only about slots. But "take the highest VORP
    available" is not a neutral baseline, it is a broken one, and the backtest
    caught it. Drafting from slot 5 in 2025 it produced:

        r2  QB Josh Allen        r3  QB Lamar Jackson
        r5  TE Travis Kelce      r6  TE David Njoku
        r7-r12  six more WRs (eight in total)
        r13 RB Joe Mixon         r14 RB Rhamondre Stevenson

    Two premium picks on a backup quarterback who can never start in a 1-QB
    league, two more on a backup tight end, eight receivers of whom three can
    play — and no running back until the end-of-draft backstop forced two. Best
    RB scored 178.8 against a league median best of 279.5. It finished 9th of 10.

    The cause is that VORP measures a player against *positional replacement*,
    which is a property of the league, not against *your roster*, which is what
    a pick actually changes. Once Josh Allen is yours, the second-best QB in
    football adds nothing: he cannot enter your lineup.

    So the policy scores a candidate by the **drop-off**: how much more does the
    best player at his position improve my starting lineup now than the best one
    still there at my next turn would? That single quantity handles both
    failures at once. A backup QB improves nothing, now or later, so his
    drop-off is zero. A position being run on has a steep drop-off, so it gets
    taken. A position nobody else wants can wait.

    Scoring by lineup improvement *alone* is not enough, and the second backtest
    proved it: an empty slot is worth replacement level (0 in VORP terms), so
    once every remaining running back is below replacement, filling RB1 scores
    as a downgrade and the policy takes a ninth receiver instead. Comparing
    against what survives to the next pick is what removes that pathology —
    below-replacement is still far better than the nothing you would otherwise
    be starting.

    Three constraints keep the policy from producing an illegal roster:

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

    def _effective(self, row) -> float:
        """A player's value on a points scale.

        `replacement_points` puts VORP back on the scale where an empty slot is
        genuinely worth zero. A board without the column (a hand-built test
        fixture) degrades to plain VORP, which is fine as long as every position
        is treated the same way.
        """
        value = float(row[self.value_col])
        repl = row.get("replacement_points", 0.0)
        try:
            repl = float(repl)
        except (TypeError, ValueError):
            repl = 0.0
        return value + (repl if repl == repl else 0.0)  # NaN-safe

    # ----------------------------------------------------------------
    # Starting-lineup value
    # ----------------------------------------------------------------

    def starting_value(self, roster: Roster,
                       extra: tuple[str, float] | None = None) -> float:
        """Total value of the best legal starting lineup, optionally with one
        more player added.

        Values here are on a **points** scale, not a VORP scale, and the
        difference is the whole reason this works. An unfilled starting slot
        scores zero actual points on Sunday. In VORP space zero is *replacement
        level*, so an empty slot and a replacement-level starter look identical
        — and the drafter concluded that taking a replacement-level running back
        for an empty RB slot gained it nothing. It then spent the pick on a
        backup quarterback with a higher VORP who could never enter the lineup.

        `EFFECTIVE_COL` (vorp + replacement_points) restores the points scale
        while preserving whatever the market anchor did to K and DEF, since the
        anchor was applied to VORP.
        """
        by_pos: dict[str, list[float]] = {}
        for p in roster.picks:
            by_pos.setdefault(p["position"], []).append(self._effective(p))
        if extra is not None:
            by_pos.setdefault(extra[0], []).append(float(extra[1]))
        for vals in by_pos.values():
            vals.sort(reverse=True)

        total = 0.0
        leftovers: list[float] = []
        for pos, vals in by_pos.items():
            n = roster.starters.get(pos, 0)
            total += sum(vals[:n])
            if pos in roster.flex_eligible:
                leftovers.extend(vals[n:])

        leftovers.sort(reverse=True)
        total += sum(leftovers[:roster.flex_slots])
        return total

    def marginal_value(self, roster: Roster, position: str, value: float
                       ) -> float:
        """How much this player would improve our starting lineup."""
        return (self.starting_value(roster, extra=(position, value))
                - self.starting_value(roster))

    def survivors(self, available: pd.DataFrame, picks_until_next: int
                  ) -> pd.DataFrame:
        """Who is plausibly still on the board at our next turn.

        Opponents draft near ADP, so the next `picks_until_next` players off the
        board are approximately the top of the remaining ADP list. This is the
        same assumption the opponent model makes, minus the noise.
        """
        if picks_until_next <= 0:
            return available
        return available.sort_values("adp_rank").iloc[picks_until_next:]

    def choose(self, available: pd.DataFrame, roster: Roster,
               roster_size: int, picks_remaining: int,
               picks_until_next: int = 0) -> int:
        """The single pick. A thin wrapper: `rank()` does the actual work and
        this returns its winner, so the two can never disagree with each
        other. Kept because every call site outside this module — the
        backtest, the live monitor's fallback, every existing test — wants
        just the index, not the full breakdown.
        """
        return self.rank(available, roster, roster_size, picks_remaining,
                         picks_until_next)[0].index

    def rank(self, available: pd.DataFrame, roster: Roster,
            roster_size: int, picks_remaining: int,
            picks_until_next: int = 0, *, top_n: int | None = None
            ) -> list["Candidate"]:
        """Every legal position's best player, ranked the same way `choose()`
        picks its winner — best first, each with why it landed where it did.

        This exists so a caller can show the runner-up and why it lost, not
        only the winner. `choose()` is exactly `self.rank(...)[0].index`; the
        selection logic below is unchanged from what it always was, only
        returning the full ordering instead of discarding everything but the
        top.
        """
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
        #    This is a forced pick, not a comparison, so there is nothing
        #    meaningful to rank it against — the list is one long.
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
                    idx = int(forced[self.value_col].idxmax())
                    pos = str(forced.at[idx, "position"])
                    left = int(supply.get(pos, 0))
                    return [Candidate(
                        index=idx, position=pos,
                        player_name=str(forced.at[idx, "player_name"]),
                        gain=float("inf"), value=self._effective(forced.loc[idx]),
                        raw_value=float(forced.at[idx, self.value_col]),
                        forced=True,
                        reason=(f"Only {left} {pos}{'s' if left != 1 else ''} left "
                               f"and the starting slot is still open — taking "
                               f"the best one now rather than risking an empty "
                               f"slot later."),
                    )]

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

        # 4. Take the pick with the most to lose by waiting.
        #
        #    Not "who improves my lineup most" — that alone still builds an
        #    absurd roster, because it says an empty slot is worth replacement
        #    level (0 in VORP terms) and therefore that filling it with a
        #    below-replacement player is a downgrade. Late in a run on a
        #    position, every remaining player is below replacement, so the
        #    policy keeps taking a ninth receiver over its first running back.
        #
        #    What matters is the *drop-off*: how much better is the best player
        #    at this position now than the best one who will still be here at my
        #    next turn? That is what a pick actually buys. A position nobody
        #    else wants can wait; a position being run on cannot.
        #
        #    Only the best available player at each position can be the answer —
        #    marginal value is monotone in value within a position — so this
        #    costs a couple of lineup evaluations per position, not one per
        #    player.
        left = self.survivors(df, picks_until_next)

        candidates: list[Candidate] = []
        for pos, group in df.groupby("position", sort=False):
            pos = str(pos)
            idx = int(group[self.value_col].idxmax())
            now = self._effective(df.loc[idx])

            later_pool = left[left["position"] == pos]
            later_name = None
            gain = self.marginal_value(roster, pos, now)
            if not later_pool.empty:
                later_idx = int(later_pool[self.value_col].idxmax())
                later_name = str(later_pool.at[later_idx, "player_name"])
                gain -= self.marginal_value(
                    roster, pos, self._effective(later_pool.loc[later_idx]))

            candidates.append(Candidate(
                index=idx, position=pos,
                player_name=str(df.at[idx, "player_name"]),
                gain=gain, value=now,
                raw_value=float(df.at[idx, self.value_col]),
                reason=_drop_off_reason(pos, gain, later_name, picks_until_next),
            ))

        if not candidates:
            idx = int(df[self.value_col].idxmax())
            return [Candidate(
                index=idx, position=str(df.at[idx, "position"]),
                player_name=str(df.at[idx, "player_name"]), gain=0.0,
                value=self._effective(df.loc[idx]),
                raw_value=float(df.at[idx, self.value_col]),
                reason="Best value left on the board.",
            )]

        # Ties break on *anchored* VORP (`raw_value`), not on lineup
        # contribution. Sorting on lineup contribution rewards filling any
        # empty slot, and once the skill positions are set that means
        # drafting a kicker and a defense in rounds 8 and 9 — undoing the
        # market anchor, which exists precisely to keep them out of the early
        # rounds. Anchored VORP already prices them below every bench body,
        # so the tie resolves the way it should and the end-of-draft backstop
        # picks them up. `sorted` is stable, so a genuine tie keeps whichever
        # position `groupby` visited first — matching the manual max-loop
        # this replaced, which only overwrote its best-so-far on a strict `>`.
        candidates.sort(key=lambda c: (c.gain, c.raw_value), reverse=True)
        return candidates[:top_n] if top_n else candidates


def simulate_draft(board: pd.DataFrame, *, teams: int, rounds: int,
                   my_slot: int, starters: dict[str, int], flex_slots: int,
                   flex_eligible: tuple[str, ...], sigma: dict[str, float],
                   need_boost: float = 1.6, need_penalty: float = 0.4,
                   value_col: str = "vorp", seed: int | None = None,
                   caps: dict[str, int] | None = None,
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
    me = ValueDrafter(value_col, caps=caps)
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
            idx = me.choose(available, roster, roster_size, picks_remaining,
                            picks_until_next_turn(pick_no, teams, my_slot))
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
            # Carried so the lineup calculation can work on a points scale.
            "replacement_points": float(row.get("replacement_points", 0.0) or 0.0),
        })
        available = available.drop(index=idx)

    return rosters
