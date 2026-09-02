"""Snake draft simulation with ADP-based opponent behavior.

The point of this module is to produce *realistic* opposition, not optimal
opposition. Opponents who draft perfectly leave a very different board than
real managers do, and a strategy tuned against perfect opponents will not
survive contact with a real league. Real managers reach, chase names, and
draft for need — so that is what we model.

Two opponent models live here, and which one runs is a flag
(`backtest.opponent_model.model`, or `--opponent` on the scripts):

**`OpponentModel` — gaussian, the fallback.** Per pick:
    1. score each available player by ADP rank with gaussian noise:
       key = adp_rank + N(0, sigma[position])
    2. multiply by a positional-need factor (unfilled starting slot -> more
       likely; already-full position -> much less likely)
    3. take the minimum key
Sigma comes from the config priors. This is the model every recorded backtest
number was produced under, so it stays here, unchanged, bit-for-bit.

**`LeagueOpponentModel` — fitted to this league's own drafts.** Same argmin,
but the key is a blend of two clocks, both in rounds:

    consensus = adp_rank / teams              # where the board says he goes
    schedule  = curve[position](k / teams)    # when this room takes the k-th
                                              # player at his position

    key = (1 - w[position]) * consensus + w[position] * schedule
        + N(0, sigma[position])
        + over_ceiling_penalty                # if the roster is already full there

`k` counts the players at that position already gone plus his own ADP rank
among those still available. Everything is measured by
`src/backtest/opponent_fit.py` from the real drafts under
`data/league_history/`, normalized per team so an 8-team draft and a 10-team
draft pool and either can drive a simulation of the other size.

**Why a blend and not one or the other.** Neither clock alone describes the
room. Fitting `pick_no ~ adp_rank` per position over this league's 2021 and
2025 drafts gives slopes of 0.97 for RB and 0.86 for WR — those go at consensus
— against 0.25 for K and 0.11 for DEF. The room does not *shift* kickers up the
board, it ignores the consensus's ordering of them almost entirely and takes
one per team in rounds 12-14 regardless, so no constant bias can express it.
But a pure schedule is board-blind in the other direction: this league opened
2021 with eight running backs and 2025 with five receivers, and it did so
because those were the two boards. Averaging those two rounds into one schedule
describes neither draft. `w` is fitted per position, closed-form, as exactly
the weight that best reconciles the two — and it lands where the regression
slopes say it should: near 0 for RB and WR, near 1 for K and DEF.

**What was wrong with the gaussian field.** Its need factor is *multiplicative
on the key*, so its size depends on where you are on the board: at ADP rank 250
the 1/1.6 boost is worth 94 picks, at rank 10 it is worth 4. That single scale
error makes the simulated field draft every defense in round 8 and every kicker
in round 9. This league takes its first defense in round 11 and its first
kicker in round 10 — and in 2021, round 13 for both. Two to four rounds of the
middle of every simulated draft are therefore spent on the wrong positions,
which is exactly the stretch where the pick policy's lookahead is deciding what
survives to the next turn.

Neither model is consulted by `ValueDrafter`, so switching them gives our side
no information it would not have had. What changes is the realism of the board
it faces. `ValueDrafter.survivors()` still assumes the field drafts straight
down ADP — see its docstring.
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


def positional_run(recent_picks: list[dict], window: int = 6,
                   threshold: float = 0.5, min_picks: int = 3
                   ) -> dict[str, float]:
    """Detect a positional run in the last `window` picks.

    Returns positions whose share of recent picks exceeds `threshold`. Lives
    here rather than in `draft.board` because the pick policy consumes it
    (`ValueDrafter.survivors`) and the simulator has to compute the same
    number the monitor does; `draft.board` re-exports it for its callers.

    `min_picks` exists because share-of-window is meaningless on a tiny sample.
    Without it the monitor announced "run in progress: WR 100%" after the first
    pick of the draft, every time, and a warning that fires on pick one is a
    warning nobody reads by pick fifty.
    """
    if len(recent_picks) < min_picks:
        return {}
    recent = recent_picks[-window:]
    counts: dict[str, int] = {}
    for p in recent:
        pos = p.get("position")
        if pos:
            counts[pos] = counts.get(pos, 0) + 1
    return {
        pos: c / len(recent)
        for pos, c in counts.items()
        if c / len(recent) >= threshold
    }


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


def picks_until_board_moves(pick_no: int, teams: int, my_slot: int) -> int:
    """Opponent picks between `pick_no` and the next turn the board has moved.

    At the picks this is ever asked about — our own — it is identical to
    `picks_until_next_turn` except at a **turn slot** (slot 1 and slot `teams`),
    where a snake gives you two picks back to back. (The two functions disagree
    at some picks belonging to other seats as well; those are never queried,
    because only our own pick number is ever passed in.)

    At a turn slot's first pick of a pair, `picks_until_next_turn` correctly
    answers 0 — nobody drafts between
    picks 10 and 11 — and that correct 0 is poison for the drop-off policy.
    `ValueDrafter.survivors()` returns the pool unfiltered at <= 0, so at every
    position the best-now and the best-later are the same player, every
    candidate's `gain` computes to exactly 0.0, and the ranking falls through to
    its raw-VORP tiebreak. That is the "take the highest VORP available" policy
    `ValueDrafter` exists to replace, and from slot 10 it runs on half our picks
    (10, 30, 50, ... 150).

    Note this is a *different* bug from the on-the-clock lookahead fixed in
    `src/draft/monitor.py`. That one fed a display value into the policy; the 0
    was wrong. Here the 0 is right, and the policy is wrong to accept it — so
    neither that fix nor its regression tests touch this.

    Skipping over our own consecutive picks makes the horizon the one that
    actually costs us something: what survives to the turn *after* the pair.
    Taking the two best drop-offs measured against that horizon is the greedy
    read of what is genuinely one joint decision over both picks.
    """
    nxt = int(pick_no)
    limit = nxt + 2 * max(1, teams)      # a run of own picks cannot exceed this
    while nxt <= limit:
        gap = picks_until_next_turn(nxt, teams, my_slot)
        if gap > 0:
            return gap
        nxt += 1                          # the very next pick is ours too
    return 0


# The two rules above, by name, for config and CLI plumbing. "next_pick" is
# what every recorded number in HANDOFF.md and tests/fixtures/ was produced
# under and is therefore the default; changing it is a pick-policy change and
# must be measured on its own (CLAUDE.md, "survivors() was deliberately left
# as it was").
LOOKAHEAD_RULES = {
    "next_pick": picks_until_next_turn,
    "next_exposed": picks_until_board_moves,
}


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


@dataclass(frozen=True)
class LeagueTendencies:
    """When this league's room takes the k-th player at each position.

    `curve[pos]` is a monotone lookup from **k per team** to **round**: the
    entry at 0.9 for QB says that by the time nine tenths of a quarterback per
    team have been drafted (nine of them in a ten-team league), the draft is
    somewhere in round 7.3. Both axes are normalized by league size so drafts of
    different sizes pool, and so a curve measured on an 8-team league can drive
    a 10-team simulation.

    Stored as two parallel lists rather than a mapping because that is what
    `np.interp` wants and what serializes to JSON without inventing a key
    format. `k_per_team` is shared by every position; `rounds[pos]` is that
    position's curve over it.

    `sigma_rounds` is the residual spread around the curve, in rounds, measured
    the same way. `max_per_team` is what a team in this league actually ends up
    holding, which is what stops a simulated manager taking a third kicker just
    because kickers are cheap in round 13.

    Fitted by `src/backtest/opponent_fit.py`; `provenance` records which drafts
    produced it, so a logged run can be traced back to them.
    """

    k_per_team: list[float]
    rounds: dict[str, list[float]]
    # How much of a position's timing the room's own schedule explains, against
    # the consensus board. 0 = drafted straight off the board, 1 = the board's
    # ordering of the position is ignored and it goes on schedule.
    curve_weight: dict[str, float]
    sigma_rounds: dict[str, float]
    max_per_team: dict[str, float]
    # Rounds added to the key once a roster is at its ceiling for the position.
    # Large enough to push it behind everything else genuinely in contention,
    # small enough that a forced pick at the very end of the draft still works.
    over_ceiling_penalty: float = 8.0
    provenance: str = ""

    def curve_for(self, position: str) -> np.ndarray | None:
        vals = self.rounds.get(position)
        return np.asarray(vals, dtype=float) if vals else None

    def to_dict(self) -> dict:
        return {
            "k_per_team": [round(float(k), 4) for k in self.k_per_team],
            "rounds": {k: [round(float(x), 3) for x in v]
                       for k, v in sorted(self.rounds.items())},
            "curve_weight": {k: round(float(v), 4)
                             for k, v in sorted(self.curve_weight.items())},
            "sigma_rounds": {k: round(float(v), 4)
                             for k, v in sorted(self.sigma_rounds.items())},
            "max_per_team": {k: round(float(v), 3)
                             for k, v in sorted(self.max_per_team.items())},
            "over_ceiling_penalty": float(self.over_ceiling_penalty),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LeagueTendencies":
        return cls(
            k_per_team=[float(k) for k in (d.get("k_per_team") or [])],
            rounds={str(k): [float(x) for x in v]
                    for k, v in (d.get("rounds") or {}).items()},
            curve_weight={str(k): float(v)
                          for k, v in (d.get("curve_weight") or {}).items()},
            sigma_rounds={str(k): float(v)
                          for k, v in (d.get("sigma_rounds") or {}).items()},
            max_per_team={str(k): float(v)
                          for k, v in (d.get("max_per_team") or {}).items()},
            over_ceiling_penalty=float(d.get("over_ceiling_penalty", 8.0)),
            provenance=str(d.get("provenance", "")),
        )


def positional_queue_numbers(positions: np.ndarray, adp: np.ndarray,
                             taken: dict[str, int]) -> np.ndarray:
    """For each available player, his k: which one at his position he would be.

    k = (already gone at that position) + (his ADP rank among those still
    available at it). So the best remaining quarterback when six are gone is the
    seventh quarterback, and the curve is asked when the seventh quarterback
    goes.

    Vectorized deliberately. The obvious loop is 500 iterations per pick and
    160 picks per draft over hundreds of drafts, which is tens of millions of
    Python-level steps in the inner loop of every backtest.
    """
    codes, inverse = np.unique(positions, return_inverse=True)
    order = np.lexsort((adp, inverse))
    sorted_inverse = inverse[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_inverse[1:] != sorted_inverse[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))

    within = np.empty(len(order), dtype=float)
    within[order] = np.arange(len(order)) - group_start + 1.0
    already = np.array([float(taken.get(str(c), 0)) for c in codes])
    return already[inverse] + within


# Past the deepest k a position was ever drafted to, the curve must keep
# climbing, and steeply: the room did not merely slow down there, it stopped.
# In rounds per player-per-team, so 10.0 is one extra round for each additional
# body at the position in a 10-team league. Flattening the tail instead — the
# obvious `np.interp` clamp — makes every remaining quarterback tie at the last
# fitted round, and the field then drafts six of them in round 14.
TAIL_SLOPE_ROUNDS_PER_K = 10.0


class LeagueOpponentModel:
    """Samples opponent picks from this league's own drafting schedule.

    key = (1 - w[pos]) * adp_rank / teams        # the consensus clock
        + w[pos] * curve[pos](k / teams)         # the room's own clock
        + N(0, sigma_rounds[pos])
        + over_ceiling_penalty   if this roster already holds its ceiling there

    Everything is in rounds, so nothing changes meaning between the top of the
    board and the bottom — which is the specific way the gaussian model's
    multiplicative need factor goes wrong (module docstring).

    The model needs to know how many players at each position are already gone
    **league-wide**, and `choose` is only handed one roster. So it snapshots the
    board's positional counts at construction and recovers the rest by
    subtraction from `available`, which is exact and costs nothing. That is why
    the factory takes the board.

    A position the fit never saw gets `w = 0` — drafted straight off the
    consensus board, which is the gaussian field's assumption without the noise,
    and the honest default when there is nothing measured.
    """

    def __init__(self, tendencies: LeagueTendencies, board: pd.DataFrame,
                 teams: int, rng: np.random.Generator | None = None):
        self.t = tendencies
        self.teams = max(1, int(teams))
        self.rng = rng or np.random.default_rng()
        self.initial = (board["position"].astype(str).value_counts()
                        .to_dict() if len(board) else {})
        self._grid = np.asarray(tendencies.k_per_team, dtype=float)

    def _rounds_for(self, position: str, k_per_team: np.ndarray,
                    adp: np.ndarray) -> np.ndarray:
        """This position's key, in rounds, before noise and the ceiling."""
        consensus = adp / self.teams
        curve = self.t.curve_for(position)
        w = float(self.t.curve_weight.get(position, 0.0))
        if curve is None or not len(self._grid) or w <= 0.0:
            return consensus

        scheduled = np.interp(k_per_team, self._grid, curve)
        beyond = k_per_team > self._grid[-1]
        if beyond.any():
            scheduled[beyond] = curve[-1] + TAIL_SLOPE_ROUNDS_PER_K * (
                k_per_team[beyond] - self._grid[-1])
        return (1.0 - w) * consensus + w * scheduled

    def choose(self, available: pd.DataFrame, roster: Roster) -> int:
        """Return the index label of the chosen player."""
        if available.empty:
            raise ValueError("no players available")

        positions = available["position"].astype(str).to_numpy()
        adp = available["adp_rank"].to_numpy(dtype=float)

        counts = available["position"].astype(str).value_counts().to_dict()
        taken = {p: int(self.initial.get(p, 0)) - int(counts.get(p, 0))
                 for p in self.initial}
        k = positional_queue_numbers(positions, adp, taken) / self.teams

        key = np.empty(len(adp), dtype=float)
        for pos in np.unique(positions):
            m = positions == pos
            key[m] = self._rounds_for(str(pos), k[m], adp[m])

        sigmas = np.array([self.t.sigma_rounds.get(p, 0.75)
                           for p in positions], dtype=float)
        key = key + self.rng.normal(0.0, sigmas)

        have = roster.counts()
        ceiling = np.array([
            have.get(p, 0) >= max(1, int(np.floor(
                self.t.max_per_team.get(p, 99.0))))
            for p in positions
        ])
        key = key + np.where(ceiling, self.t.over_ceiling_penalty, 0.0)

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
    upside: float = float("nan")   # board `upside_vorp`, if it carries one


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
    # These match `draft_policy.position_caps` in config/league.yaml — the
    # manager's actual strategy — so a caller that passes no caps (the monitor
    # on a checkout with no config) runs the same policy the backtest scored,
    # not a looser one that allows a second QB and a second DEF.
    DEFAULT_CAPS = {"QB": 1, "TE": 2, "K": 1, "DEF": 1}

    # If a position we still need a starter at drops to this few players, take
    # one now. Waiting for the end-of-draft backstop assumes the position will
    # still be there, and sometimes it isn't.
    SCARCITY_FLOOR = 3

    # `depth_tiebreak` constants. Once the starters are set every candidate's
    # drop-off is zero — nobody left improves *this* lineup — and the sort
    # falls through to raw VORP, which happily takes a second tight end six
    # turns running because his VORP edges a bench receiver's. A bench player
    # is insurance: worth something in proportion to how many starters he
    # covers and discounted for each body already ahead of him in that queue.
    # Heuristic, not measured constants; the backtest scores the whole option.
    BENCH_DECAY = 0.5

    def __init__(self, value_col: str = "vorp",
                 caps: dict[str, int] | None = None,
                 scarcity_floor: int = SCARCITY_FLOOR,
                 run_aware: bool = False,
                 depth_tiebreak: bool = False,
                 upside_tiebreak: bool = False):
        self.value_col = value_col
        self.caps = caps if caps is not None else dict(self.DEFAULT_CAPS)
        self.scarcity_floor = scarcity_floor
        # All default off: the recorded baseline and every number in
        # HANDOFF.md were measured without them. `config: draft_policy`
        # switches them on for the backtest and the monitor together.
        self.run_aware = run_aware
        self.depth_tiebreak = depth_tiebreak
        # A bench pick (drop-off zero) is ranked on the board's `upside_vorp`
        # — the best-case expert rank through the curve — instead of the
        # median. A round-12 player's median projection is worthless by
        # construction; what you are buying is the chance he is not.
        self.upside_tiebreak = upside_tiebreak

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

    def survivors(self, available: pd.DataFrame, picks_until_next: int,
                  runs: dict[str, float] | None = None) -> pd.DataFrame:
        """Who is plausibly still on the board at our next turn.

        Opponents draft near ADP, so the next `picks_until_next` players off the
        board are approximately the top of the remaining ADP list. This is the
        gaussian opponent model's assumption, minus the noise. Replayed against
        this league's real 2025 draft it was right about 45% of the players it
        named, which is the number to keep in mind when reading any drop-off.

        `runs` is `positional_run()`'s view of the last few picks. With
        `run_aware` on, a position being run on is assumed to keep going at
        its recent share of picks: if the room has taken four of the last six
        at RB and there are six picks until our turn, four more RBs are
        assumed gone even if ADP says two. Off, this is exactly the ADP list
        and `runs` is ignored — the behaviour every recorded number was
        measured under.

        It is deliberately **not** taught `LeagueOpponentModel`'s fitted
        curves. Swapping the opponent model changes only the field; keeping
        this independent of it is what makes those runs a comparison of
        fields rather than of two different drafters.
        """
        if picks_until_next <= 0:
            return available
        ordered = available.sort_values("adp_rank")
        out = ordered.iloc[picks_until_next:]
        if not self.run_aware or not runs:
            return out
        gone = ordered.iloc[:picks_until_next]
        for pos, share in runs.items():
            extra = int(round(float(share) * picks_until_next)) \
                - int((gone["position"] == pos).sum())
            if extra > 0:
                idx = out[out["position"] == pos].index[:extra]
                out = out.drop(index=idx)
        return out

    def _tiebreak(self, roster: Roster, position: str, raw_value: float,
                  gain: float = 0.0, upside: float = float("nan")) -> float:
        """Second sort key after drop-off. Raw anchored VORP by default.

        With `upside_tiebreak`, a bench candidate (drop-off zero) sorts on
        his best-case value instead of his median one; anything with a real
        drop-off keeps its raw value, because there the pick is about now.

        With `depth_tiebreak`, a candidate who would only be a backup is
        weighted by the insurance he provides: `BENCH_DECAY ** k` for being
        the k-th body behind the dedicated starters, times the number of
        starters he covers. So a third running back (covers two starters,
        first backup) keeps half his VORP; a second tight end (covers one,
        first backup) keeps a quarter. Negative values — market-anchored K and
        DEF — are left alone, or discounting them would *raise* them.
        """
        if (self.upside_tiebreak and gain <= 0 and upside == upside
                and upside > raw_value):
            raw_value = float(upside)
        if not self.depth_tiebreak or raw_value <= 0:
            return raw_value
        starters = roster.starters.get(position, 0)
        if starters <= 0:
            return raw_value
        have = roster.counts().get(position, 0)
        k = have - starters + 1              # 1 = first backup
        if k <= 0:
            return raw_value
        return raw_value * (self.BENCH_DECAY ** k) * starters

    def choose(self, available: pd.DataFrame, roster: Roster,
               roster_size: int, picks_remaining: int,
               picks_until_next: int = 0,
               runs: dict[str, float] | None = None) -> int:
        """The single pick. A thin wrapper: `rank()` does the actual work and
        this returns its winner, so the two can never disagree with each
        other. Kept because every call site outside this module — the
        backtest, the live monitor's fallback, every existing test — wants
        just the index, not the full breakdown.
        """
        return self.rank(available, roster, roster_size, picks_remaining,
                         picks_until_next, runs=runs)[0].index

    def rank(self, available: pd.DataFrame, roster: Roster,
            roster_size: int, picks_remaining: int,
            picks_until_next: int = 0, *, top_n: int | None = None,
            runs: dict[str, float] | None = None
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
        left = self.survivors(df, picks_until_next, runs)

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

            upside = float("nan")
            if "upside_vorp" in df.columns:
                try:
                    upside = float(df.at[idx, "upside_vorp"])
                except (TypeError, ValueError):
                    pass
            candidates.append(Candidate(
                index=idx, position=pos,
                player_name=str(df.at[idx, "player_name"]),
                gain=gain, value=now,
                raw_value=float(df.at[idx, self.value_col]),
                reason=_drop_off_reason(pos, gain, later_name, picks_until_next),
                upside=upside,
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
        # With `depth_tiebreak` the second key is bench-weighted VORP; see
        # `_tiebreak`. Off, it is `raw_value` exactly as before.
        candidates.sort(
            key=lambda c: (c.gain, self._tiebreak(roster, c.position, c.raw_value,
                                                  c.gain, c.upside)),
            reverse=True)
        return candidates[:top_n] if top_n else candidates


POLICY_SWITCHES = ("run_aware", "depth_tiebreak", "upside_tiebreak")


def policy_from_config(draft_policy: dict | None) -> dict | None:
    """`ValueDrafter`'s optional switches from `config: draft_policy`.

    None when every switch is off, so the recorded baseline's hash and the
    default simulator path are untouched. One parser, used by the backtest
    and the live monitor, so the two can never run different policies.
    """
    dp = draft_policy or {}
    on = {k: True for k in POLICY_SWITCHES if dp.get(k)}
    return on or None


def canonical_order(board: pd.DataFrame) -> pd.DataFrame:
    """The board in an order that does not depend on how it was valued.

    `OpponentModel.choose` draws one gaussian per available player, in row
    order, so a board sorted by VORP hands each player a *different* noise
    draw than the same board sorted by ADP — and two runs that differ only in
    `value_col` were never "same seed, only the ordering changed": the
    2026-09-01 review found the identical pure-ADP drafter finishing 4.35 in
    one run and 4.71 in another, with nothing changed but the board's row
    order. Sorting by consensus rank and then id before the first pick makes
    the noise assignment a function of the players, not of the valuation.
    """
    cols = [c for c in ("adp_rank", "player_id") if c in board.columns]
    if not cols:
        return board
    return board.sort_values(cols, kind="stable")


def gaussian_opponent(sigma: dict[str, float], need_boost: float = 1.6,
                      need_penalty: float = 0.4):
    """The default opponent factory: `(rng, board, teams) -> OpponentModel`.

    A factory rather than an instance because the draft owns the RNG — it is
    seeded per draft, and an opponent carrying its own generator would make a
    run irreproducible from its logged seed. The board and team count are in
    the signature for `LeagueOpponentModel`, which needs both; this one ignores
    them.
    """
    def make(rng: np.random.Generator, board: pd.DataFrame,
             teams: int) -> OpponentModel:
        return OpponentModel(sigma, need_boost, need_penalty, rng)
    return make


def league_opponent(tendencies: LeagueTendencies):
    """`(rng, board, teams) -> LeagueOpponentModel`, for the fitted field."""
    def make(rng: np.random.Generator, board: pd.DataFrame,
             teams: int) -> LeagueOpponentModel:
        return LeagueOpponentModel(tendencies, board, teams, rng)
    return make


def simulate_draft(board: pd.DataFrame, *, teams: int, rounds: int,
                   my_slot: int, starters: dict[str, int], flex_slots: int,
                   flex_eligible: tuple[str, ...], sigma: dict[str, float],
                   need_boost: float = 1.6, need_penalty: float = 0.4,
                   value_col: str = "vorp", seed: int | None = None,
                   caps: dict[str, int] | None = None,
                   opponent_factory=None,
                   policy: dict | None = None,
                   lookahead_rule: str = "next_pick",
                   ) -> dict[int, Roster]:
    """Run one full snake draft.

    `board` needs columns: `player_id`, `player_name`, `position`, `adp_rank`,
    and `value_col`. `my_slot` is 1-indexed.

    `policy` carries `ValueDrafter`'s optional switches (`run_aware`,
    `depth_tiebreak`) from `config: draft_policy`. `None` is the drafter as
    it always was; with `run_aware` on, the recent-picks run detector the
    monitor shows is also fed to the simulated drafter, so the backtest
    scores what the monitor would recommend.

    `opponent_factory` takes the draft's RNG and returns the model the nine
    other seats draft with. `None` builds the gaussian model from `sigma`,
    `need_boost` and `need_penalty` exactly as this function always did — the
    same object, constructed with the same generator, drawing in the same
    order — so every number recorded before the model became swappable still
    reproduces bit-for-bit.

    `lookahead_rule` picks how far ahead our side looks when scoring drop-off:
    "next_pick" (our literal next turn, which is 0 for the first of a
    back-to-back pair) or "next_exposed" (the next turn opponents have actually
    picked before). They differ only at slots 1 and `teams`. See
    `picks_until_board_moves`.

    Returns every team's roster so the scorer can compare ours against the
    league, not just measure ours in isolation.
    """
    try:
        lookahead_for = LOOKAHEAD_RULES[lookahead_rule]
    except KeyError:
        raise ValueError(
            f"unknown lookahead_rule {lookahead_rule!r}; "
            f"expected one of {sorted(LOOKAHEAD_RULES)}") from None
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

    factory = opponent_factory or gaussian_opponent(sigma, need_boost,
                                                    need_penalty)
    opponent = factory(rng, board, teams)
    me = ValueDrafter(value_col, caps=caps, **(policy or {}))
    my_team = my_slot - 1

    available = canonical_order(board).copy()
    order = snake_order(teams, rounds)
    taken_positions: list[dict] = []

    for pick_no, team in enumerate(order, start=1):
        if available.empty:
            break
        roster = rosters[team]
        if roster.is_full(roster_size):
            continue

        picks_remaining = roster_size - len(roster.picks)
        if team == my_team:
            runs = positional_run(taken_positions) if me.run_aware else None
            idx = me.choose(available, roster, roster_size, picks_remaining,
                            lookahead_for(pick_no, teams, my_slot),
                            runs=runs)
        else:
            idx = opponent.choose(available, roster)

        row = available.loc[idx]
        taken_positions.append({"position": str(row["position"])})
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
