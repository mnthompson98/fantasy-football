# CLAUDE.md

Project context for Claude Code. Read this before touching anything.

## What this is

A fantasy football draft + in-season decision system for **mnthompson98 / `marwomp`**
(Sleeper user_id `737847784316874752`), targeting the **2026 season** of the
`Camden?` league lineage.

The system is built in two tranches, deliberately:

1. **Draft board (must work first).** Value-based drafting (VORP) over a blended
   consensus projection, with a hard injury/roster filter and a live draft monitor.
2. **Backtest + ML (the long game).** Walk-forward multi-year draft simulation
   scored against real outcomes, then a gradient-boosted projection model on top.

Tranche 1 does not depend on tranche 2. Never let tranche 2 work break tranche 1.

## Verified league settings — do not re-derive these

Pulled from the Sleeper API on 2026-08-30, confirmed for both the 2021 and 2025
league instances. These were *assumptions* in the original plan; they are now facts.

| Setting | Value | Note |
|---|---|---|
| `waiver_type` | `0` | **Rolling waiver order — NOT FAAB.** |
| `waiver_budget` | `100` | Present in the payload but **unused** when `waiver_type=0`. Ignore it. |
| `playoff_week_start` | `15` | Fantasy playoffs = weeks 15–17. Week 18 unused. |
| `playoff_teams` | `6` | Top 2 seeds get a bye. |
| `trade_deadline` | `11` | Week 11. |
| Scoring | full PPR | |
| Starters | `QB, RB, RB, WR, WR, TE, FLEX, K, DEF` | 9 starters |
| Bench | 6 (2021) → 7 (2025) | Assume 7 for 2026 until confirmed. |
| League size | 8 (2021) → 10 (2025) | Assume 10 for 2026 until confirmed. |

**Consequence:** all FAAB bidding logic is out of scope. Waiver priority is a
*positional* resource — burn it only on true difference-makers. See
`src/inseason/waivers.py`.

Re-verify with `python -m scripts.verify_league_settings <league_id>` once the
2026 league exists. Do not hardcode 2026 assumptions before that.

## Code conventions

- **Player identity:** `nflverse` `gsis_id` is the canonical key (`player_key`).
  Sleeper `player_id` and FantasyPros ids are *reconciled to it* via
  `src/ingest/player_ids.py`, anchored on DynastyProcess's `ff_playerids`.
  Never join on player name — name matching exists only as a reported fallback.
  Two entities have no gsis_id and get synthetic keys: team defenses are
  `DEF_<team>` and gsis-less rookies are `FP_<fantasypros_id>`.
  The draft board carries **both** `player_key` and `player_id`: the live
  monitor matches Sleeper's feed on `player_id`, and nothing else can.
- **Timestamps:** UTC, ISO 8601. Sleeper returns epoch milliseconds — convert on ingest.
- **Caching:** every raw pull lands in `data/raw/` as parquet, partitioned by season.
  Cache is invalidated weekly in-season, never mid-draft.
- **Determinism:** every model/backtest run takes an explicit `seed`. Runs that
  cannot be reproduced from their logged config are bugs.
- **Rate limits:** Sleeper < 1000 req/min (we stay far under). Pro-Football-Reference
  is 20 req/min and will jail you for a day — prefer nflverse, which already
  ingests most PFR-derived data.

## Non-obvious decisions (and why)

- **Historical ECR exists, and the backtest uses it.**
  `nflreadpy.load_ff_rankings(type="all")` returns ~1.8M dated FantasyPros ECR
  snapshots back to 2019-12-27. That is what makes the walk-forward honest: a
  fold gets the ranking the room actually held in August of that year, not a
  hindsight ranking dressed up as a projection. Preseason `redraft-overall`
  coverage starts in 2021, so **2020 can be trained on but never targeted**.

- **Backtest window is 2020–2025, not 2010+.** Rule and environment changes make
  older seasons weakly transferable, and FantasyPros ADP coverage thins out before
  ~2014. Six seasons is enough for walk-forward with a purge gap. See
  `src/backtest/walkforward.py`.
- **2024 K/DEF is downweighted.** The 2024 kickoff rule was a one-year anomaly;
  its special-teams scoring distribution does not transfer to 2026.
- **The primary metric is playoff-week (15–17) points, not season total and not
  projection MAE.** A model that projects accurately but drafts players who miss
  weeks 15–17 has not helped. Metric priority is encoded in
  `src/backtest/metrics.py` and must not be reordered casually.
- **Aggregation beats a solo model.** The consensus blend ships before the ML
  model, and the ML model must beat the blend out-of-sample to be adopted. If it
  doesn't, we ship the blend. This is the single most important guardrail here.
- **Calibration correction is applied after projection, not learned inside it.**
  Projection spread is systematically too wide; shrink toward the positional mean.

- **Calibration slopes are measured, not assumed.** The config's priors came
  from a published study and had RB and WR backwards for this league: fit on
  2021-2025 preseason ECR against realized points, RB is 0.85 and WR 0.77, an
  exact swap of the prior 0.79 and 0.85. Since a lower slope flattens a
  position's top end, shrinking RB harder than WR pushed receivers above backs
  everywhere and the simulated drafter built 8 WR / 2 RB rosters.
  `src/features/slopes.py` fits per fold on seasons strictly earlier than the
  target; the config values are now only a fallback.

- **The drafter values a pick by drop-off, not by VORP.** "Take the highest VORP
  available" drafted a backup QB in round 3 of a 1-QB league. Two distinct
  errors, both found by the backtest and both pinned by comments in
  `src/backtest/draft_sim.py`: VORP measures a player against positional
  replacement rather than against *your roster*, and lineup value on a VORP
  scale makes an empty starting slot look identical to a replacement-level
  starter (both zero). Lineup value is on a **points** scale for that reason.

- **The blend weights were tried and are not the problem.** Rebuilding the
  2022-2025 boards under ECR-only, ECR 2/0.5/0.5, ECR 4/0.5/0.5 and equal
  weights moves the top-60 position mix by two or three slots. Even ECR-only —
  the market's own ranking as the sole input — puts ~38 QB and ~30 TE in the top
  60 against the market's 23 and 16. Do not go looking there again; the
  remaining cross-position error is downstream of the blend.

- **FantasyPros point projections are not obtainable; ECR is.** As of 2026-08-30
  fantasypros.com server-renders only the top 10 rows of each projections table,
  so the planned "FantasyPros PPR consensus projection" component does not exist.
  What we get instead is their **expert consensus ranking**, redistributed by
  ffverse (`nflreadpy.load_ff_rankings`), which `src/features/rank_curve.py`
  converts to points through a historical positional rank→points curve.

- **K and DEF are priced at the market, not at their VORP.** Value-based
  drafting says the best defense out-scores the waiver defense by ~12 points a
  season — true — and concludes you should spend a sixth-round pick on him,
  which is not. The market prices two things VORP does not model: those points
  are unpredictable in advance, and the position is streamable all year, so the
  *pick* buys far less than the *points* imply. `src/features/market_anchor.py`
  blends an anchored position's VORP rank toward its ADP rank
  (`config: market_anchor`). DEF is anchored hard (sits at ADP); K is anchored
  loosely, so a genuinely exceptional leg can still surface.

  **Anchor in rank space, never in value space.** Replacing an anchored player's
  VORP with the value the market implies for his slot does not work: that value
  is a conditional mean, so everyone scattered above the mean sorts past him and
  the position lands ~55 spots *above* its ADP instead of at it. Ranks compose
  the way the intuition expects; values do not. Both failure modes are pinned by
  tests in `tests/test_market_anchor.py`.

- **The board and the backtest share one valuation pipeline.**
  `src/features/pipeline.py` is called by both `scripts/build_draft_board.py`
  and `scripts/run_backtest.py`. If they diverge, the backtest is scoring a
  program nobody drafts with and every conclusion from it is about the wrong
  system. The *only* thing the live board adds is Sleeper injury status and
  current team — which the backtest must never touch, because that data is
  current rather than historical and using it in a fold is direct leakage.

- **The rank curve is fit on finish rank, so it must be shrunk.** Mapping a
  *projected* rank through a *finish*-rank curve assumes the projection is as
  good as hindsight, which makes the spread too wide. `calibration.py` handles
  the general case. K and DEF get a second, measured shrink on top
  (`calibration.persistence_shrink` in the config) because their rank ordering
  barely persists year to year — measured on 2020–2025, Spearman of prior-year
  rank vs next-year points is 0.26 for DEF and 0.34 for K against 0.62 for QB.
  It is scoped to those two positions on purpose; see the docstring in
  `rank_curve.persistence_shrink` for why applying it to skill positions would
  be wrong. **This is the one place a measured number stands in for a validated
  one — the walk-forward backtest should settle it.**

## Leakage rules — the failure mode that will silently ruin this

When simulating season Y:

- Features may use data through **Y-1 only**, plus Y's *preseason* ADP/injury/roster
  snapshot. Nothing from Y's regular season.
- ADP must be the **as-of-draft-day** snapshot, never end-of-season ADP.
- Training folds are **purged** — a gap season between train and test — so
  end-of-train information doesn't bleed forward.
- **Tell:** if walk-forward scores are suspiciously stable across folds, assume
  leakage before assuming skill.

`src/backtest/leakage_guard.py` enforces the temporal boundary at runtime and
raises rather than warns. Do not downgrade it to a warning.

## Realistic expectations

Best-in-class seasonal projections explain ~14–26% of within-position variance;
weekly R² is 3–23%. Pro QB systems cluster within ~0.25 MAE of each other.
Projections beat ADP by roughly half a draft slot.

The edge here is compounding small advantages — aggregation, calibration,
expected-points regression signals, and never wasting a pick or a claim on an
unavailable player. It is not weekly clairvoyance. Any result that looks
dramatically better than the above is a bug, not a breakthrough.

## Environment note

Local Python is **3.14**, which is newer than some of this stack's wheels.
`requirements.txt` pins what is known to resolve. If a package fails to build,
check `docs/environment.md` before fighting it.
