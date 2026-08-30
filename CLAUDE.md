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

- **Player identity:** `nflverse` `gsis_id` is the canonical key. Sleeper
  `player_id` and FantasyPros names are *reconciled to it* via
  `src/ingest/player_ids.py`. Never join on player name.
- **Timestamps:** UTC, ISO 8601. Sleeper returns epoch milliseconds — convert on ingest.
- **Caching:** every raw pull lands in `data/raw/` as parquet, partitioned by season.
  Cache is invalidated weekly in-season, never mid-draft.
- **Determinism:** every model/backtest run takes an explicit `seed`. Runs that
  cannot be reproduced from their logged config are bugs.
- **Rate limits:** Sleeper < 1000 req/min (we stay far under). Pro-Football-Reference
  is 20 req/min and will jail you for a day — prefer nflverse, which already
  ingests most PFR-derived data.

## Non-obvious decisions (and why)

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
