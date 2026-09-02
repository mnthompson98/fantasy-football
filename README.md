# fantasy-football

Draft and in-season decision support for a 10-team full-PPR Sleeper league.

Two tranches, built in this order on purpose:

1. **Draft board** — value-based drafting over a blended consensus projection,
   with a hard injury filter and a live draft monitor. Ships first, works alone.
2. **Walk-forward backtest + ML** — multi-year draft simulation scored against
   real outcomes, then a gradient-boosted projection model that has to *earn*
   its place by beating the blend out of sample.

Read [`CLAUDE.md`](CLAUDE.md) before changing anything — it holds the verified
league settings, the leakage rules, and the decisions that aren't obvious from
the code.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

Python 3.14 notes and package-version specifics: [`docs/environment.md`](docs/environment.md).

## Layout

```
config/league.yaml          league settings, backtest window, blend weights
data/league_history/        generated: past drafts + standings + champions
src/ingest/                 Sleeper API, nflverse (lines and weather come from
                            nflverse schedules; there is no odds or weather API)
src/features/               VORP, consensus blend, calibration
src/backtest/               leakage guard, draft sim, metrics, walk-forward
src/draft/                  board construction + live monitor
src/inseason/               start/sit, waivers, trades
scripts/                    entry points
outputs/backtests/runs.parquet   every run, with config hash + git commit
```

## League history

`data/league_history/*.json` is generated, not hand-edited. Regenerate with:

```bash
.venv/Scripts/python.exe -m scripts.build_league_history
```

Each file carries the full draft (order and every pick from every manager),
regular-season standings, resolved final standings, and the champion.

| Season | League | Teams | Champion | My finish |
|---|---|---|---|---|
| 2021 | Etown vs Louisville | 8 | cammack14 | 8th |
| 2025 | Camden? | 10 | cammack14 | 3rd |

Two 2QB novelty leagues (Drakes Miners 2024, Premier League Lads 2025) are
excluded — see `excluded:` in `config/league.yaml`. They'd pollute the
behavioral priors used to model opponents.

## Verified league settings

Pulled from the Sleeper API, not assumed:

- **Waivers: rolling priority, not FAAB.** `waiver_budget: 100` appears in the
  payload but is inert when `waiver_type: 0`. Priority is a positional resource —
  spend it only on genuine difference-makers.
- Playoffs weeks 15–17, 6 teams, top 2 seeded on bye. Week 18 unused.
- Trade deadline week 11.
- Starters `QB/RB/RB/WR/WR/TE/FLEX/K/DEF`, full PPR.

## Draft day

```bash
# build the board
.venv/Scripts/python.exe -m scripts.build_draft_board

# then, during the draft
.venv/Scripts/python.exe -m src.draft.monitor --draft-id <id> --my-slot <n>
```

The board exports three ways: parquet for the backtester, CSV as the
break-glass fallback, and a self-contained HTML file you can open on your phone
— tap a player to strike them off, and the marks persist locally.

## Backtest

```bash
.venv/Scripts/python.exe -m scripts.run_backtest
```

Walk-forward over 2020–2025 with a one-season purge gap. Every run appends to
`outputs/backtests/runs.parquet` with its config hash, seed, and git commit; a
result that can't be reproduced from its logged row is a bug.

**Metric priority is deliberate** and lives in `src/backtest/metrics.py`:

1. playoff-week (15–17) points — what actually wins a championship
2. season points
3. win rate — noisy, schedule-dependent
4. projection MAE — diagnostic only

A model that wins on (4) and loses on (1) has not helped.

## Guardrails

- **The ML model must beat the consensus blend out of sample to be adopted.**
  If it doesn't, the blend ships. Aggregation is the reliable edge; a solo model
  is not.
- **`src/backtest/leakage_guard.py` raises, it does not warn.** A leaky backtest
  produces confident wrong answers, and a warning in a log scroll is not
  protection. It also flags suspiciously stable fold scores, which is the
  classic leakage signature.
- **Realistic expectations.** Best-in-class seasonal projections explain roughly
  14–26% of within-position variance; projections beat ADP by about half a draft
  slot. The edge here is compounding small advantages, not prediction wizardry.
  A result that looks dramatically better than that is a bug.

## Status

Working and tested: Sleeper ingest, standings/bracket resolution, VORP,
consensus blend, calibration, leakage guard, draft simulator, metrics,
walk-forward orchestration, board export, live monitor. 43 tests passing.

Not yet wired: nflverse feature extraction and the projection model itself —
those need a real data pull, which is the next step.
