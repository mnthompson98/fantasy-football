# Handoff: Draft Board & Backtest System

**Last updated:** 2026-08-30
**Repo:** https://github.com/mnthompson98/fantasy-football
**Branch:** main
**Tests:** 99/99 passing on Python 3.14

---

## The four commands

```bash
# 1. Confirm the league is what the config thinks it is. Run this first.
.venv/Scripts/python.exe -m scripts.verify_league_settings <2026_league_id>

# 2. Build the board. Do this again the morning of the draft.
.venv/Scripts/python.exe -m scripts.build_draft_board --league-id <2026_league_id> --refresh

# 3. During the draft.
.venv/Scripts/python.exe -m src.draft.monitor --draft-id <draft_id> --my-slot <n>

# 4. Whenever you change the valuation, check it against 2022-2025.
.venv/Scripts/python.exe -m scripts.run_backtest --drafts 40 --label what-changed
```

Set `PYTHONIOENCODING=utf-8` first — Windows defaults to cp1252 and will crash
on the first accented player name.

## Where the board lands

`build_draft_board` writes to `outputs/projections/`:

| File | Use |
|---|---|
| `draft_board.html` | **The Thursday artifact.** Self-contained, offline, phone-sized. Tap a player to strike him; marks persist in local storage. |
| `draft_board.csv` | Spreadsheet fallback if Python breaks. |
| `draft_board.parquet` | What the live monitor and the backtester read. |
| `excluded_players.csv` | Everyone the injury gate removed, so nothing vanishes silently. |

Open `draft_board.html` in any browser — double-click it, or email it to
yourself and open it on your phone. It needs no server and no network.

The monitor is verified end-to-end against the completed 2025 draft: it read all
160 real picks off Sleeper and removed exactly those players from the board.

Backtest results append to `outputs/backtests/runs.parquet`, one row per
simulated draft plus an aggregate, tagged with the git commit and config hash.

---

## What's done

### Infrastructure
- ✅ Git repo initialized, remote pushed
- ✅ Python 3.14 venv with nflreadpy, pandas 3, xgboost, lightgbm
- ✅ League history datasets generated from Sleeper API (2021, 2025)
- ✅ League settings verified and documented (rolling waivers, weeks 15–17 playoffs)

### Core modules (all tested)
- ✅ `src/ingest/sleeper_api.py` — Sleeper read-only client, standings/bracket resolution
- ✅ `src/ingest/nflverse.py` — cached nflverse/ffverse pulls, Polars→pandas at the boundary
- ✅ `src/ingest/player_ids.py` — gsis-anchored crosswalk (FantasyPros ↔ Sleeper ↔ nflverse)
- ✅ `src/ingest/history.py` — season-parameterized history + preseason ECR snapshots
- ✅ `src/features/pipeline.py` — **the shared valuation, used by board and backtest alike**
- ✅ `src/features/market_anchor.py` — prices K and DEF at what the room pays
- ✅ `src/features/scoring.py` — league scoring settings applied to box scores, incl. K and DEF
- ✅ `src/features/rank_curve.py` — positional rank→points curves + persistence shrink
- ✅ `src/features/vorp.py` — flex-aware replacement level, cross-position comparison
- ✅ `src/features/blend.py` — within-position z-score consensus
- ✅ `src/features/calibration.py` — post-hoc spread shrink + in-season refitting
- ✅ `src/backtest/leakage_guard.py` — temporal boundary enforcement, raises (not warns)
- ✅ `src/backtest/draft_sim.py` — ADP-sampled opponent draft, positional need, scarcity backstop
- ✅ `src/backtest/metrics.py` — metric priority (playoff pts > season pts > win rate > MAE)
- ✅ `src/backtest/walkforward.py` — expanding-window orchestration, run logging
- ✅ `src/draft/board.py` — parquet + CSV + self-contained HTML
- ✅ `src/draft/monitor.py` — live Sleeper poll, best-available, positional-run detection
- ✅ `scripts/build_league_history.py` — regenerate datasets from API
- ✅ `scripts/build_draft_board.py` — **the board, end to end**
- ✅ `scripts/verify_league_settings.py` — check a live league against the config
- ✅ `scripts/run_backtest.py` — **walk-forward, on real historical ECR**
- ✅ 99 unit tests

### The pipeline the board actually runs

1. **Identity** — 13,801-player crosswalk; 515 of 517 ranked players reconcile.
   The two misses are ECR 294 and ECR 306, i.e. nobody.
2. **History** — six seasons of nflverse box scores scored under *this league's*
   settings. Offense agrees with nflverse's own `fantasy_points_ppr` to a mean
   absolute difference of 0.06 (the gap is two-point conversions, which Sleeper
   scores and nflverse does not). K and DEF are computed from scratch because
   nflverse does not score them at all.
3. **Rank curve** — monotone rank→points per position, isotonic-fit.
4. **Components** — three per player, all on a full-PPR scale:
   - `ecr_points` — expert consensus rank through the curve (515/515 coverage)
   - `xfp_points` — 2025 expected fantasy points, per-game, shrunk (365/515)
   - `prior_points` — 2025 actual points, per-game, shrunk (421/515)
5. **Blend → calibrate → persistence shrink → VORP → injury gate → export.**

### Verified this session
- Offense scoring reconciles against nflverse to 0.06 mean absolute error
- ffopportunity's expected points are **full PPR** (implied value of a reception
  is exactly 1.0), so they need no rescaling for this league
- The live monitor correctly consumes the board and matches real Sleeper picks

### Bugs found and fixed
1. *(earlier)* Summing per-position need triple-counted flex → empty QB slots
2. *(earlier)* Backstop assumed the needed position would still be available
3. **The board had a null `player_id` column.** Everything joined on the
   canonical `player_key`, but the live monitor matches Sleeper's draft feed on
   `player_id` — so the monitor would have struck out *nobody* and happily
   recommended players who were already drafted. Mid-draft, silently. The board
   now carries both ids and a test pins the contract.
4. **The HTML board pushed VORP off-screen on a phone.** Ten columns at 375px
   meant the number that decides every pick required a sideways scroll. Team,
   bye and injury now ride under the player name; VORP and ADP fit.

---

## What's NOT done (next steps)

### Before Thursday
1. **Confirm the 2026 league settings** once the league exists —
   `python -m scripts.verify_league_settings <league_id>`, then promote
   `assumed_2026` (10 teams, 7 bench) to `verified` in `config/league.yaml`.
2. **Rebuild the board the morning of the draft** with `--refresh`. The ECR
   snapshot is the perishable input; anything older than a day has missed news.
3. **Do a mock draft with the monitor running** to shake out the ergonomics.

### After the draft
4. **Settle the open modelling questions below with the backtest**, not by
   argument. `--label` tags a run so two configurations can be compared in
   `outputs/backtests/runs.parquet`. The obvious first experiments:
   - blend weights (ECR 2.0/0.5/0.5 vs equal vs ECR-only)
   - market anchor on vs off
   - persistence shrink on vs off
5. **XGBoost projection model** — must beat the blend out-of-sample or the blend
   ships.
6. **In-season automation** — `start_sit.py`, `waivers.py`, `trades.py`.

---

## Backtest baseline

160 simulated drafts (40 per season, 2022-2025), varied slots and seeds,
`--label fitted-slopes+dropoff`:

| metric | first run | after the fixes below |
|---|---|---|
| league rank by points (of 10, lower better) | 6.60 | **4.90** |
| win rate | 0.456 | **0.532** |
| playoff points (weeks 15-17) | 355.8 | **389.9** |
| season points | 2093 | **2200** |

5.50 is what a coin flip gets. The first run was meaningfully *worse* than an
ADP-following field; it is now meaningfully better. Two fixes did it:

1. **Calibration slopes fit from history** rather than the config's generic
   priors, which had RB and WR inverted (see CLAUDE.md).
2. **The drafter values picks by drop-off**, on a points scale, rather than by
   raw VORP.

Per season: 2022 4.35, 2023 2.75, 2024 4.58, **2025 7.92**. 2025 is the one
season it still loses badly, and nobody has looked at why. That is the most
promising thread left.

Caveat worth keeping in mind: the two fixes went in together and were measured
together. Neither has been scored on its own.

---

## Open questions the backtest should settle

These are places where the board makes a defensible choice that has not been
*validated*. They are listed in the order they are likely to cost points.

### 1. RESOLVED — K and DEF are now priced at the market

Naive VBD put a defense at overall rank 32 and a kicker at 35. Two corrections,
in sequence:

- **Persistence shrink** (measured): DEF and K rank orderings barely persist
  year to year, so a finish-rank curve overstates their spread. Moved them to
  56 and 53.
- **Market anchor** (`src/features/market_anchor.py`): blends an anchored
  position's VORP *rank* toward its *ADP rank*. DEF at weight 1.0 sits exactly
  at its ADP; K at 0.7 stays mostly market-priced but lets a genuinely good leg
  climb.

Where they land now:

| | board | ADP | round (10-team) |
|---|---|---|---|
| Houston DEF | 146 | 146 | 15 |
| Denver DEF | 156 | 155 | 16 |
| Brandon Aubrey (K) | 140 | 176 | 14 |
| Ka'imi Fairbairn (K) | 149 | 188 | 15 |

Ordering *within* each position still follows our own valuation — Houston comes
off the board before Denver, Aubrey before Fairbairn — which is the point of
capping the anchor just below 1.0.

**Still worth backtesting:** the anchor weights (DEF 1.0, K 0.7) are a stated
preference, not a fitted result. `run_backtest.py --label` exists so you can
score a board with and without them and compare.

### 2. TRIED AND REJECTED — the blend weights are not the problem

Rebuilding the 2022-2025 boards under four weightings, and comparing each one's
top-60 position mix against the market's:

| weights | QB | RB | WR | TE |
|---|---|---|---|---|
| ECR-only | 38 | 64 | 108 | 30 |
| ECR 2 / .5 / .5 (current) | 38 | 67 | 103 | 32 |
| ECR 4 / .5 / .5 | 35 | 70 | 105 | 30 |
| equal 1/1/1 | 35 | 68 | 106 | 31 |
| **ADP (market)** | **23** | **82** | **119** | **16** |

(totals over four seasons; divide by four for per-season)

The weights move the mix by two or three slots out of sixty. Even ECR-only —
the market's own ranking as the *sole* input — lands 38 QB and 30 TE in the top
60 against the market's 23 and 16. Whatever is wrong is downstream of the blend.

A calibration hypothesis was also tried and rejected: calibration shrinks each
position toward its *pool* mean, and the pools differ hugely in depth (147 RB vs
62 QB), so the RB top end should get dragged toward a hundred undraftable backs.
Shrinking toward replacement level instead changed almost nothing, because the
fitted slopes are near-uniform (0.68-0.72) and uniform scaling is
rank-preserving. It was reverted rather than kept as an unjustified change.

### 2b. The live question: QB and TE are overdrafted

The board takes ~9.5 QB and ~7.5 TE per season in its top 60; the market takes
~5.75 and ~4. That ~7-slot overdraft is what squeezes out the backs.

Note the board's RB:WR *ratio* (0.65) is close to the market's (0.68) — so the
8 WR / 2 RB rosters were the drafter compounding a smaller board error, not the
board being receiver-mad.

**Nobody has established whether the board or the market is right about QB/TE.**
The test to run: realized value-above-replacement for the board's top-60
selection versus ADP's top-60 selection, per position. If the board's QBs
genuinely deliver more points above QB replacement than the market's extra
RB/WR do above theirs, the board is right and the market is wrong. Since the
error survives ECR-only, it lives in the rank curve or the replacement levels —
`fit_rank_curve` and `replacement_ranks` are where to look.

### 3. The rank curve conflates projected rank with finish rank

Documented in `rank_curve.py` and partially corrected by calibration. The clean
fix needs historical *preseason* ECR snapshots, which ffverse does not keep — it
serves only the current scrape. If a historical ADP source turns up, that curve
can be fit properly and this whole correction chain gets simpler.

---

## Known limitations and caveats

### Before you trust the backtest
- **Four scorable folds, and that is all there will be.** Targets are 2022-2025:
  ffverse preseason ECR starts in 2021 and the purge gap costs one more season.
  With ~40 drafts each that is enough to rank two configurations against each
  other, and nowhere near enough to justify a subtle parameter.
- **The lineup scorer is hindsight-optimal.** `optimal_lineup_points` starts the
  best legal lineup in each week, which no real manager achieves. It is applied
  identically to every simulated team, so the *comparison* is fair; treat the
  absolute point totals as an upper bound.
- **Opponents follow ADP with gaussian noise.** Real managers are worse than
  that in some ways and better in others. A strategy that only beats this field
  has not proven much.
- **Rule changes.** The 2024 kickoff rule was a one-year anomaly; the config
  downweights 2024 K/DEF to 0.3.

### Realistic expectations
- Best-in-class seasonal projections explain ~14–26% of within-position variance
- Projections beat ADP by ~half a draft slot
- The profit is compounding small edges, not prediction wizardry
- A result that looks dramatically better than the above is a bug

### Waiver system — verified as rolling priority
- Both leagues use `waiver_type: 0` (rolling priority), **not FAAB**
- `waiver_budget: 100` is in the payload but inert at this setting
- Priority is a positional resource — spend it only on true difference-makers
- Confirm in the 2026 settings before the season starts

---

## Environment

**Python:** 3.14.7 (no 3.14 wheels for `nfl_data_py`; use `nflreadpy`)

```bash
cd "C:\Users\markt\OneDrive\Desktop\Repositories\Fantasy Football"
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

Set `PYTHONIOENCODING=utf-8` before any script that prints player names —
Windows defaults to cp1252 and will crash on the first accented name.

**Caching.** Every raw pull lands in `data/raw/` as parquet. Rankings go stale
after 12 hours; completed-season stats never change. If a pull fails and a cache
exists at any age, the cache is used and the staleness is printed — losing the
board on draft morning because ffverse is briefly down is the worse failure.

---

## Project decisions (encoded as guardrails, not prose)

1. **Aggregation over solo model.** Consensus blend ships first. ML must beat it
   out-of-sample to be adopted.
2. **Metric priority is deliberate.** Playoff-week (15–17) points, then season
   points, then win rate, then projection MAE.
3. **Leakage raises, not warns.** The board build asserts its own temporal
   boundary before it uses a single historical row.
4. **Roster legality is enforced.** The simulator never leaves a starting slot
   unfilled.
5. **Rolling waivers, not FAAB.**
6. **Nothing is dropped silently.** Unreconciled players, injury-gated players
   and players missing a Sleeper id are all counted and printed. A board that
   quietly loses a top-40 rookie is worse than no board.

---

## Quick reference

| File/Command | Purpose |
|---|---|
| `CLAUDE.md` | Project context, verified settings, decision rationale (read first) |
| `config/league.yaml` | League settings, blend weights, calibration, backtest window |
| `scripts/build_draft_board.py` | **Build the board** |
| `src/draft/monitor.py` | Live Sleeper poll during the draft |
| `src/ingest/player_ids.py` | The crosswalk. Read this before touching any join |
| `src/features/scoring.py` | League scoring applied to nflverse box scores |
| `src/features/rank_curve.py` | Rank→points curves, persistence shrink |
| `src/features/vorp.py` | VORP with flex-aware replacement level |
| `src/features/pipeline.py` | The shared valuation. Change this, not a copy of it |
| `src/features/market_anchor.py` | K/DEF priced at market. Read the docstring first |
| `scripts/verify_league_settings.py` | Check a live league against the config |
| `scripts/run_backtest.py` | Walk-forward over 2022–2025 |
| `scripts/build_league_history.py` | Regenerate league history from API |
| `.venv/Scripts/python.exe -m pytest tests/ -q` | 99 tests |

---

## Contact & continuity

**Principles this was built on:**
- Two tranches: lean draft tool (ship first) + full backtest (ship after)
- Realistic edge is compounding small advantages, not prediction magic
- Leakage is the thing that will silently ruin everything, so guard aggressively
- Aggregation beats a solo model, so the consensus blend is the fallback
- Anything dropped gets counted and printed

If you're resuming this later, re-read `CLAUDE.md` and check `git log --oneline`.
