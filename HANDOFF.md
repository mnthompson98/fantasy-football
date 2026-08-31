# Handoff: Draft Board & Backtest System

**Last updated:** 2026-08-30
**Repo:** https://github.com/mnthompson98/fantasy-football
**Branch:** main
**Tests:** 153/153 passing on Python 3.14
**2026 league:** `Camden?` · league `1389723592727461888` · draft `1389723592727461889` · 10 teams · 16 rounds · snake · pre_draft

---

## The four commands

```bash
# 1. Confirm nothing changed. Run this first, every time.
.venv/Scripts/python.exe -m scripts.verify_league_settings 1389723592727461888

# 2. Build the board. Do this again the morning of the draft.
.venv/Scripts/python.exe -m scripts.build_draft_board --league-id 1389723592727461888 --refresh

# 3. During the draft. `--my-slot` is your draft position, 1-10.
.venv/Scripts/python.exe -m src.draft.monitor --draft-id 1389723592727461889 --my-slot <n>

# 4. Whenever you change the valuation, check it against 2022-2025.
.venv/Scripts/python.exe -m scripts.run_backtest --drafts 40 --label what-changed
```

**You do not know `--my-slot` yet.** The commissioner has not set the draft
order (`draft_order: null`, no start time). Check Sleeper once it is drawn.

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
- ✅ `src/features/lineup.py` — best legal lineup; shared by all three in-season modules
- ✅ `src/inseason/projections.py` — weekly FantasyPros points, canonically keyed
- ✅ `src/inseason/roster.py` — who owns whom, from Sleeper
- ✅ `src/inseason/start_sit.py` — lineup + close calls, with uncertainty
- ✅ `src/inseason/waivers.py` — rolling-priority claims vs free streaming
- ✅ `src/inseason/trades.py` — trades valued by lineup change, not rankings
- ✅ `scripts/weekly_update.py` — **the weekly run, incl. `--check-only` ingestion health**
- ✅ 143 unit tests

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
1. ✅ **2026 league confirmed** — `Camden?`, league `1389723592727461888`,
   draft `1389723592727461889`, 10 teams, 16 rounds, snake. All nine checked
   settings matched and the roster shape has been promoted into `verified`.
   Re-run `verify_league_settings` if the commissioner changes anything.
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
6. ✅ **In-season automation built** — `start_sit.py`, `waivers.py`,
   `trades.py`, driven by `scripts/weekly_update.py`. **Unvalidated by
   construction:** the weekly FantasyPros feed is a live snapshot with no
   archive, so unlike the draft board none of it can be backtested.

---

## Audit against the original plan

Checked item by item, 2026-08-31.

### "By Thursday" — all five shipped

| Plan item | State |
|---|---|
| 1. nflverse seasonal/weekly + xFP + FantasyPros ranks | ✅ `ingest/nflverse.py` |
| 2. Full-PPR VORP cheat sheet, exact roster settings | ✅ verified against the live 2026 league |
| 3. Blend with consensus (aggregation > solo model) | ✅ `features/blend.py` |
| 4. Injury/roster hard filter (Sleeper **+ nflverse**) | ✅ — see note below |
| 5. Live draft-day poll of `/draft/{id}/picks` | ✅ `draft/monitor.py`, verified on the real 2025 draft |

**On "Sleeper + nflverse":** these cover different phases and neither replaces
the other. nflverse's injury report starts at **week 1** and does not exist for
2026 at all yet, so at draft time Sleeper is the only live source — it is what
carries IR/PUP/NFI/suspended. nflverse becomes the authority once games are
played, and `ingest/injuries.py` now supplies it to the weekly run.

### "Weeks 1–3" — shipped, with one scope correction

| Plan item | State |
|---|---|
| Walk-forward backtester + draft simulator | ✅ |
| Validate on **2019–2025** | ⚠️ **2022–2025 only** |
| Wire up weekly ingestion | ✅ + `--check-only` health check |

**The 2019–2021 folds are not obtainable.** ffverse's preseason `redraft-overall`
ECR archive begins in 2021, and the one-season purge gap costs another, so the
earliest targetable season is 2022. Four folds, not seven. Everything concluded
from the backtest rests on that sample — it is the single biggest limitation in
the project and no amount of further tuning changes it.

### "Weeks 4+" — partially shipped

| Plan item | State |
|---|---|
| Calibration | ✅ and *measured*, not assumed (found RB/WR priors inverted) |
| Uncertainty ranges | ✅ ECR `sd`/`best`/`worst` on the board; weekly `sd` drives start/sit toss-ups |
| Weekly automation | ✅ `scripts/schedule_weekly.py` (Task Scheduler / cron) |
| **ML projection model** | ❌ **not built** |
| **Injury-risk feature** | ❌ **not built** |

The ML model must beat the blend out-of-sample to ship at all (CLAUDE.md), and
it would be validated on those same four folds. That is a weak basis for
adopting a model over a consensus blend, and worth knowing before starting.

### In-season automation — shipped

Every element of the scheduled job the plan describes:

| Plan item | State |
|---|---|
| Runs Tue night / Wed morning | ✅ `schedule_weekly.py`, default TUE 20:00 |
| Injuries **+ practice reports** | ✅ practice participation drives the `risky` flag |
| Trending adds **and drops** | ✅ adds → waivers; drops → buy-low trade targets |
| This week's matchups / odds / weather | ✅ implied team total, wind, cold, dome |
| Recomputes projections | ✅ weekly FantasyPros points |
| Recomputes **VORP** | ◑ see note |
| Emits a markdown report | ✅ `outputs/reports/weekNN.md` |

**Two deliberate departures from the plan, both worth arguing with:**

1. **Odds and weather are flags, not adjustments.** The obvious move is to nudge
   projections for a high total or a windy game. Don't: the FantasyPros experts
   who made those projections had already seen the spread and the forecast, so
   adjusting on top double-counts the matchup and makes the number worse. The
   context explains a number you are about to act on; it does not modify it.

2. **VORP is not recomputed weekly, because in-season it is the wrong quantity.**
   VORP prices a player against a *draftable replacement* — the right question in
   August. In week 6 the question is whether he improves *your* starting lineup
   this Sunday, which is what `features/lineup.py` computes and what every weekly
   recommendation is built on. Recomputing VORP would produce a number nothing
   should act on.

---

## Backtest baseline

160 simulated drafts per run (40 per season, 2022-2025), varied slots and seeds.
5.50 is what a coin flip gets on league rank.

| label | scorer | rank | win% | playoff | season |
|---|---|---|---|---|---|
| `baseline-anchored` | old | 6.60 | 0.456 | 355.8 | 2093 |
| `fitted-slopes+dropoff` | old | 4.90 | 0.532 | 389.9 | 2200 |
| `caps-only-no-qbte-anchor` | old | 5.61 | 0.517 | 374.8 | 2167 |
| `qb-te-anchor+caps` | old | 5.74 | 0.498 | 369.7 | 2141 |
| `caps+waiver-aware-scoring` | corrected | 4.64 | 0.548 | 405.3 | 2294 |
| `qbcap2+waiver-aware-CONTROL` | corrected | 4.73 | 0.528 | 411.1 | 2292 |
| `qbte-anchor-RETEST-waiver-aware` | corrected | 4.78 | 0.531 | 404.2 | 2287 |
| `rolling-window-3seasons` | corrected | 4.39 | 0.563 | 404.7 | 2307 |
| **`rolling3+prekickoff-ecr`** (shipping) | **corrected** | **4.42** | **0.556** | **407.8** | **2310** |

The shipping row beats `caps+waiver-aware-scoring` on all four metrics, so
CLAUDE.md's adoption rule (improve playoff points without hurting season points)
is satisfied without interpretation.

**Two caveats on reading this table.** Rows under different scorers are not
comparable at all. And `rolling-window-3seasons` and `rolling3+prekickoff-ecr`
share a `config_hash` *and* a `git_commit` despite differing — the snapshot-date
fix was a code change made in an uncommitted working tree. Commit between runs
that change code, or the log cannot tell them apart.

**Rows under different scorers are not comparable to each other.** The corrected
scorer credits every team for streaming, so absolute points rise for everyone;
only same-scorer rows can be compared.

What moved the needle, in order:

1. **Calibration slopes fit from history** rather than the config's generic
   priors, which had RB and WR inverted (see CLAUDE.md).
2. **The drafter values picks by drop-off**, on a points scale, not raw VORP.
3. **Waiver-aware scoring.** See below — this was a measurement bug, not a
   strategy improvement, and it was worth about a place in the standings.

Tried and reverted: the QB/TE market anchor (cost 0.13 places).

### 2025 IS an outlier, and it is the most valuable open question

**This supersedes an earlier reading in this file that called 2025 "the end of a
trend". The draft-outcome evidence says otherwise.**

Running the simulator with the player ordering swapped for raw ADP — same pick
policy, same caps, same scorer, only the ranking changed — isolates what the
valuation pipeline is worth:

| season | board | pure ADP | board edge |
|---|---|---|---|
| 2022 | 3.28 | 4.28 | **+1.00** |
| 2023 | 2.95 | 4.75 | **+1.80** |
| 2024 | 3.82 | 5.00 | **+1.17** |
| 2025 | 7.62 | 4.00 | **−3.62** |
| all | 4.42 | 4.51 | +0.09 |

The board beats the market by roughly **1.3 places in three seasons of four**,
then loses by 3.6 in one. The aggregate near-tie is entirely that one season
dragging down three good ones — and reading only the aggregate would have led
to exactly the wrong conclusion, which is that the valuation adds nothing.

At 40 drafts per season the per-season standard error on league rank is roughly
0.4, so both the ~1.3 wins and the 3.6 loss are well outside noise *within the
simulation*. Four folds is still four folds.

**Reproduce with:** `--value-col adp_value` against the default `vorp`.

The earlier rank-correlation table (below) showed a smooth decay and pointed at
a gradual erosion. The draft outcomes do not agree: 2024 is a clear win, not a
slide into parity. Where the two disagree, prefer the draft outcomes — they
measure the thing the system actually does.

### The rank-correlation view (kept, but it misled)

Our whole edge is board-versus-market, since opponents draft near ADP. Rank
correlation with realized points over each season's draftable top 160:

| season | board ρ | ADP ρ | board edge | fold rank |
|---|---|---|---|---|
| 2022 | −0.582 | −0.486 | **+0.096** | 3.27 |
| 2023 | −0.524 | −0.489 | +0.034 | 2.95 |
| 2024 | −0.454 | −0.472 | −0.019 | 3.83 |
| 2025 | −0.394 | −0.426 | **−0.033** | 7.62 |

The board beat the market in 2022–23 and has lost to it since, decaying
monotonically, and the fold ranks track it. So the question is not "what broke in
2025" but "why is the edge eroding".

**Ruled out so far:**

- *QB/TE overdraft.* Real (the board takes ~10 QB in its top 60 against the
  market's 5) but anchoring it moved 2025 by 0.03 — nothing.
- *Stale training seasons.* The rolling 3-season window fixed **2024**
  (4.88 → 3.83) and left 2025 alone.
- *Post-kickoff ECR snapshot.* 2025 was genuinely using a snapshot taken after
  the season opener. Fixed — and 2025 got slightly *worse* (7.53 → 7.62), which
  is what removing a game of hindsight should do.

**Still untested:** that the market simply got better and the edge was never
durable. Four folds cannot separate that from a modelling flaw, and two of them
now come from a rolling window that only differs in the last two seasons.
Anything concluded from 2025 alone rests on 40 simulated drafts of one season.

### The scorer was punishing thin rosters for a missing feature

`optimal_lineup_points` used to score an unfilled starting slot as zero. The
simulator has no waiver wire, so a roster carrying one quarterback fielded
**nobody** at QB 3.10 weeks a season (one bye plus injuries, against 0.85 weeks
for a two-QB roster) and was charged the entire loss — about 75 QB points a
year. A real manager streams a replacement for free.

That made a one-QB strategy look 0.71 places worse than it is. It would have
biased *every* roster-shape question the same way: "carry a second TE?", "how
deep at RB?". An unfilled slot is now credited with what a freely-available
player at that position actually scored that week — the median of the tier just
below what the league rosters as starters (QB11-QB20 in a 10-team, 1-QB league),
not the best available, which would assume foresight nobody has.

### One QB costs nothing measurable

With the scorer corrected, capping QB at 1 versus 2:

| | rank | win% | playoff | season |
|---|---|---|---|---|
| QB cap 1 | 4.64 | 0.548 | 405.3 | 2294 |
| QB cap 2 | 4.73 | 0.528 | 411.1 | 2292 |

Mixed in direction and small in every case — cap 1 is ahead on rank and win
rate, cap 2 on playoff points, season points a tie. Treat it as **no measurable
difference over 160 drafts**, not as evidence either way. The entire apparent
penalty on one-QB rosters was the scorer artifact.

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

### 2b. SETTLED — the market was right about QB and TE

Realized value above that season's realized positional replacement, by board
rank bucket, 2022-2025:

| bucket | QB | RB | WR | TE |
|---|---|---|---|---|
| 1–24 | **+19** | +62 | +55 | +57 |
| 25–48 | **−12** | +45 | +2 | −5 |
| 49–72 | **−71** | +9 | −24 | −5 |

The same table by ADP rank is roughly level (1–24: QB +64, RB +59, WR +53,
TE +93), which is what a well-calibrated cross-position ranking looks like. The
board was putting ~4.75 QB per season into its top 24 and they returned a third
of what everything else at the same rank returned.

Fixed by extending `market_anchor` to QB (0.6) and TE (0.5) — the same machinery
K and DEF use, for a different reason: not streamability, but that our
cross-position pricing of them is measurably worse than the market's. The 2026
board's top-60 mix now matches ADP almost exactly:

| | WR | RB | QB | TE |
|---|---|---|---|---|
| board | 29 | 20 | 6 | 5 |
| ADP | 29 | 21 | 6 | 4 |

Josh Allen still sits at board 20 against ADP 26 — the one quarterback the board
is allowed to like more than the room does.

**The two anchor weights are a reasoned starting point, not a fitted result.**
Four folds cannot tune them. If you revisit, change one and compare labelled
runs rather than eyeballing a board.

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
- **Season downweighting is off.** The 2024 K/DEF downweight was removed on
  the manager's call rather than tested. `backtest.downweight` is `[]`.

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
