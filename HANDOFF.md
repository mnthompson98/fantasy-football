# Handoff: Draft Board & In-Season System

**Last updated:** 2026-08-31
**Repo:** https://github.com/mnthompson98/fantasy-football
**Branch:** main · **Tests:** 196 passing + 1 gated skip, Python 3.14
**2026 league:** `Camden?` · league `1389723592727461888` · draft `1389723592727461889`
10 teams · 16 rounds · snake · status `pre_draft`

Read `CLAUDE.md` first — it holds the verified league settings and the
decisions-with-reasons. This file is state, progress and what to do next.

---

## The commands

```bash
# Confirm nothing changed. Run first, every time.
.venv/Scripts/python.exe -m scripts.verify_league_settings 1389723592727461888

# Build the board. Run again the morning of the draft.
.venv/Scripts/python.exe -m scripts.build_draft_board --league-id 1389723592727461888 --refresh

# During the draft. --my-slot is your draft position, 1-10. It is checked
# against Sleeper's own draft_order and refuses to run if they disagree.
.venv/Scripts/python.exe -m src.draft.monitor --draft-id 1389723592727461889 --my-slot <n>

# Rehearse the above against a real completed draft, at full speed.
.venv/Scripts/python.exe -m scripts.replay_draft --draft-id 1262443300887994368 --my-slot 4

# Weekly, in season. --check-only verifies ingestion and stops.
.venv/Scripts/python.exe -m scripts.weekly_update

# Whenever the valuation changes. Compare against --value-col adp_value.
.venv/Scripts/python.exe -m scripts.run_backtest --drafts 40 --label what-changed

# ...or let the parity guard run both orderings and diff them for you.
.venv/Scripts/python.exe -m scripts.check_backtest_baseline
```

Set `PYTHONIOENCODING=utf-8` first — Windows defaults to cp1252 and dies on the
first accented player name.

**You still do not know `--my-slot`.** The commissioner has not drawn the order
(`draft_order: null`, no start time). Check Sleeper once it is up.

### Outputs

| Path | What |
|---|---|
| `outputs/projections/draft_board.html` | **The draft-day artifact.** Self-contained, offline, phone-sized. The monitor rewrites it as picks land, so every team's picks strike out, not just yours. Tap anyone else to mark them; taps persist locally. |
| `outputs/projections/draft_board.{csv,parquet}` | Spreadsheet fallback; parquet is what the monitor and backtester read. |
| `outputs/projections/excluded_players.csv` | Everyone the injury gate removed, so nothing vanishes silently. |
| `outputs/reports/weekNN.md` | The weekly in-season report. |
| `outputs/backtests/runs.parquet` | Every backtest run, tagged with label, config hash and commit. |

---

## State: what is built

All of it tested; 196 unit tests.

**Ingest** — `sleeper_api` (read-only client, standings/bracket resolution),
`nflverse` (cached pulls, season-aware freshness), `player_ids` (gsis-anchored
crosswalk), `history` (season-parameterized history + preseason ECR),
`injuries` (weekly report + practice participation).

**Features** — `pipeline` (**the shared valuation, board and backtest both**),
`blend`, `calibration`, `slopes` (fitted per fold), `rank_curve`,
`market_anchor`, `vorp`, `lineup` (**shared by all three in-season modules**),
`availability` (durability), `scoring` (league rules applied to box scores).

**Backtest** — `walkforward`, `draft_sim`, `metrics`, `leakage_guard`.

**Draft** — `board` (parquet/CSV/HTML + live rewrite), `monitor` (live Sleeper
poll; runs the simulator's own pick policy, not a second copy).

**In-season** — `projections`, `roster`, `start_sit`, `waivers`, `trades`,
`matchup`, `report`.

**Scripts** — `build_draft_board`, `run_backtest`, `verify_league_settings`,
`weekly_update`, `schedule_weekly`, `build_league_history`, `replay_draft`
(rehearse the monitor against a real completed draft),
`check_backtest_baseline` (board-vs-ADP parity guard).

---

## Where the project actually stands

### The board works and is ready to draft from

Rebuilt against the live 2026 league. 515 of 517 ranked players reconcile; the
two misses are ECR 294 and 306. Nine players gated out by injury status, Josh
Jacobs (DNR, ECR 45) the notable one.

### The live draft path was broken in eight ways, and now is not

It had no tests at all. Driving it against this league's real 2025 and 2021
Sleeper feeds found the following, in rough order of what each would have cost:

| What | Effect on draft day |
|---|---|
| Recommended by **raw VORP**, ignoring your roster and the caps | The whole documented edge — drop-off valuation, roster-aware marginal value, position caps — ran only inside the backtest. Late rounds offered a backup QB. |
| `picks_until_next_turn` looped forever on an out-of-range `--my-slot` | Monitor hangs, silently, with no output at all |
| Printed nothing until pick 1 landed | From slot 1 you are on the clock for pick 1 staring at a blank terminal |
| Pick counter was one behind | "pick 3/160 · YOU ARE ON THE CLOCK" while making pick 4 |
| Detection keyed on `len(picks)` | An undo + replace between two polls is invisible; the wrong player stays struck off for the rest of the draft |
| Phone board only reflected picks **you tapped** | Nine other managers were invisible on the artifact you actually draft from |

Plus two smaller ones: a "run in progress: WR 100%" warning that fired off a
single pick every draft, and a countdown that kept promising turns after your
last pick.

**What changed.** `monitor.py` now imports `Roster`, `ValueDrafter` and
`picks_until_next_turn` from the simulator rather than owning copies, so the
live tool and the backtest cannot disagree about the policy or whose turn it is.
It rebuilds your roster from the feed's `draft_slot`, prints the pick the policy
would make, marks capped positions, and rewrites the phone board with every
drafted player struck through — the union of the live feed and your own taps,
where a player the feed says is gone cannot be un-tapped. It cross-checks
`--my-slot` against Sleeper's `draft_order` and refuses to start on a mismatch.

**How it is held.** `tests/test_draft_monitor.py` — 23 tests, each pinned to one
of the above, driven off `tests/fixtures/sleeper_draft_picks_2025.json`, the
genuine 160-pick payload from this league's 2025 draft trimmed to the fields the
monitor reads. Synthetic picks would not have caught the parts that depend on
Sleeper's own shape: `draft_slot`, string player ids, defenses keyed by team
abbreviation rather than a numeric id.

**What is still not proven.** Every replay feeds a *completed* draft. A live
draft can do things a replay cannot: a pick landing mid-poll, an autopick, a
commissioner undo. Run one real mock — step 1 under "Before the draft".

### But the valuation is not demonstrably better than ADP

This is the single most important thing to know, and it took a purpose-built
baseline to find out. `--value-col adp_value` swaps only the player ordering,
holding the pick policy, caps and scorer constant:

| season | board | pure ADP | gap | ECR-only |
|---|---|---|---|---|
| 2022 | 3.88 | 4.28 | −0.40 | 4.03 |
| 2023 | 2.27 | 4.75 | −2.48 | 5.18 |
| 2024 | 4.00 | 5.00 | −1.00 | 3.82 |
| 2025 | **7.80** | 4.00 | **+3.80** | 5.48 |
| mean | 4.49 | 4.51 | −0.02 | 4.62 |
| playoff pts | 410.1 | **411.5** | −1.4 | 407.4 |

The board and ADP columns are the checked-in baseline
(`tests/fixtures/backtest_baseline.json`, 40 drafts, seed 20260830, re-measured
2026-08-31); ECR-only is the earlier one-off and was not re-run.

Note for whoever re-measures next: **the ADP column reproduces the previously
recorded run exactly** (4.275 / 4.75 / 5.00 / 4.00, mean 4.5062, 411.47) while
the board column moved by up to 0.7 in a fold (previously 3.28 / 2.95 / 3.82 /
7.62, mean 4.42, 407.8). ADP being bit-identical rules out the seed, the
opponent model, the scorer and the pick policy — the movement is in the
valuation, and the most likely cause is the durability commit (609565a) adding
columns that perturbed the board's ordering. That is a hypothesis, not a
finding; nobody has bisected it. **It does not change the conclusion** — the
gap went from −0.09 to −0.02, which is more of a tie than before, and ADP still
has the better playoff points.

The board beats the market by ~1.3 places in three seasons and loses by 3.6 in
the fourth. All three configurations have the same expected finish — a spread of
0.2 places — and **pure ADP has the best playoff points**, which is the primary
metric.

2025 was diagnosed: the production components (`xfp_points`, `prior_points`)
cause ~2.1 of that 3.6, but removing them costs 0.75 in 2022 and 2.2 in 2023.
They are a variance position, not a bug. There is no free fix.

**Seven configuration changes have now been measured on four folds. That is the
noise floor. Further tuning is fitting the backtest, not learning.** The blend
sits at its shipping weights and should be left alone.

### What the edge actually is

The simulated team finishes ~4.4 of 10 against a field of ADP drafters, where
5.5 is a coin flip. Since the board ≈ ADP in ordering, that edge comes from the
**pick policy** — drop-off valuation, roster-aware marginal value, position
caps — not from the projections. That is worth knowing before investing more in
projection quality.

---

## Next steps

### Before the draft

1. **Run one real Sleeper mock with the monitor attached.** The live path is no
   longer untested — see "The live draft path" below — but everything that
   validated it replays a *completed* draft, so nothing has yet exercised a
   draft that is genuinely in progress: a pick landing while we poll, an
   autopick, a commissioner undo. Start a mock in the Sleeper app, take its
   `draft_id` from the URL, and run the monitor against it for the full sixteen
   rounds. Point `--html` somewhere disposable so the mock's picks do not end up
   struck through on the real phone board.
2. **Get your draft slot** once the order is drawn; the monitor needs it. It
   cross-checks the flag against Sleeper's `draft_order` and refuses to start on
   a mismatch, so this is now a hard error rather than a silent wrong roster.
3. **Rebuild the board that morning** with `--refresh`. ECR is the perishable
   input.

### After the draft

4. **Run `weekly_update` for real.** Built and tested against 2025 rosters, but
   never against a live week. Consider `schedule_weekly --install`.
5. **Injury-risk feature** — partly done (see below); the predictive half is
   built, a full model is not.
6. **ML projection model** — *recommended against for now.* It would improve a
   layer that cannot demonstrate value over ADP, validated on the same four
   folds that failed to separate three very different configurations.

---

## Availability / durability — what was measured

Built because the primary metric is playoff-week points and a player who misses
weeks 15-17 scores zero regardless of projection quality. Unlike everything
else here it rests on ~4,800 player-seasons rather than four folds.

**It is predictable, modestly.** 2012-2025 year-over-year persistence of
availability: QB 0.55, TE 0.38, RB 0.32, WR 0.29, overall 0.36. Three seasons of
history beat one (0.39 vs 0.35).

**The market already prices it, where it is paying attention.** Inside ECR rank
buckets, probability of playing 2+ of weeks 15-17, fragile vs durable history:

| ECR bucket | fragile | durable | gap |
|---|---|---|---|
| 1–24 | 0.863 | 0.880 | +0.017 |
| 25–60 | 0.759 | 0.781 | +0.022 |
| 61–120 | 0.726 | 0.777 | +0.051 |
| **121–250** | **0.663** | **0.769** | **+0.106** |

Raw correlation 0.380; controlling for ECR bucket, **0.148**.

**So it does not adjust projections.** Scaling a top-60 player's projection by
durability double-counts what ECR has already priced — the same error as
adjusting for weather the experts have already seen. It ships as a board column
plus a flag that fires only past ECR 120, where the residual signal lives.

**Not built:** a fitted availability *model* (features like age, workload,
injury type, position interactions). The measurement above says the ceiling on
that work is low for early picks and real but modest for late ones. If you build
it, validate on the 4,800-row sample, not on the backtest.

---

## Guardrails — do not undo these

1. **Benchmark against `--value-col adp_value` before believing any change.**
   The comparison is checked in — `tests/fixtures/backtest_baseline.json` — and
   `scripts/check_backtest_baseline.py` re-runs both orderings and diffs them.
   The guarded quantity is the **gap**, not either level, plus an absolute floor
   that fails if the board finishes more than 0.5 places behind pure ADP no
   matter what the fixture says. Re-record only with `--write`, and justify the
   move in the commit message.
2. **The board and backtest share `features/pipeline.py`.** If they diverge the
   backtest scores a program nobody drafts with.
3. **Leakage raises, not warns.** The board asserts its own temporal boundary.
4. **Rows scored under different scorers are not comparable** (waiver-aware vs
   not). Same for different `config_hash`.
5. **Metric priority:** playoff points → season points → win rate → projection
   MAE. Do not reorder casually.
6. **Nothing is dropped silently.** Unreconciled players, injury-gated players,
   players missing a Sleeper id are all counted and printed.
7. **Rolling waivers, not FAAB.** A claim costs your queue position; the bar is
   high, not positive.
8. **Aggregation beats a solo model.** The ML model must beat the blend
   out-of-sample or the blend ships.
9. **The monitor runs the simulator's pick policy, imported not copied.** If it
   grows its own `ValueDrafter`, `Roster` or `picks_until_next_turn`, the
   backtest is again scoring a program nobody drafts with — this time the pick
   policy rather than the valuation, which is where the measured edge actually
   lives.

---

## Known limitations

- **Four scorable folds, permanently.** ffverse preseason ECR starts 2021 and
  the purge gap costs one more, so 2022–2025 is the whole window. Every
  backtest conclusion rests on it.
- **In-season recommendations are unvalidated by construction.** The weekly
  FantasyPros feed is a live snapshot with no archive, so nothing built on it
  can be scored against history. `weekly_update` says so in its own output.
- **The lineup scorer is hindsight-optimal.** Applied identically to every
  simulated team, so comparisons are fair; absolute totals are an upper bound.
- **Opponents draft ADP with gaussian noise.** Beating that field is not the
  same as beating your league.
- **FantasyPros draft projections are not obtainable** (top-10 server-rendered
  only). ECR through a rank curve is the substitute. Their *weekly* product does
  include real point projections.

### Realistic expectations

Best-in-class seasonal projections explain ~14–26% of within-position variance;
weekly R² is 3–23%. Projections beat ADP by roughly half a draft slot. The edge
here is compounding small advantages and never wasting a pick or a claim on an
unavailable player — not weekly clairvoyance. Anything that looks dramatically
better than this is a bug.

---

## Environment

Python 3.14.7. `nflreadpy` (not `nfl_data_py` — no 3.14 wheels), pandas 3,
Polars→pandas at the ingest boundary. See `docs/environment.md`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

Caching: every raw pull lands in `data/raw/` as parquet. Completed seasons cache
for a week; any range including the live season gets six hours. A failed pull
falls back to a stale cache and says so — losing the board on draft morning
because ffverse is briefly down is the worse failure.
