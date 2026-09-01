# Handoff: Draft Board & In-Season System

**Last updated:** 2026-08-31
**Repo:** https://github.com/mnthompson98/fantasy-football
**Branch:** main · **Tests:** 219 passing + 1 gated skip, Python 3.14
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

All of it tested; 219 unit tests.

**Ingest** — `sleeper_api` (read-only client, standings/bracket resolution),
`nflverse` (cached pulls, season-aware freshness), `player_ids` (gsis-anchored
crosswalk), `history` (season-parameterized history + preseason ECR),
`injuries` (weekly report + practice participation).

**Features** — `pipeline` (**the shared valuation, board and backtest both**),
`blend`, `calibration`, `slopes` (fitted per fold), `rank_curve`,
`market_anchor`, `vorp`, `lineup` (**shared by all three in-season modules**),
`availability` (durability), `scoring` (league rules applied to box scores).

**Backtest** — `walkforward`, `draft_sim`, `metrics`, `leakage_guard`.

**Draft** — `board` (parquet/CSV/HTML, live rewrite, auto-reloading phone board
with persisted filters and a baked-in recommendation), `monitor` (live Sleeper
poll; runs the simulator's own `ValueDrafter.rank()` for the pick *and* a
couple of ranked alternatives with individual reasoning — not a second copy of
the policy, and not just the winner).

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

### The live draft path was broken in nine ways. Now it has been run live
### three times, clean, and is what will run Thursday.

It had no tests at all when this started. Driving it first against replays,
then against three real, in-progress Sleeper mock drafts, found the following.

| What | Effect on draft day | Found by |
|---|---|---|
| Recommended by **raw VORP**, ignoring your roster and the caps | Whole documented edge ran only inside the backtest. Backup QB offered late. | Replay |
| `picks_until_next_turn` looped forever on a bad `--my-slot` | Monitor hangs, silently, no output at all | Replay |
| Printed nothing until pick 1 landed | Blank terminal while on the clock for pick 1 | Replay |
| Pick counter was one behind | "pick 3/160 · ON THE CLOCK" while making pick 4 | Replay |
| Detection keyed on `len(picks)` | An undo+replace between polls is invisible | Replay |
| Phone board only reflected picks **you tapped** | Nine other managers invisible on the artifact you draft from | Replay |
| **The on-the-clock lookahead was 0, always** | Every live recommendation silently fell back to raw VORP — the first bug, resurrected, only while it actually mattered. Josh Allen recommended round 2 of a 1-QB league. | **Live mock #2** |
| `--html` defaulted to the real board regardless of `--draft-id` | A mock's picks overwrote `draft_board.html`, the actual draft-day file | Live mock #1 |
| No warning on a scoring mismatch | A full-PPR board misprices every reception in a standard room, silently | Live mock #1 |

Plus smaller ones: output block-buffered when redirected (looked like a hang
for 15 picks), a positional-run warning that fired off one pick, a countdown
that kept promising turns after your last pick, and — found by you, live,
mock #3 — the phone board's filter/search/hide state reset on every 6-second
auto-refresh, and the board carried no recommendation at all, only the
terminal did.

**The lookahead bug is the one to understand.** `draft_state()` used one
number, "picks until your turn" (correctly 0 the instant you're up), as the
*drafter's* lookahead too. A lookahead `<= 0` makes `ValueDrafter` treat the
whole remaining pool as "still there next turn," zeroing every position's
drop-off and falling back to raw VORP — silently, only while a recommendation
was live, i.e. every pick that mattered. The backtest was never affected: it
always computes the lookahead from the pick *being decided*, never from picks
already completed. Fixed and pinned by two regression tests engineered so the
display value and the correct lookahead are forced to disagree.

**What changed, in total.** `monitor.py` imports `Roster`, `ValueDrafter` and
`picks_until_next_turn` from the simulator, never reimplementing them. It
rebuilds your roster from the feed's `draft_slot`, computes the *correct*
lookahead for the pick being decided, prints the pick plus a couple of
runners-up (`ValueDrafter.rank()`, not just `choose()`) each with its own
plain-English reasoning, rewrites the phone board with every drafted player
struck through and the same recommendation-plus-alternatives baked into the
page, and auto-reloads that page every 6s with the filter/search/hide state
persisted across the reload. It cross-checks `--my-slot` against Sleeper's
`draft_order`, refuses on a mismatch, warns loudly and continuously on a
scoring-format mismatch, and only ever writes to the real `draft_board.html`
when `--draft-id` matches `config.current.draft_id` — anything else gets a
sibling `*.mock.html` so a practice draft cannot overwrite the real artifact.

**How it is held.** `tests/test_draft_monitor.py` and `tests/test_draft_policy.py`
— 60+ tests between them, each pinned to one specific thing that broke,
including the lookahead bug reproduced from first principles (a fixture where
the display value and the correct lookahead are engineered to disagree) and an
exact equivalence check (`choose() == rank()[0].index`) guarding the refactor
that exposed alternatives. `tests/fixtures/sleeper_draft_picks_2025.json` is
this league's genuine 160-pick payload, trimmed to the fields the monitor
reads — synthetic picks would not catch what depends on Sleeper's own shape.

**What is proven now.** Three full, real, in-progress Sleeper mock drafts,
start to finish, 150-160 picks each, across two different room configurations
(one deliberately mismatched — standard scoring — to prove the warning fires;
one matching Thursday's settings almost exactly). The real `draft_board.html`
was checked byte-identical before and after all three. Every defect found live
is fixed and has a regression test.

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

1. ✅ **Three real Sleeper mocks run with the monitor attached**, start to
   finish — see "The live draft path" above. Found and fixed a critical bug no
   replay could have caught. Nothing further to prove here.
2. **Get your draft slot** once the order is drawn; the monitor needs it. It
   cross-checks the flag against Sleeper's `draft_order` and refuses to start on
   a mismatch, so this is now a hard error rather than a silent wrong roster.
3. **Rebuild the board that morning** with `--refresh`. ECR is the perishable
   input.
4. **Do not pass `--html`.** The default now does the right thing on its own:
   `--draft-id 1389723592727461889` matches `config.current.draft_id`, so it
   writes straight to the real `draft_board.html` — the file to have open on
   your phone. Passing `--draft-id` for anything else (another mock, a typo)
   automatically redirects to a sibling `*.mock.html` instead.

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
