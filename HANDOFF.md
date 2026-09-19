# Handoff: Draft Board & In-Season System

**Last updated:** 2026-09-01
**Repo:** https://github.com/mnthompson98/fantasy-football
**Branch:** `main` (uncommitted review work, 2026-09-01) · **Tests:** 275 passing + 1 gated skip, Python 3.14
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

# During the draft. --my-slot defaults to config current.my_slot (10) and is
# checked against Sleeper's own draft_order, refusing to run if they disagree.
# Pass --my-slot only for a mock, where you will be at some other seat.
.venv/Scripts/python.exe -m src.draft.monitor --draft-id 1389723592727461889

# Rehearse the above against a real completed draft, at full speed.
.venv/Scripts/python.exe -m scripts.replay_draft --draft-id 1262443300887994368 --my-slot 4

# Weekly, in season. --check-only verifies ingestion and stops.
.venv/Scripts/python.exe -m scripts.weekly_update

# Whenever the valuation changes. Compare against --value-col adp_value.
.venv/Scripts/python.exe -m scripts.run_backtest --drafts 40 --label what-changed

# ...or let the parity guard run both orderings and diff them for you.
.venv/Scripts/python.exe -m scripts.check_backtest_baseline

# How this league drafts vs national consensus, and the field fitted to it.
.venv/Scripts/python.exe -m scripts.fit_opponent_model

# Either of the two above against that fitted field instead of ADP-plus-noise.
# Read "The opponent model" below before interpreting the result.
.venv/Scripts/python.exe -m scripts.check_backtest_baseline --opponent league
```

Set `PYTHONIOENCODING=utf-8` first — Windows defaults to cp1252 and dies on the
first accented player name.

**Your slot is 10**, announced by the commissioner out of band on 2026-09-02 and
recorded at `config current.my_slot`. Sleeper still has `draft_order: null` and
no start time, so the monitor's cross-check cannot confirm it yet and says so on
startup. It confirms itself the moment the commissioner saves the order.

Slot 10 is a **turn slot**: picks 10, 11, 30, 31, 50, 51 ... 150, 151 - pairs of
back-to-back picks separated by an 18-pick wait. See "The turn-slot lookahead"
below; it is the one thing about this seat that changes the pick policy.

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

All of it tested; 238 unit tests.

**Ingest** — `sleeper_api` (read-only client, standings/bracket resolution),
`nflverse` (cached pulls, season-aware freshness), `player_ids` (gsis-anchored
crosswalk), `history` (season-parameterized history + preseason ECR),
`injuries` (weekly report + practice participation).

**Features** — `pipeline` (**the shared valuation, board and backtest both**),
`blend`, `calibration`, `slopes` (fitted per fold), `rank_curve`,
`market_anchor`, `vorp`, `lineup` (**shared by all three in-season modules**),
`availability` (durability), `scoring` (league rules applied to box scores).

**Draft** — `board` (parquet/CSV/HTML, live rewrite, auto-reloading phone board
with persisted filters and a baked-in recommendation), `monitor` (live Sleeper
poll; runs the simulator's own `ValueDrafter.rank()` for the pick *and* a
couple of ranked alternatives with individual reasoning — not a second copy of
the policy, and not just the winner).

**In-season** — `projections`, `roster`, `start_sit`, `waivers`, `trades`,
`matchup`, `report`.

**Backtest** — `walkforward`, `draft_sim`, `metrics`, `leakage_guard`,
`opponent_fit` (the field, fitted to this league's own drafts).

**Scripts** — `build_draft_board`, `run_backtest`, `verify_league_settings`,
`weekly_update`, `schedule_weekly`, `build_league_history`, `replay_draft`
(rehearse the monitor against a real completed draft),
`check_backtest_baseline` (board-vs-ADP parity guard),
`fit_opponent_model` (how this league drafts against consensus).

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

> **Superseded 2026-09-01.** The table below is the pre-review measurement.
> After the review fixes the board *trails* the consensus ordering by
> 0.5-1.2 places in every configuration; see "Review, audit and draft-day
> stress test" for the re-measurement and the recommendation. The
> reasoning in this section about noise still stands.

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

**And most of it does not survive a realistic field.** See the next section.

---

## Review, audit and draft-day stress test — 2026-09-01

A file-by-file review, a statistical audit of the walk-forward, and a replay
of this league's real 2025 draft through the monitor. Everything below was
applied, not just filed; `tests/test_review_fixes.py` and
`tests/test_draft_policy_options.py` pin it.

### What was broken

| Where | Defect | Fix |
|---|---|---|
| `scripts/schedule_weekly.py` | Task ran in System32; `-m scripts.weekly_update` could not import. `nflverse.RAW_DIR` was cwd-relative on top. | `cd /d` in the command; `RAW_DIR` anchored to the repo. |
| `scripts/weekly_update.py` | Week inferred from the latest injury report, which on Tuesday night is last week's: wrong title, overwrote last week's file, stale matchup flags. | `nflreadpy.get_current_week()`; season from `current.season`. |
| `src/draft/monitor.py` | `get_draft(...) or {}` turned an unknown draft id into a 10x16 draft polling an empty feed. | Raise on null; require `settings.teams/rounds`. |
| `src/features/scoring.py` | DEF fumble recoveries read the special-teams key (`def_st_fum_rec`, 1.0). The league pays `fum_rec` 2.0 (verified live). | Read `fum_rec` / `ff`; missing nflverse columns are named on stderr instead of zeroed silently. |
| `src/ingest/history.py` | ffopportunity has no `season_type` and runs to week 22 (626 of 6054 rows in 2025); `xfp_points` counted playoff games, `prior_points` did not. | Weeks 1-17 only. |
| `src/features/slopes.py` | Slopes fit on the ECR-only curve, applied to the blend. | Pairs built through `ecr_to_pool` + `blend_projections`. |
| `scripts/run_backtest.py` | The "pure ADP" baseline mixed `-adp_rank` with per-position `replacement_points` (QB 218, TE 118): a hidden positional bias in every flex comparison. | One constant replacement level for that ordering. |
| `src/inseason/waivers.py` | Every bench player costs zero to drop, so "drop X" named whichever sorted first — possibly your best handcuff. | Tie-break on projection. |
| `src/features/availability.py` | `min_games=4` dropped the seasons where a starter got hurt in week 2 — the fragile observations themselves. | Filter players, not seasons. |
| `src/ingest/sleeper_api.py` | No retry, no stale-cache fallback on the 5 MB player map. | Backoff on 429/5xx/timeouts; stale cache with a warning. |
| Docs | CLAUDE.md promised a purged gap season. `build_folds` trains through Y-1 and must (last season is a blend component). | Docs corrected; folds now tested. |

Smaller: playoff weeks and rounds now flow from config into the scorer;
`ValueDrafter.DEFAULT_CAPS` matches the configured policy; the NaN path
(a position with no rank curve) raises in `ecr_to_pool` instead of crashing
`rank()` mid-draft; `blend` z-scores and re-maps with the same ddof; cache
names cannot collide for gapped season sets; the fingerprint now covers
`history.py`, `scoring.py`, `walkforward.py` and `metrics.py`.

Not in the repo at all, despite the README: The Odds API and Open-Meteo.
`matchup.py` reads lines and weather from nflverse schedules, where `temp`
and `wind` are recorded after the game — so weather flags for the coming
week are always empty. No code models the Wednesday waiver clear; nothing
reads `waiver_day_of_week`. The Tuesday 20:00 job is before it, which is
what matters.

### Walk-forward audit

**Leakage.** Traced every fold. Training totals, the rank curve, the
persistence shrink, the calibration pairs and the opponent fit all use
seasons strictly before the target; the ECR snapshot is the last one before
the schedule-derived opener (2025: 08-29, opener 09-04). The crosswalk is
current but used for identity only. Nothing reaches the target season's
regular season. Two things are worth knowing: the fold boards carry no
injury gate (Sleeper's historical status is not available), so the backtest
drafts players who were known-out in August — pessimistic for the tool, not
optimistic; and ffverse scrapes ECR weekly, so "rebuild the morning of the
draft" gives you the 08-28 snapshot until 09-04 lands.

**Survivorship.** The boards are the full ~500-player ECR snapshot, busts
included, and a drafted player with no stats scores zero in the scorer.
The calibration pairs treat a top-40 player with no stats as zero (28 of
1146, mostly camp-cut kickers), which is the right treatment. The one real
survivorship defect was durability's per-season `min_games` filter, above.

**Calibration slopes, leave-one-season-out** (top 40 per position, MSE of
the calibrated projection against realized points; slope fit on the other
four seasons):

| pos | MSE no shrink | MSE priors | MSE LOO fit | LOO vs none | LOO vs priors |
|---|---|---|---|---|---|
| QB | 8289 | 7388 | 7416 | −10.5% | −0.4% |
| RB | 6084 | 5858 | 5938 | −2.4% | −1.4% |
| WR | 5717 | 5529 | 5464 | −4.4% | +1.2% |
| TE | 2590 | 2408 | 2416 | −6.7% | −0.3% |
| K | 2445 | 2184 | 2195 | −10.2% | −0.5% |
| DEF | 1134 | 828 | 771 | −32.0% | +6.9% |

Shrinking helps everywhere. *Which* slope does not matter except at DEF,
where 0.30-0.38 fits every season and the 0.60 prior is wrong. Single-season
fits are noise: RB ranges 0.50-1.14 across 2021-2025 (sd 0.27). The 2022
fold's whole history is one season of 40 pairs, and `min_obs: 40` let it
overwrite the prior with that. `prior_weight: 80` now shrinks the fit toward
the prior. The "RB and WR were backwards" finding is inside this noise. The
priors you asked about (QB ~0.67, TE ~0.72) are well supported: QB LOO fits
0.66-0.71, TE 0.70-0.75.

### Draft-day stress test: the real 2025 draft replayed from slot 4

The 2025 *fold* board (history through 2024, no hindsight), re-keyed to
Sleeper ids, run through `draft_state` against the genuine 160-pick feed.
All 160 picks reconcile.

- **The ADP survivor assumption is right 45% of the time** against this
  room (share of the players `survivors()` assumed gone who were actually
  gone by our next turn). The room takes QB and TE 2-3 picks *earlier* than
  consensus and K/DEF 40-80 picks earlier.
- **A run on RB in round 2 did not move the RB drop-off at all.** With
  picks 11-16 all RBs the tool still said "no rush" on RB; the RB it passed
  (Josh Jacobs, 237 pts) beat the one left at its next turn (Omarion
  Hampton, 136). The run detector was display-only.
- **When nobody takes a QB or TE for five rounds the tool reaches anyway**,
  because ADP says three QBs and three TEs go in the next six picks: "Big
  drop-off: Jalen Hurts looks like the best QB still around" with Allen,
  Jackson and Daniels all sitting there. The lookahead trusts ADP over what
  the room has done for 56 picks. And when four QBs then go in a row it
  says "no rush" on the next one.
- **The rec shown on the phone right after your pick was gone by your
  next turn 9 times in 15.** It was computed as if it were your turn now.
- **K and DEF are handled right.** First K/DEF in the tool's top 3 at pick
  157 (round 16); the room took its first K at 95 and DEF at 103. The
  end-of-draft backstop spends the last two picks on them, exactly as it
  should. No horsepower is wasted there.
- **The waste is rounds 10-15.** Once the starters are set every
  candidate's drop-off is zero and the tie-break is raw VORP, which
  recommended the same second tight end at six consecutive turns (Jonnu
  Smith, 83 pts). The policy has no notion of bench upside or insurance.

Three switches were added to `ValueDrafter`, all off by default, all shared
with the monitor through `draft_sim.policy_from_config`: `run_aware`
(a detected run is assumed to continue through `survivors()`),
`depth_tiebreak` (bench picks weighted by the insurance they provide) and
`upside_tiebreak` (bench picks ranked on the most bullish expert's rank
through the curve, `upside_vorp` on the board). The monitor now shows a
labelled preview between turns, ranked over the expected survivors.

Replayed per turn on the real manager's roster (realized 2025 points of
the recommended player, summed over 16 turns): recorded policy 2712,
`run_aware` 2827 (Mayfield 268 over Kittle 153 at pick 37), `depth` 2712
(no change: the roster already held four RBs and four WRs), `upside` 2376
(reached for a 2025 Joe Mixon the fold board could not know was hurt). One
draft; the walk-forward below is the arbiter.

### Edge decomposition — the walk-forward, 40 drafts per fold, seed 20260830

Every row: same data, same seed, same gaussian field, same pick policy,
same caps, same scorer. Only the valuation config changes. `se` is the
per-draft standard error of the mean rank (160 drafts, which overstates
independence — 40 share each board); `sd fold` is the spread of the four
fold means, which is the honest noise floor. Lower rank is better; 5.50 is
a coin flip.

| valuation | rank | se | sd fold | playoff pts | win | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| **pure consensus ordering** (`adp_value`) | **4.35** | 0.21 | 0.64 | **417.3** | 0.55 | 3.90 | 4.35 | 5.25 | 3.90 |
| ECR-only blend, full pipeline | 4.71 | 0.23 | 1.27 | 408.8 | 0.55 | 3.70 | 4.03 | 4.58 | 6.53 |
| shipping blend, slopes fit raw | 4.90 | 0.21 | 1.81 | 405.8 | 0.53 | 3.38 | 3.55 | 5.45 | 7.23 |
| shipping blend, no calibration at all | 5.06 | 0.23 | 1.94 | 406.8 | 0.52 | 4.73 | 2.80 | 5.18 | 7.53 |
| **shipping blend, slopes shrunk to priors (what ships)** | **5.15** | 0.24 | 1.62 | 404.6 | 0.53 | 3.93 | 4.25 | 4.93 | 7.50 |
| shipping blend, no market anchor | 5.41 | 0.22 | 1.65 | 401.1 | 0.51 | 4.18 | 3.88 | 6.30 | 7.28 |
| shipping blend, priors only (fit off) | 5.54 | 0.22 | 1.66 | 404.5 | 0.50 | 4.83 | 3.98 | 5.50 | 7.85 |
| production-led (ECR weight 0.01) | 6.95 | 0.21 | 1.65 | 384.8 | 0.43 | 5.58 | 5.55 | 7.85 | 8.83 |

Isolating two of the review fixes, shipping config otherwise: the old
ECR-only slope-fit path scores 5.51 and ffopportunity unfiltered scores
5.39, against 5.15 with both fixes — neither fix caused the gap to the
market; both narrowed it.

**A confound found while recording the baseline, and what it does to the
tables above.** `OpponentModel.choose` draws its gaussian noise per row in
board order, so the identical pure-ADP drafter finished 4.35 in the ablation
and 4.71 in the baseline re-record with nothing changed but the board's row
order (the shrunk slopes re-sorted it). Permuting rows changes every pick of
a seeded draft; `tests/test_draft_policy_options.py` pins the fix
(`draft_sim.canonical_order`). **Every table in this section was measured
before the fix**, so each configuration also carried its own noise
assignment: differences under ~0.4 places are not evidence of anything. The
comparisons that matter were re-run under the fix and are in the last table
of this section; the large effects (production-led, no anchor, the league
field) survive, the small ones should be read as ties.

**What this says, honestly.**

1. **The board trails the raw consensus ordering by 0.5-1.2 places in every
   configuration tried**, and pure consensus has the best playoff points
   (the primary metric) by 8-30 points. The old fixture's tie (4.42 vs 4.51)
   does not reproduce: the ADP baseline gained ~0.16 when its hidden
   positional offset was removed, and the board's level had already moved
   0.7 in a fold between two earlier recordings that nobody bisected. The
   board's level is unstable at the ±0.5 scale under innocuous changes.
   That *is* the noise floor, and it is larger than any edge claimed here.
2. **The 2025 fold is the whole story.** Excluding it, the shipping board
   averages 4.37 against the market's 4.50. Including it, 5.15 against 4.35.
   Four folds cannot tell you whether 2025 is a tail or a trend.
3. **Cut two of three, keep the consensus.** Production-led valuation
   collapses (6.95, 385 playoff points, 0.43 win rate): the consensus is
   structurally required, not just helpful. ECR-only (4.71) beats the full
   blend (4.90 / 5.15): the production components add nothing measurable
   and cost ~0.2-0.4. Calibration is worth ~0.15 over none (4.90 vs 5.06)
   and the market anchor ~0.5 (5.15 vs 5.41) — the anchor is the one
   transformation that clearly earns its place, because it is the one that
   *defers* to the market. If you had to keep one thing beyond the
   consensus, keep the anchor; the calibration slope choice is noise
   (leave-one-season-out agrees); the production blend should go.
4. **The pick policy is worth less than HANDOFF claimed, and its
   lookahead is the reason.** Against the fitted league field the shipping
   board finishes **7.09** (0.39 win rate; 2023-2025 all 7.5-8.6), against
   5.58 in the earlier measurement, while pure consensus ordering finishes
   **5.61** against the same field. The gap to the market is ~1.5 places
   there, three times what it is against the gaussian field. The policy
   switches recover part of it (table below); nothing recovers all of it.

### Re-measured under the canonical row order (the numbers to quote)

Same seed, same field, noise now assigned by consensus rank rather than by
whatever order the valuation left the rows in. This is what
`tests/fixtures/backtest_baseline.json` records.

| valuation | field | rank | playoff pts | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| shipping blend (what ships) | gaussian | 4.91 | 407.7 | 3.77 | 3.52 | 4.90 | 7.42 |
| pure consensus ordering | gaussian | 4.35 | 416.5 | 3.77 | 4.78 | 4.70 | 4.15 |
| **ECR-only blend, full pipeline** | gaussian | **4.25** | 411.4 | 3.40 | 4.28 | 4.10 | 5.23 |
| shipping blend | league | 7.18 | 408.7 | 3.90 | 7.80 | 8.60 | 8.43 |
| pure consensus ordering | league | 5.39 | 405.3 | 3.85 | 6.45 | 5.20 | 6.05 |
| shipping blend + `run_aware` | league | 6.87 | 413.2 | 3.95 | 7.13 | 7.60 | 8.80 |

The board trails consensus by 0.56 on the gaussian field — past the repo's
own `MAX_LOSS_TO_ADP` floor of 0.50, so `check_backtest_baseline` exits 1
and `test_the_recorded_board_is_past_the_floor_and_handoff_says_so` pins
that this is known. Drop the two production components and the same
pipeline finishes 4.25: ahead of the full blend by 0.66 (past the noise
threshold) and level with pure consensus. 2025 is still the fold that
decides everything.

**Applied 2026-09-02, on the manager's go-ahead.** `blend.components` is
now `{ecr_points: 1.0}` — the consensus through the rank curve,
calibration, VORP and the market anchor, without last season's actual and
expected points — and `draft_policy.run_aware` is `true`. The baseline
fixture was re-recorded under that config (table in "Shipping config" just
below). The valuation change is the one this whole audit points at (the
production components cost 0.2-0.7 places in every configuration tried, on
both noise regimes); the policy change is the one the replay and the league
field both supported. Drafting straight off `adp_rank` with the pick policy
remains within noise of this and is the fallback if the board misbehaves on
the day.

Against the league field, `run_aware` is worth 0.31 places and 4.5 playoff
points under the canonical order (0.45 and 7 points before it) — the same
direction in both noise regimes, modest in size, and the gap to pure
consensus on that field (1.8 places) is far larger than anything a policy
switch recovers. The valuation is the problem; the policy is a second-order
fix to it.

### Shipping config — what `tests/fixtures/backtest_baseline.json` records

ECR-only blend, `run_aware` on for *both* drafters (the policy applies to
whichever ordering is being scored), canonical row order, gaussian field,
40 drafts, seed 20260830, `lookahead_rule: next_exposed`. Re-recorded
2026-09-19 after the lookahead flip (the fixture had been red on `main`
since 2026-09-02 because the flip shipped without the re-record it was
documented to need):

| | rank | playoff pts | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| board (ships) | 4.47 | 417.0 | 4.00 | 4.20 | 4.25 | 5.45 |
| pure consensus | 4.69 | 417.2 | 4.28 | 5.08 | 4.90 | 4.53 |

The previous record (2026-09-02, `next_pick`) was board 4.65 / 413.8 against
consensus 4.59 / 418.1, per season 3.98 / 4.45 / 4.22 / 5.95 and 3.90 / 4.83 /
4.85 / 4.80. The gap moved +0.06 → −0.22 and the playoff-points gap −4.3 →
−0.3: both inside the fixture's tolerance and inside the ~0.4-place noise
floor, so this is a re-record, not a finding. Read it as the same picture:
a tie on rank, ADP still (barely) ahead on playoff points, 2025 still the
fold the board loses, by a place instead of three. Note what `run_aware` did on this
field: ECR-only without it was 4.25 and consensus 4.35, so on the gaussian
field it costs both drafters ~0.3-0.4, not the one standard error quoted
above — the gaussian field never runs on a position, so the switch only
ever removes survivors that would have been there. It ships on because the
league field, which does run, is the one that looks like your room; that
is a judgment, and this table is what it costs if the judgment is wrong.

### Pick-policy switches — the walk-forward

| policy | field | rank | se | playoff pts | win | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| off (ships) | gaussian | 5.15 | 0.24 | 404.6 | 0.53 | 3.93 | 4.25 | 4.93 | 7.50 |
| `run_aware` | gaussian | 5.33 | 0.23 | 408.8 | 0.51 | 4.38 | 4.50 | 4.85 | 7.60 |
| `depth_tiebreak` | gaussian | 5.27 | 0.23 | 403.7 | 0.52 | 3.90 | 4.33 | 5.20 | 7.65 |
| both | gaussian | 5.39 | 0.22 | 407.5 | 0.51 | 4.30 | 4.63 | 5.13 | 7.53 |
| off | league | 7.09 | 0.22 | 406.9 | 0.39 | 3.88 | 7.53 | 8.40 | 8.58 |
| **both** | **league** | **6.47** | 0.22 | **412.8** | 0.43 | 3.23 | 6.35 | 7.68 | 8.63 |
| `run_aware` alone | league | 6.64 | 0.23 | **413.9** | 0.43 | 3.65 | 7.08 | 7.18 | 8.68 |
| pure consensus, off | league | 5.61 | 0.20 | 404.1 | 0.50 | 4.15 | 6.85 | 4.93 | 6.50 |

Under the gaussian field none of the switches helps; that field does not
produce runs, so `run_aware` has nothing to react to there and just adds
noise (+0.24, one standard error). **Against the fitted league field both
switches together are worth 0.62 places and 6 playoff points, better in
three folds of four.** That is the field that behaves like your room —
K/DEF forty to eighty picks early, QB/TE a few picks early, RB/WR at
consensus — and it is the one the review found the ADP lookahead being
right 45% of the time against. It is also one 8-team draft's worth of fit
per fold, so treat 0.62 as a direction, not a quantity. `run_aware` alone
carries most of it (0.45 places, and the best playoff points of any policy
row); `depth_tiebreak` adds ~0.18 on top.

**Recommendation.** For Thursday, set `draft_policy.run_aware: true`. It is
the one change that responds to the specific way this room deviates from
consensus, it helped in the replay of the real 2025 draft, and it helped
against the field fitted to that draft. Its cost against the gaussian field
is one standard error. `depth_tiebreak` is optional; it did nothing in the
replay because that roster already carried four backs and four receivers.
Both ship **off** here so the recorded baseline describes the recorded
policy; flipping either changes `BacktestConfig.hash()` and the cheap
fixture test will ask you to re-record, which is the point. `upside_tiebreak` was not run in the walk-forward: in the
2025 replay it reached for a Joe Mixon the fold board could not know was on
PUP, and it needs an injury-gated historical board before it is a fair
test. All three stay off.

## The opponent model — measured, fitted, and what it cost

"Opponents draft ADP with gaussian noise" was a Known Limitation. It is now a
measurement. `python -m scripts.fit_opponent_model` prints all of the below.

### How this league actually drafts

Both of this league's drafts under `data/league_history/` — 2021 (8 teams, 15
rounds, `Etown vs Louisville`) and 2025 (10 teams, 16 rounds, `Camden?`, five
seats shared with 2021) — joined to the same preseason ECR `adp_rank` the
simulator drafts off. All 280 picks reconcile.

`delta = pick_no - adp_rank`, centered on the pooled mean (only relative
differences are identifiable; the raw mean is structurally negative because
consensus ranks run past 500 and picks stop at 160):

| pos | n | centered delta | t | 2021 | 2025 |
|---|---|---|---|---|---|
| **K** | 18 | **−80.8** | −9.7 | −89.3 | −73.8 |
| **DEF** | 18 | **−47.0** | −10.3 | −45.3 | −48.1 |
| QB | 34 | −1.3 | −0.2 | −16.2 | +12.5 |
| TE | 31 | +0.7 | +0.2 | +7.5 | −3.7 |
| RB | 82 | +10.4 | +7.9 | +14.1 | +7.7 |
| WR | 97 | +15.1 | +9.7 | +18.1 | +12.9 |

**The deviation is K and DEF, and it is enormous and replicates.** A kicker goes
81 picks and a defense 47 picks earlier than his consensus rank relative to
everyone else, in both drafts. Everything else is the arithmetic consequence:
picks are conserved, so pulling 20 kickers and defenses forward pushes RB and WR
back.

Among the skill positions the pattern is smaller but consistent across both
drafts. Cumulative counts against the consensus board taken straight down
("chalk"), through round 6: **2021 was QB +3, TE +3, RB −1, WR −5; 2025 was
QB +2, TE +3, RB −1, WR −4.** So this league takes two or three *more* QBs and
about three more TEs in the first six rounds than the consensus implies, and
four or five fewer WRs. **That is the opposite of the hypothesis in the brief**
— QB and TE go *earlier* here, not later, and RB is at chalk rather than ahead
of it.

### Why the model is a run curve and not a bias

Regressing pick number on consensus rank per position (robust, one 3σ refit):

| pos | a | b | resid sd |
|---|---|---|---|
| RB | −1.5 | **1.027** | 8.9 |
| WR | 6.9 | **0.959** | 9.1 |
| TE | 4.9 | 0.832 | 13.2 |
| QB | 39.6 | 0.402 | 32.8 |
| K | 68.7 | **0.253** | 19.9 |
| DEF | 90.7 | **0.114** | 8.3 |

RB and WR go at consensus. The room does not *shift* kickers up the board — it
ignores the consensus's ordering of them almost entirely and takes one per team
in rounds 12–14 regardless of which kicker. No constant offset can express that,
and fitting one by simulated moments drives QB to a sigma of 123 picks trying to
reproduce a spread that is structure rather than noise.

So `LeagueOpponentModel` keys on two clocks, both in rounds, blended per
position by a closed-form weight:

```
key = (1 - w) * adp_rank/teams  +  w * curve[pos](k/teams)  +  N(0, sigma)
      + over_ceiling_penalty if the roster is already full at that position
```

Fitted `w`: RB 0.78, WR 0.94, QB 1.00, TE 1.00, K 0.99, DEF 0.97. The pure
schedule alone is board-blind in the other direction — this league opened 2021
with eight running backs and 2025 with five receivers *because those were the
two boards* — which is why the consensus clock stays in the blend.

### What was wrong with the gaussian field

Its need factor multiplies the key, so its size depends on where you are on the
board: at ADP rank 250 the 1/1.6 boost is worth 94 picks, at rank 10 it is worth
4. Consequence, simulated over the same boards:

| | first K | first DEF |
|---|---|---|
| league 2021 | round 13 | round 13 |
| league 2025 | round 10 | round 11 |
| **gaussian field** | **round 9** | **round 8** |
| fitted field | round 9–10 | round 10 |

The gaussian field takes *every* defense in round 8 and *every* kicker in round
9. Two to four rounds in the middle of every simulated draft are spent on the
wrong positions — exactly the stretch where the pick policy's lookahead decides
what survives to the next turn.

### The delta: what a realistic field costs

`check_backtest_baseline`, 40 drafts/season, seed 20260830, same data, same
valuation, same pick policy, same caps, same scorer. Only the field changes.
League rank by points, of 10; **5.50 is a coin flip; lower is better.**

Three runs. The third fits the field on *both* drafts instead of only on the
prior one — leaky for the 2025 fold and therefore not a shippable
configuration, but the only way to ask whether the answer survives a
better-identified field.

| season | board (gauss) | ADP (gauss) | gap | board (league, prior) | ADP | gap | board (league, both) | ADP | gap |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 3.48 | 3.85 | −0.38 | 1.90 | 3.65 | −1.75 | 5.03 | 6.80 | −1.77 |
| 2023 | 2.42 | 4.45 | −2.03 | 4.47 | 6.47 | −2.00 | 3.15 | 5.35 | −2.20 |
| 2024 | 3.73 | 5.12 | −1.40 | 7.72 | 4.35 | +3.38 | 6.05 | 3.90 | +2.15 |
| 2025 | 7.85 | 4.05 | +3.80 | 8.20 | 6.10 | +2.10 | 8.05 | 6.30 | +1.75 |
| **mean** | **4.37** | **4.37** | **−0.00** | **5.58** | **5.14** | **+0.43** | **5.57** | **5.59** | **−0.02** |
| playoff pts | 408.3 | 413.5 | −5.2 | 404.8 | 406.3 | −1.6 | 401.0 | 400.9 | +0.1 |

The gaussian column reproduced the checked-in fixture (`parity holds against
2026-08-31 (07f5b2a)`), so the columns differ only in the field.

**Two findings, and only one of them holds.**

**1. The board-vs-ADP gap does not reliably move. Answer: no measurable
effect.** It goes −0.00 → **+0.43** fitting on the prior draft only, and
−0.00 → **−0.02** fitting on both. The two fit scopes disagree by 0.45, which is
the entire size of the effect. The honest reading is that a four-fold
evaluation cannot separate them — the same conclusion CLAUDE.md reaches about
seven other configuration changes. Do not quote +0.43 as the delta; quote the
pair.

**2. The absolute level moves, consistently, and by a lot. Answer: about 1.2
places.** The board goes 4.37 → 5.58 and 5.57; pure ADP goes 4.37 → 5.14 and
5.59. Both orderings, both fit scopes, land essentially on the coin flip.
**Most of the ~1.1-place edge the backtest reports is an artifact of the
gaussian field, not something either ordering earns against a room that drafts
like this one.** HANDOFF has said since `df921da` that the edge lives in the
pick policy rather than the projections; this says the pick policy's edge is
largely against gaussian opponents.

That second result is the one worth acting on, and it is not a valuation
problem — swapping the valuation is exactly what finding 1 says does nothing
here.

### How much to believe it

- **The shipping fit each fold sees is one 8-team draft.** `fit_scope: prior`
  means every scorable fold (2022–2025) fits on the 2021 draft alone: 120 picks,
  a different league size, half a different room. That is leakage-clean and it
  is thin, and the two fit scopes disagreeing on the gap is exactly what being
  that thin looks like. The K/DEF tendencies replicate across both drafts, and
  those dominate the field's behaviour, which is why the *level* result is
  stable while the *gap* result is not.
- **Four folds, and they swing hard.** 2024 goes −1.40 → +3.38 → +2.15 on
  changes that touch nobody's valuation. The direction of the level shift is
  real; none of the per-fold numbers should be read as a quantity.
- **The fitted field is *less* dispersed than the real one** (realized delta sd:
  WR 6.9 vs the league's 15.4, DEF 8.4 vs 19.3). A more predictable field should
  help a drafter whose lookahead is right and hurt one whose lookahead is wrong,
  so this probably *overstates* the level damage.
- **`ValueDrafter.survivors()` was deliberately left alone.** It still assumes
  the field drafts straight down ADP, which against the fitted field is now a
  known-wrong assumption — and it is the most likely single cause of the level
  drop. Keeping it fixed is what makes these a comparison of fields rather than
  of two different drafters. It is the obvious next experiment, below.

### Next

1. **Teach `survivors()` the fitted field and re-measure.** The policy's
   lookahead is the mechanism the whole edge runs through, and against this
   league it is currently mis-specified: it assumes the room takes the top
   `picks_until_next` by ADP, and the room demonstrably does not. This is the
   change most likely to recover the 1.2 places, and unlike everything in
   finding 1 it is not a valuation change. It *is* a policy change, so it also
   moves what `src/draft/monitor.py` recommends live — measure before shipping,
   and expect the four-fold noise floor to make it hard to prove.
2. **Do not re-record the baseline under the league field.** The fixture guards
   the valuation and is measured under the gaussian field;
   `check_backtest_baseline --write --opponent league` refuses for that reason.
3. **`opponent_model.model` still ships as `gaussian`.** Nothing about the
   default run changed — verified bit-for-bit, and the parity guard passes.
4. **A third draft would help more than any modelling here.** The gap result
   flipped sign on one extra draft. `data/league_history/` holds two;
   `config/league.yaml` `excluded:` lists two more that were excluded as
   novelty leagues, correctly. If earlier `Camden?` instances exist on Sleeper,
   `scripts/build_league_history.py` will pull them.

---

## Next steps

### Draft day, start to finish

Every command runs from the repo root in PowerShell. Set the encoding once per
terminal or the first accented player name kills it:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**The morning of**

1. Confirm nothing moved under you:
   `.venv\Scripts\python.exe -m scripts.verify_league_settings 1389723592727461888`
   Expect "All 10 checked settings match."
2. Open the Sleeper draft room and **read your slot off the board.** If it is
   not 10, change `current.my_slot` in `config/league.yaml`. This is the one
   input nothing can check for you until Sleeper publishes `draft_order`.
3. Rebuild the board - ECR is the perishable input:
   `.venv\Scripts\python.exe -m scripts.build_draft_board --league-id 1389723592727461888 --refresh`
   Skim `outputs/projections/excluded_players.csv`. A top-40 name in there means
   the injury gate caught something; a top-40 name in there *wrongly* is the
   only failure that silently costs you a round.

**Fifteen minutes before**

4. Start the monitor. **Do not pass `--html` and do not pass `--my-slot`** -
   both defaults are correct for this draft and wrong to retype under pressure:
   `.venv\Scripts\python.exe -m src.draft.monitor --draft-id 1389723592727461889`
5. **Read the first four lines before the draft starts.** They are the whole
   pre-flight:
   - `slot 10 from config current.my_slot; confirmed by Sleeper's draft_order.`
     If it says **NOT confirmed**, the order is not up yet - re-check against the
     room. If it *refuses to start*, Sleeper disagrees with you; believe Sleeper.
   - `10 teams - 16 rounds - your slot 10`
   - `turn slot: back-to-back picks - lookahead rule 'next_pick'`
   - the caps line, which should read `run_aware` among the policy switches -
     it was switched on 2026-09-02 and is worth 0.3-0.45 places against the
     field fitted to this room.
   - **No `SCORING MISMATCH` banner.** If one appears, the board is priced for a
     different game than the room is playing and every VORP is wrong.

**During**

6. The banner tells you which kind of advice you are looking at. `>> TAKE:`
   means you are on the clock and it is a live recommendation.
   `>> LIKELY AT YOUR TURN (pick N)` is a preview: ranked over who plausibly
   survives to pick N, with the lookahead measured from *that* pick. It is a
   real forecast, not a pretend on-the-clock call - but it is still a forecast,
   so act on `TAKE`.
7. Make the pick in the Sleeper app. The monitor sees it within 5 seconds and
   reprints; you never tell it anything.
8. Read `also considered:` - each alternative carries its own reasoning from the
   same drop-off comparison that produced the winner. **There is no pick timer
   this year, so this is free.** It is the biggest practical change: what the
   system is worst at is knowing what you know about a player, and you now have
   unlimited time to overrule it deliberately rather than in a panic.

**If something breaks**

- **Monitor crashes or you close it:** restart it. It holds no state - it
  rebuilds your roster from the feed's `draft_slot` every poll.
- **`[warn] poll failed`:** Sleeper hiccuped. It keeps polling; ignore it.
- **You disagree with the board:** `outputs/projections/draft_board.csv` is the
  same data, sortable, and the draft does not depend on the monitor at all.

**The phone board (optional, needs setup in advance)**

`outputs/projections/draft_board.html` is self-contained and the monitor
rewrites it every pick with a 6-second meta-refresh. To read it on a phone,
serve the directory over the LAN:

```powershell
.venv\Scripts\python.exe -m http.server 8000 --directory outputs\projections
```

then open `http://<laptop-LAN-IP>:8000/draft_board.html` on the phone.
**This does not work out of the box here:** the Wi-Fi is classified as a
*Public* network and there are no inbound firewall rules for Python, so the
phone cannot reach the port. Fixing it means reclassifying the network as
Private or adding an inbound rule - a deliberate security change, so make it
and *test it from the phone* days before, never on draft night. With no pick
timer the laptop terminal carries everything the HTML does; the phone board is
a convenience, not a requirement.

### The turn-slot lookahead

Slot 10 gets back-to-back picks, so at picks 10, 30, 50 ... 150 the literal
answer to "how many picks until my next turn" is 0. `ValueDrafter.survivors()`
returns the pool unfiltered at `<= 0`, so every candidate's drop-off computes to
exactly 0.0 and the ranking falls through to its raw-VORP tiebreak - the policy
`ValueDrafter` exists to replace, on half of our picks.

This is the **same degenerate path** the 2026-09-01 audit found from the other
end of the draft: once the starters are set every drop-off is zero too, which is
what `depth_tiebreak` was built for ("recommended a second tight end at six
consecutive turns of the 2025 replay"). Two triggers, one failure - `gain == 0`
falling through to a tiebreak that was never meant to be a policy. `depth_tiebreak`
and `upside_tiebreak` change *what* that tiebreak is; the lookahead rule changes
whether we land in it at all.

`draft_policy.lookahead_rule` selects between `next_pick` (that behaviour) and
`next_exposed` (`draft_sim.picks_until_board_moves`, which skips past our own
consecutive picks). **It ships as `next_pick`** - see the config comment for the
paired measurement.

The case for flipping it is correctness, not points: at those eight picks the
drop-off policy is not running. In the slot-10 mock it is the difference between
taking QB Joe Burrow at pick 30 (14 slots ahead of his ADP, on a 0.3-point VORP
tiebreak over Omarion Hampton) and taking Hampton at value with a quarterback at
market in round 7 - which is what "the answer is not 'later', it is 'at market,
once'" in CLAUDE.md already concluded by a different route.

Flipping it is one config line, but not a free one: the monitor and the backtest
read the same key on purpose, so it moves live recommendations *and* invalidates
`tests/fixtures/backtest_baseline.json` (which randomizes slots, 2 of 10 being
turn slots). It needs `check_backtest_baseline` and a deliberate re-record.

### After the draft

4. ~~**Run `weekly_update` for real.**~~ **Done 2026-09-19**, against live
   2026 week 2: `--check-only --refresh` and the full job both ran clean on
   Sleeper + nflverse (roster found, week-2 injury report attached, lineup and
   waiver sections produced, nothing needed fixing). The waiver module was
   audited against guardrail #7 at the same time: a claim at +3.9 against the
   4.0 bar is surfaced as a free add and refused as a claim, and `evaluate()`
   now rejects a `priority_threshold` that does not exceed the stream floor,
   so `--waiver-threshold 0` can no longer turn "claim" into "any add".

   **The run now also writes a brief** — `outputs/reports/weekNN.brief.md`,
   copied to `latest.brief.md` and printed last on the terminal — with a
   `STATUS: OK | DEGRADED | FAILED` line first. DEGRADED means an input was
   stale: a pull that fell back to cache (`nflverse.FALLBACKS`), a weekly feed
   whose games have all already kicked off, or one not scraped in 2+ days.
   This is what a scheduled reader relays — and the reader **cannot run
   anything**: Claude Cowork's first attempt reported that a linked computer
   exposes file access only, no shell. So it is two tasks. Windows Task
   Scheduler runs `scripts/run_weekly.cmd` at Wednesday 06:00
   (`python -m scripts.schedule_weekly --install`, then `--run-now` to prove
   it fires; registered with wake-to-run and run-if-missed, log in
   `outputs/reports/last_run.log`), and Cowork reads `latest.brief.md` at
   07:00 under the prompt in `docs/cowork_weekly_prompt.md`, which has it
   check the brief's `Generated` timestamp before believing it.

   **Timing caveat for that schedule:** this league processes waivers on
   Wednesday (`waiver_day_of_week: 2`, `waiver_clear_days: 2`). A Wednesday
   7 AM run lands *after* processing, when the wire is mostly free agents
   until players lock at their own kickoff — so its "claim" labels describe
   what a pickup *would* cost once locked, and its free adds are genuinely
   free right then. To place claims *before* processing, run Tuesday night
   instead. The module does not read per-player waiver status from Sleeper;
   that would be the next in-season feature if the distinction starts costing
   priority.
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

## The fingerprint is version-stable now, and the baseline was re-recorded

`test_the_valuation_code_has_not_changed_under_the_baseline` used to hash
`ast.dump` output, whose node field names change between Python releases, so
a fixture recorded on 3.14 fired on every module under 3.12. It hashes
`ast.unparse` output now (docstrings stripped), which is stable across
versions and still ignores comments and layout. The baseline fixture was
re-recorded on 2026-09-02 under the shipping config (ECR-only blend,
`run_aware` on) and the gaussian field; see the review section for what
moved and why.

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
- **The shipping field is still gaussian, and beating it is not the same as
  beating your league** — now with a number attached rather than a worry. See
  "The opponent model" above: swapping in a field fitted to this league's own
  drafts costs the board 1.2 places and turns its tie with raw ADP into a
  0.43-place loss. `--opponent league` runs it. The fitted field itself rests on
  one prior 8-team draft per fold.
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

The opponent-model work above was developed and measured on **Python 3.12**,
which is why the AST fingerprint test fails (see "One failing test" above).
Everything else reproduces: the gaussian field was verified bit-for-bit against
a pre-change run, and `check_backtest_baseline` reports `parity holds` there.
Note that 3.13+ enables `ssl.VERIFY_X509_STRICT` by default, which some
corporate/proxy TLS chains fail — if `nflreadpy` starts raising
`CERTIFICATE_VERIFY_FAILED` behind a proxy, that is the reason, not the data
source.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

Caching: every raw pull lands in `data/raw/` as parquet. Completed seasons cache
for a week; any range including the live season gets six hours. A failed pull
falls back to a stale cache and says so — losing the board on draft morning
because ffverse is briefly down is the worse failure.
