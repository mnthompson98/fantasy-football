# Handoff: Draft Board & In-Season System

**Last updated:** 2026-09-01
**Repo:** https://github.com/mnthompson98/fantasy-football
**Branch:** `claude/league-opponent-model-kd1fy3` · **Tests:** 238 passing + 1 gated skip
(plus one pre-existing failure, see below), Python 3.14
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

# How this league drafts vs national consensus, and the field fitted to it.
.venv/Scripts/python.exe -m scripts.fit_opponent_model

# Either of the two above against that fitted field instead of ADP-plus-noise.
# Read "The opponent model" below before interpreting the result.
.venv/Scripts/python.exe -m scripts.check_backtest_baseline --opponent league
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

## One failing test, and it is not this branch's

`test_the_valuation_code_has_not_changed_under_the_baseline` fails. It hashes
the AST of eight modules and compares against the fixture, and it reports **all
eight** as moved — including `blend.py`, `calibration.py`, `vorp.py` and four
others that nothing here touches.

**It fails identically on an unmodified `main`.** `ast.dump` output changed
between Python versions, and the fixture was recorded on 3.14 while this branch
was developed on 3.12 (see Environment below). The tripwire is
interpreter-sensitive, which is a real defect in it — a guard that fires on the
Python you happen to be running teaches people to re-record without looking,
which is exactly what its own docstring says it exists to prevent. Fixing it
means hashing something version-stable (`ast.unparse` output, or the source with
comments and docstrings stripped) and re-recording once. Left alone here because
it is not what this branch is about, and because re-recording the fixture was
explicitly out of scope.

`draft_sim.py` would legitimately fire it in any case — it is fingerprinted and
it changed. The other seven are what prove the cause is environmental.

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
