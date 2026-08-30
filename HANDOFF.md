# Handoff: Draft Board & Backtest System

**Last updated:** 2026-08-30  
**Repo:** https://github.com/mnthompson98/fantasy-football  
**Branch:** main  
**Tests:** 43/43 passing on Python 3.14  

---

## What's done

### Infrastructure
- ✅ Git repo initialized, remote pushed
- ✅ Python 3.14 venv with nflreadpy, pandas 3, xgboost, lightgbm
- ✅ League history datasets generated from Sleeper API (2021, 2025)
- ✅ League settings verified and documented (rolling waivers, weeks 15–17 playoffs)

### Core modules (all tested)
- ✅ `src/ingest/sleeper_api.py` — Sleeper read-only client, standings/bracket resolution
- ✅ `src/features/vorp.py` — flex-aware replacement level, cross-position comparison
- ✅ `src/features/blend.py` — within-position z-score consensus (xFP + FantasyPros + ADP)
- ✅ `src/features/calibration.py` — post-hoc spread shrink (per-position slopes) + in-season refitting
- ✅ `src/backtest/leakage_guard.py` — temporal boundary enforcement, raises (not warns)
- ✅ `src/backtest/draft_sim.py` — ADP-sampled opponent draft, positional need, scarcity backstop
- ✅ `src/backtest/metrics.py` — metric priority (playoff pts > season pts > win rate > projection MAE)
- ✅ `src/backtest/walkforward.py` — expanding-window orchestration, run logging with config hash + git commit
- ✅ `src/draft/board.py` — parquet + CSV + self-contained HTML (phone-readable, offline)
- ✅ `src/draft/monitor.py` — live Sleeper poll, best-available, positional-run detection
- ✅ `scripts/build_league_history.py` — regenerate datasets from API
- ✅ 43 unit tests covering VORP, blend, calibration, leakage, draft sim, roster legality

### Bugs found and fixed
1. Summing per-position need dict triple-counted flex → rosters finished with empty QB slots
   - **Fixed:** separate `unfilled_dedicated_slots()` + `flex_slots_open()` + `total_unfilled_starting_slots()`
2. Backstop assumed needed position would be available → high-demand positions exhausted early
   - **Fixed:** scarcity trigger grabs a needed starter when supply drops to 3 remaining

### Data
- 2021 Etown vs Louisville: 8 teams, 120 picks, champion cammack14, you finished 8th
- 2025 Camden?: 10 teams, 160 picks, champion cammack14, you finished **3rd**
- Excluded: Drakes Miners 2024, Premier League Lads 2025 (2QB novelty leagues, would pollute opponent priors)

---

## What's NOT done (next steps)

### Immediate (Thursday before draft)
1. **nflverse feature extraction** — pull historical play-by-play, weekly stats, injuries, depth charts, snap counts
2. **Build the draft board** — run `scripts/build_draft_board.py` (not yet written)
   - Fetch current season's preseason ADP from FantasyPros
   - Pull consensus projections + xFP
   - Compute VORP for your exact roster settings
   - Export parquet + CSV + HTML
3. **Test the live monitor** — run `src.draft.monitor` against a mock draft once the board exists

### Weeks 1–3 post-draft
4. **Run walk-forward backtest** — `scripts/run_backtest.py` (not yet written)
   - Seasons 2020–2025, one-season purge gap
   - Simulate 40 drafts per season from varied slots/seeds
   - Score against actual weekly points (weeks 1–17)
   - Log results to `outputs/backtests/runs.parquet`

### Weeks 4+
5. **XGBoost projection model** — train on historical features, cross-validate
   - Must beat consensus blend out-of-sample or it ships as fallback
6. **In-season automation** — `src/inseason/start_sit.py`, `waivers.py`, `trades.py`
   - Weekly refit of calibration slopes
   - FAAB/waiver decision support (remember: rolling priority, not FAAB)
   - Trade value calculator

---

## Known limitations and caveats

### Before you trust the backtest
- **Synthetic smoke test only.** I ran the draft sim end-to-end with synthetic data to prove the mechanics. It has NOT been validated against real nflverse data.
- **ADP realism matters.** The fixture that initially broke was ranking ADP by raw points, which put all QBs in the top 40. Real ADP is value-aware. When you build the real board, use FantasyPros PPR consensus ADP, not raw projection rank.
- **Rule changes.** 2024 kickoff rule was a one-year anomaly (special teams scoring doesn't transfer to 2026). The backtest config downweights 2024 K/DEF; if you add 2026 K/DEF data later, reassess.

### Realistic expectations
- Best-in-class seasonal projections explain ~14–26% of within-position variance
- Projections beat ADP by ~half a draft slot
- Profit margin is compounding small edges, not prediction wizardry
- A result that looks dramatically better than the above is a bug, not a breakthrough

### Waiver system — verified as rolling priority
- Both your leagues use `waiver_type: 0` (rolling priority), NOT FAAB
- `waiver_budget: 100` is present but inert at this setting
- Priority is a positional resource — spend it only on true difference-makers
- Confirm this in the 2026 league settings before the season starts

---

## Environment

**Python:** 3.14.7 (no 3.14 wheels for `nfl_data_py`, use `nflreadpy` instead)

**Setup:**
```bash
cd "C:\Users\markt\OneDrive\Desktop\Repositories\Fantasy Football"
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

**Key files to modify for 2026:**
- `config/league.yaml` — league_id, draft_id, assumed_2026 (teams, bench), calibration slopes
- `scripts/build_draft_board.py` — entry point for Thursday board (stub: not yet written)
- `scripts/run_backtest.py` — entry point for backtest orchestration (stub: not yet written)

---

## Project decisions (encoded as guardrails, not prose)

1. **Aggregation over solo model.** Consensus blend ships first. ML must beat it out-of-sample to be adopted.
2. **Metric priority is deliberate.** Playoff-week (15–17) points, then season points, then win rate, then projection MAE. A model that wins on MAE and loses on playoffs is useless.
3. **Leakage raises, not warns.** A leaky backtest produces confident wrong answers. `src/backtest/leakage_guard.py` raises rather than logging.
4. **Roster legality is enforced.** The draft simulator will never produce an unfilled starting slot or a roster where you can't legally start 9 players.
5. **Rolling waivers, not FAAB.** All FAAB logic is explicitly out of scope. Verify before the season.

---

## Quick reference

| File/Command | Purpose |
|---|---|
| `CLAUDE.md` | Project context, verified settings, decision rationale (read first) |
| `config/league.yaml` | League settings, backtest window, calibration slopes |
| `data/league_history/*.json` | Generated: past drafts, standings, champions |
| `src/ingest/sleeper_api.py` | Sleeper API client, no auth needed, self-throttles |
| `src/features/vorp.py` | VORP with flex-aware replacement level |
| `src/features/blend.py` | Consensus blend (z-score within position) |
| `src/backtest/draft_sim.py` | Simulator; both real and opponent logic |
| `src/draft/board.py` | Board export: parquet, CSV, HTML |
| `src/draft/monitor.py` | Live Sleeper poll during draft |
| `.venv/Scripts/python.exe -m pytest tests/ -q` | Run 43 tests (should all pass) |
| `scripts/build_league_history.py` | Regenerate league history from API |
| `scripts/build_draft_board.py` | **TODO:** Build Thursday board (not yet written) |
| `scripts/run_backtest.py` | **TODO:** Run walk-forward backtest (not yet written) |

---

## Next immediate steps

1. **Verify the setup works locally:**
   ```bash
   .venv/Scripts/python.exe -m pytest tests/ -q
   ```
   Should print: `43 passed in X.XXs`

2. **Write `scripts/build_draft_board.py`** to wire together:
   - nflverse seasonal/weekly + xFP pulls
   - FantasyPros PPR consensus
   - Your league's VORP with the exact roster (1 QB / 2 RB / 2 WR / 1 TE / 1 FLEX / 1 K / 1 DEF)
   - Output: `outputs/projections/draft_board.{parquet,csv,html}`

3. **Test with a mock Sleeper draft** once the board exists:
   ```bash
   .venv/Scripts/python.exe -m src.draft.monitor --draft-id <test-id> --my-slot <n>
   ```

4. **After the draft:** run the backtest to establish a baseline of what the blend+drafter combination can do over 2020–2025.

That's the critical path to Thursday. Everything after that is post-draft work.

---

## Contact & continuity

**This work was scaffolded by Claude Opus with the following principles:**
- Two tranches: lean draft tool (ship first) + full backtest (ship after)
- Realistic edge is compounding small advantages, not prediction magic
- Leakage is the thing that will silently ruin everything, so guard aggressively
- Aggregation beats a solo model, so the consensus blend is the fallback
- Test everything with synthetic data before trusting real outcomes

If you're resuming this session later, re-read `CLAUDE.md` and check `git log --oneline` to see what changed.
