# Claude Cowork — weekly in-season prompt

The recurring task Cowork runs every Wednesday at 7 AM. Paste everything
below the rule into the Cowork chat as-is. It relies on `weekly_update`
writing `outputs/reports/latest.brief.md` with a `STATUS:` line first, which
is what the agent leads with; see `scripts/weekly_update.py` and
`src/inseason/report.py::brief`.

Why the edits from the original draft:

- Cowork runs on this machine, so it needs the **local path**, not the GitHub
  URL. The URL is kept for reference only.
- `.venv` is not "activated"; the venv's `python.exe` is called directly. That
  is what every command in HANDOFF does and it cannot pick up the wrong
  interpreter.
- `PYTHONIOENCODING=utf-8` is set first, or the first accented player name
  kills the run (HANDOFF, "Draft day, start to finish").
- `--refresh` forces live pulls. Once a week is the right time to insist on
  them; if a host is down the script falls back to cache *and says so* in the
  brief's STATUS line, which is the case the summary must lead with.
- The agent reads the **brief**, not the full report. The brief was built for
  exactly this: status first, then only the verdicts, so a stale input can
  never be buried under a tidy lineup.
- HANDOFF.md is not needed to do this task and each scheduled run starts
  fresh, so "read it once" is rephrased as "read it only if something in the
  brief does not make sense".

---

This is my fantasy football repo. Local path: `C:\Users\markt\OneDrive\Desktop\Repositories\Fantasy Football` (GitHub: https://github.com/mnthompson98/fantasy-football, for reference only — work in the local checkout). You do not need to read HANDOFF.md to do this task; the brief you will read is self-contained. Open HANDOFF.md only if something in the brief does not make sense.

Recurring task, every Wednesday morning at 7 AM:

1. In PowerShell, from the repo root, run exactly:

   ```
   $env:PYTHONIOENCODING = "utf-8"
   git pull --ff-only
   .venv\Scripts\python.exe -m scripts.weekly_update --refresh
   ```

   Do not activate the venv or use a different `python`. If `git pull` fails, continue with the checkout as it is and mention it.

2. Confirm both files were written for the current week: `outputs/reports/weekNN.md` (full report) and `outputs/reports/weekNN.brief.md` (also copied to `outputs/reports/latest.brief.md`). The script prints their paths near the end.

3. Read `outputs/reports/latest.brief.md`. Its first line is `# STATUS: OK`, `# STATUS: DEGRADED`, or `# STATUS: FAILED`.

4. Summarize for me in chat, in this order, as a few bullets — not a re-paste of the report:
   - **First, unconditionally:** the STATUS line. If the command exited non-zero, crashed, or STATUS is DEGRADED or FAILED, put that at the very top in plain words with the reasons the brief lists (stale cache, rankings for a week already played, feed not scraped, ingestion failure). Say "this may be stale" before saying anything else.
   - Top start/sit calls: projected total, any UNFILLED slot, the close calls, any injury flags on my roster.
   - Waivers: whether anything clears the priority bar, and the free-agent adds.
   - Trade flags: the surplus players.

5. Do NOT submit any waiver claim, add, drop, or trade offer on Sleeper, and do not open Sleeper at all. This league uses rolling-priority waivers, not FAAB — a claim costs my queue position, so any submission goes through me first. Surface recommendations only. Note that a Wednesday 7 AM run lands after this league's waivers process, so "free add" means free right now, and "claim" means what it would cost once players lock at kickoff.

6. Do not edit, commit, or push anything in the repo. If the script errors, report the error text and stop; do not try to fix the code.
