# Claude Cowork — weekly in-season prompt

The recurring task Cowork runs every Wednesday at 7 AM. Paste everything
below the rule into the Cowork chat as-is.

**Cowork does not run anything.** The first version of this prompt asked it
to execute `weekly_update`, and it reported back that a linked computer
exposes file access only — no shell. So the job is split in two:

| when | who | what |
|---|---|---|
| Wed 06:00 | Windows Task Scheduler (`python -m scripts.schedule_weekly --install`) | runs `scripts/run_weekly.cmd` → writes `outputs/reports/latest.brief.md` and `outputs/reports/last_run.log` |
| Wed 07:00 | Claude Cowork (this prompt) | reads the brief, relays it |

The hand-off is a file, and the brief was built for exactly that: a
`STATUS:` line first (OK / DEGRADED / FAILED), a `Generated` timestamp on the
line after the title, then only the verdicts (`src/inseason/report.py::brief`).

A second, daily prompt reads the injury brief: `docs/cowork_injury_prompt.md`.

The one failure the brief cannot report on itself is *not having been
written* — the laptop was asleep, the task did not fire, the wrapper died
before Python started. So the prompt makes the agent check the `Generated`
timestamp before believing anything, and read the log when the brief is
missing or old. The scheduled task is registered with wake-to-run and
run-if-missed so that case should be rare; the check is for when it is not.

Timing: this league processes waivers on Wednesday (`waiver_day_of_week: 2`),
so a 06:00 run lands after processing. "Free add" in the brief means free
right then; "claim" means what a pickup would cost once players lock at
kickoff. To read a pre-processing brief instead, register the task with
`--day TUE --time 20:00`; nothing in this prompt changes.

---

This is my fantasy football repo, and you have file access to it at `C:\Users\markt\OneDrive\Desktop\Repositories\Fantasy Football`. You cannot run commands on that computer and do not need to: a scheduled task on the machine already ran the weekly update at 6 AM and wrote its output to files. Your job is to read those files and relay them. Do not try to execute anything. You do not need to read HANDOFF.md; the brief is self-contained. Open it only if something in the brief does not make sense.

Recurring task, every Wednesday morning at 7 AM:

1. Read `outputs\reports\latest.brief.md`.
   - Its first line is `# STATUS: OK`, `# STATUS: DEGRADED`, or `# STATUS: FAILED`.
   - The next line of text starts `*Generated YYYY-MM-DD HH:MM UTC.` — that is when the update actually ran. If that timestamp is more than 24 hours old, or the file does not exist, the scheduled run did not happen this week: say so at the very top of your summary, then read `outputs\reports\last_run.log` and quote its last error lines. Do not summarize a stale brief as if it were current.

2. Summarize for me in chat, in this order, as a few bullets — not a re-paste of the file:
   - **First, unconditionally:** the STATUS line and the Generated time. If STATUS is DEGRADED or FAILED, put that at the very top in plain words with the reasons the brief lists under "Read this first" (stale cache, rankings for a week already played, feed not scraped, ingestion failure). Say "this may be stale" before saying anything else.
   - Top start/sit calls: projected total, any UNFILLED slot, the close calls, any injury flags on my roster.
   - Waivers: whether anything clears the priority bar, and the free-agent adds.
   - **Dropped this week by other teams**, as its own bullet group: the brief lists anyone another manager dropped in the last three days and whether he is worth picking up. Relay every one marked "Worth it" with the cost the brief gives him — "CLAIM (burns priority)" if he is still on waivers, "free agent now" if he has cleared — and the "Not worth it" line as-is. If it says nobody was dropped, say that in one line.
   - Trade flags: the surplus players.

3. If you want detail beyond the brief, the full report for the week is `outputs\reports\weekNN.md` (the brief names it). Read it only to answer a question; do not summarize it in place of the brief.

4. Do NOT submit any waiver claim, add, drop, or trade offer on Sleeper, and do not open Sleeper at all. This league uses rolling-priority waivers, not FAAB — a claim costs my queue position, so any submission goes through me first. Surface recommendations only. The 6 AM run lands after this league's waivers process, so "free add" means free right now, and "claim" means what it would cost once players lock at kickoff.

5. Do not edit, create, or delete any file in the repo. If the brief or log shows an error, report the error text and stop; do not try to fix anything.
