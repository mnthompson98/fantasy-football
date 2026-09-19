# Claude Cowork — daily injury prompt

The recurring task Cowork runs every morning at 7 AM. Paste everything below
the rule into the Cowork chat as-is.

Same arrangement as the weekly brief (`docs/cowork_weekly_prompt.md`):
Cowork cannot run anything, so a Task Scheduler job runs
`scripts/injury_watch.py` at 06:30 daily (`python -m scripts.schedule_weekly
--task injuries --install`) and writes `outputs/reports/injuries.brief.md`;
Cowork reads it half an hour later.

The brief is a **diff**: it leads with what changed since yesterday's run,
and says "No changes" when nothing did. The point of a daily message is to
be ignorable on quiet days and impossible to miss when a starter's status
moves, so the prompt tells Cowork to be short when the brief is short.

---

This is my fantasy football repo, and you have file access to it at `C:\Users\markt\OneDrive\Desktop\Repositories\Fantasy Football`. You cannot run commands on that computer and do not need to: a scheduled task already ran the daily injury check at 6:30 AM and wrote its output to a file. Your job is to read that file and relay it. Do not try to execute anything, and do not read HANDOFF.md for this — the brief is self-contained.

Recurring task, every morning at 7 AM:

1. Read `outputs\reports\injuries.brief.md`.
   - Its first line is `# STATUS: OK`, `# STATUS: DEGRADED`, or `# STATUS: FAILED`.
   - The next line of text starts `*Generated YYYY-MM-DD HH:MM UTC.` If that timestamp is more than 24 hours old, or the file does not exist, the scheduled run did not happen: say so at the top, read `outputs\reports\injury_watch.log` and quote its last error lines, and stop. Do not relay a stale brief as if it were today's.

2. Relay it in chat:
   - If STATUS is DEGRADED or FAILED, say that first, with the reasons the brief lists.
   - Then the **"Changes since last check"** section, verbatim in substance: each player who moved, what he moved from and to. This is the whole point of the message.
   - If that section says **No changes**, your entire message is one line: "Injuries: no changes since yesterday. Flagged: <the names under "Currently flagged", or "none">." Do not pad it.
   - If there are changes, add the "Currently flagged" list after them so I can see the full picture.

3. Do NOT change my lineup, submit a waiver claim, add or drop anyone, or open Sleeper. This is a status report, not a decision — the lineup call is the Wednesday brief, and on game day it is mine.

4. Do not edit, create, or delete any file in the repo. If the brief or log shows an error, report the error text and stop.
