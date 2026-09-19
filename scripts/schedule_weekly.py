"""Register the weekly run with Windows Task Scheduler.

    python -m scripts.schedule_weekly            # show what would be created
    python -m scripts.schedule_weekly --install
    python -m scripts.schedule_weekly --remove

The task runs `scripts/run_weekly.cmd`, which writes
`outputs/reports/latest.brief.md` and `outputs/reports/last_run.log`. It is
half of a two-part arrangement: this machine *runs* the update, and Claude
Cowork's scheduled task *reads* the brief an hour later and relays it
(`docs/cowork_weekly_prompt.md`). Cowork cannot execute anything on a linked
computer — it has file access only — which is why the run has to be scheduled
here and the hand-off is a file.

The default is Wednesday 06:00, an hour before that read. This league
processes waivers on Wednesday (`waiver_day_of_week: 2`), so a 06:00 run lands
after processing, when the wire is free agents until kickoff locks. To place
claims *before* processing, register it with `--day TUE --time 20:00`
instead; the brief will then be the one Cowork reads on Wednesday morning.

Two settings `schtasks` cannot express are applied afterwards through
PowerShell: wake the machine to run, and run as soon as possible if the
scheduled time was missed (a laptop asleep or shut at 06:00). Without the
second one a missed Wednesday is simply skipped, and the brief Cowork reads is
last week's — which the brief's own STATUS line will say, but a run is better
than a warning.

This registers the task; it does not invent a scheduler. On Windows that is
`schtasks`. The equivalent line for cron is printed too, so the same job can be
moved to a machine that is actually on at 6am.

Nothing is created without `--install`, and the command is printed either way —
a scheduled task that silently appears is worse than one you had to approve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "FantasyWeeklyUpdate"
DEFAULT_DAY = "WED"
DEFAULT_TIME = "06:00"
RUNNER = ROOT / "scripts" / "run_weekly.cmd"
LOG = ROOT / "outputs" / "reports" / "last_run.log"
BRIEF = ROOT / "outputs" / "reports" / "latest.brief.md"


def _python() -> Path:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return venv if venv.exists() else Path(sys.executable)


def _command() -> str:
    """The command the task runs: the checked-in wrapper, quoted.

    The wrapper carries what a scheduled task does not inherit — `cd /d` into
    the repo (tasks start in System32, where `-m scripts.weekly_update` cannot
    import), `PYTHONIOENCODING` (an unset console dies on the first accented
    player name), the venv interpreter, and the log redirect. It used to be
    one `cmd /c "..."` string here; with a space in the repo path that is
    three layers of quoting and it broke in ways that looked like the
    script's fault.
    """
    return f'"{RUNNER}"'


# `schtasks /Create` cannot set these; the PowerShell cmdlets can. WakeToRun
# for a sleeping laptop, StartWhenAvailable so a missed 06:00 runs at the
# next opportunity instead of waiting a week.
_SETTINGS_PS = (
    "Set-ScheduledTask -TaskName '{task}' -Settings "
    "(New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable "
    "-ExecutionTimeLimit (New-TimeSpan -Hours 2))"
)


def _schtasks_args(day: str, time: str) -> list[str]:
    return [
        "schtasks", "/Create", "/TN", TASK_NAME,
        "/TR", _command(),
        "/SC", "WEEKLY", "/D", day, "/ST", time,
        "/RL", "LIMITED", "/F",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Schedule the weekly update")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--day", default=DEFAULT_DAY,
                    help="MON..SUN (default TUE)")
    ap.add_argument("--time", default=DEFAULT_TIME,
                    help="HH:MM 24-hour (default 06:00)")
    ap.add_argument("--run-now", action="store_true",
                    help="trigger the registered task immediately and wait "
                         "for it, then show the log's tail — the only real "
                         "test that Task Scheduler can run it")
    args = ap.parse_args()

    if args.run_now:
        return _run_now()

    if args.remove:
        out = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                             capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
        return out.returncode

    if not RUNNER.exists():
        print(f"missing {RUNNER}", file=sys.stderr)
        return 1

    argv = _schtasks_args(args.day, args.time)
    cron_dow = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4,
                "FRI": 5, "SAT": 6}.get(args.day.upper(), "*")
    print(f"working directory : {ROOT}")
    print(f"interpreter       : {_python()}")
    print(f"runner            : {RUNNER}")
    print(f"schedule          : every {args.day} at {args.time}")
    print(f"writes            : {BRIEF.relative_to(ROOT)}, "
          f"{LOG.relative_to(ROOT)}")
    print(f"\nWindows Task Scheduler:\n  {' '.join(argv)}")
    print(f"  powershell -NoProfile -Command "
          f"\"{_SETTINGS_PS.format(task=TASK_NAME)}\"")
    print(f"\ncron equivalent (Linux/macOS):\n"
          f"  {int(args.time.split(':')[1])} {int(args.time.split(':')[0])} "
          f"* * {cron_dow}  cd '{ROOT}' && PYTHONIOENCODING=utf-8 "
          f"{_python()} -m scripts.weekly_update --refresh "
          f"> outputs/reports/last_run.log 2>&1")

    if not args.install:
        print("\nDry run. Re-run with --install to register it.")
        return 0

    # `cwd=ROOT` here only affects this `schtasks /Create` call; the task
    # itself gets its working directory from the `cd /d` in `_command()`.
    out = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    print(f"\n{out.stdout.strip() or out.stderr.strip()}")
    if out.returncode != 0:
        return out.returncode

    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         _SETTINGS_PS.format(task=TASK_NAME)],
        capture_output=True, text=True, cwd=ROOT)
    if ps.returncode != 0:
        # The task exists and will run on time; it just will not wake the
        # machine or catch up a missed slot. Say so rather than fail.
        print(f"[warn] could not set wake/catch-up settings: "
              f"{ps.stderr.strip() or ps.stdout.strip()}")
    else:
        print("wake-to-run and run-if-missed: set")

    print(f"\nRegistered as '{TASK_NAME}'. Each run writes "
          f"{BRIEF.relative_to(ROOT)} and {LOG.relative_to(ROOT)}.")
    print("Test it now with: python -m scripts.schedule_weekly --run-now")
    print("Remove it with:   python -m scripts.schedule_weekly --remove")
    return 0


def _task_status() -> str:
    """The `Status:` field of `schtasks /Query /V` — "Ready", "Running",
    "Disabled". Searching the whole verbose dump for the word "Running" does
    not work: it always contains a "Stop If Still Running:" line, so the
    first version of `--run-now` waited out its full deadline on every run
    that had in fact finished in thirty seconds."""
    q = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME,
                        "/FO", "LIST", "/V"], capture_output=True, text=True)
    for line in q.stdout.splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _run_now() -> int:
    """Fire the registered task and wait for it, the way Wednesday will."""
    import time

    before = LOG.stat().st_mtime if LOG.exists() else 0.0
    out = subprocess.run(["schtasks", "/Run", "/TN", TASK_NAME],
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode != 0:
        return out.returncode

    print("waiting for the run to finish", end="", flush=True)
    deadline = time.time() + 15 * 60
    while time.time() < deadline:
        time.sleep(5)
        print(".", end="", flush=True)
        if (_task_status() != "Running" and LOG.exists()
                and LOG.stat().st_mtime > before):
            break
    print()

    if not LOG.exists():
        print(f"no log at {LOG} — the task did not start", file=sys.stderr)
        return 1
    tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"--- {LOG.relative_to(ROOT)} (last 25 lines) ---")
    print("\n".join(tail[-25:]))
    if BRIEF.exists():
        head = BRIEF.read_text(encoding="utf-8").splitlines()[0]
        print(f"\n{BRIEF.relative_to(ROOT)}: {head}")
        return 0 if "STATUS: OK" in head or "DEGRADED" in head else 1
    print(f"\n{BRIEF.relative_to(ROOT)} was not written", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
