"""Register the scheduled runs with Windows Task Scheduler.

    python -m scripts.schedule_weekly                        # dry run, weekly job
    python -m scripts.schedule_weekly --install              # register it
    python -m scripts.schedule_weekly --run-now              # fire it, show the log
    python -m scripts.schedule_weekly --remove
    python -m scripts.schedule_weekly --task injuries ...    # the daily job

Two jobs, one shape each:

| task       | runs                       | default     | writes                      |
|------------|----------------------------|-------------|-----------------------------|
| `weekly`   | scripts/run_weekly.cmd       | Wed 06:00   | latest.brief.md, last_run.log |
| `injuries` | scripts/run_injury_watch.cmd | daily 06:30 | injuries.brief.md, injury_watch.log |

Each is half of a two-part arrangement: this machine *runs* the script, and
Claude Cowork's scheduled task *reads* the brief afterwards and relays it
(`docs/cowork_weekly_prompt.md`, `docs/cowork_injury_prompt.md`). Cowork
cannot execute anything on a linked computer — it has file access only —
which is why the runs are scheduled here and the hand-off is a file.

Wednesday 06:00 is an hour before Cowork's read. This league processes
waivers on Wednesday (`waiver_day_of_week: 2`, and the log shows it happening
around 03:00 ET), so a 06:00 run lands after processing, when the wire is
free agents until kickoff locks. To place claims *before* processing,
register it with `--day TUE --time 20:00` instead; the brief will then be
the one Cowork reads on Wednesday morning.

Four settings `schtasks` cannot express are applied afterwards through
PowerShell: wake the machine to run; run as soon as possible if the
scheduled time was missed (a laptop asleep or shut at 06:00); and start on —
and do not stop on — battery power. Without run-if-missed a missed Wednesday
is simply skipped, and the brief Cowork reads is last week's — which the
brief's own STATUS line will say, but a run is better than a warning. Without
the battery pair the task is killed at launch (see `_SETTINGS_PS`).

One setting this script cannot apply is "run whether user is logged on or
not": it needs your Windows password, which is not going through a script.
Set it in Task Scheduler's GUI after `--install` (task → Properties →
General). The task then runs as a batch logon; if the Windows password ever
changes, the task fails silently with result 0x8007052E until it is re-entered.

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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"


@dataclass(frozen=True)
class Job:
    task_name: str
    runner: Path
    log: Path
    brief: Path
    schedule: str          # WEEKLY | DAILY
    day: str | None        # for WEEKLY
    time: str
    module: str            # for the cron line
    extra: str = ""        # extra args for the cron line

    @property
    def cron_dow(self) -> str:
        if self.schedule != "WEEKLY" or not self.day:
            return "*"
        return str({"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4,
                    "FRI": 5, "SAT": 6}.get(self.day.upper(), "*"))


JOBS: dict[str, Job] = {
    "weekly": Job(
        task_name="FantasyWeeklyUpdate",
        runner=ROOT / "scripts" / "run_weekly.cmd",
        log=REPORTS / "last_run.log",
        brief=REPORTS / "latest.brief.md",
        schedule="WEEKLY", day="WED", time="06:00",
        module="scripts.weekly_update", extra="--refresh",
    ),
    "injuries": Job(
        task_name="FantasyInjuryWatch",
        runner=ROOT / "scripts" / "run_injury_watch.cmd",
        log=REPORTS / "injury_watch.log",
        brief=REPORTS / "injuries.brief.md",
        schedule="DAILY", day=None, time="06:30",
        module="scripts.injury_watch", extra="--refresh",
    ),
}

# The weekly job's names, kept at module level: HANDOFF and the tests refer
# to them, and the weekly job is the one that existed first.
TASK_NAME = JOBS["weekly"].task_name
DEFAULT_DAY = JOBS["weekly"].day
DEFAULT_TIME = JOBS["weekly"].time
RUNNER = JOBS["weekly"].runner
LOG = JOBS["weekly"].log
BRIEF = JOBS["weekly"].brief


def _python() -> Path:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return venv if venv.exists() else Path(sys.executable)


def _command(job: Job = JOBS["weekly"]) -> str:
    """The command the task runs: the checked-in wrapper, quoted.

    The wrapper carries what a scheduled task does not inherit — `cd /d` into
    the repo (tasks start in System32, where `-m scripts.weekly_update` cannot
    import), `PYTHONIOENCODING` (an unset console dies on the first accented
    player name), the venv interpreter, and the log redirect. It used to be
    one `cmd /c "..."` string here; with a space in the repo path that is
    three layers of quoting and it broke in ways that looked like the
    script's fault.
    """
    return f'"{job.runner}"'


# `schtasks /Create` cannot set these; the PowerShell cmdlets can. WakeToRun
# for a sleeping laptop, StartWhenAvailable so a missed slot runs at the
# next opportunity instead of waiting for the next one.
#
# The two battery flags are not optional on a laptop. `schtasks /Create`
# defaults to "do not start on battery" AND "stop if going on battery", and
# the second one killed FantasyInjuryWatch on every run — an empty log
# ending in `^C`, result 0xC000013A — with the machine reporting itself
# plugged in. The weekly task only ever worked because those boxes had been
# unticked by hand in the GUI.
_SETTINGS_PS = (
    "Set-ScheduledTask -TaskName '{task}' -Settings "
    "(New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable "
    "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
    "-ExecutionTimeLimit (New-TimeSpan -Hours 2))"
)


def _schtasks_args(job: Job, day: str | None, time: str) -> list[str]:
    argv = ["schtasks", "/Create", "/TN", job.task_name, "/TR", _command(job)]
    if job.schedule == "WEEKLY":
        argv += ["/SC", "WEEKLY", "/D", day or job.day or "WED"]
    else:
        argv += ["/SC", "DAILY"]
    argv += ["/ST", time, "/RL", "LIMITED", "/F"]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description="Schedule the weekly update")
    ap.add_argument("--task", default="weekly", choices=sorted(JOBS),
                    help="which job (default weekly)")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--day", default=None,
                    help="MON..SUN for the weekly job (default WED)")
    ap.add_argument("--time", default=None,
                    help="HH:MM 24-hour (default 06:00 weekly, 06:30 daily)")
    ap.add_argument("--run-now", action="store_true",
                    help="trigger the registered task immediately and wait "
                         "for it, then show the log's tail — the only real "
                         "test that Task Scheduler can run it")
    args = ap.parse_args()
    job = JOBS[args.task]
    day = args.day or job.day
    time = args.time or job.time

    if args.run_now:
        return _run_now(job)

    if args.remove:
        out = subprocess.run(["schtasks", "/Delete", "/TN", job.task_name, "/F"],
                             capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
        return out.returncode

    if not job.runner.exists():
        print(f"missing {job.runner}", file=sys.stderr)
        return 1

    argv = _schtasks_args(job, day, time)
    when = f"every {day} at {time}" if job.schedule == "WEEKLY" else f"daily at {time}"
    print(f"task              : {job.task_name}")
    print(f"working directory : {ROOT}")
    print(f"interpreter       : {_python()}")
    print(f"runner            : {job.runner}")
    print(f"schedule          : {when}")
    print(f"writes            : {job.brief.relative_to(ROOT)}, "
          f"{job.log.relative_to(ROOT)}")
    print(f"\nWindows Task Scheduler:\n  {' '.join(argv)}")
    print(f"  powershell -NoProfile -Command "
          f"\"{_SETTINGS_PS.format(task=job.task_name)}\"")
    print(f"\ncron equivalent (Linux/macOS):\n"
          f"  {int(time.split(':')[1])} {int(time.split(':')[0])} "
          f"* * {job.cron_dow}  cd '{ROOT}' && PYTHONIOENCODING=utf-8 "
          f"{_python()} -m {job.module} {job.extra} "
          f"> {job.log.relative_to(ROOT).as_posix()} 2>&1")

    if not args.install:
        print("\nDry run. Re-run with --install to register it.")
        return 0

    # `cwd=ROOT` here only affects this `schtasks /Create` call; the task
    # itself gets its working directory from the `cd /d` in the wrapper.
    out = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    print(f"\n{out.stdout.strip() or out.stderr.strip()}")
    if out.returncode != 0:
        return out.returncode

    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         _SETTINGS_PS.format(task=job.task_name)],
        capture_output=True, text=True, cwd=ROOT)
    if ps.returncode != 0:
        # The task exists and will run on time; it just will not wake the
        # machine or catch up a missed slot. Say so rather than fail.
        print(f"[warn] could not set wake/catch-up settings: "
              f"{ps.stderr.strip() or ps.stdout.strip()}")
    else:
        print("wake-to-run and run-if-missed: set")

    print(f"\nRegistered as '{job.task_name}'. Each run writes "
          f"{job.brief.relative_to(ROOT)} and {job.log.relative_to(ROOT)}.")
    print("To run while logged off: Task Scheduler → task → Properties → "
          "General → 'Run whether user is logged on or not' (needs your "
          "Windows password; this script will not ask for it).")
    print(f"Test it now with: python -m scripts.schedule_weekly "
          f"--task {args.task} --run-now")
    print(f"Remove it with:   python -m scripts.schedule_weekly "
          f"--task {args.task} --remove")
    return 0


def _task_status(job: Job) -> str:
    """The `Status:` field of `schtasks /Query /V` — "Ready", "Running",
    "Disabled". Searching the whole verbose dump for the word "Running" does
    not work: it always contains a "Stop If Still Running:" line, so the
    first version of `--run-now` waited out its full deadline on every run
    that had in fact finished in thirty seconds."""
    q = subprocess.run(["schtasks", "/Query", "/TN", job.task_name,
                        "/FO", "LIST", "/V"], capture_output=True, text=True)
    for line in q.stdout.splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _run_now(job: Job) -> int:
    """Fire the registered task and wait for it, the way the schedule will."""
    import time as _time

    log, brief = job.log, job.brief
    before = log.stat().st_mtime if log.exists() else 0.0
    out = subprocess.run(["schtasks", "/Run", "/TN", job.task_name],
                         capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    if out.returncode != 0:
        return out.returncode

    print("waiting for the run to finish", end="", flush=True)
    deadline = _time.time() + 15 * 60
    while _time.time() < deadline:
        _time.sleep(5)
        print(".", end="", flush=True)
        if (_task_status(job) != "Running" and log.exists()
                and log.stat().st_mtime > before):
            break
    print()

    if not log.exists():
        print(f"no log at {log} — the task did not start", file=sys.stderr)
        return 1
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"--- {log.relative_to(ROOT)} (last 25 lines) ---")
    print("\n".join(tail[-25:]))
    if brief.exists():
        head = brief.read_text(encoding="utf-8").splitlines()[0]
        print(f"\n{brief.relative_to(ROOT)}: {head}")
        return 0 if "STATUS: OK" in head or "DEGRADED" in head else 1
    print(f"\n{brief.relative_to(ROOT)} was not written", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
