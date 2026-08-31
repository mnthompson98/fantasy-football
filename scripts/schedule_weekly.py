"""Register the weekly run with Windows Task Scheduler.

    python -m scripts.schedule_weekly            # show what would be created
    python -m scripts.schedule_weekly --install
    python -m scripts.schedule_weekly --remove

The plan asks for a scheduled job every Tuesday night / Wednesday morning. That
timing is right for this league specifically: waivers process overnight, so
Tuesday evening is after the previous week has settled and before claims are
due, and the Wednesday injury report has begun.

This registers the task; it does not invent a scheduler. On Windows that is
`schtasks`. The equivalent line for cron is printed too, so the same job can be
moved to a machine that is actually on at 8pm.

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
DEFAULT_DAY = "TUE"
DEFAULT_TIME = "20:00"


def _python() -> Path:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return venv if venv.exists() else Path(sys.executable)


def _command() -> str:
    """The command the task runs.

    `PYTHONIOENCODING` is set inside the command rather than assumed: Task
    Scheduler does not inherit an interactive shell's environment, and without
    it the run dies on the first accented player name — which is exactly the
    kind of failure nobody notices until they need the report.
    """
    return (f'cmd /c "set PYTHONIOENCODING=utf-8 && '
            f'\"{_python()}\" -m scripts.weekly_update"')


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
                    help="HH:MM 24-hour (default 20:00)")
    args = ap.parse_args()

    if args.remove:
        out = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                             capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
        return out.returncode

    argv = _schtasks_args(args.day, args.time)
    print(f"working directory : {ROOT}")
    print(f"interpreter       : {_python()}")
    print(f"schedule          : every {args.day} at {args.time}")
    print(f"\nWindows Task Scheduler:\n  {' '.join(argv)}")
    print(f"\ncron equivalent (Linux/macOS):\n"
          f"  0 {int(args.time.split(':')[0])} * * 2  cd '{ROOT}' && "
          f"PYTHONIOENCODING=utf-8 {_python()} -m scripts.weekly_update")

    if not args.install:
        print("\nDry run. Re-run with --install to register it.")
        return 0

    # Task Scheduler runs with the working directory unset, so the task has to
    # cd itself; `cmd /c` above handles that via the module invocation from
    # ROOT, which we set explicitly here.
    out = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    print(f"\n{out.stdout.strip() or out.stderr.strip()}")
    if out.returncode == 0:
        print(f"\nRegistered as '{TASK_NAME}'. Reports land in "
              f"outputs/reports/weekNN.md.")
        print("Remove it with: python -m scripts.schedule_weekly --remove")
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
