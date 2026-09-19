@echo off
rem The scheduled daily injury check. Registered by
rem `python -m scripts.schedule_weekly --task injuries --install`. Same shape
rem as run_weekly.cmd and for the same reasons: Task Scheduler starts in
rem System32 with no environment, so the working directory, the encoding and
rem the interpreter are all set here, relative to this file.
rem
rem Output goes to outputs\reports\injury_watch.log. The brief the reader
rem relays is outputs\reports\injuries.brief.md.

cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
if not exist outputs\reports mkdir outputs\reports

rem No git pull here: the Wednesday job pulls, and a daily pull racing a
rem checkout you are working in is a way to get a surprise merge at 06:30.

".venv\Scripts\python.exe" -m scripts.injury_watch --refresh > outputs\reports\injury_watch.log 2>&1
exit /b %ERRORLEVEL%
