@echo off
rem The scheduled weekly run. Registered by `python -m scripts.schedule_weekly
rem --install`; Task Scheduler calls this file, not python directly, because a
rem repo path with a space inside a `schtasks /TR` string is three layers of
rem quoting deep and fails in ways that look like the script's fault.
rem
rem Everything a scheduled task does not inherit is set here: the working
rem directory (tasks start in System32), the encoding (the first accented
rem player name kills an unset console), and the interpreter (the venv's,
rem by relative path, so the checkout can move).
rem
rem Output goes to outputs\reports\last_run.log. The brief the reader relays
rem is outputs\reports\latest.brief.md; when that file is missing or old,
rem the log is where the reason is.

cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
if not exist outputs\reports mkdir outputs\reports

rem Pull whatever was merged on GitHub since the last run. The repo is public,
rem so a fetch needs no credentials and cannot hang on a prompt; --ff-only
rem refuses to touch a checkout that has diverged, and a refused pull is
rem logged and does not stop the run.
git pull --ff-only > outputs\reports\last_run.log 2>&1

".venv\Scripts\python.exe" -m scripts.weekly_update --refresh >> outputs\reports\last_run.log 2>&1
exit /b %ERRORLEVEL%
