@echo off
cd /d %~dp0..
python scripts\reproduce.py --check > results\reproduce_run.log 2>&1
echo === done >> results\reproduce_run.log
