@echo off
cd /d %~dp0..
echo === new wordings queue start %date% %time% > results\newwordings_runs.log
python scripts\phase_a_tabulate.py --all configs\phase_a_newwordings --model qwen0.6b >> results\newwordings_runs.log 2>&1
echo === 0.6B done %time% >> results\newwordings_runs.log
python scripts\phase_a_tabulate.py --all configs\phase_a_newwordings --model qwen1.7b --dtype bfloat16 >> results\newwordings_runs.log 2>&1
echo === queue end %date% %time% >> results\newwordings_runs.log
