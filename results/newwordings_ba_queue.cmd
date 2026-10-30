@echo off
cd /d %~dp0..
echo === new wordings B/A queue start %date% %time% > results\newwordings_ba_runs.log
python scripts\phase_a_tabulate.py configs\phase_a_newwordings\A_qwen0.6b_list_k0_e4_Tan_f0_allnwba_N8.yaml >> results\newwordings_ba_runs.log 2>&1
echo === 0.6B done %time% >> results\newwordings_ba_runs.log
python scripts\phase_a_tabulate.py configs\phase_a_newwordings\A_qwen1.7b_list_k0_e4_Tan_f0_allnwba_N8.yaml --dtype bfloat16 >> results\newwordings_ba_runs.log 2>&1
echo === queue end %date% %time% >> results\newwordings_ba_runs.log
