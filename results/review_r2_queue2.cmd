@echo off
cd /d %~dp0..
echo === queue2 start %date% %time% > results\review_r2_runs.log
python scripts\review_r2_generation_check.py --model qwen0.6b >> results\review_r2_runs.log 2>&1
echo === 0.6B generation done %time% >> results\review_r2_runs.log
python scripts\phase_a_tabulate.py configs\phase_a_review_r2\A_qwen1.7b_list_k0_e4_Tan_f0_allba_N8.yaml --dtype bfloat16 >> results\review_r2_runs.log 2>&1
echo === 1.7B cell done %time% >> results\review_r2_runs.log
python scripts\review_r2_generation_check.py --model qwen1.7b >> results\review_r2_runs.log 2>&1
echo === queue end %date% %time% >> results\review_r2_runs.log
