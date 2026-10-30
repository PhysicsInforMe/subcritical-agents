@echo off
cd /d %~dp0..
echo === queue3 start %date% %time% > results\review_r2_runs3.log
python scripts\review_r2_generation_check.py --model qwen0.6b --max-new-tokens 40 --out results\phase_a\generation_check_40.json >> results\review_r2_runs3.log 2>&1
echo === queue end %date% %time% >> results\review_r2_runs3.log
