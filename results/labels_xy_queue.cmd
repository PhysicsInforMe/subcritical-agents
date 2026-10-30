@echo off
cd /d %~dp0..
echo === labels X/Y queue start %date% %time% > results\labels_xy_runs.log
python scripts\phase_a_tabulate.py configs\phase_a_labels_xy\A_qwen0.6b_list_k0_e4_Tan_f0_allxy_N8.yaml configs\phase_a_labels_xy\A_qwen0.6b_list_k0_e4_Tan_f0_allxyba_N8.yaml configs\phase_a_labels_xy\A_qwen0.6b_list_k0_e4_Tan_f0_allnwxy_N8.yaml configs\phase_a_labels_xy\A_qwen0.6b_list_k0_e4_Tan_f0_allnwxyba_N8.yaml >> results\labels_xy_runs.log 2>&1
echo === 0.6B done %time% >> results\labels_xy_runs.log
python scripts\phase_a_tabulate.py configs\phase_a_labels_xy\A_qwen1.7b_list_k0_e4_Tan_f0_allxy_N8.yaml configs\phase_a_labels_xy\A_qwen1.7b_list_k0_e4_Tan_f0_allxyba_N8.yaml configs\phase_a_labels_xy\A_qwen1.7b_list_k0_e4_Tan_f0_allnwxy_N8.yaml configs\phase_a_labels_xy\A_qwen1.7b_list_k0_e4_Tan_f0_allnwxyba_N8.yaml --dtype bfloat16 >> results\labels_xy_runs.log 2>&1
echo === queue end %date% %time% >> results\labels_xy_runs.log
