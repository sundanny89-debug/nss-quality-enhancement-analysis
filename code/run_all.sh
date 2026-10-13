#!/bin/bash
cd "$(dirname "$0")"
echo "start $(date)" > ../out/run_status.txt
python3 03_fit_holdout.py main > ../out/log_holdout_main.txt 2>&1 && echo "holdout main done $(date)" >> ../out/run_status.txt
python3 04_fit_full.py > ../out/log_full.txt 2>&1 && echo "full done $(date)" >> ../out/run_status.txt
python3 05_sems.py > ../out/log_sems.txt 2>&1 && echo "sems done $(date)" >> ../out/run_status.txt
python3 03_fit_holdout.py rolling > ../out/log_holdout_rolling.txt 2>&1 && echo "holdout rolling done $(date)" >> ../out/run_status.txt
echo "ALL DONE $(date)" >> ../out/run_status.txt
