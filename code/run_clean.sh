#!/bin/bash
cd "$(dirname "$0")"
echo "start $(date)" > ../out/status_clean.txt
python3 run_M4_clean_refit_ext.py > ../out/log_clean.txt 2>&1 && echo "clean refit done $(date)" >> ../out/status_clean.txt
echo "ALL DONE $(date)" >> ../out/status_clean.txt
