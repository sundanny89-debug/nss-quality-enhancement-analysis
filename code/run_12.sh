#!/bin/bash
cd "$(dirname "$0")"
echo "start $(date)" > ../out/status_12.txt
NW=2000 NS=2000 python3 10b_model_v4_holdout.py > ../out/log_10b.txt 2>&1 && echo "10b done $(date)" >> ../out/status_12.txt
python3 09c_sensitivity.py > ../out/log_09c.txt 2>&1 && echo "09c done $(date)" >> ../out/status_12.txt
echo "ALL DONE $(date)" >> ../out/status_12.txt
