#!/bin/bash
cd "$(dirname "$0")"
echo "start $(date)" > ../out/status_11.txt
NW=2000 NS=2000 python3 10_model_v4.py > ../out/log_10_2000.txt 2>&1 && echo "M4-2000 done $(date)" >> ../out/status_11.txt
python3 09c_sensitivity.py > ../out/log_09c.txt 2>&1 && echo "09c done $(date)" >> ../out/status_11.txt
echo "ALL DONE $(date)" >> ../out/status_11.txt
