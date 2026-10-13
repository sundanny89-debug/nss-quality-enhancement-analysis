#!/bin/bash
cd "$(dirname "$0")"
echo "start $(date)" > ../out/status_10.txt
python3 10_model_v4.py > ../out/log_10.txt 2>&1 && echo "10 done $(date)" >> ../out/status_10.txt
python3 09c_sensitivity.py > ../out/log_09c.txt 2>&1 && echo "09c done $(date)" >> ../out/status_10.txt
echo "ALL DONE $(date)" >> ../out/status_10.txt
