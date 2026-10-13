#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f 09a_final_modelB.py > /dev/null; do sleep 15; done
echo "09a finished $(date)" > ../out/status_09.txt
python3 09b_final_modelA.py > ../out/log_09b.txt 2>&1 && echo "09b done $(date)" >> ../out/status_09.txt
python3 09c_sensitivity.py > ../out/log_09c.txt 2>&1 && echo "09c done $(date)" >> ../out/status_09.txt
echo "ALL 09 DONE $(date)" >> ../out/status_09.txt
