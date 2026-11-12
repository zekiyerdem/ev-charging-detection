#!/usr/bin/env bash
# Stress test: 16 A (3.3-3.8 kW) Level 2 chargers in every third Level 2 home and 4.5-6.5 kW hot tubs.
# Same pipeline as run_all.sh; results go to reports/hard_test/.
set -e
cd "$(dirname "$0")"
python3 src/generate_data.py --homes 600 --days 365 --out data_hard --hard
mkdir -p reports/hard_test
cd src && python3 detect.py --data ../data_hard --reports ../reports/hard_test
python3 sessions.py --data ../data_hard --reports ../reports/hard_test
