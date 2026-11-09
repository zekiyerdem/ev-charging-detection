#!/usr/bin/env bash
# Generate synthetic data, cluster homes, compare baselines, check stability, and write results to reports/.
set -e
cd "$(dirname "$0")"
python3 src/generate_data.py --homes 600 --days 365 --out data
cd src && python3 detect.py --data ../data --reports ../reports
python3 compare_baselines.py --data ../data --reports ../reports
python3 stability.py --data ../data --reports ../reports
