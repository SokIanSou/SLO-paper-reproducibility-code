#!/bin/sh
# Regenerates every table, figure and quoted number of the paper (about 5 minutes on two cores).
set -e
python3 run_validation.py | tee results/validation.txt
python3 run_policy.py     | tee results/policy.txt
python3 run_config.py     | tee results/config.txt
python3 run_scaling.py    | tee results/scaling.txt
python3 run_robustness.py | tee results/robustness.txt
python3 make_figures.py
