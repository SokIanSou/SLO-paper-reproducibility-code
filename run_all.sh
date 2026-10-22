#!/usr/bin/env bash
# Regenerates every number and figure in the paper. ~2 h, dominated by the
# Monte Carlo validation sweep (144 runs x 20,000 sessions).
set -e
cd "$(dirname "$0")/src"
mkdir -p ../results ../figures res figs
python3 validate_120.py     # Table IV, and the SLO tail at D_min
python3 e0_validation.py    # Table IV summary
python3 valfig.py           # Fig. 3  (continuity-time CCDF)
python3 e1_ablation.py      # Table V (activation mechanisms)
python3 e3_policy.py        # data for Fig. 4
python3 e3b.py              # Fig. 4  (policy parameters)
python3 e5_config.py        # Table VI (SLO-constrained configuration)
python3 e5b.py              # Fig. 5  (configuration)
python3 e4_scal.py          # timings quoted in Sec. IV-B
python3 ell_sens.py         # locality-length sensitivity, Sec. V-A
python3 check_table3.py     # verifies every case probability in Table II sums to 1
cp res/*.json ../results/ ; cp figs/*.pdf ../figures/
echo "done"
