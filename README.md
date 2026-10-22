# Session-level performance model of candidate-driven camera activation

Code and data for the paper *Session-Level Performance Modeling of
Candidate-Driven Camera Activation for SLO-Aware Tracking Services*.

The paper uses **no external dataset**. Every number in it is produced by the
model and the Monte Carlo simulator in this repository, from the testbed and
the parameters printed in Section V-A. Nothing has to be downloaded.

## Quick start

```
pip install -r requirements.txt
./run_all.sh                 # ~2 h; the Monte Carlo sweep dominates
```

Results land in `results/` as JSON and figures in `figures/` as PDF.

## What produces what

| Paper item | Script | Output |
| --- | --- | --- |
| Table II (case probabilities) | `src/check_table3.py` | verifies each row sums to 1 |
| Table IV (model vs simulation) | `src/validate_120.py`, `src/e0_validation.py` | `results/validation120.json`, `results/e0.json` |
| Fig. 3 (continuity-time CCDF) | `src/valfig.py` | `figures/val_ccdf_wide.pdf` |
| Table V (activation mechanisms) | `src/e1_ablation.py` | `results/e1.json` |
| Fig. 4 (policy parameters) | `src/e3_policy.py`, `src/e3b.py` | `results/e3.json`, `figures/e3_policy_1col.pdf` |
| Table VI, Fig. 5 (configuration) | `src/e5_config.py`, `src/e5b.py` | `results/e5.json`, `figures/e5_config_1col.pdf` |
| Timings in Sec. IV-B | `src/e4_scal.py` | `results/e4.json` |
| Locality-length check, Sec. V-A | `src/ell_sens.py` | `results/ell_sens.json` |

## The model

- `src/model.py` — the model of Section III. `Deployment` builds the transition
  kernel from camera positions; `metrics()` solves the renewal-reward system for
  the expected session metrics; `Fast.ccdf()` inverts the Laplace transform for
  `Pr(D >= d)` by the Abate–Whitt Euler algorithm; `greedy_config()` is
  Algorithm 1; `simulate()` is the discrete-event simulator, which shares none
  of the model's equations.
- `src/common.py` — plotting defaults and the SLO threshold.
- `TESTBED_XY` and `TESTBED_P0` in `model.py` are the ten camera positions and
  out-of-coverage probabilities printed in Section V-A.

## Determinism

Every simulation call takes an explicit seed, so `run_all.sh` reproduces the
reported numbers exactly on any machine. Timings in `e4_scal.py` are wall-clock
and will differ with hardware; the growth exponents will not.

## License

See `LICENSE`.
