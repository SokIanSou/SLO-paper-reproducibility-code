# Reproducibility package

Model, simulator and experiment scripts for "Session-Level Performance Modeling of Candidate-Driven
Camera Activation for SLO-Aware Tracking Services". Requires Python 3 with NumPy, SciPy and Matplotlib.

    ./run_all.sh

| File | Content | Paper |
|---|---|---|
| `slo_model.py` | mobility kernel, handover probabilities, session chain, rewards, SLO probability | Section 3 |
| `simulator.py` | Monte Carlo simulation of the tracking service (uses none of the model's equations) | Section 5.2 |
| `config.py` | configuration problem (P), Algorithm 1, uniform and proportional rules | Section 4 |
| `run_validation.py` | model versus simulation, 48 cases | Table 4, Figure 3 |
| `run_policy.py` | activation mechanisms and policy sweeps | Table 5, Figure 4 |
| `run_config.py` | SLO-constrained configuration, robustness checks, camera failure | Table 6, Figure 5 |
| `run_robustness.py` | first-order model against a target with directional inertia | Section 5.3, Table 5 |
| `run_scaling.py` | cost of an evaluation on random deployments; local-search check | Section 4.2 |
| `make_figures.py` | draws the figures from `results/*.json` | Figures 3-5 |

The transform equations are solved in the reduced form (non-candidate cameras only). All random seeds are fixed. Timings in `run_scaling.py` depend on the machine.
