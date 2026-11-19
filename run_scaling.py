"""Section 4.2: cost of one evaluation on random deployments, and a local-search check of Algorithm 1."""
import json, time, numpy as np
from slo_model import *
from config import *

out = {"timing": []}
rng = np.random.default_rng(42)
print("   N | all handover rows | one row | expected metrics | SLO probabilities | locations per row")
for N in (10, 20, 40, 80, 160, 320):
    side = 100 * np.sqrt(N / 22.0); pos = rng.random((N, 2)) * side
    dd = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    m3 = np.sort(dd, axis=1)[:, 1:4].mean(1); p0 = 0.3 + 0.4 * (m3 - m3.min()) / (m3.max() - m3.min())
    d, Pdir, Pgap = mobility(pos, p0); fam = exponential_family(N); Phi = nearest_sets(d, 5)
    reps = 3 if N <= 80 else 1
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); a, w = handover(Phi, 20.0, Pdir, Pgap, p0, fam)
        t1 = time.perf_counter(); m = metrics(a, w, Phi, Pdir, 30.0, 2.5)
        t2 = time.perf_counter(); slo = slo_ccdf(a, m["q"], fam.lstH, 120.0); t3 = time.perf_counter()
        t.append((t1 - t0, t2 - t1, t3 - t2))
    t = np.min(t, axis=0)
    out["timing"].append(dict(N=N, rows=t[0], row=t[0] / N, metrics=t[1], slo=t[2]))
    print(f"{N:4d} | {t[0]*1e3:12.1f} ms | {t[0]/N*1e3:7.2f} ms | {t[1]*1e3:10.2f} ms | {t[2]*1e3:10.1f} ms | {2*(N-6)+1}")
tt = out["timing"]; lg = lambda key: np.polyfit(np.log([x["N"] for x in tt[2:]]), np.log([x[key] for x in tt[2:]]), 1)[0]
out["exponents"] = dict(rows=lg("rows"), row=lg("row"), slo=lg("slo"))
print("fitted growth exponents (N >= 40):", {k: round(v, 2) for k, v in out["exponents"].items()})

# ---- local-search check: from the configuration returned by Algorithm 1, try every single move and every
#      exchange (k_i + 1, k_j - 1) at the chosen grace period and its neighbours in the grid
T = [10, 20, 30, 40, 60, 80, 100, 120]; N = 10
d, Pdir, Pgap = mobility(); prob = Problem(d, Pdir, Pgap, P0, exponential_family(N)); out["local"] = []
for tg in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    (gl, gk, gT), _, _ = greedy(prob, T, tg, 9)
    best = (gl, gk.copy(), gT); improved = True
    while improved:
        improved = False
        for Th in [x for x in T if abs(T.index(x) - T.index(best[2])) <= 1]:
            k = best[1]
            moves = [(i, s, None) for i in range(N) for s in (-1, 1)] + [(i, 1, j) for i in range(N) for j in range(N) if i != j] \
                    + [(None, 0, None)]
            for i, s, j in moves:
                e = k.copy()
                if i is not None: e[i] += s
                if j is not None: e[j] -= 1
                if e.min() < 0 or e.max() > 9: continue
                p, l = prob.evaluate(e, Th)
                if p >= tg and l < best[0] - 1e-9:
                    best = (l, e, Th); improved = True
    out["local"].append(dict(target=tg, greedy=gl, local=best[0], T=int(best[2]), k=best[1].tolist()))
    print(f"target {tg}: Algorithm 1 {gl:.2f} Mbps -> after local search {best[0]:.2f} Mbps ({100*(1-best[0]/gl):.2f}% lower), T={best[2]}")
json.dump(out, open("results/scaling.json", "w"), default=float)
