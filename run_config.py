"""Section 6.3: SLO-constrained configuration (Table 'config', Figure 'config'), sensitivity and checks."""
import json, time, numpy as np
from slo_model import *
from config import *

T = [10, 20, 30, 40, 60, 80, 100, 120]; KMAX = 9; N = 10
d, Pdir, Pgap = mobility(); fam = exponential_family(N)
prob = Problem(d, Pdir, Pgap, P0, fam)
out = {"targets": [], "T": T}
t0 = time.time()
print("target | uniform        | proportional   | model-driven   | red. vs U  vs P | k (model-driven)")
for tg in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    n0 = len(prob.cache); t1 = time.time()
    (gl, gk, gT), pimax, _ = greedy(prob, T, tg, KMAX)
    ng, tg_time = len(prob.cache) - n0, time.time() - t1
    ul, uk, uT = uniform(prob, T, tg, KMAX); pl, pk, pT = proportional(prob, T, tg, KMAX)
    out["targets"].append(dict(target=tg, uniform=[ul, int(uT), uk.tolist()], prop=[pl, int(pT), pk.tolist()],
                               model=[gl, int(gT), gk.tolist()], new_evals=ng, seconds=tg_time,
                               pi=[prob.evaluate(uk, uT)[0], prob.evaluate(pk, pT)[0], prob.evaluate(gk, gT)[0]]))
    print(f"  {tg:.1f}  | {ul:6.1f} ({uT:3d}) k={uk[0]} | {pl:6.1f} ({pT:3d})   | {gl:6.1f} ({gT:3d})   | {100*(1-gl/ul):5.1f}% {100*(1-gl/pl):5.1f}% | {gk.tolist()}  prop k={pk.tolist()}  [{ng} new evals, {tg_time:.1f}s]")
out["pimax"] = pimax; out["distinct_evals"] = len(prob.cache); out["rows"] = len(prob.rows); out["seconds"] = time.time() - t0
print(f"Pi_max = {pimax:.3f}; distinct configurations evaluated {len(prob.cache)}, handover rows computed {len(prob.rows)}, total {time.time()-t0:.1f}s")
# a cold run of Algorithm 1 alone at target 0.5
p2 = Problem(d, Pdir, Pgap, P0, fam); t1 = time.time(); greedy(p2, T, 0.5, KMAX)
out["cold"] = dict(evals=len(p2.cache), rows=len(p2.rows), seconds=time.time() - t1)
print("cold run of Algorithm 1 at 0.5:", out["cold"])

# ---- is Pi monotone in k on the nested nearest-first family?  (not guaranteed: Proposition (iii))
rng = np.random.default_rng(5); neg = 0; tot = 0; worst = 0.0
for _ in range(3000):
    k = rng.integers(0, 9, N); Th = int(rng.choice(T)); i = rng.integers(N)
    if k[i] >= 9: continue
    e = k.copy(); e[i] += 1
    dp = prob.evaluate(e, Th)[0] - prob.evaluate(k, Th)[0]; tot += 1
    if dp < -1e-12: neg += 1; worst = min(worst, dp)
out["mono_check"] = dict(trials=tot, decreases=neg, worst=worst)
print(f"single-camera increments on nearest-first sets: Pi decreased in {neg} of {tot} (worst {worst:.2e})")

# ---- sensitivity to the locality length
out["ell"] = []
for ell in (10, 15, 20, 25):
    dd, Pd, Pg = mobility(ell=ell); pr = Problem(dd, Pd, Pg, P0, fam)
    (gl, gk, gT), _, _ = greedy(pr, T, 0.5, KMAX); ul, uk, uT = uniform(pr, T, 0.5, KMAX); pl, pk, pT = proportional(pr, T, 0.5, KMAX)
    out["ell"].append(dict(ell=ell, model=gl, Tm=int(gT), uniform=ul, Tu=int(uT), prop=pl))
    print(f"  ell={ell}: model {gl:.1f} (T={gT}) uniform {ul:.1f} (T={uT}, k={uk[0]}) prop {pl:.1f}: reduction {100*(1-gl/ul):.1f}% / {100*(1-gl/pl):.1f}%")

# ---- run-time reconfiguration: camera 6 fails (cannot see; the target still walks through its field of view)
dead = 5; alive = [j for j in range(N) if j != dead]
nu = np.array([0 if j == dead else 1 / (N - 1) for j in range(N)])          # nu renormalised over live cameras; Lambda unchanged
cur = out["targets"][2]["model"]; Tcur = cur[1]
old_sets = prob.sets(np.array(cur[2]))
deg_sets = [old_sets[i] - {dead} if i != dead else set() for i in range(N)]  # degraded service: the failed camera simply drops out
m = evaluate(deg_sets, Tcur, Pdir, Pgap, P0, fam)
pi_deg, load_deg = float(nu @ m["slo"]), float(0.05 * nu @ m["K"])
k0 = np.array([len(sg) for sg in deg_sets])                                  # warm start: each camera keeps its live candidates
pf = Problem(d, Pdir, Pgap, P0, fam, nu=nu, alive=alive)
assert all(pf.sets(k0)[i] == deg_sets[i] for i in range(N))
full = np.array([len(pf.order[i]) if i != dead else 0 for i in range(N)])
pi_full_same_T = pf.evaluate(full, Tcur)[0]
t1 = time.time(); (gl, gk, gT), pim, st = greedy(pf, T, 0.5, KMAX); cold = (len(pf.cache), time.time() - t1)
pw = Problem(d, Pdir, Pgap, P0, fam, nu=nu, alive=alive); t1 = time.time()
(wl, wk, wT), _, _ = greedy(pw, T, 0.5, KMAX, k0=k0); warm = (len(pw.cache), time.time() - t1)
out["failure"] = dict(camera=dead + 1, pi_before=out["targets"][2]["pi"][2], pi_degraded=pi_deg, load_degraded=load_deg,
                      load_before=cur[0], pi_full_same_T=pi_full_same_T, k0=k0.tolist(),
                      reopt=[gl, int(gT), gk.tolist()], pimax=pim, cold=cold, warm=[wl, int(wT), wk.tolist()], warm_cost=warm)
print(f"camera {dead+1} fails: Pi {out['targets'][2]['pi'][2]:.3f} -> {pi_deg:.3f} (load {cur[0]:.1f} -> {load_deg:.1f}); best Pi at the same T_h with all live cameras as candidates: {pi_full_same_T:.3f}; Pi_max {pim:.3f}")
print(f"  re-optimised (cold): {gl:.1f} Mbps, T={gT}, k={gk.tolist()}; {cold[0]} evals {cold[1]:.1f}s")
print(f"  warm start k0={k0.tolist()}: {wl:.1f} Mbps, T={wT}, k={wk.tolist()}; {warm[0]} evals {warm[1]:.1f}s; changed sizes at {int((wk!=k0).sum())} cameras")
# kmax < N-1: the test at the maximal allocation is sufficient only
for km in (3, 5):
    r = greedy(Problem(d, Pdir, Pgap, P0, fam), T, 0.9, km); print(f"  kmax={km}, target 0.9: status {r[2]}, Pi at maximal allocation {r[1]:.3f}")
# per-session cost behind the load figures
out["per_session"] = [dict(target=t["target"], K=t["model"][0] / 0.05, streams=t["model"][0] / 2.5) for t in out["targets"]]
print("  per-session cost (Mbps*s) and unshared concurrent streams:", [(t["target"], round(t["K"]), round(t["streams"], 1)) for t in out["per_session"]])
json.dump(out, open("results/config.json", "w"), default=float)
