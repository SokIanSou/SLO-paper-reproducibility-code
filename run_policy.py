"""Section 6.1-6.2: activation mechanisms (Table 'ablation') and policy sweeps (Figure 'policy')."""
import json, numpy as np
from slo_model import *

d, Pdir, Pgap = mobility(); N = 10; fam = exponential_family(N); RC = DEFAULTS["rc"]
ev = lambda Phi, Th: evaluate(Phi, Th, Pdir, Pgap, P0, fam, rc=RC)
row = lambda m, i: dict(C=m["C"][i], slo=m["slo"][i], R=m["R"][i], kappa=m["kappa"][i], Vh=m["Vc"][i] - 1,
                        L=m["L"][i], K=m["K"][i], V0=m["V0"][i])
out = {}
# ---- Table: mechanisms
rng = np.random.default_rng(2026)
pol = {"none": ev(nearest_sets(d, 0), 20), "default": ev(nearest_sets(d, 5), 20),
       "all": ev(nearest_sets(d, 9), 20), "long": ev(nearest_sets(d, 5), 120)}
NR = 200
rnd = [ev([set(rng.choice([j for j in range(N) if j != i], 5, replace=False).tolist()) for i in range(N)], 20)
       for _ in range(NR)]
tab = {}
for name, st in (("dense", DENSE), ("sparse", SPARSE)):
    tab[name] = {p: row(m, st) for p, m in pol.items()}
    tab[name]["random"] = {k: float(np.mean([row(m, st)[k] for m in rnd])) for k in row(rnd[0], st)}
    tab[name]["random_slo_minmax"] = [float(min(m["slo"][st] for m in rnd)), float(max(m["slo"][st] for m in rnd))]
out["table"] = tab
print("Policy                    E[C]    Pr      R     kappa   V^h")
for name in ("dense", "sparse"):
    print(name)
    for p in ("none", "default", "all", "random", "long"):
        r = tab[name][p]; print(f"  {p:10s} {r['C']:10.1f} {r['slo']:7.3f} {r['R']:7.3f} {r['kappa']:7.2f} {r['Vh']:6.2f}")
    print("   random draws: SLO range", np.round(tab[name]["random_slo_minmax"], 3))
print("revisit f_i default: cam3", round(pol["default"]["f"][DENSE], 3), "cam9", round(pol["default"]["f"][SPARSE], 3))

# ---- sweeps
Ths = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]
grid = {(k, T): ev(nearest_sets(d, k), T) for k in range(10) for T in Ths}
sw = {name: {key: [[float(grid[(k, T)][key][st]) for T in Ths] for k in range(10)]
             for key in ("slo", "R", "kappa", "C")} for name, st in (("dense", DENSE), ("sparse", SPARSE))}
out["sweep"] = dict(Ths=Ths, ks=list(range(10)), **sw)
json.dump(out, open("results/policy.json", "w"), default=float)
for name in ("dense", "sparse"):
    S = np.array(sw[name]["slo"]); R = np.array(sw[name]["R"]); Kp = np.array(sw[name]["kappa"]); Cm = np.array(sw[name]["C"])
    i20, i60 = Ths.index(20), Ths.index(60)
    print(f"\n{name}:")
    print("  SLO vs k at Th=20:", np.round(S[:, i20], 3)); print("  SLO vs k at Th=60:", np.round(S[:, i60], 3))
    print("  R   vs k at Th=20:", np.round(R[:, i20], 3)); print("  R   vs k at Th=60:", np.round(R[:, i60], 3))
    print("  kap vs k at Th=20:", np.round(Kp[:, i20], 2)); print("  kap vs k at Th=60:", np.round(Kp[:, i60], 2))
    print("  SLO vs Th at k=5 :", dict(zip(Ths, np.round(S[5], 3)))); print("  R   vs Th at k=5 :", dict(zip(Ths, np.round(R[5], 3))))
    print("  monotone in Th (SLO, E[C]) at every k:", bool(np.all(np.diff(S, axis=1) >= -1e-9) and np.all(np.diff(Cm, axis=1) >= -1e-9)),
          "| monotone in k at every Th:", bool(np.all(np.diff(S, axis=0) >= -1e-9) and np.all(np.diff(Cm, axis=0) >= -1e-9)))
    # matched cost: SLO probability reachable at a given cost per tracked second, per set size
    fine = np.arange(5, 161, 1.0)
    print("  matched cost (best SLO at kappa <= level, by k=1..9):")
    for lev in (6, 8, 10, 12, 14):
        best = []
        for k in range(1, 10):
            kap = np.interp(fine, Ths, Kp[k]); slo = np.interp(fine, Ths, S[k]); ok = kap <= lev
            best.append(slo[ok].max() if ok.any() else np.nan)
        print(f"    kappa<={lev:2d}: ", np.round(best, 3), " best k =", int(np.nanargmax(best)) + 1)
