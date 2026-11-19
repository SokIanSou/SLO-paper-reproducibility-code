"""Section 5: model versus simulation (Table 'validation', Figure 'val-ccdf')."""
import json, itertools, numpy as np
from multiprocessing import Pool
from slo_model import *
from simulator import simulate

NSIM = 100000
KEYS = ("S", "C", "L", "K", "Vc", "V0")
d, Pdir, Pgap = mobility()
FAMS = {"exponential": exponential_family(10), "general": general_family(d)}


def case(args):
    c, (fname, k, Th, start) = args
    fam = FAMS[fname]; Phi = nearest_sets(d, k)
    m = evaluate(Phi, Th, Pdir, Pgap, P0, fam)
    s = simulate(Phi, Th, Pdir, Pgap, P0, fam, start, NSIM, seed=1000 + c)
    out = dict(family=fname, k=k, Th=Th, start=start + 1)
    for key in KEYS:
        out[key] = dict(model=float(m[key][start]), sim=float(s[key].mean()),
                        se=float(s[key].std(ddof=1) / np.sqrt(NSIM)))
    p = float((s["C"] >= 120).mean())
    out["slo"] = dict(model=float(m["slo"][start]), sim=p, se=float(np.sqrt(p * (1 - p) / NSIM)))
    if k == 5 and Th == 20:                                   # full distribution for the figure
        ts = np.arange(5, 805, 5.0)
        out["ccdf"] = dict(t=ts.tolist(),
                           model=[float(slo_ccdf(m["a"], m["q"], fam.lstH, t)[start]) for t in ts],
                           sim=[float((s["C"] > t).mean()) for t in ts])
    return out


if __name__ == "__main__":
    cases = list(enumerate(itertools.product(FAMS, (2, 5, 8), (10, 20, 60, 120), (DENSE, SPARSE))))
    with Pool(2) as pool:
        res = pool.map(case, cases)
    json.dump(res, open("results/validation.json", "w"))
    print(f"{len(res)} cases, {NSIM} sessions each")
    dev = {k: np.array([abs(r[k]["model"] / r[k]["sim"] - 1) * 100 for r in res]) for k in KEYS}
    z = np.array([[abs(r[k]["model"] - r[k]["sim"]) / r[k]["se"] for k in KEYS] for r in res])
    print("          " + "".join(f"{k:>7s}" for k in KEYS))
    print("max (%)   " + "".join(f"{dev[k].max():7.2f}" for k in KEYS))
    print("mean (%)  " + "".join(f"{dev[k].mean():7.2f}" for k in KEYS))
    for f in FAMS:
        idx = [i for i, r in enumerate(res) if r["family"] == f]
        print(f"  {f}: max deviation {max(dev[k][idx].max() for k in KEYS):.2f}%")
    for T in (10, 20, 60, 120):
        idx = [i for i, r in enumerate(res) if r["Th"] == T]
        print(f"  Th={T}: max deviation {max(dev[k][idx].max() for k in KEYS):.2f}%")
    print(f"within 2 standard errors: {100 * (z <= 2).mean():.1f}% of {z.size} comparisons; "
          f"signed mean z = {np.mean([[(r[k]['model'] - r[k]['sim']) / r[k]['se'] for k in KEYS] for r in res]):+.2f}")
    sd = np.array([abs(r["slo"]["model"] - r["slo"]["sim"]) for r in res])
    print(f"SLO probability: max |model - sim| = {sd.max():.4f}, mean {sd.mean():.4f}")
    cc = [r for r in res if "ccdf" in r]
    for r in cc:
        mo, si = np.array(r["ccdf"]["model"]), np.array(r["ccdf"]["sim"])
        ok = si > 1e-3
        print(f"  ccdf {r['family']} start {r['start']}: max abs diff {np.abs(mo - si).max():.4f}; "
              f"max rel diff where sim > 1e-3: {np.abs(mo[ok] / si[ok] - 1).max() * 100:.1f}%")
