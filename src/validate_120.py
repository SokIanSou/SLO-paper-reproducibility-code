from model import *
import json, time
cfgs = []
for gap, soj in (("exp","exp"), ("lognorm","gamma")):
    for k in (2, 5, 8):
        for Th in (10.0, 20.0, 60.0):
            for ps in (0.8, 0.95):
                cfgs.append((gap, soj, k, Th, ps))
rows = []
t0=time.time()
for gap, soj, k, Th, ps in cfgs:
    dep = Deployment(TESTBED_XY, TESTBED_P0, gap=gap, soj=soj, ps=ps)
    Phi = nearest_sets(dep.D, k); m = metrics(dep, Phi, Th)
    tail = ccdf(dep, Phi, Th, 120.0 + Th) if (gap=="exp" or k==5) else None
    for c in (DENSE, SPARSE):
        sim = simulate(dep, Phi, Th, c, n=20000, seed=k*100+int(Th)+c)
        mean = sim.mean(0); se = sim.std(0)/np.sqrt(len(sim))
        an = [m[q][c] for q in ("S","C","L","K","Vc","V0","D")]
        row = dict(gap=gap, k=k, Th=Th, ps=ps, cam=int(c),
                   an=[float(x) for x in an], sim=[float(x) for x in mean], se=[float(x) for x in se])
        if tail is not None:
            row["slo_an"] = float(tail[c]); row["slo_sim"] = float((sim[:,6] > 120.0 + Th).mean())
        rows.append(row)
json.dump(rows, open("validation120.json","w"))
err = np.array([[abs(a-s)/s for a,s in zip(r["an"],r["sim"]) if s>0] for r in rows])
z = np.array([[abs(a-s)/e for a,s,e in zip(r["an"],r["sim"],r["se"]) if e>0] for r in rows])
print(len(rows), "max rel err", err.max(0).round(4), "mean", err.mean().round(4))
print("frac |z|<2:", (z<2).mean().round(3), "max z", z.max().round(2))
slo = [(r["slo_an"], r["slo_sim"]) for r in rows if "slo_an" in r]
print("slo abs err max", max(abs(a-b) for a,b in slo), len(slo), time.time()-t0)
