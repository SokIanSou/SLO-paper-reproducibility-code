from model import *; from common import *; import json
rows = json.load(open("validation.json"))
names = ["S","C","L","K","V^c","V^0"]
err = np.array([[abs(a-s)/s for a,s in zip(r["an"],r["sim"])] for r in rows])
z = np.array([[abs(a-s)/e for a,s,e in zip(r["an"],r["sim"],r["se"])] for r in rows])
res = {"n_cfg": len(rows), "max_rel": dict(zip(names, err.max(0).round(4).tolist())),
       "mean_rel": dict(zip(names, err.mean(0).round(4).tolist())),
       "frac_within_2se": float((z<2).mean())}
for g in ("exp","lognorm"):
    sel=[i for i,r in enumerate(rows) if r["gap"]==g]
    res[f"max_rel_{g}"]=float(err[sel].max())
slo=[abs(r["slo_an"]-r["slo_sim"]) for r in rows if "slo_an" in r]
res["slo_max_abs"]=float(max(slo)); res["slo_n"]=len(slo)
json.dump(res, open("res/e0.json","w"), indent=1); print(res)
# CCDF figure of continuity time D
fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True)
for ax, (gap, soj, title) in zip(axs, (("exp","exp","Exponential sojourns and gaps"),
                                      ("lognorm","gamma","Gamma sojourns, lognormal gaps"))):
    dep = Deployment(TESTBED_XY, TESTBED_P0, gap=gap, soj=soj)
    Phi = nearest_sets(dep.D, 5); Th = 20.0
    ts = np.linspace(5, 400, 16)
    an = np.array([ccdf(dep, Phi, Th, t) for t in ts])
    for c, col, lab in ((DENSE, CD, "dense start"), (SPARSE, CS, "sparse start")):
        sim = simulate(dep, Phi, Th, c, n=20000, seed=7 + c)
        Dsim = sim[:,6]          # column 6 is the continuity time D
        ax.plot(ts, an[:, c], color=col, lw=1.4, label=f"{lab}: model")
        tt = np.linspace(0, 400, 200)
        ax.plot(tt, [(Dsim > t).mean() for t in tt], color=col, ls="--", lw=1.0, label=f"{lab}: simulation")
    ax.axvline(DMIN, color="gray", lw=0.8, ls=":"); ax.set_yscale("log"); ax.set_ylim(1e-3, 1.05)
    ax.set_title(title); ax.set_xlabel("")
axs[1].set_xlabel("continuity time $d$ (s)")
for ax in axs: ax.set_ylabel(r"$\Pr(D > d)$")
axs[0].legend(loc="lower left", fontsize=6.2)
fig.tight_layout(h_pad=0.6)
save(fig, "val_ccdf")
