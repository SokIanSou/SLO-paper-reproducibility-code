from model import *; from common import *; import json
dep = Deployment(TESTBED_XY, TESTBED_P0); rng = np.random.default_rng(3)
def evalc(Phi, Th):
    m = metrics(dep, Phi, Th); p = Fast(dep, Phi, Th).ccdf(DMIN)
    return {c: dict(ED=float(m["D"][c]), P=float(p[c]), R=float(m["R"][c]),
                    Krate=float(m["Krate"][c]), kappa=float(m["kappa"][c]), K=float(m["K"][c]),
                    H=float(m["H"][c])) for c in (DENSE, SPARSE)}
conds = {"Proposed ($|\\Phi_i|{=}5$, $T_h{=}20$ s)": evalc(nearest_sets(dep.D,5), 20.0),
         "No candidate set ($|\\Phi_i|{=}0$)": evalc(nearest_sets(dep.D,0), 20.0),
         "All cameras ($|\\Phi_i|{=}N{-}1$)": evalc(nearest_sets(dep.D,9), 20.0),
         "Long grace period ($T_h{=}120$ s)": evalc(nearest_sets(dep.D,5), 120.0)}
draws=[]
for d in range(30):
    Phi=[list(rng.choice([j for j in range(10) if j!=i], 5, replace=False)) for i in range(10)]
    draws.append(evalc(Phi, 20.0))
conds["Random candidates ($|\\Phi_i|{=}5$)"] = {c:{k:float(np.mean([d[c][k] for d in draws])) for k in draws[0][c]} for c in (DENSE,SPARSE)}
json.dump({k:{str(c):v for c,v in d.items()} for k,d in conds.items()}, open("res/e1.json","w"), indent=1)
for k,d in conds.items(): print(k, {c:{q:round(v,3) for q,v in d[c].items()} for c in d})
# trade-off frontier: Pr(D>=Dmin) vs cost per tracked second, sweeping Th, per k; nearest vs random
Ths = [5,10,20,30,40,60,80,100,120,160]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, c, title in ((axs[0], DENSE, "Dense start (camera 3)"), (axs[1], SPARSE, "Sparse start (camera 9)")):
    for idx, k in enumerate((1,3,5,7,9)):
        pts=[evalc(nearest_sets(dep.D,k),Th)[c] for Th in Ths]
        ax.plot([p["kappa"] for p in pts],[p["P"] for p in pts],"-o",ms=2.5,lw=1.1,color=cmap(idx/4.5),label=f"nearest, $|\\Phi_i|={k}$")
    rpts=[]
    for Th in Ths:
        acc=[]
        for d in range(10):
            r2=np.random.default_rng(100+d)
            Phi=[list(r2.choice([j for j in range(10) if j!=i],5,replace=False)) for i in range(10)]
            acc.append(evalc(Phi,Th)[c])
        rpts.append({q:np.mean([a[q] for a in acc]) for q in acc[0]})
    ax.plot([p["kappa"] for p in rpts],[p["P"] for p in rpts],"--s",ms=2.5,lw=1.0,color="gray",label="random, $|\\Phi_i|=5$")
    p0=evalc(nearest_sets(dep.D,0),20.0)[c]; ax.plot(p0["kappa"],p0["P"],"kx",ms=6,label="no candidate set")
    ax.set_title(title); ax.set_xlabel("streaming cost per tracked second $\\kappa$ (Mbps)")
axs[0].set_ylabel(r"SLO probability $\Pr(D\geq 120$ s$)$")
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(0.5,-0.2),frameon=False,columnspacing=0.9,handletextpad=0.3)
save(fig,"e1_tradeoff")
