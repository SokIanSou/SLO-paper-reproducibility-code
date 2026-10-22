from model import *; from common import *; import json
dep = Deployment(TESTBED_XY, TESTBED_P0)
Ths = np.array([5,10,15,20,30,40,60,80,100,120,160]); ks = np.arange(0,10)
def ev(k,Th):
    Phi=nearest_sets(dep.D,k); m=metrics(dep,Phi,Th); p=Fast(dep,Phi,Th).ccdf(DMIN)
    return {q:(m[q][[DENSE,SPARSE]]) for q in ("S","R","kappa","Krate","K")} | {"P":p[[DENSE,SPARSE]], "ED":m["D"][[DENSE,SPARSE]]}
A = {k:[ev(k,Th) for Th in Ths] for k in (3,5,7)}
B = {Th:[ev(k,Th) for k in ks] for Th in (20.0,60.0)}
res = {"R_vs_k_Th20": {"dense":[float(b["R"][0]) for b in B[20.0]], "sparse":[float(b["R"][1]) for b in B[20.0]]},
       "R_vs_k_Th60": {"dense":[float(b["R"][0]) for b in B[60.0]], "sparse":[float(b["R"][1]) for b in B[60.0]]},
       "P_vs_k_Th20": {"dense":[float(b["P"][0]) for b in B[20.0]], "sparse":[float(b["P"][1]) for b in B[20.0]]},
       "kappa_vs_k_Th20": {"dense":[float(b["kappa"][0]) for b in B[20.0]], "sparse":[float(b["kappa"][1]) for b in B[20.0]]},
       "R_vs_Th_k5": {"dense":[float(a["R"][0]) for a in A[5]], "sparse":[float(a["R"][1]) for a in A[5]]},
       "P_vs_Th_k5": {"dense":[float(a["P"][0]) for a in A[5]], "sparse":[float(a["P"][1]) for a in A[5]]},
       "ED_vs_Th_k5": {"dense":[float(a["ED"][0]) for a in A[5]], "sparse":[float(a["ED"][1]) for a in A[5]]}}
# check monotonicity propositions over the grid
mono_Th = all(np.all(np.diff([a["S"][c] for a in A[k]])>=-1e-9) and np.all(np.diff([a["P"][c] for a in A[k]])>=-1e-6) for k in A for c in (0,1))
mono_k = all(np.all(np.diff([b["S"][c] for b in B[T]])>=-1e-9) and np.all(np.diff([b["P"][c] for b in B[T]])>=-1e-6) for T in B for c in (0,1))
res["mono_Th"]=bool(mono_Th); res["mono_k"]=bool(mono_k)
json.dump(res, open("res/e3.json","w"), indent=1)
for k,v in res.items(): print(k, v if isinstance(v,bool) else {a:np.round(b,3).tolist() for a,b in v.items()})
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.55))
for k, ls in ((3,":"),(5,"-"),(7,"--")):
    for c,col in ((0,CD),(1,CS)):
        axs[0].plot(Ths,[a["P"][c] for a in A[k]],ls=ls,color=col,lw=1.1)
axs[0].set_xlabel("grace period $T_h$ (s)"); axs[0].set_ylabel(r"$\Pr(D\geq120$ s$)$"); axs[0].set_title("(a) SLO vs. $T_h$")
for T, ls in ((20.0,"-"),(60.0,"--")):
    for c,col in ((0,CD),(1,CS)):
        axs[1].plot(ks,[b["P"][c] for b in B[T]],ls=ls,color=col,marker="o",ms=2,lw=1.1)
        axs[2].plot(ks,[b["R"][c] for b in B[T]],ls=ls,color=col,marker="o",ms=2,lw=1.1)
        axs[3].plot(ks,[b["kappa"][c] for b in B[T]],ls=ls,color=col,marker="o",ms=2,lw=1.1)
axs[1].set_title("(b) SLO vs. $|\\Phi_i|$"); axs[2].set_title("(c) tracking ratio $R$"); axs[3].set_title("(d) $\\kappa$ (Mbps)")
for ax in axs[1:]: ax.set_xlabel("candidate-set size $|\\Phi_i|$")
[ax.set_xticks([0,3,6,9]) for ax in axs[1:]]
from matplotlib.lines import Line2D
hs=[Line2D([],[],color=CD,lw=1.5),Line2D([],[],color=CS,lw=1.5),Line2D([],[],color="k",ls=":",lw=1),Line2D([],[],color="k",lw=1),Line2D([],[],color="k",ls="--",lw=1),Line2D([],[],color="gray",marker="o",lw=1,ms=2),Line2D([],[],color="gray",ls="--",marker="o",lw=1,ms=2)]
fig.legend(hs,["dense start","sparse start","(a) $|\\Phi_i|=3$","(a) $|\\Phi_i|=5$","(a) $|\\Phi_i|=7$","(b–d) $T_h=20$ s","(b–d) $T_h=60$ s"],loc="lower center",ncol=7,bbox_to_anchor=(0.5,0.0),frameon=False,columnspacing=0.8,handletextpad=0.3)
fig.tight_layout(w_pad=0.6, rect=[0,0.13,1,1]); save(fig,"e3_policy")
