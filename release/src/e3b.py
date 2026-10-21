"""Single-column (2x2) redraw of the policy-parameter figure (same data as e3_policy.py)."""
from model import *
from common import *
import numpy as np
from matplotlib.lines import Line2D

dep = Deployment(TESTBED_XY, TESTBED_P0)
Ths = np.array([5,10,15,20,30,40,60,80,100,120,160]); ks = np.arange(0,10)

def ev(k, Th):
    Phi = nearest_sets(dep.D, int(k)); m = metrics(dep, Phi, Th)
    p = Fast(dep, Phi, Th).ccdf(DMIN)
    return p[[DENSE,SPARSE]], m["R"][[DENSE,SPARSE]], m["kappa"][[DENSE,SPARSE]]

A = {k: np.array([ev(k, Th)[0] for Th in Ths]) for k in (3,5,7)}
B = {Th: [ev(k, Th) for k in ks] for Th in (20.0, 60.0)}

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.2, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8})
fig, axs = plt.subplots(2, 2, figsize=(3.45, 3.10)); axs = axs.ravel()

for k, ls in ((3,":"),(5,"-"),(7,"--")):
    for c, col in ((0,CD),(1,CS)):
        axs[0].plot(Ths, A[k][:,c], ls=ls, color=col, lw=1.0)
axs[0].set_xlabel("$T_h$ (s)"); axs[0].set_ylabel(r"$\Pr(D\geq120$ s$)$")
axs[0].set_title("(a) SLO vs. $T_h$")

for Th, ls in ((20.0,"-"),(60.0,"--")):
    P = np.array([b[0] for b in B[Th]]); R = np.array([b[1] for b in B[Th]]); K = np.array([b[2] for b in B[Th]])
    for c, col in ((0,CD),(1,CS)):
        axs[1].plot(ks, P[:,c], ls=ls, color=col, marker="o", ms=1.8, lw=1.0)
        axs[2].plot(ks, R[:,c], ls=ls, color=col, marker="o", ms=1.8, lw=1.0)
        axs[3].plot(ks, K[:,c], ls=ls, color=col, marker="o", ms=1.8, lw=1.0)

axs[1].set_title("(b) SLO vs. $|\\Phi_i|$"); axs[1].set_ylabel(r"$\Pr(D\geq120$ s$)$")
axs[2].set_title("(c) tracking ratio $R$");  axs[2].set_ylabel("$R$")
axs[3].set_title("(d) $\\kappa$ (Mbps)");    axs[3].set_ylabel("$\\kappa$")
for ax in axs[1:]:
    ax.set_xlabel("$|\\Phi_i|$"); ax.set_xticks([0,3,6,9])

hs = [Line2D([],[],color=CD,lw=1.4), Line2D([],[],color=CS,lw=1.4),
      Line2D([],[],color="k",ls=":",lw=1), Line2D([],[],color="k",lw=1), Line2D([],[],color="k",ls="--",lw=1),
      Line2D([],[],color="gray",marker="o",lw=1,ms=1.8), Line2D([],[],color="gray",ls="--",marker="o",lw=1,ms=1.8)]
fig.legend(hs, ["dense start","sparse start","(a) $|\\Phi_i|{=}3$","(a) $|\\Phi_i|{=}5$","(a) $|\\Phi_i|{=}7$",
                "(b)–(d) $T_h{=}20$ s","(b)–(d) $T_h{=}60$ s"],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5,0.0), frameon=False,
           columnspacing=0.7, handletextpad=0.3, handlelength=1.4)
fig.tight_layout(w_pad=0.6, h_pad=0.7, rect=[0,0.12,1,1])
save(fig, "e3_policy_1col")
print("saved")
