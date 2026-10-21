# side-by-side redraw of the continuity-time CCDF validation figure
from model import *; from common import *
import numpy as np
plt.rcParams.update({"font.size":7.5,"axes.titlesize":7.5,"axes.labelsize":7.5,
                     "legend.fontsize":5.8,"xtick.labelsize":6.8,"ytick.labelsize":6.8})
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
for ax,(gap,soj,title) in zip(axs,(("exp","exp","(a) exponential"),
                                   ("lognorm","gamma","(b) general"))):
    dep = Deployment(TESTBED_XY, TESTBED_P0, gap=gap, soj=soj)
    Phi = nearest_sets(dep.D,5); Th = 20.0
    ts = np.linspace(5,800,18)
    an = np.array([ccdf(dep,Phi,Th,t) for t in ts])
    for c,col,lab in ((DENSE,CD,"dense"),(SPARSE,CS,"sparse")):
        sim = simulate(dep,Phi,Th,c,n=20000,seed=7+c); Dsim = sim[:,6]          # column 6 is the continuity time D
        ax.plot(ts,an[:,c],color=col,lw=1.2,label=f"{lab}: model")
        tt=np.linspace(0,800,240)
        ax.plot(tt,[(Dsim>t).mean() for t in tt],color=col,ls="--",lw=0.9,label=f"{lab}: sim.")
    ax.axvline(DMIN,color="gray",lw=0.8,ls=":"); ax.set_yscale("log"); ax.set_ylim(1e-3,1.05)
    ax.set_title(title); ax.set_xlabel("continuity time $d$ (s)"); ax.set_xticks([0,400,800])
axs[0].set_ylabel(r"$\Pr(D > d)$")
axs[0].legend(loc="lower left",fontsize=5.6,framealpha=0.9,handlelength=1.3,borderpad=0.25,labelspacing=0.25)
fig.tight_layout(w_pad=0.5)
save(fig,"val_ccdf_wide"); print("ok")
