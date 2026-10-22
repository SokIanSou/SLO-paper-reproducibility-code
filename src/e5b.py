# single-column (a)+(b) redraw of the configuration figure, with one shared legend
from common import *; import json, numpy as np
from model import TESTBED_P0
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

N = 10; e5 = json.load(open("res/e5.json")); T = sorted(e5, key=float)
METHODS = (("uniform", "gray", "s", "Uniform"),
           ("proportional", "#d9822b", "^", "Proportional"),
           ("greedy", CD, "o", "Model-driven"))

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.2, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.15))

# (a) minimum load vs SLO target
for m, c, mk, lab in METHODS:
    axs[0].plot([float(t) for t in T], [e5[t][m]["load"] for t in T],
                marker=mk, ms=2.5, color=c, lw=1.0)
axs[0].set_xlabel("SLO target $1-\\epsilon$"); axs[0].set_ylabel("load (Mbps)")
axs[0].set_ylim(0, 215); axs[0].set_title("(a) minimum load")

# (b) per-camera allocation at 1-eps = 0.5, all three rules
x = np.arange(N)
for k, (m, c, mk, lab) in enumerate(METHODS):
    axs[1].bar(x + (k - 1) * 0.27, e5["0.5"][m]["k"], 0.27, color=c)
ax2 = axs[1].twinx(); ax2.plot(x, TESTBED_P0, "k.--", lw=0.7, ms=3)
ax2.set_ylabel("$P_{i,0}$ (dashed)"); ax2.set_ylim(0, 1.05); ax2.grid(False)
axs[1].set_xticks([0, 2, 4, 6, 8]); axs[1].set_xticklabels(["1", "3", "5", "7", "9"])
axs[1].set_xlabel("camera"); axs[1].set_ylabel("$|\\Phi_i|$")
axs[1].set_title("(b) allocation, $1-\\epsilon{=}0.5$")
axs[1].set_ylim(0, 10.5); axs[1].set_yticks([0, 3, 6, 9])

handles = [(Line2D([], [], color=c, marker=mk, ms=2.5, lw=1.0), lab) for m, c, mk, lab in METHODS]
fig.legend([h for h, _ in handles], [l for _, l in handles],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False,
           columnspacing=1.2, handletextpad=0.4, handlelength=1.6)
fig.tight_layout(w_pad=1.1, rect=[0, 0.10, 1, 1])
save(fig, "e5_config_1col"); print("ok")
