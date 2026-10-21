"""Plan view of the 10-camera testbed used in Sec. V-A."""
from common import *
from model import TESTBED_XY, TESTBED_P0, DENSE, SPARSE
import numpy as np

plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.2})
fig, ax = plt.subplots(figsize=(3.45, 1.80))
xy = np.asarray(TESTBED_XY); p0 = np.asarray(TESTBED_P0)

sc = ax.scatter(xy[:, 0], xy[:, 1], c=p0, cmap="YlOrRd", vmin=0.15, vmax=0.85,
                s=78, edgecolors="0.25", linewidths=0.7, zorder=3)
for i, (x, y) in enumerate(xy):
    ax.annotate(str(i + 1), (x, y), ha="center", va="center", fontsize=5.8,
                color="0.1", zorder=4)
for c, mark in ((DENSE, "dense start"), (SPARSE, "sparse start")):
    ax.scatter(*xy[c], s=190, facecolors="none", edgecolors=CD if c == DENSE else CS,
               linewidths=1.1, zorder=2, label=mark)

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_xlim(-2, 106); ax.set_ylim(-2, 74); ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", frameon=True, framealpha=0.92, handletextpad=0.1,
          borderpad=0.22, labelspacing=0.15, markerscale=0.42, scatterpoints=1)
cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
cb.set_label("$P_{i,0}$", fontsize=7.5); cb.ax.tick_params(labelsize=6.2)
fig.tight_layout()
save(fig, "testbed_plan")
print("ok", xy.shape)
