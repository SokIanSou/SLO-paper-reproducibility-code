import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3,
    "savefig.dpi": 300})
CD, CS = "#1f5fa8", "#c0392b"
DMIN = 120.0
def save(fig, name):
    fig.savefig(f"figs/{name}.pdf"); fig.savefig(f"figs/{name}.png"); plt.close(fig)
