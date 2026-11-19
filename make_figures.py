"""Figures of Sections 5 and 6 from results/*.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from slo_model import P0

BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#333333", "#dddddd"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6,
                     "axes.edgecolor": INK, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
                     "axes.axisbelow": True, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False})
W = 248.4 / 72

# ------------------------------------------------------------------ validation: distribution of C
val = json.load(open("results/validation.json"))
fig, axs = plt.subplots(1, 2, figsize=(W, 140 / 72), sharey=True, layout="constrained")
for ax, famname, title in zip(axs, ("exponential", "general"), ("(a) exponential", "(b) general")):
    for start, col, lab in ((3, BLUE, "dense"), (9, ORANGE, "sparse")):
        r = next(x for x in val if x["family"] == famname and x["start"] == start and "ccdf" in x)
        t, mo, si = (np.array(r["ccdf"][k]) for k in ("t", "model", "sim"))
        ax.semilogy(t, mo, color=col, lw=1.4, label=f"{lab}: model")
        ok = si > 0
        ax.semilogy(t[ok], si[ok], color=INK, lw=0.8, ls=(0, (3, 2)), label="simulation" if start == 9 else None)
    ax.axvline(120, color=INK, lw=0.7, ls=":")
    ax.set_ylim(1e-3, 1); ax.set_xlim(0, 800); ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_title(title); ax.set_xlabel("track continuity $c$ (s)")
axs[0].set_ylabel(r"$\Pr(C > c)$")
axs[0].text(128, 1.4e-3, r"$C_{\min}$", fontsize=6, color=INK)
h = [Line2D([], [], color=BLUE, lw=1.6, label="dense start: model"), Line2D([], [], color=ORANGE, lw=1.6, label="sparse start: model"),
     Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label="simulation")]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, handlelength=2.0, columnspacing=1.4)
fig.savefig("figs/val_ccdf_wide.pdf"); plt.close(fig)

# ------------------------------------------------------------------ policy sweeps
pol = json.load(open("results/policy.json"))["sweep"]; Ths, ks = pol["Ths"], pol["ks"]
fig, axs = plt.subplots(2, 2, figsize=(W, 223.2 / 72), layout="constrained")
ax = axs[0, 0]
for name, col in (("dense", BLUE), ("sparse", ORANGE)):
    for k, ls in ((3, ":"), (5, "-"), (7, "--")):
        ax.plot(Ths, pol[name]["slo"][k], color=col, ls=ls, lw=1.3)
ax.set_title(r"(a) SLO vs. $T_h$"); ax.set_xlabel(r"$T_h$ (s)"); ax.set_ylabel(r"$\Pr(C \geq 120\,\mathrm{s})$")
ax.set_xlim(0, 160); ax.set_ylim(0, 1); ax.set_xticks([0, 40, 80, 120, 160])
for (r, c), key, title, yl in (((0, 1), "slo", r"(b) SLO vs. $|\Phi_i|$", r"$\Pr(C \geq 120\,\mathrm{s})$"),
                               ((1, 0), "R", r"(c) tracking ratio $R$", r"$R$"),
                               ((1, 1), "kappa", r"(d) $\kappa$ (Mbps)", r"$\kappa$ (Mbps)")):
    ax = axs[r, c]
    for name, col in (("dense", BLUE), ("sparse", ORANGE)):
        for T, ls, mk in ((20, "-", "o"), (60, "--", "s")):
            ax.plot(ks, [pol[name][key][k][Ths.index(T)] for k in ks], color=col, ls=ls, lw=1.2, marker=mk, ms=2.6)
    ax.set_title(title); ax.set_xlabel(r"$|\Phi_i|$"); ax.set_ylabel(yl); ax.set_xticks([0, 3, 6, 9])
axs[0, 1].set_ylim(0, 1); axs[1, 0].set_ylim(0.3, 0.9); axs[1, 1].set_ylim(3, 13)
h = [Line2D([], [], color=BLUE, lw=1.6, label="dense start"), Line2D([], [], color=ORANGE, lw=1.6, label="sparse start"),
     Line2D([], [], color=INK, ls=":", lw=1.2, label=r"(a) $|\Phi_i|=3$"), Line2D([], [], color=INK, ls="-", lw=1.2, label=r"(a) $|\Phi_i|=5$"),
     Line2D([], [], color=INK, ls="--", lw=1.2, label=r"(a) $|\Phi_i|=7$"),
     Line2D([], [], color=INK, ls="-", marker="o", ms=2.6, lw=1.2, label=r"(b)–(d) $T_h=20$ s"),
     Line2D([], [], color=INK, ls="--", marker="s", ms=2.6, lw=1.2, label=r"(b)–(d) $T_h=60$ s")]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, handlelength=2.2, columnspacing=1.2)
fig.savefig("figs/e3_policy_1col.pdf"); plt.close(fig)

# ------------------------------------------------------------------ configuration
cfg = json.load(open("results/config.json"))["targets"]
fig, axs = plt.subplots(1, 2, figsize=(W, 154.8 / 72), gridspec_kw=dict(width_ratios=[1, 1.35]), layout="constrained")
ax = axs[0]; tg = [c["target"] for c in cfg]
for key, col, mk, ls, lab in (("uniform", BLUE, "s", "--", "Uniform"), ("prop", ORANGE, "^", ":", "Proportional"),
                              ("model", AQUA, "o", "-", "Model-driven")):
    ax.plot(tg, [c[key][0] for c in cfg], color=col, marker=mk, ms=3.2, ls=ls, lw=1.3, label=lab)
ax.set_title("(a) minimum load"); ax.set_xlabel(r"SLO target $1-\epsilon$"); ax.set_ylabel("load (Mbps)")
ax.set_ylim(0, 340); ax.set_xticks([0.3, 0.5, 0.7])
ax = axs[1]; c5 = next(c for c in cfg if abs(c["target"] - 0.5) < 1e-9); x = np.arange(10); bw = 0.27
for off, key, col, hatch in ((-bw, "uniform", BLUE, None), (0, "prop", ORANGE, "////"), (bw, "model", AQUA, None)):
    ax.bar(x + off, c5[key][2], bw - 0.03, color=col, hatch=hatch, edgecolor="white", linewidth=0.3)
ax.set_title(r"(b) allocation at $1-\epsilon = 0.5$"); ax.set_ylabel(r"$|\Phi_i|$"); ax.set_yticks([0, 3, 6, 9]); ax.set_ylim(0, 9.6)
ax.set_xticks(x); ax.set_xticklabels([f"{i+1}\n{str(P0[i])[1:]}" for i in x], fontsize=5.2); ax.grid(axis="x", visible=False)
ax.set_xlabel(r"camera, with its $P_{i,0}$ below")
h = [Line2D([], [], color=BLUE, marker="s", ms=3.2, ls="--", lw=1.3, label="Uniform"),
     Line2D([], [], color=ORANGE, marker="^", ms=3.2, ls=":", lw=1.3, label="Proportional"),
     Line2D([], [], color=AQUA, marker="o", ms=3.2, ls="-", lw=1.3, label="Model-driven")]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, handlelength=2.4)
fig.savefig("figs/e5_config_1col.pdf"); plt.close(fig)
print("figures written")
