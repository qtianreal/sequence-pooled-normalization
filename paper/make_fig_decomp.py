"""The information landscape: what each route can supply, exactly computed."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

op = json.loads(Path("results/optima.json").read_text())
syn = json.loads(Path("results/synth_dilated_cnn.json").read_text())
POOLED, LOCAL, ORACLE, MUTED, INK = "#D55E00", "#0072B2", "#009E73", "#888888", "#222222"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.dpi": 200})

n = np.array([r["switches_per_window"] for r in op])
# The canvas is the full text width but the axes are not: the plot keeps the size
# it had at 0.62 of the width, and the extra width is spent on the legend, which
# needs three columns to fit on two lines.
fig = plt.figure(figsize=(6.5, 3.35))
ax = fig.add_axes([0.245, 0.30, 0.52, 0.66])
ax.plot(n, [r["full"] for r in op], color=MUTED, marker="d", ms=4, ls="-",
        label="whole sequence (optimum)")
ax.plot(n, [r["global_only"] for r in op], color=ORACLE, marker="^", ms=4, ls="-.",
        label="class proportion / summary (optimum)")
ax.plot(n, 0.5 + 1 / np.sqrt(2 * np.pi * n), color=ORACLE, ls=":", lw=1,
        label=r"$1/2+1/\sqrt{2\pi n}$ (closed form)")
ax.plot(n, [r["local"]["9"] for r in op], color=LOCAL, marker="s", ms=4, ls="--",
        label="9 positions (bound)")

def net(row, norm, k):
    """Match on the switch probability the run was generated with, not on the
    realised switch count: the latter is per-seed while the optima table stores
    the seed mean, so an absolute tolerance silently drops the dense end."""
    h = [x["retrained"] for x in syn if x["switch_prob"] == row["switch_prob"]
         and x["norm"] == norm and x["prefix"] == k]
    return np.mean(h) if h else np.nan

ax.scatter(n, [net(r, "group", 1) for r in op], color=POOLED, marker="o", s=22,
           zorder=5, label="trained network (reach 9, sequence-pooled)")
ax.set_xscale("log"); ax.set_ylim(0.45, 1.02)
ax.set_xlabel("label switches per sequence"); ax.set_ylabel("accuracy")
ax.grid(True, color="#E8E8E8", lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
# Legend below the axes: the flat 9-position curve runs through every in-axes
# corner that would otherwise be free. Anchored to the figure rather than the
# axes, so it may use the full width instead of being clipped at the axes
# edge -- the y-label margin is otherwise wasted on it.
# matplotlib fills a multi-column legend column by column, so the order below is
# chosen to make each column mean something: the two reach bounds, then the summary
# exactly and in closed form (the pair that share a colour), then the network.
_h, _l = ax.get_legend_handles_labels()
_order = [0, 3, 1, 2, 4]
fig.legend([_h[i] for i in _order], [_l[i] for i in _order],
           frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 0.19), ncol=3, fontsize=7,
           columnspacing=0.9, handlelength=2.0)
fig.savefig("paper/figures/fig4_decomp.pdf")
fig.savefig("paper/figures/fig4_decomp.png", dpi=200)
print("wrote fig4_decomp")
