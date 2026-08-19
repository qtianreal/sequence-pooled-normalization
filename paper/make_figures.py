"""Figures for the paper. Run from the repository root.

Every series is encoded three ways -- hue, marker and line style -- so identity
never rests on colour alone. Hues are Okabe-Ito, which is validated for
colour-vision deficiency; the skill's own validator was unavailable in this
environment, so an externally validated palette is used rather than a
hand-picked one.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Every figure is authored at exactly the width it is placed at, and saved
# without a tight bounding box, so LaTeX scales it by 1.0 and the point sizes
# below are the point sizes on the page. Saving with bbox_inches="tight" lets
# the width depend on how long the tick labels happen to be, which is how one
# figure in this paper ended up with 6.8 pt labels next to another's 9.1 pt.
TEXT = 6.5   # \textwidth of the tmlr style, in inches

# Okabe-Ito. Length-pooled normalization is the treatment, per-position the
# control; the causal variant is a third identity.
POOLED = "#D55E00"    # vermillion
LOCAL = "#0072B2"     # blue
CAUSAL = "#E69F00"    # orange
INK = "#222222"
MUTED = "#888888"
# Vermillion and blue mean "with the path" and "without it" throughout, so
# a panel that distinguishes processes rather than normalizations needs its own
# pair: otherwise a reader carries the wrong key across panels of one figure.
SYNTH = "#009E73"     # bluish green
GENOM = "#CC79A7"     # reddish purple

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "figure.dpi": 200,
})


def save(fig, name):
    """Fixed figure width, content fitted inside it."""
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def load(n):
    return json.loads((RESULTS / n).read_text())


DIL = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def reach(k, per_block=4):
    """Positions visible to one output. per_block is 4 for a kernel-5
    convolution and 2 for TasNet's depthwise kernel-3."""
    return 1 + per_block * sum(DIL[:k]) + 4


def curve(rows, norm, prefixes, per_block=4):
    xs, ys = [], []
    for k in prefixes:
        h = [r for r in rows if r["norm"] == norm and r["prefix"] == k]
        if not h:
            continue
        xs.append(reach(k, per_block))
        ys.append((np.mean([r["retrained"] for r in h]),
                   np.std([r["retrained"] for r in h], ddof=1) if len(h) > 1 else 0))
    return np.array(xs), np.array([y for y, _ in ys]), np.array([e for _, e in ys])


def band(ax, x, y, e, colour, marker, ls, label=None):
    """A shaded +/- one standard deviation band. Reads more cleanly than capped
    error bars where the series are dense, and does not add a third glyph to a
    panel that already encodes identity by hue, marker and line style."""
    ax.plot(x, y, color=colour, marker=marker, linestyle=ls, markersize=4,
            label=label)
    ax.fill_between(x, np.asarray(y) - e, np.asarray(y) + e, color=colour,
                    alpha=0.15, linewidth=0)


def style(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=4)
    ax.grid(True, which="major", axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------- figure 0
def figure_mechanism():
    """The mechanism, drawn rather than derived: the bounded cone the
    receptive-field calculation describes, and the sequence-spanning path a
    pooled normalization statistic opens beside it. No data; the figure exists
    so a reader holds the two paths before meeting Equation 1."""
    fig, ax = plt.subplots(figsize=(TEXT, 2.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    t = 72          # the output position the paths converge on
    half = 7        # half-width of the drawn receptive field

    # Input and output sequences, drawn as strips of positions.
    for y0, label in ((1.0, r"input $x_1 \ldots x_L$"),
                      (8.0, r"output $z_1 \ldots z_L$")):
        ax.add_patch(plt.Rectangle((5, y0), 90, 0.8, facecolor="#F2F2F2",
                                   edgecolor=MUTED, linewidth=0.6))
        for x in np.arange(7, 95, 2.5):
            ax.plot([x, x], [y0 + 0.12, y0 + 0.68], color="#CCCCCC",
                    linewidth=0.5)
        ax.text(3.6, y0 + 0.4, label, ha="right", va="center", fontsize=7)
    ax.add_patch(plt.Rectangle((t - 0.9, 8.0), 1.8, 0.8, facecolor=INK,
                               edgecolor="none"))
    ax.text(t, 9.35, r"$z_t$", ha="center", va="bottom", fontsize=7)

    # The receptive field: a bounded cone from R input positions to z_t.
    ax.add_patch(plt.Polygon([(t - half, 1.8), (t + half, 1.8),
                              (t + 0.9, 8.0), (t - 0.9, 8.0)],
                             closed=True, facecolor=LOCAL, alpha=0.20,
                             edgecolor=LOCAL, linewidth=0.8))
    ax.plot([t - half, t + half], [1.8, 1.8], color=LOCAL, linewidth=2.2,
            solid_capstyle="butt")
    ax.text(t + half + 1.5, 3.4, "receptive field:\nthe $R$ positions\n"
            "the convolutions reach", ha="left", va="center", fontsize=7,
            color=LOCAL)

    # The path: every input position feeds the pooled statistic, which is
    # delivered back to every output position.
    from matplotlib.patches import Ellipse
    node = (18, 4.9)
    for x in np.arange(7, 95, 4.4):
        ax.plot([x, node[0]], [1.8, node[1] - 1.1], color=POOLED,
                linewidth=0.5, alpha=0.30, zorder=1)
    # Axis units differ 3.1-fold, so a circle needs the ellipse drawn to match.
    ax.add_patch(Ellipse(node, width=9.5, height=3.0, facecolor="white",
                         edgecolor=POOLED, linewidth=1.1, zorder=3))
    ax.text(node[0], node[1], r"$\mu_S,\,\sigma_S$", ha="center", va="center",
            fontsize=7.5, color=POOLED, zorder=4)
    for x_out, alpha, lw in ((t - 2.2, 1.0, 1.1), (36, 0.35, 0.7),
                             (50, 0.35, 0.7), (88, 0.35, 0.7)):
        ax.annotate("", xy=(x_out, 7.95), xytext=(node[0] + 2.5, node[1] + 1.3),
                    arrowprops=dict(arrowstyle="->", color=POOLED,
                                    linewidth=lw, alpha=alpha,
                                    connectionstyle="arc3,rad=-0.14"), zorder=2)
    ax.text(26, 5.0, "statistics pooled along the sequence:\n"
            "a summary of the whole input,\ndelivered to every position",
            ha="left", va="center", fontsize=7, color=POOLED, zorder=5)
    ax.text(50, 0.25, r"each term $\partial z_t/\partial x_s$ is $O(1/|S|)$; "
            r"together they equal the direct path (Eq. 1)",
            ha="center", va="center", fontsize=6.5, color=INK)

    fig.tight_layout(pad=0.4)
    save(fig, "fig0_mechanism")


# --------------------------------------------------------------- figure 1
def figure_reach():
    fig, axes = plt.subplots(1, 3, figsize=(TEXT, 2.35))
    pref = [1, 3, 5, 7, 9]

    # All five divergence levels: the two families never meet at short reach,
    # and the gap between them is what the ratio measures. Plotting one level
    # would hide that the ratio is invariant to difficulty.
    tr = load("tradeoff.json")
    Ts = sorted({r["split_time"] for r in tr},
                key=lambda T: -np.mean([r["fst"] for r in tr if r["split_time"] == T]))
    for T, alpha in zip(Ts, np.linspace(1.0, 0.34, len(Ts))):
        h = [r for r in tr if r["split_time"] == T]
        for norm, c, m, ls, lab in [("group", POOLED, "o", "-", "pooled along the sequence"),
                                    ("positionwise", LOCAL, "s", "--", "per position")]:
            x, y, _ = curve(h, norm, pref)
            axes[0].plot(x, y, color=c, marker=m, linestyle=ls, markersize=2.6,
                         linewidth=1.0, alpha=alpha,
                         label=lab if T == Ts[0] else None)
    style(axes[0], "receptive field (positions)", "accuracy",
          "(a) simulated genomes")

    rl = load("real_norm.json")
    for norm, c, m, ls in [("group", POOLED, "o", "-"),
                           ("positionwise", LOCAL, "s", "--")]:
        x, y, e = curve(rl, norm, pref)
        band(axes[1], x, y, e, c, m, ls)
    style(axes[1], "receptive field (positions)", "", "(b) real 1000 Genomes")

    tn = load("synth_tasnet.json")
    for norm, c, m, ls, lab in [("gln", POOLED, "o", "-", "pooled along the sequence"),
                                ("cln", CAUSAL, "^", "-.", "pooled over the past only"),
                                ("gln_pos", LOCAL, "s", "--", "per position")]:
        x, y, e = curve(tn, norm, pref, per_block=2)
        band(axes[2], x, y, e, c, m, ls, label=lab)
    ceil = np.mean([r["bayes_full_context"] for r in tn])
    axes[2].axhline(ceil, color=MUTED, linestyle=":", linewidth=1,
                    label="whole sequence (optimum)")
    style(axes[2], "receptive field (positions)", "", "(c) Conv-TasNet")

    handles = [h for ax in (axes[0], axes[2]) for h in ax.get_legend_handles_labels()[0]]
    labels = [l for ax in (axes[0], axes[2]) for l in ax.get_legend_handles_labels()[1]]
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); hh.append(h); ll.append(l)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(hh, ll, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4)
    save(fig, "fig1_reach")


# --------------------------------------------------------------- figure 2
def figure_dose():
    """(a) the two terms the ratio is built from, (b) the ratio itself.

    (a) is what makes the exponent in (b) legible: both the numerator and the
    denominator of the ratio depend on the switch rate, in opposite directions
    at first, so the ratio's $1/n$ is not the $1/\\sqrt{n}$ of the path's
    value alone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.5))

    def dose(rows, key):
        out = []
        for p in sorted({r[key] for r in rows}):
            h = [r for r in rows if r[key] == p]
            w, e = {}, {}
            for nm in ("group", "positionwise"):
                d = [next(r["retrained"] for r in h if r["norm"] == nm
                          and r["seed"] == s and r["prefix"] == 9)
                     - next(r["retrained"] for r in h if r["norm"] == nm
                            and r["seed"] == s and r["prefix"] == 1)
                     for s in sorted({r["seed"] for r in h})]
                w[nm], e[nm] = np.mean(d), np.std(d, ddof=1)
            out.append((np.mean([r["switches_per_window"] for r in h]),
                        w["positionwise"] / w["group"],
                        w["group"], e["group"],
                        w["positionwise"], e["positionwise"]))
        return np.array(out)

    syn = dose(load("synth_dilated_cnn.json"), "switch_prob")
    gen = dose(load("tracts_generations.json"), "generations")

    ax = axes[0]
    for d, ls, mk in ((syn, "-", "o"), (gen, "--", "s")):
        band(ax, d[:, 0], d[:, 4], d[:, 5], LOCAL, mk, ls)
        band(ax, d[:, 0], d[:, 2], d[:, 3], POOLED, mk, ls)
    ax.set_xscale("log")
    ax.set_xlabel("label switches per sequence")
    ax.set_ylabel("reach worth")
    ax.set_title("(a) the two terms", loc="left", pad=4)
    ax.text(0.05, 0.74, "per position", color=LOCAL, fontsize=7,
            transform=ax.transAxes, weight="bold")
    ax.text(0.05, 0.36, "pooled along the sequence", color=POOLED, fontsize=7,
            transform=ax.transAxes, weight="bold")
    ax.grid(True, which="major", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    c = np.mean([(r - 1) * n for n, r in syn[:, :2]])
    xs = np.logspace(np.log10(0.07), np.log10(25), 100)
    ax.plot(xs, c / xs, color=MUTED, linestyle=":", linewidth=1,
            label=f"$c/n$, $c={c:.1f}$", zorder=1)
    ax.plot(syn[:, 0], syn[:, 1] - 1, color=SYNTH, marker="o", linestyle="-",
            markersize=4, label="synthetic", zorder=3)
    ax.plot(gen[:, 0], gen[:, 1] - 1, color=GENOM, marker="s", linestyle="--",
            markersize=4, label="genomic", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("label switches per sequence")
    ax.set_ylabel("ratio $-$ 1")
    ax.set_title("(b) their ratio", loc="left", pad=4)
    ax.grid(True, which="major", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, "fig2_dose")


# --------------------------------------------------------------- figure 3
def figure_ablation():
    """Two views of the same disagreement.

    (a) is the causal test: the gap between the two measurements is a function
    of which axes the normalization statistics span, and of nothing else, since
    every other row of the panel is the same task, architecture, seeds and data.
    (b) shows that ablation gets the shape wrong as well as the level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.8),
                             gridspec_kw=dict(width_ratios=[1.95, 1.0]))
    rl, tr, nm = load("real_norm.json"), load("tradeoff.json"), load("tracts_norms.json")

    ax = axes[0]
    ORDER = [("none", "none"), ("positionwise", "channels"),
             ("batch", "batch $+$ length,\nrunning at eval"),
             ("instance", "length"), ("group", "channels $+$ length")]
    rows = []
    for key, lab in ORDER:
        h = [r for r in nm if r["norm"] == key and r["prefix"] == 1]
        rows.append((lab, np.mean([r["full"] - r["ablated"] for r in h]),
                     np.mean([r["full"] - r["retrained"] for r in h])))
    for key, lab in (("positionwise", "channels"), ("group", "channels $+$ length")):
        h = [r for r in rl if r["norm"] == key and r["prefix"] == 1]
        rows.append((lab + ", real 1000G",
                     np.mean([r["full"] - r["ablated"] for r in h]),
                     np.mean([r["full"] - r["retrained"] for r in h])))

    y = np.arange(len(rows))[::-1]
    for i, (lab, abl, ret) in zip(y, rows):
        ax.plot([ret, abl], [i, i], color=MUTED, linewidth=1, zorder=1)
        ax.scatter([abl], [i], color=POOLED, marker="o", s=24, zorder=3,
                   label="ablation" if i == y[0] else None)
        ax.scatter([ret], [i], color=LOCAL, marker="s", s=20, zorder=3,
                   label="retraining" if i == y[0] else None)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 0.52)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axhline(1.5, color="#DDDDDD", linewidth=0.8)
    ax.set_xlabel("accuracy lost by removing the long-range blocks")
    ax.set_title("(a) by the axes the statistics are pooled over",
                 loc="left", pad=4)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    Ts = sorted({r["split_time"] for r in tr},
                key=lambda T: -np.mean([r["fst"] for r in tr if r["split_time"] == T]))
    fst = [np.mean([r["fst"] for r in tr if r["split_time"] == T]) for T in Ts]
    for key, colour, marker, ls, lab in [
            ("retention_ablated", POOLED, "o", "-", "ablation"),
            ("retention_retrained", LOCAL, "s", "--", "retraining")]:
        m, e = [], []
        for T in Ts:
            v = [r[key] for r in tr if r["split_time"] == T
                 and r["norm"] == "group" and r["prefix"] == 3
                 and not np.isnan(r[key])]
            m.append(np.mean(v))
            e.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        band(ax, fst, m, np.array(e), colour, marker, ls, label=lab)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$F_{ST}$ (harder $\rightarrow$)")
    ax.set_ylabel("above-chance accuracy retained")
    ax.set_title("(b) across difficulty", loc="left", pad=4)
    ax.grid(True, axis="y", color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6)
    save(fig, "fig3_ablation")


if __name__ == "__main__":
    figure_mechanism()
    figure_reach()
    figure_dose()
    figure_ablation()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
