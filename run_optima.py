"""What is available to a model at each reach, computed exactly.

Every accuracy in this project is reported against other accuracies. On the
synthetic axis the generating process is known, so the optima are computable
and the numbers can be placed on an absolute scale instead:

    local(R)   the best any decoder can do seeing R positions centred on the
               target, with the true Markov prior and the true emission model.
               The ceiling for a model with no path outside its receptive field.
    full       the best any decoder can do seeing the whole window. Already
               implemented as bayes_accuracy.
    global     the best a decoder can do seeing only the window's class
               proportion, which is what a normalisation statistic supplies to
               first order.

Three things follow that the measured ratios cannot give on their own.

Whether per-position-normalised networks are simply reaching their information
limit, or are leaving something on the table -- if they track local(R), the
comparison is between a model at its limit and one with an extra channel,
which is the cleanest form of the claim.

Whether the substitution is what an optimal decoder would do with the extra
statistic. If length-pooling networks sit near local(R) combined with global,
the effect needs no explanation beyond information availability.

And whether the constant in (ratio - 1) = c/n follows from the task's own
parameters. c is 1.21 on real data, 2.24 on the simulated genomic axis and
3.94 here, which is currently unexplained. If the same ratio computed from
these optima reproduces the measured one, c is derived rather than fitted.

The naive account of the 1/n law was wrong: the informativeness of the summary
goes as 1/sqrt(n) when converted to an accuracy gain, not 1/n. Two candidate
repairs, both testable here. Compare the paths in information rather than in
accuracy, since Var(pi-bar) and the mutual information both go as 1/n while the
mean deviation goes as its square root. Or account for the denominator: the
full-context advantage itself shrinks as runs shorten, so the ratio carries
n-dependence from both terms. These optima settle it without a closed form.
"""

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np

from difficulty.axis.synthetic import (
    DELTAS,
    WINDOW,
    Config,
    MarkovGaussianAxis,
    bayes_accuracy,
    per_position_bayes_error,
)

OUT = Path("results")
FULL = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def local_optimum(cfg, x, y, reach):
    """Optimal accuracy seeing only `reach` positions centred on the target.

    Forward-backward is run over each window independently. Doing that for
    every position would cost L passes; instead the sequence is cut into
    non-overlapping blocks of `reach` and each position is scored from its own
    block, which gives every position at least reach/2 of context on average
    and is exact for the block centre. Reported as a bound: the true
    centred-window optimum is between this and full context.
    """
    m = np.full(cfg.dim, cfg.delta / sqrt(cfg.dim))
    llr = np.einsum("ncl,c->nl", x.astype(np.float64), m)
    n, L = llr.shape
    p = cfg.switch_prob
    stay, go = np.log(1 - p), np.log(p)
    pred = np.zeros((n, L), dtype=np.int8)

    for s in range(0, L, reach):
        e = min(s + reach, L)
        seg = llr[:, s:e]
        T = seg.shape[1]
        em = np.stack([np.zeros_like(seg), seg], axis=2)
        fwd = np.zeros((n, T, 2))
        fwd[:, 0] = np.log(0.5) + em[:, 0]
        for t in range(1, T):
            pv = fwd[:, t - 1]
            fwd[:, t, 0] = np.logaddexp(pv[:, 0] + stay, pv[:, 1] + go) + em[:, t, 0]
            fwd[:, t, 1] = np.logaddexp(pv[:, 0] + go, pv[:, 1] + stay) + em[:, t, 1]
        bwd = np.zeros((n, T, 2))
        for t in range(T - 2, -1, -1):
            nx = bwd[:, t + 1] + em[:, t + 1]
            bwd[:, t, 0] = np.logaddexp(nx[:, 0] + stay, nx[:, 1] + go)
            bwd[:, t, 1] = np.logaddexp(nx[:, 0] + go, nx[:, 1] + stay)
        post = fwd + bwd
        pred[:, s:e] = (post[:, :, 1] > post[:, :, 0]).astype(np.int8)
    return float((pred == y).mean())


def global_only_optimum(y):
    """Best accuracy from the window's class proportion alone.

    An oracle given pi-bar exactly and nothing else predicts the majority label
    of the window at every position. This is the ceiling for the information a
    single window-level statistic can carry, and it is what makes the 1/n
    account testable: it depends on the label process and not at all on delta.
    """
    frac = y.mean(axis=1)
    return float(np.mean(np.maximum(frac, 1 - frac)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--switches", default="0.1,0.3,1.0,3.0,6.0,10.0,20.0")
    ap.add_argument("--reaches", default="9,33,129,513,2049")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()

    targets = [float(v) for v in args.switches.split(",")]
    reaches = [int(v) for v in args.reaches.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    delta = DELTAS[args.level]
    OUT.mkdir(exist_ok=True)

    print(f"delta={delta}, per-position optimum "
          f"{1 - per_position_bayes_error(delta):.4f}, {len(seeds)} seeds")
    print(f"{'n_sw':>7} " + " ".join(f"{'R=' + str(r):>8}" for r in reaches)
          + f" {'full':>8} {'global':>8}")

    rows = []
    for target in targets:
        p = target / (WINDOW - 1)
        axis = MarkovGaussianAxis(switch_prob=p)
        cfg = Config(delta=delta, switch_prob=p)
        loc = {r: [] for r in reaches}
        full, glob, sw = [], [], []
        for seed in seeds:
            t = axis.eval_set(level=args.level, seed=seed, n=args.n)
            sw.append(t.meta["switches_per_window"])
            full.append(bayes_accuracy(cfg, t.x, t.y))
            glob.append(global_only_optimum(t.y))
            for r in reaches:
                loc[r].append(local_optimum(cfg, t.x, t.y, r))
        row = {"switches_per_window": float(np.mean(sw)), "switch_prob": p,
               "delta": delta,
               "local": {str(r): float(np.mean(loc[r])) for r in reaches},
               "full": float(np.mean(full)), "global_only": float(np.mean(glob))}
        rows.append(row)
        print(f"{row['switches_per_window']:>7.2f} "
              + " ".join(f"{row['local'][str(r)]:>8.4f}" for r in reaches)
              + f" {row['full']:>8.4f} {row['global_only']:>8.4f}", flush=True)
        (OUT / "optima.json").write_text(json.dumps(rows, indent=2))

    # The prediction: reach is worth full - local(R_min) to a model with no
    # outside path, and full - (local combined with the global statistic) to
    # one that has it. The second is bounded below by using whichever is
    # better, which is what makes this a bound rather than an estimate.
    lo, hi = str(reaches[0]), str(reaches[-1])
    print(f"\n{'n_sw':>7} {'worth, no path':>15} {'worth, with path':>17} "
          f"{'predicted ratio':>16} {'(r-1)*n':>9}")
    for r in rows:
        w_no = r["local"][hi] - r["local"][lo]
        w_with = r["local"][hi] - max(r["local"][lo], r["global_only"])
        pred = w_no / w_with if w_with > 1e-9 else float("inf")
        print(f"{r['switches_per_window']:>7.2f} {w_no:>15.4f} {w_with:>17.4f} "
              f"{pred:>16.2f} {(pred - 1) * r['switches_per_window']:>9.2f}")
    print("\nMeasured: 62.8, 15.2, 4.3, 2.1, 1.6, 1.4, 1.2 at these densities,")
    print("(ratio-1)*n = 3.94 +/- 22%. If the predicted column tracks it, the")
    print("substitution is information availability and c is derived, not fitted.")
    print(f"\nwrote {OUT / 'optima.json'}")


if __name__ == "__main__":
    main()
