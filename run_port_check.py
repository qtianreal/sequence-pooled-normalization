"""Does the ported axis reproduce the numbers it was ported from?

Nothing here is a result. This is the check that the port is faithful before
any new architecture is trained on it, and it exists because the LAI project
lost most of a day to a harness that disagreed with the sweep it was supposed
to reproduce -- for configuration reasons, not code reasons, which took a
dedicated control run to establish.

Three things are compared against IDEA.md, which records them from
lai-lowdiv at 8 levels x 3 seeds:

    F_ST per level      0.0030 0.0050 0.0097 0.0200 0.0384 0.0736 0.1376 0.2434
    CNN accuracy        0.518  0.606  0.796  0.891  0.949  0.977  0.991  0.995
    d_eff, final block   25.5   36.0   39.0   40.0   36.7   35.2   35.7   33.7

F_ST should match closely: same demography, same estimator, different draws.
Accuracy and d_eff need a trained model per level and are the slower half, so
they are opt-in.

    python run_port_check.py --fst-only          # minutes
    python run_port_check.py --levels 0,3,7      # trains, ~1 h

Training and evaluation use separate draws from the axis. They must: sequences
within one replicate share reference panels, so a split inside a single draw
scores memorisation of those panels. The first version of the port did exactly
that and read below chance at levels the original solves at 0.89.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from difficulty.axis.genomic import SPLIT_TIMES, genomic
from difficulty.measure import profile
from difficulty.models import DilatedCNN, accuracy, fit, n_parameters

OUT = Path("results")

# From the LAI benchmark: pruning_seeds.json for accuracy, lda_results.json for
# d_eff, both per seed rather than averaged. Per seed, because the comparison
# that matters is distribution against distribution -- one draw against a mean
# reads a 3.5 SD deficit where three against three read none.
REFERENCE = {
    3200: {"fst": 0.2434, "acc": [0.9985, 0.9983, 0.9986], "d_eff": [35.1, 33.2, 33.0]},
    1600: {"fst": 0.1376, "acc": [0.9948, 0.9966, 0.9951], "d_eff": [35.7, 32.9, 38.5]},
    800:  {"fst": 0.0736, "acc": [0.9891, 0.9847, 0.9867], "d_eff": [37.3, 34.1, 34.4]},
    400:  {"fst": 0.0384, "acc": [0.9520, 0.9619, 0.9603], "d_eff": [38.7, 33.8, 37.5]},
    200:  {"fst": 0.0200, "acc": [0.8794, 0.8922, 0.9045], "d_eff": [40.7, 42.0, 37.3]},
    100:  {"fst": 0.0097, "acc": [0.7713, 0.7605, 0.7945], "d_eff": [36.5, 42.2, 38.4]},
    50:   {"fst": 0.0050, "acc": [0.6697, 0.6837, 0.6383], "d_eff": [29.4, 39.2, 39.4]},
    25:   {"fst": 0.0030, "acc": [0.5549, 0.5565, 0.6101], "d_eff": [23.0, 23.8, 29.6]},
}


def welch(a, b):
    """Difference of means in pooled-SE units. Small n, so this is a rough
    scale, not a p-value -- three seeds cannot support a real test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fst-only", action="store_true")
    args = ap.parse_args()

    levels = (list(range(len(SPLIT_TIMES))) if args.levels == "all"
              else [int(v) for v in args.levels.split(",")])
    seeds = [int(v) for v in args.seeds.split(",")]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)

    head = f"{'T':>5} {'s':>2} {'Fst':>8} {'ref':>8} {'diff':>8}"
    if not args.fst_only:
        head += f" {'acc':>7} {'d_eff':>7}"
    print(head)

    rows = []
    for lv in levels:
        T = SPLIT_TIMES[lv]
        ref = REFERENCE[T]
        for seed in seeds:
            # Evaluation draw first: it is what F_ST is reported on when only
            # F_ST is asked for, and what the rest is measured on otherwise.
            te = genomic.eval_set(level=lv, seed=seed, n=args.n)
            r = {"split_time": T, "level": lv, "seed": seed, "fst": te.difficulty,
                 "fst_reference": ref["fst"],
                 "fst_diff": te.difficulty - ref["fst"],
                 "base_rate": te.base_rate}
            print(f"{T:>5} {seed:>2} {r['fst']:>8.4f} {ref['fst']:>8.4f} "
                  f"{r['fst_diff']:>+8.4f}", end="", flush=True)

            if not args.fst_only:
                # A different seed, so the training replicates are independent
                # simulations with their own reference panels.
                tr = genomic.train_set(level=lv, seed=seed + 500, n=args.n)
                # Seed before construction: weight initialisation is drawn at
                # __init__, so seeding inside fit() would leave it to whatever
                # global state the process happened to be in. d_eff varies by
                # ~10 across seeds at fixed accuracy, so this is not a detail.
                torch.manual_seed(seed)
                model = DilatedCNN(in_ch=tr.n_channels)
                model = fit(model, tr.x, tr.y, device, epochs=args.epochs, seed=seed)
                acc = accuracy(model, te.x, te.y, device)
                stats = profile(model, te.x, te.y, device)
                d_eff = stats[-1]["d_eff"]
                r.update(accuracy=acc, d_eff=d_eff, J=stats[-1]["J"],
                         n90=stats[-1]["n90"],
                         dead_channels=stats[-1].get("dropped"),
                         train_examples=int(tr.x.shape[0]),
                         test_examples=int(te.x.shape[0]),
                         n_parameters=n_parameters(model))
                print(f" {acc:>7.4f} {d_eff:>7.1f}", end="", flush=True)
            print()
            rows.append(r)
            (OUT / "port_check.json").write_text(json.dumps(rows, indent=2))

    f = np.array([r["fst_diff"] for r in rows])
    print(f"\nF_ST: mean |difference| {np.abs(f).mean():.4f}, worst {np.abs(f).max():.4f}")

    if not args.fst_only:
        print(f"\n{'T':>5} {'acc':>15} {'reference':>15} {'t':>6}"
              f" {'d_eff':>15} {'reference':>15} {'t':>6}")
        for lv in levels:
            T = SPLIT_TIMES[lv]
            ref, g = REFERENCE[T], [r for r in rows if r["level"] == lv]
            a = [r["accuracy"] for r in g]
            d = [r["d_eff"] for r in g]
            print(f"{T:>5} {np.mean(a):>8.4f}+/-{np.std(a, ddof=1):<6.4f}"
                  f" {np.mean(ref['acc']):>8.4f}+/-{np.std(ref['acc'], ddof=1):<6.4f}"
                  f" {welch(a, ref['acc']):>+6.1f}"
                  f" {np.mean(d):>8.1f}+/-{np.std(d, ddof=1):<6.1f}"
                  f" {np.mean(ref['d_eff']):>8.1f}+/-{np.std(ref['d_eff'], ddof=1):<6.1f}"
                  f" {welch(d, ref['d_eff']):>+6.1f}")
        print("\nt is the difference in pooled-SE units at n=3 per side: a rough"
              "\nscale, not a test. |t| under about 2 is what a faithful port"
              "\nlooks like given how much both quantities move across seeds.")

    if not args.fst_only and len(levels) > 2:
        d = np.array([np.mean([r["d_eff"] for r in rows if r["level"] == lv])
                      for lv in levels])
        print(f"\nd_eff by level, {d.min():.1f} to {d.max():.1f}, "
              f"peak at level {levels[int(d.argmax())]} "
              f"(T={SPLIT_TIMES[levels[int(d.argmax())]]})")
        print("The port is faithful if the peak sits at intermediate difficulty,")
        print("not at either end. Absolute values may differ; the shape is the claim.")
    print(f"\nwrote {OUT / 'port_check.json'}")


if __name__ == "__main__":
    main()
