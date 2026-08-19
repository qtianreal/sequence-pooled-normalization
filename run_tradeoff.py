"""How much is receptive field worth, and to whom.

Three measurements of "what the long-range blocks contribute", on identical
data, at each difficulty:

    ablated    the trained full network evaluated with blocks k..8 skipped
    retrained  a fresh network built with only the first k dilations
    norm       the same, with normalisation statistics taken over channels
               only instead of over channels and length

They disagree, and the disagreement is the result. At T=200 (4 epochs, one
seed) ablation charges the long-range blocks 0.375 of accuracy while training
without them costs 0.027 -- fourteen-fold, and the first is the number this
kind of analysis usually reports. The reason is the third row: GroupNorm pools
statistics over the length axis, so every output position sees a summary of the
whole window regardless of what the convolutions can reach. Removing blocks
removes that path along with the reach; never having the blocks does not.

The effect is large and one-directional. Across a 227-fold range of receptive
field, accuracy moves 0.039 with global normalisation and 0.281 without it, and
the advantage of global normalisation decays monotonically as reach grows,
+0.285 at receptive field 9 down to +0.044 at 2049. Whether "reach matters" is
therefore a statement about the normalisation, not about the architecture.

    prefix k   dilations (1, 2, ..., 2^(k-1)), receptive field 1 + 4*sum + 4
    norm       group = channels and length (as published)
               positionwise = the same channel groups, per position

A controlled difficulty axis is what makes this quantifiable rather than
anecdotal: the substitution should matter most where the task needs evidence
the convolutions cannot reach, and vanish where local evidence suffices.

One simulation draw per (level, seed) is shared across every configuration, so
comparisons are paired on data as well as on seed, and the coalescent cost is
paid once instead of once per cell.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from difficulty.ablation import CHANCE, MIN_HEADROOM, retention
from difficulty.axis.genomic import SPLIT_TIMES, genomic
from difficulty.models import DilatedCNN, accuracy, fit, n_parameters

OUT = Path("results")
FULL = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    # Default levels span easy to hard-but-feasible. The floor levels are
    # excluded: with above-chance accuracy under 0.20 the ratio of interest has
    # no readable denominator, the same rule ablation.py applies.
    ap.add_argument("--levels", default="0,2,3,4,5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prefixes", default="1,3,5,7,9")
    ap.add_argument("--norms", default="group,positionwise")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save-weights", action="store_true")
    args = ap.parse_args()

    levels = [int(v) for v in args.levels.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    prefixes = sorted(int(v) for v in args.prefixes.split(","))
    norms = args.norms.split(",")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    if args.save_weights:
        (OUT / "weights").mkdir(exist_ok=True)
    path = OUT / "tradeoff.json"

    print(f"prefixes {prefixes} x norms {norms} x {len(levels)} levels "
          f"x {len(seeds)} seeds on {device}")
    print(f"{'T':>5} {'s':>2} {'k':>2} {'RF':>5} {'norm':>13} {'retrained':>10} "
          f"{'ablated':>8} {'gap':>7} {'min':>5}")

    rows = []
    for lv in levels:
        T = SPLIT_TIMES[lv]
        for seed in seeds:
            # One draw, shared by every prefix: paired on data and on seed.
            te = genomic.eval_set(level=lv, seed=seed, n=args.n)
            tr = genomic.train_set(level=lv, seed=seed + 500, n=args.n)

            fulls = {}
            for nm in norms:
                torch.manual_seed(seed)
                fulls[nm] = fit(DilatedCNN(in_ch=te.n_channels, norm=nm),
                                tr.x, tr.y, device, epochs=args.epochs,
                                batch=args.batch, seed=seed)
                if args.save_weights:
                    torch.save(fulls[nm].state_dict(),
                               OUT / "weights" / f"full_{nm}_T{T}_s{seed}.pt")

            for nm in norms:
                full = fulls[nm]
                full_acc = accuracy(full, te.x, te.y, device)
                for k in prefixes:
                    t0 = time.time()
                    dil = FULL[:k]
                    # Ablated: the trained full network with blocks k..8
                    # skipped. Each is h <- h + f(h), so skipping is exactly
                    # the identity and no retraining is involved.
                    abl = accuracy(full, te.x, te.y, device,
                                   skip=set(range(k, len(FULL))))
                    if k == len(FULL):
                        ret_acc, model = full_acc, full
                    else:
                        torch.manual_seed(seed)
                        model = fit(DilatedCNN(in_ch=te.n_channels,
                                               dilations=dil, norm=nm),
                                    tr.x, tr.y, device, epochs=args.epochs,
                                    batch=args.batch, seed=seed)
                        ret_acc = accuracy(model, te.x, te.y, device)
                    rf = 1 + 4 * sum(dil) + 4
                    rows.append({
                        "split_time": T, "level": lv, "seed": seed,
                        "prefix": k, "norm": nm, "dilations": list(dil),
                        "receptive_field": rf, "fst": te.difficulty,
                        "retrained": ret_acc, "ablated": abl, "full": full_acc,
                        "gap": ret_acc - abl,
                        "cost_retrained": full_acc - ret_acc,
                        "cost_ablated": full_acc - abl,
                        "retention_ablated": retention(abl, full_acc),
                        "retention_retrained": retention(ret_acc, full_acc),
                        "n_parameters": n_parameters(model),
                        "minutes": round((time.time() - t0) / 60, 1),
                    })
                    r = rows[-1]
                    print(f"{T:>5} {seed:>2} {k:>2} {rf:>5} {nm:>13} "
                          f"{ret_acc:>10.4f} {abl:>8.4f} {r['gap']:>+7.4f} "
                          f"{r['minutes']:>5.1f}", flush=True)
                    path.write_text(json.dumps(rows, indent=2))
            del fulls

    def cell(sub, key):
        return np.mean([r[key] for r in sub]) if sub else float("nan")

    for nm in norms:
        print(f"\nnorm = {nm}: retrained accuracy by reach")
        print(f"{'T':>5} " + " ".join(f"{'k=' + str(k):>8}" for k in prefixes)
              + f" {'reach worth':>12}")
        for lv in levels:
            g = [r for r in rows if r["level"] == lv and r["norm"] == nm]
            if not g:
                continue
            vals = [cell([r for r in g if r["prefix"] == k], "retrained")
                    for k in prefixes]
            print(f"{SPLIT_TIMES[lv]:>5} " + " ".join(f"{v:>8.4f}" for v in vals)
                  + f" {vals[-1] - vals[0]:>+12.4f}")

    print("\nWhat receptive field is worth, by normalisation (last k minus first)")
    print("and what ablation charges for the same blocks:")
    print(f"{'T':>5} " + " ".join(f"{'reach ' + nm[:5]:>13}" for nm in norms)
          + f" {'ablation says':>14}")
    for lv in levels:
        line = f"{SPLIT_TIMES[lv]:>5}"
        for nm in norms:
            g = [r for r in rows if r["level"] == lv and r["norm"] == nm]
            lo = cell([r for r in g if r["prefix"] == prefixes[0]], "retrained")
            hi = cell([r for r in g if r["prefix"] == prefixes[-1]], "retrained")
            line += f" {hi - lo:>+13.4f}"
        g = [r for r in rows if r["level"] == lv and r["norm"] == norms[0]
             and r["prefix"] == prefixes[0]]
        line += f" {cell(g, 'cost_ablated'):>+14.4f}"
        print(line)

    print("\nThree numbers for one quantity. If they disagree, receptive field")
    print("is not what the ablation was measuring, and normalisation is")
    print("supplying the reach that the dilations appear to supply.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
