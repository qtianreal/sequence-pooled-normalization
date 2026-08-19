"""Does normalisation substitute for reach only when labels barely switch?

The substitution result has an obvious alternative explanation. In the default
setup 72% of windows are single-ancestry and a window averages 0.35 ancestry
switches, so a summary statistic over the whole window nearly gives the answer
away. Normalisation supplies exactly such a summary. If that is the whole
story, the effect should vanish once labels switch often, and the claim would
be about tasks with long label runs rather than about normalisation.

Tract length is set by the time since admixture: crossovers accumulate at
`generations * RHO * L`, so raising `generations` shortens tracts without
touching divergence, the architecture, or anything else. Difficulty is held
fixed at one level and the label process alone is varied.

    generations   30   the published setting, ~0.35 switches per window
                 100   ~1.2
                 300   ~3.5
                1000   ~12

At each setting, reach is worth (k=9 minus k=1) under both normalisations. The
prediction under the alternative explanation is that the group-norm curve rises
towards the positionwise one as switches increase, closing the gap. The
prediction under the substitution account is that the gap persists: pooled
statistics still carry usable information about which ancestries are present
and in what proportion, even when the window is a mosaic.

Ablated accuracy is recorded alongside, so the ablation-versus-retraining
discrepancy can be read as a function of tract length too.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from difficulty.axis.genomic import SPLIT_TIMES, Config, GenomicAxis
from difficulty.models import DilatedCNN, accuracy, fit

OUT = Path("results")
FULL = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def label_stats(y):
    """How often ancestry switches within a window, and how often it never does."""
    sw = (np.diff(y.astype(int), axis=1) != 0).sum(axis=1)
    return float(sw.mean()), float((sw == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3, help="index into SPLIT_TIMES")
    ap.add_argument("--generations", default="30,100,300,1000")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prefixes", default="1,9")
    ap.add_argument("--norms", default="group,positionwise")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    gens = [int(v) for v in args.generations.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    prefixes = sorted(int(v) for v in args.prefixes.split(","))
    norms = args.norms.split(",")
    T = SPLIT_TIMES[args.level]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    path = OUT / "tracts.json"

    print(f"T={T}, generations {gens} x norms {norms} x prefixes {prefixes} "
          f"x {len(seeds)} seeds on {device}")
    print(f"{'gen':>5} {'s':>2} {'sw/win':>7} {'const':>6} {'k':>2} {'norm':>13} "
          f"{'retrained':>10} {'ablated':>8} {'min':>5}")

    rows = []
    for g in gens:
        for seed in seeds:
            axis = GenomicAxis(cfg=Config(split_time=T, generations=g))
            te = axis.eval_set(level=args.level, seed=seed, n=args.n)
            tr = axis.train_set(level=args.level, seed=seed + 500, n=args.n)
            sw, const = label_stats(te.y)

            for nm in norms:
                torch.manual_seed(seed)
                full = fit(DilatedCNN(in_ch=te.n_channels, norm=nm), tr.x, tr.y,
                           device, epochs=args.epochs, batch=args.batch, seed=seed)
                full_acc = accuracy(full, te.x, te.y, device)
                for k in prefixes:
                    t0 = time.time()
                    abl = accuracy(full, te.x, te.y, device,
                                   skip=set(range(k, len(FULL))))
                    if k == len(FULL):
                        ret = full_acc
                    else:
                        torch.manual_seed(seed)
                        ret = accuracy(
                            fit(DilatedCNN(in_ch=te.n_channels,
                                           dilations=FULL[:k], norm=nm),
                                tr.x, tr.y, device, epochs=args.epochs,
                                batch=args.batch, seed=seed),
                            te.x, te.y, device)
                    rows.append({
                        "split_time": T, "level": args.level, "generations": g,
                        "seed": seed, "norm": nm, "prefix": k,
                        "switches_per_window": sw, "frac_single_ancestry": const,
                        "fst": te.difficulty, "retrained": ret, "ablated": abl,
                        "full": full_acc, "base_rate": te.base_rate,
                        "minutes": round((time.time() - t0) / 60, 1),
                    })
                    print(f"{g:>5} {seed:>2} {sw:>7.2f} {const:>6.2f} {k:>2} "
                          f"{nm:>13} {ret:>10.4f} {abl:>8.4f} "
                          f"{rows[-1]['minutes']:>5.1f}", flush=True)
                    path.write_text(json.dumps(rows, indent=2))

    print(f"\n{'gen':>5} {'sw/win':>7} {'const':>6} " +
          " ".join(f"{'reach ' + nm[:5]:>14}" for nm in norms) + f" {'ratio':>7}")
    for g in gens:
        h = [r for r in rows if r["generations"] == g]
        if not h:
            continue
        worth = []
        for nm in norms:
            d = []
            for s in seeds:
                lo = [r["retrained"] for r in h
                      if r["norm"] == nm and r["seed"] == s and r["prefix"] == prefixes[0]]
                hi = [r["retrained"] for r in h
                      if r["norm"] == nm and r["seed"] == s and r["prefix"] == prefixes[-1]]
                if lo and hi:
                    d.append(hi[0] - lo[0])
            worth.append((np.mean(d), np.std(d, ddof=1) if len(d) > 1 else 0.0))
        line = (f"{g:>5} {np.mean([r['switches_per_window'] for r in h]):>7.2f} "
                f"{np.mean([r['frac_single_ancestry'] for r in h]):>6.2f} ")
        line += " ".join(f"{m:>+8.4f}+/-{sd:<5.4f}" for m, sd in worth)
        line += f" {worth[1][0] / worth[0][0]:>7.1f}x" if worth[0][0] else "    n/a"
        print(line)

    print("\nIf the ratio collapses towards 1 as switches rise, the substitution")
    print("needs labels that barely switch, and the claim is about tasks with")
    print("long label runs. If it holds, it is about normalisation.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
