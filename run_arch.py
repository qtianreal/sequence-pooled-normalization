"""Step 2: one architecture across the whole difficulty axis.

Trains a named architecture at every difficulty level and seed, and records
accuracy plus the discriminant profile of every layer. The question is not
which architecture is most accurate -- the LAI work already found accuracy
nearly independent of architecture -- but whether d_eff still peaks at
intermediate difficulty when the mechanism for reaching across the sequence
changes.

    python run_arch.py --arch plain_cnn
    python run_arch.py --arch dilated_cnn --levels 4 --seeds 0

Writes results/arch_<name>.json incrementally, so a run that is interrupted
leaves usable partial results rather than nothing.

Parameter counts are matched to the dilated CNN's 226,177 across all
architectures, within about 1%. Depth is the free variable; width is held at
64 everywhere, because d_eff counts channels and a wider model would raise its
ceiling rather than its meaning.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from difficulty.axis.genomic import SPLIT_TIMES, genomic
from difficulty.measure import profile
from difficulty.models import (
    BiSSM,
    DilatedCNN,
    PlainCNN,
    Transformer,
    accuracy,
    fit,
    n_parameters,
)

OUT = Path("results")

ARCHITECTURES = {
    "dilated_cnn": lambda ch: DilatedCNN(in_ch=ch),
    "plain_cnn": lambda ch: PlainCNN(in_ch=ch),
    "transformer": lambda ch: Transformer(in_ch=ch, width=64, depth=9, ff_mult=1),
    "bissm": lambda ch: BiSSM(in_ch=ch, width=64, depth=17, d_state=1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHITECTURES))
    ap.add_argument("--levels", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    levels = (list(range(len(SPLIT_TIMES))) if args.levels == "all"
              else [int(v) for v in args.levels.split(",")])
    seeds = [int(v) for v in args.seeds.split(",")]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"arch_{args.arch}.json"

    print(f"{args.arch}: {len(levels)} levels x {len(seeds)} seeds on {device}")
    print(f"{'T':>5} {'s':>2} {'Fst':>8} {'acc':>8} {'d_eff':>7} {'J':>8} "
          f"{'n90':>4} {'peak':>5} {'min':>6}")

    rows = []
    for lv in levels:
        T = SPLIT_TIMES[lv]
        for seed in seeds:
            t0 = time.time()
            te = genomic.eval_set(level=lv, seed=seed, n=args.n)
            tr = genomic.train_set(level=lv, seed=seed + 500, n=args.n)
            # Seed before construction: initialisation is drawn at __init__.
            torch.manual_seed(seed)
            model = ARCHITECTURES[args.arch](te.n_channels)
            model = fit(model, tr.x, tr.y, device, epochs=args.epochs,
                        batch=args.batch, seed=seed)
            acc = accuracy(model, te.x, te.y, device)
            stats = profile(model, te.x, te.y, device)
            last = stats[-1]
            # Which block adds the most discriminant signal: the layer where
            # the architecture is doing its work, comparable across families
            # even when depth differs.
            inc = np.diff([s["J"] for s in stats])
            rows.append({
                "arch": args.arch, "split_time": T, "level": lv, "seed": seed,
                "fst": te.difficulty, "accuracy": acc,
                "d_eff": last["d_eff"], "J": last["J"], "n90": last["n90"],
                "d_eff_by_layer": [s["d_eff"] for s in stats],
                "J_by_layer": [s["J"] for s in stats],
                "n90_by_layer": [s["n90"] for s in stats],
                "peak_block": int(inc.argmax()) if len(inc) else -1,
                "dead_channels": last.get("dropped"),
                "n_parameters": n_parameters(model),
                "base_rate": te.base_rate,
                "minutes": round((time.time() - t0) / 60, 1),
            })
            r = rows[-1]
            print(f"{T:>5} {seed:>2} {r['fst']:>8.4f} {acc:>8.4f} "
                  f"{r['d_eff']:>7.1f} {r['J']:>8.2f} {r['n90']:>4d} "
                  f"{r['peak_block']:>5d} {r['minutes']:>6.1f}", flush=True)
            path.write_text(json.dumps(rows, indent=2))

    print(f"\n{'T':>5} {'Fst':>8} {'accuracy':>17} {'d_eff':>15}")
    for lv in levels:
        g = [r for r in rows if r["level"] == lv]
        a = [r["accuracy"] for r in g]
        d = [r["d_eff"] for r in g]
        sd = (lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0)
        print(f"{SPLIT_TIMES[lv]:>5} {np.mean([r['fst'] for r in g]):>8.4f} "
              f"{np.mean(a):>8.4f}+/-{sd(a):<7.4f} {np.mean(d):>7.1f}+/-{sd(d):<6.1f}")

    d = np.array([np.mean([r["d_eff"] for r in rows if r["level"] == lv])
                  for lv in levels])
    if len(d) > 2:
        i = int(d.argmax())
        interior = 0 < i < len(d) - 1
        print(f"\nd_eff peaks at level {levels[i]} (T={SPLIT_TIMES[levels[i]]}, "
              f"{d.max():.1f}), {'interior' if interior else 'AT AN END'}")
        print("The dilated CNN peaks at T=200. A peak in a comparable band is")
        print("what generality looks like; a peak at an end is not a peak.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
