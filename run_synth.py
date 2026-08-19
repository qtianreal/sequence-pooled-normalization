"""Does the substitution leave genomics?

The same experiment as run_tracts.py, on an axis with no coalescent simulation
in it: Gaussian per-position evidence, Markov labels, switch rate set directly.
If reach is worth several times more under per-position normalisation than
under normalisation that pools along the sequence, and if that ratio decays as
labels switch more often, then the effect is a property of sequence labelling
rather than of this genomic task.

Switch rates are matched to the densities measured on the genomic axis, so the
two dose-response curves can be plotted on the same x-axis.

Everything here is also reported against the exact optimal decoder for the
generating process, which the genomic axis cannot supply. That turns "reach is
worth +0.04" into a fraction of a knowable maximum.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from difficulty.axis.synthetic import (
    DELTAS,
    WINDOW,
    Config,
    MarkovGaussianAxis,
    bayes_accuracy,
    per_position_bayes_error,
)
from difficulty.models import DilatedCNN, TasNetTCN, Transformer, accuracy, fit

OUT = Path("results")
FULL = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3, help="index into DELTAS")
    ap.add_argument("--switches", default="0.29,0.99,3.00,9.89")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prefixes", default="1,9")
    ap.add_argument("--norms", default="group,positionwise")
    ap.add_argument("--arch", default="dilated_cnn",
                    choices=["dilated_cnn", "transformer", "tasnet"])
    ap.add_argument("--train", type=int, default=3840)
    ap.add_argument("--eval", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    targets = [float(v) for v in args.switches.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    prefixes = sorted(int(v) for v in args.prefixes.split(","))
    norms = args.norms.split(",")
    delta = DELTAS[args.level]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"synth_{args.arch}.json"

    print(f"delta={delta} (per-position Bayes accuracy "
          f"{1 - per_position_bayes_error(delta):.4f}), "
          f"switch targets {targets}, norms {norms}, {len(seeds)} seeds")
    print(f"{'sw/win':>7} {'s':>2} {'k':>2} {'norm':>13} {'retrained':>10} "
          f"{'ablated':>8} {'bayes':>7} {'min':>5}")

    rows = []
    for target in targets:
        p = target / (WINDOW - 1)
        for seed in seeds:
            axis = MarkovGaussianAxis(switch_prob=p)
            te = axis.eval_set(level=args.level, seed=seed, n=args.eval)
            tr = axis.train_set(level=args.level, seed=seed, n=args.train)
            cfg = Config(delta=delta, switch_prob=p)
            ceiling = bayes_accuracy(cfg, te.x, te.y)
            sw = te.meta["switches_per_window"]

            for nm in norms:
                torch.manual_seed(seed)
                build = {"transformer": Transformer, "tasnet": TasNetTCN}.get(
                    args.arch, DilatedCNN)
                full = fit(build(in_ch=te.n_channels, norm=nm), tr.x, tr.y,
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
                            fit(build(in_ch=te.n_channels,
                                      dilations=FULL[:k], norm=nm),
                                tr.x, tr.y, device, epochs=args.epochs,
                                batch=args.batch, seed=seed),
                            te.x, te.y, device)
                    rows.append({
                        "axis": "synthetic", "arch": args.arch, "level": args.level,
                        "delta": delta,
                        "switch_prob": p, "switches_per_window": sw,
                        "seed": seed, "norm": nm, "prefix": k,
                        "retrained": ret, "ablated": abl, "full": full_acc,
                        "bayes_full_context": ceiling,
                        "bayes_no_context": 1 - per_position_bayes_error(delta),
                        "minutes": round((time.time() - t0) / 60, 1),
                    })
                    print(f"{sw:>7.2f} {seed:>2} {k:>2} {nm:>13} {ret:>10.4f} "
                          f"{abl:>8.4f} {ceiling:>7.4f} "
                          f"{rows[-1]['minutes']:>5.1f}", flush=True)
                    path.write_text(json.dumps(rows, indent=2))

    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    print(f"\n{'sw/win':>7} {'ceiling':>8} " +
          " ".join(f"{'reach ' + nm[:5]:>16}" for nm in norms) + f" {'ratio':>7}")
    for target in targets:
        h = [r for r in rows if abs(r["switch_prob"] - target / (WINDOW - 1)) < 1e-12]
        if not h:
            continue
        worth = []
        for nm in norms:
            d = [next(r["retrained"] for r in h if r["norm"] == nm and r["seed"] == s
                      and r["prefix"] == prefixes[-1])
                 - next(r["retrained"] for r in h if r["norm"] == nm and r["seed"] == s
                        and r["prefix"] == prefixes[0])
                 for s in seeds]
            worth.append((np.mean(d), sd(d)))
        line = (f"{np.mean([r['switches_per_window'] for r in h]):>7.2f} "
                f"{np.mean([r['bayes_full_context'] for r in h]):>8.4f} ")
        line += " ".join(f"{m:>+9.4f}+/-{s:<6.4f}" for m, s in worth)
        line += f" {worth[1][0] / worth[0][0]:>6.1f}x" if worth[0][0] else "   n/a"
        print(line)

    print("\nThe genomic axis gave 8.3x, 3.1x, 1.7x, 1.3x at these switch")
    print("densities. A similar decay here makes the claim about sequence")
    print("labelling; a flat or absent curve makes it about that task.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
