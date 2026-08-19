"""Does the substitution appear where the pooling extent changes with depth?

Every architecture tested so far normalises over a sequence of one length. A
U-Net does not, and that is why two reviews and the prior work all point at it:
at depth d the sequence has been halved d times, so a normalisation layer there
pools over L/2^d values while each value summarises 2^d input positions. The
extent in input positions is unchanged; the number of values the statistic is
taken over falls geometrically. Section 3 puts the per-position strength of the
channel at O(1/|S|), so if the effect is going to come apart from the criterion
anywhere, it is here.

    python run_unet.py                    the full grid
    python run_unet.py --depths 1,4 --switches 0.29

Writes results/unet.json incrementally, so an interrupted run leaves usable
partial results.

Reach is set by depth rather than by a dilation schedule, and the receptive
field is measured by autograd rather than derived: an off-by-one in a hand-
written recursion through down- and up-sampling would be silent, and the whole
experiment is about that number. The measurement is taken twice, once with
normalisation that pools along the sequence and once without, because the
difference between those two numbers is the paper's claim stated as an
observation rather than an argument.

Matched to synth_dilated_cnn.json: level 3 (delta=0.08), the same switch
densities, three seeds, 3840 training and 512 evaluation sequences, so the two
architectures are directly comparable.
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
from difficulty.models import UNet, accuracy, fit, n_parameters

OUT = Path("results")


def measured_receptive_field(model, length=WINDOW, in_ch=4):
    """Input positions the centre output actually depends on.

    A gradient that is exactly zero means no path, which is the definition we
    want: with per-position statistics this returns the convolutional span, and
    with statistics pooled along the sequence it returns the whole window,
    because the normalisation is itself a path.
    """
    model.eval()
    x = torch.zeros(1, in_ch, length, requires_grad=True)
    model(x)[0, length // 2].backward()
    nz = (x.grad.abs().sum(0).sum(0) > 0).nonzero().flatten()
    return int(nz[-1] - nz[0] + 1) if len(nz) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3, help="index into DELTAS")
    ap.add_argument("--switches", default="0.29,3.00")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--depths", default="1,2,3,4,5,6")
    ap.add_argument("--norms", default="group,positionwise")
    ap.add_argument("--train", type=int, default=3840)
    ap.add_argument("--eval", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    targets = [float(v) for v in args.switches.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    depths = sorted(int(v) for v in args.depths.split(","))
    norms = args.norms.split(",")
    delta = DELTAS[args.level]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    path = OUT / "unet.json"

    # Measured once: it is a property of the architecture, not of a run.
    reach = {}
    for d in depths:
        torch.manual_seed(0)
        reach[d] = {
            nm: measured_receptive_field(UNet(depth=d, norm=nm))
            for nm in ("none", "group", "positionwise")
        }
        print(f"depth {d}: receptive field {reach[d]['none']:>5} by convolution, "
              f"{reach[d]['positionwise']:>5} with per-position statistics, "
              f"{reach[d]['group']:>5} with statistics pooled along the sequence")

    print(f"\ndelta={delta}, switch targets {targets}, depths {depths}, "
          f"norms {norms}, {len(seeds)} seeds on {device}")
    print(f"{'sw/win':>7} {'s':>2} {'d':>2} {'RF':>5} {'norm':>13} "
          f"{'accuracy':>9} {'bayes':>7} {'min':>5}")

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
                for d in depths:
                    t0 = time.time()
                    torch.manual_seed(seed)
                    model = UNet(in_ch=te.n_channels, depth=d, norm=nm)
                    model = fit(model, tr.x, tr.y, device,
                                epochs=args.epochs, batch=args.batch, seed=seed)
                    acc = accuracy(model, te.x, te.y, device)
                    rows.append({
                        "axis": "synthetic", "arch": "unet",
                        "level": args.level, "delta": delta,
                        "switch_prob": p, "switches_per_window": sw,
                        "seed": seed, "norm": nm, "depth": d,
                        "receptive_field": reach[d][nm],
                        "receptive_field_conv": reach[d]["none"],
                        "retrained": acc,
                        "bayes_full_context": ceiling,
                        "bayes_no_context": 1 - per_position_bayes_error(delta),
                        "n_parameters": n_parameters(model),
                        "minutes": round((time.time() - t0) / 60, 1),
                    })
                    r = rows[-1]
                    print(f"{sw:>7.2f} {seed:>2} {d:>2} {r['receptive_field']:>5} "
                          f"{nm:>13} {acc:>9.4f} {ceiling:>7.4f} "
                          f"{r['minutes']:>5.1f}", flush=True)
                    path.write_text(json.dumps(rows, indent=2))

    # Reach worth, paired on seed, exactly as elsewhere in the paper.
    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    lo, hi = min(depths), max(depths)
    print(f"\n{'sw/win':>7} " + " ".join(f"{'reach ' + nm[:6]:>17}" for nm in norms)
          + f" {'ratio':>7}")
    for target in targets:
        p = target / (WINDOW - 1)
        h = [r for r in rows if r["switch_prob"] == p]
        if not h:
            continue
        worth = {}
        for nm in norms:
            per_seed = []
            for s in seeds:
                a = [r["retrained"] for r in h
                     if r["norm"] == nm and r["seed"] == s and r["depth"] == lo]
                b = [r["retrained"] for r in h
                     if r["norm"] == nm and r["seed"] == s and r["depth"] == hi]
                if a and b:
                    per_seed.append(b[0] - a[0])
            worth[nm] = np.array(per_seed)
        cells = " ".join(f"{worth[nm].mean():>+10.4f} ± {sd(worth[nm]):.4f}"
                         for nm in norms)
        ratio = (worth[norms[1]].mean() / worth[norms[0]].mean()
                 if len(norms) > 1 and worth[norms[0]].mean() else float("nan"))
        print(f"{h[0]['switches_per_window']:>7.2f} {cells} {ratio:>7.1f}")


if __name__ == "__main__":
    main()
