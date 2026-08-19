"""What d_eff reads when there is nothing to read.

d_eff is a participation ratio over per-channel Fisher ratios. When a
representation carries class signal, it counts how widely that signal is
spread. When it carries none, the Fisher ratios are noise and the ratio still
returns a number -- plausibly a large one, since noise spread evenly across 64
channels looks exactly like signal spread evenly across 64 channels.

That number is the noise floor, and without it the low-difficulty end of the
axis cannot be interpreted. At the two hardest levels J falls to 0.1-0.2 and
the seed spread of d_eff reaches +/-16, which is what a statistic being driven
by noise looks like. The collapse of d_eff at the floor is one of the two
things that make the difficulty curve non-monotonic, so whether it is real
decides whether the phenomenon is real.

Three nulls, all applied to the same activations as the observed value, so the
only thing that changes is the labels:

    shuffle   labels permuted freely across sequences and positions. Destroys
              every association, including the tract autocorrelation. The
              strict "no relationship whatsoever" null, and the one that
              defines the floor.
    rowperm   sequences keep their own label vectors, but the vectors are
              dealt to different sequences. Tract structure exactly preserved;
              the link between a sequence's activations and its ancestry
              destroyed.
    crossseed labels taken from an independent replicate at the same
              difficulty -- a genuine second draw from the same generative
              process, statistically identical and causally unrelated to these
              activations. The most principled of the three.

A circular-shift null was tried first and discarded: 72% of these windows are
single-ancestry, so rolling leaves 46 of 64 label vectors bit-identical and
changes 8.8% of labels overall. It agreed closely with the observed value,
which looked alarming and meant nothing -- a null that preserves the thing it
is supposed to destroy tests nothing. Recorded here because the failure is easy
to repeat.

Run against the LAI project's cached models, which are the exact networks the
published d_eff figures came from -- so the answer applies to that evidence and
not merely to a reimplementation of it.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from difficulty.measure import discriminant_stats, layer_activations
from difficulty.models import DilatedCNN

# The d_eff appendix re-reads the LAI project's cached models and evaluation
# sets: those are 261 MB of derived artefacts belonging to that project, and
# re-deriving them here would mean retraining its networks. Declared rather
# than assumed, so a missing checkout fails with a sentence instead of a
# FileNotFoundError on a path nobody chose.
CACHE = Path(os.environ.get("LAI_CACHE",
                            Path.home() / "lai-lowdiv/results/cache"))
OUT = Path("results")
SPLIT_TIMES = [3200, 1600, 800, 400, 200, 100, 50, 25]


def rowperm_labels(y, rng):
    """Deal the label vectors to different sequences."""
    return y[rng.permutation(len(y))]


def shuffle_labels(y, rng):
    flat = y.reshape(-1).copy()
    rng.shuffle(flat)
    return flat.reshape(y.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--draws", type=int, default=5, help="null draws per model")
    ap.add_argument("--n", type=int, default=64, help="sequences, as in run_lda")
    args = ap.parse_args()

    seeds = [int(v) for v in args.seeds.split(",")]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)

    kinds = ("shuffle", "rowperm", "crossseed")
    print(f"{'T':>5} {'s':>2} {'J':>7} {'d_eff':>7} | " +
          " | ".join(f"{'d_eff ' + k:>16}" for k in kinds))

    rows = []
    for T in SPLIT_TIMES:
        for s in seeds:
            mp, tp = CACHE / f"cnn_T{T}_s{s}.pt", CACHE / f"test_T{T}_s{s}.npz"
            if not (mp.exists() and tp.exists()):
                continue
            model = DilatedCNN()
            model.load_state_dict(torch.load(mp, map_location="cpu"))
            model = model.to(device).eval()
            d = np.load(tp)
            x, y = d["x"][: args.n].astype(np.float32), d["y"][: args.n]

            # Activations once; only the labels differ between observed and null.
            act = layer_activations(model, x, device)[-1]
            obs = discriminant_stats(act, y)

            # Independent label draws at the same difficulty, from the other
            # seeds' evaluation sets. Same process, unrelated to these weights.
            others = []
            for o in seeds:
                q = CACHE / f"test_T{T}_s{o}.npz"
                if o != s and q.exists():
                    others.append(np.load(q)["y"][: args.n])

            rng = np.random.default_rng(1000 * T + s)
            null = {k: [] for k in kinds}
            for i in range(args.draws):
                null["shuffle"].append(discriminant_stats(act, shuffle_labels(y, rng)))
                null["rowperm"].append(discriminant_stats(act, rowperm_labels(y, rng)))
                if others:
                    yo = others[i % len(others)]
                    null["crossseed"].append(discriminant_stats(act, rowperm_labels(yo, rng)))

            r = {"split_time": T, "seed": s, "J": obs["J"], "d_eff": obs["d_eff"],
                 "n90": obs["n90"]}
            for kind in kinds:
                r[f"J_{kind}"] = [v["J"] for v in null[kind]]
                r[f"d_eff_{kind}"] = [v["d_eff"] for v in null[kind]]
            rows.append(r)
            line = f"{T:>5} {s:>2} {obs['J']:>7.2f} {obs['d_eff']:>7.1f} |"
            for kind in kinds:
                v = r[f"d_eff_{kind}"]
                line += (f" {np.mean(v):>7.1f}+/-{np.std(v):<4.1f} (J{np.mean(r['J_' + kind]):>5.2f}) |"
                         if v else f" {'--':>16} |")
            print(line, flush=True)
            (OUT / "null_deff.json").write_text(json.dumps(rows, indent=2))
            del act

    print(f"\n{'T':>5} {'J':>8} {'d_eff obs':>11} {'d_eff null (crossseed)':>24} {'margin':>8}")
    for T in SPLIT_TIMES:
        g = [r for r in rows if r["split_time"] == T]
        if not g:
            continue
        obs = np.array([r["d_eff"] for r in g])
        nul = np.concatenate([r["d_eff_crossseed"] or r["d_eff_shuffle"] for r in g])
        # In units of the null's own spread: how far outside noise the
        # observed value sits. Under about 2, d_eff is not measuring signal.
        margin = (obs.mean() - nul.mean()) / nul.std() if nul.std() > 0 else np.nan
        print(f"{T:>5} {np.mean([r['J'] for r in g]):>8.2f} "
              f"{obs.mean():>7.1f}+/-{obs.std(ddof=1):<3.1f} "
              f"{nul.mean():>11.1f}+/-{nul.std():<5.1f} {margin:>+8.1f}")
    print("\nmargin = (observed - null) / null SD, at the same activations.")
    print("A level whose margin is small is not evidence about anything: the")
    print("statistic is reporting the shape of noise, not of a representation.")
    print(f"\nwrote {OUT / 'null_deff.json'}")


if __name__ == "__main__":
    main()
