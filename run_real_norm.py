"""The substitution on real haplotypes.

Everything else in this project runs on simulated sequences: a coalescent
process or a Gaussian one. Both give exact labels and a tunable switch rate,
which is what makes the measurements possible, and neither shows that any of it
survives real linkage disequilibrium, real allele-frequency spectra, or real
population structure.

This arm uses 1000 Genomes chromosome 22 haplotypes for the source panels. The
ancestry mosaics are still constructed -- that is the only way to have exact
per-position labels on real data, and it is what the LAI benchmark did too --
but the sequences being mosaicked, the reference panels they are classified
against, and all the correlation structure among sites are real.

Train and evaluation windows come from genomically disjoint segments separated
by a buffer, because linkage disequilibrium makes nearby windows correlated and
sampling both from the same stretch would leak. Replication is by independent
reference/donor partitions and independent mosaics, which is weaker than
independent draws from a generative process and is reported as such: there is
only one chromosome 22.

The VCF reader is vendored in difficulty/real.py; the mosaicking and the
features come from this project's own genomic axis, so the two arms are built
the same way. The 1000 Genomes VCF itself is 425 MB of public data and is not
copied here -- set KG_VCF or symlink it into data/ (difficulty.real.require_vcf
says so if it is missing).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from difficulty.real import load_region, require_vcf
from difficulty.axis.genomic import Config, _features, _mosaic, hudson_fst  # noqa: E402
from difficulty.models import DilatedCNN, accuracy, fit, n_parameters  # noqa: E402

OUT = Path("results")
FULL = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WINDOW = 4096


def split_panel(haps, n_ref, n_donor, rng):
    """Disjoint reference and donor haplotypes, so a donor is never in the
    panel used to classify it."""
    idx = rng.permutation(haps.shape[1])
    return haps[:, idx[:n_ref]], haps[:, idx[n_ref:n_ref + n_donor]]


def build(haps, positions, sl, pops, n_ref, n_donor, n_seq, n_windows, rng):
    """One mosaic over a genomic segment, cut into windows."""
    ref_a, don_a = split_panel(haps[pops[0]][sl], n_ref, n_donor, rng)
    ref_b, don_b = split_panel(haps[pops[1]][sl], n_ref, n_donor, rng)
    pos = positions[sl] - positions[sl][0]
    cfg = Config(split_time=0.0, seq_length=float(pos[-1]), n_seq=n_seq)
    seqs, labels = _mosaic(cfg, don_a, don_b, pos, rng)

    xs, ys = [], []
    n_sites = seqs.shape[0]
    for _ in range(n_windows):
        s = int(rng.integers(0, n_sites - WINDOW))
        w = slice(s, s + WINDOW)
        xs.append(_features(seqs[w], ref_a[w], ref_b[w]))
        ys.append(labels[w].T.astype(np.int8))
    y = np.concatenate(ys)
    sw = float((np.diff(y.astype(int), axis=1) != 0).sum(axis=1).mean())
    return np.concatenate(xs), y, sw, hudson_fst(ref_a, ref_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", default=None,
                    help="1000 Genomes VCF; defaults to data/chr22.vcf.gz or $KG_VCF")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--start", type=int, default=16_000_000)
    ap.add_argument("--end", type=int, default=51_000_000)
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--n-ref", type=int, default=80)
    ap.add_argument("--n-donor", type=int, default=80)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--prefixes", default="1,3,5,7,9")
    ap.add_argument("--norms", default="group,positionwise")
    ap.add_argument("--train-windows", type=int, default=60)   # x64 = 3840
    ap.add_argument("--eval-windows", type=int, default=8)     # x64 = 512
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    pops = args.pops.split(",")
    seeds = [int(v) for v in args.seeds.split(",")]
    prefixes = sorted(int(v) for v in args.prefixes.split(","))
    norms = args.norms.split(",")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    path = OUT / "real_norm.json"

    # The vendored reader resolves its panel from the repository root, so no
    # working-directory dance is needed any more.
    haps, positions = load_region(require_vcf(args.vcf), args.chrom,
                                  args.start, args.end, pops)
    n_sites = positions.size
    print(f"{pops[0]}/{pops[1]}: {n_sites:,} biallelic SNVs, "
          f"Fst {hudson_fst(haps[pops[0]], haps[pops[1]]):.5f}")

    # Disjoint segments with a buffer: LD correlates nearby windows.
    cut, buf = int(n_sites * 0.60), int(n_sites * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n_sites)
    print(f"train sites 0-{cut:,}, test sites {cut + buf:,}-{n_sites:,} "
          f"(buffer {buf:,})")
    print(f"{'s':>2} {'k':>2} {'RF':>5} {'norm':>13} {'retrained':>10} "
          f"{'ablated':>8} {'sw/win':>7} {'min':>5}")

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(4242 + seed)
        xtr, ytr, sw_tr, _ = build(haps, positions, tr_sl, pops, args.n_ref,
                                   args.n_donor, 64, args.train_windows, rng)
        xte, yte, sw, fst = build(haps, positions, te_sl, pops, args.n_ref,
                                  args.n_donor, 64, args.eval_windows, rng)

        for nm in norms:
            torch.manual_seed(seed)
            full = fit(DilatedCNN(in_ch=xtr.shape[1], norm=nm), xtr, ytr, device,
                       epochs=args.epochs, batch=args.batch, seed=seed)
            full_acc = accuracy(full, xte, yte, device)
            for k in prefixes:
                t0 = time.time()
                abl = accuracy(full, xte, yte, device,
                               skip=set(range(k, len(FULL))))
                if k == len(FULL):
                    ret = full_acc
                else:
                    torch.manual_seed(seed)
                    ret = accuracy(
                        fit(DilatedCNN(in_ch=xtr.shape[1], dilations=FULL[:k],
                                       norm=nm),
                            xtr, ytr, device, epochs=args.epochs,
                            batch=args.batch, seed=seed),
                        xte, yte, device)
                rows.append({
                    "data": "1000G_chr22", "pops": pops, "fst": fst,
                    "seed": seed, "norm": nm, "prefix": k,
                    "receptive_field": 1 + 4 * sum(FULL[:k]) + 4,
                    "retrained": ret, "ablated": abl, "full": full_acc,
                    "switches_per_window": sw,
                    "train_examples": int(xtr.shape[0]),
                    "test_examples": int(xte.shape[0]),
                    "minutes": round((time.time() - t0) / 60, 1),
                })
                r = rows[-1]
                print(f"{seed:>2} {k:>2} {r['receptive_field']:>5} {nm:>13} "
                      f"{ret:>10.4f} {abl:>8.4f} {sw:>7.2f} "
                      f"{r['minutes']:>5.1f}", flush=True)
                path.write_text(json.dumps(rows, indent=2))
        del xtr, ytr, xte, yte

    sd = lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0
    print(f"\n{'norm':>13} " + " ".join(f"{'k=' + str(k):>8}" for k in prefixes)
          + f" {'reach worth':>16}")
    for nm in norms:
        vals = [np.mean([r["retrained"] for r in rows
                         if r["norm"] == nm and r["prefix"] == k]) for k in prefixes]
        d = [next(r["retrained"] for r in rows if r["norm"] == nm
                  and r["seed"] == s and r["prefix"] == prefixes[-1])
             - next(r["retrained"] for r in rows if r["norm"] == nm
                    and r["seed"] == s and r["prefix"] == prefixes[0]) for s in seeds]
        print(f"{nm:>13} " + " ".join(f"{v:>8.4f}" for v in vals)
              + f" {np.mean(d):>+9.4f}+/-{sd(d):<6.4f}")
    g = [r for r in rows if r["norm"] == norms[0] and r["prefix"] == prefixes[0]]
    print(f"\nablation says the removed blocks are worth "
          f"{np.mean([r['full'] - r['ablated'] for r in g]):+.4f}, "
          f"retraining says {np.mean([r['full'] - r['retrained'] for r in g]):+.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
