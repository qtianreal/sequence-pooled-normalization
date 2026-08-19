"""Load phased 1000 Genomes haplotypes as source panels.

Vendored from lai-lowdiv/lai/real.py so this project does not import from
another checkout at runtime. Only the two functions the real-data arm needs
are here, copied unchanged: the published numbers came from that code, and a
reimplementation would have to be revalidated against them before it could be
trusted to reproduce them.

The panel of sample-to-population assignments is small and ships in data/. The
VCF does not: it is 425 MB of public 1000 Genomes data, so its location is a
parameter rather than a copy. Point KG_VCF at it, or symlink it into data/.

Parsing note: the phased panel carries FORMAT=GT only, so for biallelic sites
every sample field is exactly three characters ("a|b") on a four-character
stride. That makes the genotype block fixed-width and lets us index it with
numpy instead of splitting several hundred thousand lines in Python.
"""

import os
from pathlib import Path

import numpy as np
import pysam

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "kg.panel"
# Overridable because the VCF is too large to duplicate per project.
VCF = Path(os.environ.get("KG_VCF", ROOT / "data" / "chr22.vcf.gz"))


def require_vcf(path=None):
    """The VCF path, with a message that says what to do if it is absent."""
    p = Path(path or VCF)
    if not p.exists():
        raise SystemExit(
            f"1000 Genomes VCF not found at {p}.\n"
            "It is 425 MB of public data and is deliberately not copied into "
            "this repository.\nEither set KG_VCF to an existing copy:\n"
            "    export KG_VCF=~/lai-lowdiv/data/chr22.vcf.gz\n"
            "or symlink it in:\n"
            "    ln -s ~/lai-lowdiv/data/chr22.vcf.gz data/chr22.vcf.gz\n"
            "The .tbi index must sit beside it.")
    return p


def population_samples(pop: str):
    """Sample IDs belonging to a 1000 Genomes population code."""
    ids = []
    with open(PANEL) as fh:
        next(fh)
        for line in fh:
            f = line.split()
            if f[1] == pop:
                ids.append(f[0])
    return ids


def load_region(vcf_path, chrom, start, end, pops, maf=0.01, max_sites=None):
    """Return phased haplotypes for the requested populations over a region.

    Sites are retained if they are biallelic SNVs, fully called, and
    polymorphic above ``maf`` in the pooled sample. Pooling the populations
    for the frequency filter avoids ascertaining sites on the very frequency
    difference the task depends on.

    Returns
    -------
    haps : dict pop -> (n_sites, 2 * n_individuals) int8
    positions : (n_sites,) float
    """
    tbx = pysam.TabixFile(str(vcf_path))
    header = [l for l in tbx.header if l.startswith("#CHROM")][-1].split("\t")
    col = {s: i for i, s in enumerate(header)}

    wanted, slices = {}, []
    offset = 0
    for pop in pops:
        ids = [s for s in population_samples(pop) if s in col]
        wanted[pop] = ids
        slices.append((pop, offset, offset + 2 * len(ids)))
        offset += 2 * len(ids)

    # Column index within the genotype block (block starts after 9 fixed cols).
    flat = [col[s] - 9 for pop in pops for s in wanted[pop]]
    # Two characters per sample: allele 1 at 4j, allele 2 at 4j+2.
    take = np.empty(2 * len(flat), dtype=np.int64)
    take[0::2] = [4 * j for j in flat]
    take[1::2] = [4 * j + 2 for j in flat]

    zero = ord("0")
    rows, pos = [], []
    for line in tbx.fetch(chrom, start, end):
        f = line.split("\t", 9)
        ref, alt = f[3], f[4]
        if len(ref) != 1 or len(alt) != 1 or "," in alt:
            continue  # biallelic SNVs only
        blob = np.frombuffer(f[9].encode("ascii"), dtype=np.uint8)
        if take[-1] >= blob.size:
            continue
        g = blob[take]
        if np.any((g != zero) & (g != zero + 1)):
            continue  # missing or non-0/1 allele coding
        g = (g - zero).astype(np.int8)
        p = g.mean()
        if p < maf or p > 1 - maf:
            continue
        rows.append(g)
        pos.append(float(f[1]))
        if max_sites and len(rows) >= max_sites:
            break

    tbx.close()
    if not rows:
        raise RuntimeError("no sites passed filtering")

    mat = np.vstack(rows)  # (n_sites, total_haplotypes)
    positions = np.asarray(pos, dtype=np.float64)
    return {pop: mat[:, a:b] for pop, a, b in slices}, positions
