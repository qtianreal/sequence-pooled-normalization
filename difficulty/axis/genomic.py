"""A difficulty axis whose parameter is a property of the generating process.

Two populations split T generations ago and are then mixed into mosaic
sequences whose segment origins are recorded exactly. The task is to recover
the origin of each position. Difficulty is Hudson's F_ST between the two
sources: as the split becomes recent the populations share more variation, the
per-position evidence shrinks, and below a floor around 0.002 no method
recovers anything.

Why this axis rather than a corruption level. Label noise, blur radius and
severity scales have no units, so "twice as hard" is undefined and the hard end
of the range is merely annoying rather than impossible. F_ST is set by the
demography, is measurable from the data, and comes with a floor -- which is
what lets a representation that has run out of information be distinguished
from one that is merely struggling.

A draw is many independent replicates, not one. Each replicate is its own
coalescent simulation with its own reference panels, and windows are cut from
it at random offsets. Training and evaluation must come from *different*
replicates: within a replicate the reference panels are shared by every
sequence, so a split inside one simulation measures memorisation of those
panels rather than generalisation at that difficulty. The benchmark this is
ported from used 20 training replicates x 3 windows and 8 evaluation
replicates x 1 window -- 3840 and 512 sequences -- and those are the defaults
here. An earlier version of this file split one replicate 75/25 and produced
below-chance accuracy at levels the original solves easily.

Ported from lai-lowdiv/lai/sim.py, lai/methods.py and run_pilot.py. The
genomics is the instrument; nothing downstream of Task should know it is here.
"""

from dataclasses import dataclass

import msprime
import numpy as np

from difficulty.task import Task

RHO = 1e-8   # per-base per-generation recombination rate
MU = 1.25e-8  # per-base per-generation mutation rate
PSEUDO = 1e-3  # frequency pseudocount, guards log(0) at fixed sites
WINDOW = 4096  # positions per window

# Split times, ascending in difficulty (recent split = hard). These are the
# eight levels the LAI benchmark swept; the port is checked against its
# published F_ST and d_eff values.
SPLIT_TIMES = [3200, 1600, 800, 400, 200, 100, 50, 25]

# The training and evaluation budgets of the original sweep.
TRAIN_REPLICATES, TRAIN_WINDOWS = 20, 3
EVAL_REPLICATES, EVAL_WINDOWS = 8, 1


@dataclass
class Config:
    split_time: float
    ne: int = 10_000
    seq_length: float = 1e7
    n_ref: int = 100      # reference sequences per population
    n_donor: int = 100    # donor sequences per population
    n_seq: int = 64       # mosaic sequences per replicate
    generations: int = 30  # since the mixing pulse
    prop: float = 0.5     # expected fraction from population A


def _panels(cfg: Config, seed: int):
    """Coalescent simulation of two populations after a clean split."""
    n_per_pop = cfg.n_ref + cfg.n_donor
    dem = msprime.Demography()
    dem.add_population(name="ANC", initial_size=cfg.ne)
    dem.add_population(name="A", initial_size=cfg.ne)
    dem.add_population(name="B", initial_size=cfg.ne)
    dem.add_population_split(time=cfg.split_time, derived=["A", "B"], ancestral="ANC")

    ts = msprime.sim_ancestry(
        samples={"A": n_per_pop, "B": n_per_pop}, demography=dem,
        sequence_length=cfg.seq_length, recombination_rate=RHO,
        ploidy=1, random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)

    geno = ts.genotype_matrix().astype(np.int8)
    pos = ts.sites_position.astype(np.float64)
    # Recurrent mutation puts allele states above 1 at ~0.05% of sites; the
    # binary coding downstream assumes 0/1, so those sites are dropped.
    keep = (geno <= 1).all(axis=1)
    geno, pos = geno[keep], pos[keep]
    return geno[:, :n_per_pop], geno[:, n_per_pop:], pos


def hudson_fst(a: np.ndarray, b: np.ndarray) -> float:
    """Hudson's estimator as a ratio of averages over sites."""
    n1, n2 = a.shape[1], b.shape[1]
    p1, p2 = a.mean(axis=1), b.mean(axis=1)
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    keep = den > 0
    return float(num[keep].sum() / den[keep].sum())


def _mosaic(cfg: Config, donor_a, donor_b, pos, rng):
    """Splice donor sequences into mosaics, recording each segment's origin."""
    n_sites = len(pos)
    rate = cfg.generations * RHO * cfg.seq_length
    seqs = np.zeros((n_sites, cfg.n_seq), dtype=np.int8)
    labels = np.zeros((n_sites, cfg.n_seq), dtype=np.int8)
    for j in range(cfg.n_seq):
        breaks = np.sort(rng.uniform(0, cfg.seq_length, size=rng.poisson(rate)))
        edges = np.concatenate([[0.0], breaks, [cfg.seq_length]])
        for k in range(len(edges) - 1):
            lo, hi = np.searchsorted(pos, [edges[k], edges[k + 1]])
            if hi <= lo:
                continue
            from_a = rng.random() < cfg.prop
            pool = donor_a if from_a else donor_b
            seqs[lo:hi, j] = pool[lo:hi, rng.integers(pool.shape[1])]
            labels[lo:hi, j] = 0 if from_a else 1
    return seqs, labels


def _features(seqs, ref_a, ref_b):
    """Four channels per position: the observed allele, the two reference
    frequencies, and their log likelihood ratio. Deliberately the weakest of
    the representations the LAI work compared -- it carries no information the
    references do not, which keeps the difficulty axis clean.

    Called on the cropped window, not the whole simulation, so the frequencies
    are estimated from exactly the sites the model sees.
    """
    p_a = np.clip(ref_a.mean(axis=1), PSEUDO, 1 - PSEUDO)
    p_b = np.clip(ref_b.mean(axis=1), PSEUDO, 1 - PSEUDO)
    n_sites, n = seqs.shape
    x = np.zeros((n, 4, n_sites), dtype=np.float32)
    x[:, 0, :] = seqs.T
    x[:, 1, :] = p_a
    x[:, 2, :] = p_b
    lr = seqs.T * np.log(p_a / p_b) + (1 - seqs.T) * np.log((1 - p_a) / (1 - p_b))
    x[:, 3, :] = lr
    return x


def _replicate(cfg: Config, seed: int):
    """One independent simulation: panels, mosaic sequences, exact labels."""
    a, b, pos = _panels(cfg, seed)
    ref_a, donor_a = a[:, : cfg.n_ref], a[:, cfg.n_ref :]
    ref_b, donor_b = b[:, : cfg.n_ref], b[:, cfg.n_ref :]
    rng = np.random.default_rng(seed + 7919)
    seqs, labels = _mosaic(cfg, donor_a, donor_b, pos, rng)
    return {"ref_a": ref_a, "ref_b": ref_b, "seqs": seqs, "labels": labels,
            "fst": hudson_fst(ref_a, ref_b)}


class GenomicAxis:
    name = "genomic"

    def __init__(self, window=WINDOW, cfg=None):
        self.window = window
        self._cfg = cfg

    def levels(self):
        """Split times, hardest last. Difficulty is reported as F_ST, which is
        measured per replicate rather than set, so the levels are indices."""
        return list(range(len(SPLIT_TIMES)))

    def sample(self, level: int, seed: int, n: int = 64,
               replicates: int = EVAL_REPLICATES,
               windows: int = EVAL_WINDOWS) -> Task:
        """Draw `replicates` independent simulations, `windows` cut from each.

        The returned Task holds replicates * windows * n sequences. Two calls
        with different `seed` are independent and may be used as train and
        evaluation sets; a split *inside* one Task may not, because sequences
        from the same replicate share reference panels.
        """
        cfg = self._cfg or Config(split_time=SPLIT_TIMES[level])
        cfg.n_seq = n
        xs, ys, fsts = [], [], []
        for i in range(replicates):
            # msprime rejects seed 0, and callers naturally start seeds at 0.
            # Offsetting by level and replicate keeps every coalescent draw
            # independent across the axis and within a draw.
            sim_seed = 1 + seed * 100_000 + level * 1_000 + i
            rep = _replicate(cfg, sim_seed)
            fsts.append(rep["fst"])
            rng = np.random.default_rng(sim_seed + 31)
            n_sites = rep["seqs"].shape[0]
            for _ in range(windows):
                # A random offset, not a fixed one: segments are long relative
                # to the window, so the same stretch of every replicate would
                # give the class balance a positional bias.
                off = int(rng.integers(0, max(n_sites - self.window, 1)))
                sl = slice(off, off + self.window)
                xs.append(_features(rep["seqs"][sl], rep["ref_a"][sl],
                                    rep["ref_b"][sl]))
                ys.append(rep["labels"][sl].T.astype(np.int8))

        return Task(
            x=np.concatenate(xs), y=np.concatenate(ys),
            difficulty=float(np.mean(fsts)),
            floor=0.0022,
            meta={"axis": self.name, "level": level,
                  "split_time": cfg.split_time, "seed": seed,
                  "replicates": replicates, "windows_per_replicate": windows,
                  "fst_sd": float(np.std(fsts))},
        )

    def train_set(self, level: int, seed: int, n: int = 64) -> Task:
        return self.sample(level, seed, n, TRAIN_REPLICATES, TRAIN_WINDOWS)

    def eval_set(self, level: int, seed: int, n: int = 64) -> Task:
        return self.sample(level, seed, n, EVAL_REPLICATES, EVAL_WINDOWS)


genomic = GenomicAxis()
