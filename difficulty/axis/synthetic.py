"""A sequence-labelling axis with no genomics in it.

Two isotropic Gaussians separated by delta supply the per-position evidence;
per-position labels come from a two-state Markov chain whose switch probability
is set directly. That gives independent control of the two things the genomic
axis confounds:

    delta          how much evidence a single position carries. Per-position
                   Bayes error is Phi(-delta/2), exactly.
    switch_prob    how often the label changes, and so how much evidence can be
                   pooled before it stops being about the same label.

The genomic axis ties these together through demography and time since
admixture. Here they are orthogonal, which is what makes it possible to say
whether the normalisation-substitutes-for-reach effect tracks label run length
or per-position difficulty.

Both bounds are computable rather than estimated:

    no context      Phi(-delta/2), the per-position Bayes error
    full context    exact forward-backward over the true Markov prior and the
                    true emission model, i.e. what an optimal decoder that
                    integrates the whole window achieves

The gap between them is the headroom that reach is competing for. Reporting a
network's accuracy against the second is what makes "reach is worth +0.04"
interpretable: it is +0.04 out of a knowable maximum, not out of nothing.

Ported from nothing. The genomic axis is difficulty/axis/genomic.py; this one
exists so the claim can be about sequence labelling rather than about genomes.
"""

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np

from difficulty.task import Task

WINDOW = 4096

# Separations chosen so that the achievable accuracy at each reach matches the
# genomic axis, not so that per-position Bayes error spans a tidy range. The
# regime this project is about is the one where a single position is nearly
# uninformative and the task is solvable only by integration: at delta = 0.08 a
# single position gives 0.516, nine positions give 0.548, and 2049 give 0.965,
# against the genomic axis's 0.577 and 0.951 at those two reaches. Choosing
# delta by per-position error instead put nine positions at 0.73 and left
# nothing for reach to buy. Easiest first, so levels() ascends in difficulty.
DELTAS = [0.40, 0.20, 0.12, 0.08, 0.06, 0.04, 0.02, 0.0]

# Switch probabilities matching the switch densities measured on the genomic
# axis: 0.29, 0.99, 3.00 and 9.89 changes per 4096-position window.
SWITCH_RATES = {0.29: 7.1e-5, 0.99: 2.4e-4, 3.00: 7.3e-4, 9.89: 2.4e-3}

TRAIN_SEQUENCES, EVAL_SEQUENCES = 3840, 512  # the genomic axis's budgets


def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def per_position_bayes_error(delta):
    """Error of the optimal rule that sees one position and no context."""
    return normal_cdf(-delta / 2.0)


@dataclass
class Config:
    delta: float
    dim: int = 4          # channels, matching the genomic axis's four
    switch_prob: float = 7.1e-5
    window: int = WINDOW


def _labels(cfg, n, rng):
    """Two-state Markov chain per sequence, stationary and symmetric."""
    switches = rng.random((n, cfg.window)) < cfg.switch_prob
    switches[:, 0] = False
    y = np.cumsum(switches, axis=1) % 2
    start = rng.integers(0, 2, size=(n, 1))
    return (y ^ start).astype(np.int8)


def _emissions(cfg, y, rng):
    """x_t ~ N(+/- m/2, I) with ||m|| = delta, signal spread over all channels.

    Spread rather than concentrated in one channel: a single informative
    channel would let a network solve the task without mixing channels at all,
    which is not the regime any of this is about.
    """
    m = np.full(cfg.dim, cfg.delta / sqrt(cfg.dim), dtype=np.float32)
    x = rng.standard_normal((y.shape[0], cfg.dim, cfg.window)).astype(np.float32)
    sign = (2 * y - 1).astype(np.float32)          # (n, window)
    return x + 0.5 * m[None, :, None] * sign[:, None, :]


def bayes_accuracy(cfg, x, y):
    """Exact forward-backward accuracy: the ceiling an optimal decoder reaches.

    Emissions are equal-covariance Gaussians, so the per-position log
    likelihood ratio reduces to a projection, x . m. The chain is symmetric
    with switch probability p, and both states are equally likely a priori.
    """
    m = np.full(cfg.dim, cfg.delta / sqrt(cfg.dim))
    llr = np.einsum("ncl,c->nl", x.astype(np.float64), m)   # log p(x|1) - log p(x|0)
    n, L = llr.shape
    p = cfg.switch_prob
    stay, go = np.log(1 - p), np.log(p) if p > 0 else -np.inf

    em = np.stack([np.zeros_like(llr), llr], axis=2)        # (n, L, 2), up to a constant
    fwd = np.zeros((n, L, 2))
    fwd[:, 0] = np.log(0.5) + em[:, 0]
    for t in range(1, L):
        prev = fwd[:, t - 1]
        fwd[:, t, 0] = np.logaddexp(prev[:, 0] + stay, prev[:, 1] + go) + em[:, t, 0]
        fwd[:, t, 1] = np.logaddexp(prev[:, 0] + go, prev[:, 1] + stay) + em[:, t, 1]
    bwd = np.zeros((n, L, 2))
    for t in range(L - 2, -1, -1):
        nxt = bwd[:, t + 1] + em[:, t + 1]
        bwd[:, t, 0] = np.logaddexp(nxt[:, 0] + stay, nxt[:, 1] + go)
        bwd[:, t, 1] = np.logaddexp(nxt[:, 0] + go, nxt[:, 1] + stay)
    post = fwd + bwd
    return float(((post[:, :, 1] > post[:, :, 0]).astype(np.int8) == y).mean())


class MarkovGaussianAxis:
    name = "synthetic"

    def __init__(self, switch_prob=7.1e-5, dim=4, window=WINDOW):
        self.switch_prob, self.dim, self.window = switch_prob, dim, window

    def levels(self):
        return list(range(len(DELTAS)))

    def sample(self, level, seed, n=EVAL_SEQUENCES) -> Task:
        """Draw n independent sequences. No shared latent structure here --
        unlike the genomic axis, where sequences within a replicate share
        reference panels -- so a split inside one draw would be legitimate.
        Separate seeds are still used for train and evaluation, to keep the two
        axes interchangeable behind the same interface."""
        cfg = Config(delta=DELTAS[level], dim=self.dim,
                     switch_prob=self.switch_prob, window=self.window)
        rng = np.random.default_rng(1 + seed * 100_000 + level * 1_000)
        y = _labels(cfg, n, rng)
        x = _emissions(cfg, y, rng)
        sw = float((np.diff(y.astype(int), axis=1) != 0).sum(axis=1).mean())
        return Task(
            x=x, y=y,
            difficulty=per_position_bayes_error(cfg.delta),
            floor=0.5,
            meta={"axis": self.name, "level": level, "seed": seed,
                  "delta": cfg.delta, "switch_prob": cfg.switch_prob,
                  "switches_per_window": sw,
                  "frac_single_label": float((np.diff(y.astype(int), axis=1) != 0)
                                             .sum(axis=1).__eq__(0).mean()),
                  "bayes_no_context": 1 - per_position_bayes_error(cfg.delta)},
        )

    def train_set(self, level, seed, n=TRAIN_SEQUENCES) -> Task:
        return self.sample(level, seed + 500, n)

    def eval_set(self, level, seed, n=EVAL_SEQUENCES) -> Task:
        return self.sample(level, seed, n)


synthetic = MarkovGaussianAxis()
