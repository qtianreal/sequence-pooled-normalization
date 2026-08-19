"""Discriminant structure of a representation.

Three quantities per layer, computed on activations pooled over sequences and
positions:

    J        trace((S_W + ridge I)^-1 S_B), how much class-separating signal
             the representation carries at all
    d_eff    participation ratio of the per-channel Fisher ratios, how widely
             that signal is spread across channels
    n90      channels needed to reach 90% of the summed Fisher ratio, a
             discrete companion to d_eff that is easier to interpret

A formulation error worth not repeating. The eigen-spectrum of
(S_W + ridge I)^-1 S_B cannot give effective dimensionality in a two-class
problem: S_B has rank one there, so the participation ratio of its eigenvalues
is identically 1 whatever the representation does. The per-channel Fisher ratio
is the working substitute and is what d_eff and n90 use below. With more than
two classes the eigen-spectrum becomes usable and is the better measure --
worth revisiting when a multi-class axis is added.

Ported from lai-lowdiv/run_lda.py.
"""

import numpy as np
import torch

RIDGE = 1e-3  # relative ridge on S_W, which is rank-deficient otherwise


def layer_activations(model, x, device, batch=16, max_positions=2048):
    """Stem output and every residual block output, subsampled over positions.

    Returns a list of (n, channels, positions) arrays, stem first.
    """
    acts = [[] for _ in range(len(model.blocks) + 1)]
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = torch.from_numpy(x[i : i + batch].astype(np.float32)).to(device)
            # Through the model's own iterator: the residual convention differs
            # by architecture -- the CNN's blocks return f(h) and are added
            # here, an SSM block returns h + f(h) itself -- and applying the
            # wrong one silently doubles the skip path.
            for j, h in enumerate(model.layer_outputs(xb)):
                acts[j].append(h[:, :, :max_positions].cpu().numpy())
    return [np.concatenate(a) for a in acts]


def discriminant_stats(act: np.ndarray, labels: np.ndarray) -> dict:
    """J, d_eff, n90 and the largest per-channel Fisher ratio.

    act    (n, channels, positions)
    labels (n, positions) binary
    """
    n, c0, s = act.shape
    X = act.transpose(0, 2, 1).reshape(-1, c0).astype(np.float64)
    y = labels[:, :s].reshape(-1).astype(np.int8)

    # A diverged model produces non-finite activations. Standardising them
    # yields NaN, which then propagates silently through the scatter matrices
    # and out as a plausible-looking number, so it is reported instead.
    if not np.isfinite(X).all():
        return dict(J=float("nan"), d_eff=float("nan"), n90=-1,
                    fisher_max=float("nan"), channels=0, dropped=c0,
                    finite=False)

    # Residual accumulation leaves channels on very different scales, which
    # overflows the scatter accumulation. trace(S_W^-1 S_B) is invariant to
    # per-channel rescaling, so standardising is numerically necessary and
    # analytically free.
    X -= X.mean(axis=0)
    sd = X.std(axis=0)
    # Dead channels carry no class information and would make S_W singular
    # beyond what the ridge is there for. Dropping them changes neither J nor
    # d_eff -- a constant channel contributes zero to both sums -- but it is
    # recorded, because a layer that is mostly dead is a finding, not a detail.
    live = sd > 0
    c = int(live.sum())
    if c == 0:
        return dict(J=0.0, d_eff=float("nan"), n90=-1, fisher_max=0.0,
                    channels=0, dropped=c0, finite=True)
    X = X[:, live] / sd[live]

    mu = X.mean(axis=0)
    S_B = np.zeros((c, c))
    S_W = np.zeros((c, c))
    means, varis = [], []

    for k in (0, 1):
        m = y == k
        nk = int(m.sum())
        if nk < 2:
            return dict(J=float("nan"), d_eff=float("nan"), n90=-1,
                        fisher_max=float("nan"), channels=c,
                        dropped=c0 - c, finite=True)
        Xk = X[m]
        muk = Xk.mean(axis=0)
        d = (muk - mu)[:, None]
        S_B += nk * (d @ d.T)
        Xc = Xk - muk
        S_W += Xc.T @ Xc
        means.append(muk)
        varis.append(Xk.var(axis=0))

    S_B /= len(y)
    S_W /= len(y)

    ridge = RIDGE * np.trace(S_W) / c
    M = np.linalg.solve(S_W + ridge * np.eye(c), S_B)
    J = float(np.clip(np.linalg.eigvals(M).real, 0, None).sum())

    f = (means[0] - means[1]) ** 2 / (varis[0] + varis[1] + 1e-12)
    d_eff = float(f.sum() ** 2 / (f ** 2).sum()) if (f ** 2).sum() > 0 else float("nan")
    order = np.sort(f)[::-1]
    csum = np.cumsum(order) / max(order.sum(), 1e-12)
    n90 = int(np.searchsorted(csum, 0.90) + 1)

    return dict(J=J, d_eff=d_eff, n90=n90, fisher_max=float(f.max()),
                channels=c, dropped=c0 - c, finite=True)


MAX_SEQUENCES = 64  # sequences the statistics are computed on


def profile(model, x, labels, device, max_sequences=MAX_SEQUENCES) -> list[dict]:
    """discriminant_stats for every layer, stem first.

    Only the first `max_sequences` sequences are used, matching the original,
    where 64 was one evaluation replicate. 64 sequences x 2048 positions is
    131k rows for a 64x64 scatter matrix, which is ample; the whole evaluation
    set would hold every layer's activations in memory at once for no gain in
    precision. Since a draw is ordered by replicate, this reads one replicate --
    seed-to-seed spread, not within-draw spread, is what the SDs in the
    reference table describe.
    """
    x, labels = x[:max_sequences], labels[:max_sequences]
    return [discriminant_stats(a, labels) for a in layer_activations(model, x, device)]
