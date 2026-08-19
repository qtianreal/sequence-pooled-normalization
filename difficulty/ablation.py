"""Group ablation: how much of the task rests on long-range integration.

Blocks are removed from the long-dilation end and the model re-evaluated
without retraining. Because each block enters as h <- h + f(h), removal is
exactly the identity map, so no surgery or fine-tuning is involved and the
difference is attributable to the removed range alone.

Single-unit ablation is uninformative here and must not be substituted. With
residual connections the remaining blocks compensate for any one removal, so
per-block importance reads near zero even where the blocks collectively
matter. That mistake cost a reversed conclusion in the LAI work; it is a
general hazard of using pruning as measurement rather than as compression.

Accuracy is bounded below by chance, so raw drops are not comparable across
difficulty levels -- a model barely above chance cannot lose much. Retention is
therefore reported as the share of *above-chance* accuracy that survives, and
is meaningless once the denominator approaches zero.

Ported from lai-lowdiv/run_pruning_seeds.py.
"""

import numpy as np

from difficulty.models import accuracy

CHANCE = 0.5
MIN_HEADROOM = 0.20  # above-chance accuracy below which retention is not read


def retention(pruned: float, full: float) -> float:
    """Share of above-chance accuracy retained. NaN when unreadable."""
    head = full - CHANCE
    if head < MIN_HEADROOM:
        return float("nan")
    return float((pruned - CHANCE) / head)


def group_ablation(model, x, y, device, thresholds=(8, 4, 2)) -> dict:
    """Remove every block with dilation >= t, for each t.

    Returns full accuracy plus, per threshold, the ablated accuracy, the
    retention, and which blocks were removed.
    """
    full = accuracy(model, x, y, device)
    out = {"full": full, "readable": (full - CHANCE) >= MIN_HEADROOM, "groups": {}}
    for t in thresholds:
        skip = {i for i, d in enumerate(model.dilations) if d >= t}
        acc = accuracy(model, x, y, device, skip=skip)
        out["groups"][t] = {
            "accuracy": acc,
            "retention": retention(acc, full),
            "removed": sorted(skip),
            "dilations_removed": [d for d in model.dilations if d >= t],
        }
    return out
