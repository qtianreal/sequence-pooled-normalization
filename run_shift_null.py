"""How much did the circular-shift null actually destroy?

The first null tried against the d_eff premise rolled each window's label vector
and recomputed the statistic. It agreed closely with the observed value, which
looked like a finding and was an artefact: a circular shift is exactly the
identity on a window whose label never changes, and most of these windows are
like that. This measures the two quantities that say so, on the same cached
label vectors run_null.py uses, so the appendix can report them rather than
recall them.

    python run_shift_null.py     ->  results/shift_null.json

The single-label fraction is exact and does not depend on the shift. The
fraction of positions whose label a shift changes does, so it is averaged over
shifts drawn uniformly at random, which is the null as a procedure rather than
one arbitrary instance of it.

Reads the LAI project's cached evaluation sets, as run_null.py does; the result
file is committed so the paper does not depend on that checkout.
"""

import json
import os
from pathlib import Path

import numpy as np

# The d_eff appendix re-reads the LAI project's cached models and evaluation
# sets: those are 261 MB of derived artefacts belonging to that project, and
# re-deriving them here would mean retraining its networks. Declared rather
# than assumed, so a missing checkout fails with a sentence instead of a
# FileNotFoundError on a path nobody chose.
CACHE = Path(os.environ.get("LAI_CACHE",
                            Path.home() / "lai-lowdiv/results/cache"))
OUT = Path("results")
SPLIT_TIMES = [3200, 1600, 800, 400, 200, 100, 50, 25]
SEEDS = (0, 1, 2)
N = 64      # sequences per draw, matching run_null.py
DRAWS = 20  # random shifts averaged over


def main():
    rng = np.random.default_rng(0)
    rows = []
    for T in SPLIT_TIMES:
        for s in SEEDS:
            path = CACHE / f"test_T{T}_s{s}.npz"
            if not path.exists():
                continue
            y = np.load(path)["y"][:N].astype(np.int8)
            n_seq, L = y.shape

            # A circular shift cannot alter a window whose label is constant.
            constant = (np.diff(y.astype(int), axis=1) != 0).sum(axis=1) == 0
            identical, changed = [], []
            for _ in range(DRAWS):
                k = rng.integers(1, L, size=n_seq)
                rolled = np.stack([np.roll(y[i], int(k[i])) for i in range(n_seq)])
                identical.append(float((rolled == y).all(axis=1).mean()))
                changed.append(float((rolled != y).mean()))
            rows.append({
                "split_time": T, "seed": s, "sequences": int(n_seq),
                "frac_single_label": float(constant.mean()),
                "frac_vectors_unchanged": float(np.mean(identical)),
                "frac_labels_changed": float(np.mean(changed)),
            })

    OUT.mkdir(exist_ok=True)
    (OUT / "shift_null.json").write_text(json.dumps(rows, indent=2))
    sl = np.mean([r["frac_single_label"] for r in rows])
    un = np.mean([r["frac_vectors_unchanged"] for r in rows])
    ch = np.mean([r["frac_labels_changed"] for r in rows])
    print(f"{len(rows)} cells over {len(SPLIT_TIMES)} split times")
    print(f"  windows carrying a single label      {sl:.3f}")
    print(f"  label vectors a shift leaves intact  {un:.3f}")
    print(f"  positions whose label a shift alters {ch:.3f}")
    print(f"\nwrote {OUT / 'shift_null.json'}")


if __name__ == "__main__":
    main()
