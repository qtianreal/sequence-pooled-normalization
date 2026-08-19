"""Does the pooled mean literally carry pi-bar, or only plausibly?

Section 3 calls the window's class proportion "what a normalisation statistic
taken along the sequence supplies to first order" without deriving why. For the
Gaussian-Markov process the emission model gives an exact answer: since the
signal is spread evenly across channels (m_c = delta/sqrt(C) for every c), the
mean pooled over channels and the window is

    mu_S = delta/(2*sqrt(C)) * (2*pi_bar - 1) + eta,   eta ~ N(0, 1/(C*L))

with nothing else contributing. This checks that identity against the same
draws run_synth.py trains on, rather than asserting it.

    python run_signal_check.py     ->  results/signal_check.json

Matches synth_dilated_cnn.json's config exactly (delta=0.08, the same seven
switch densities, three seeds, 512 evaluation sequences) so the two are
directly comparable.
"""

import json
from pathlib import Path

import numpy as np

from difficulty.axis.synthetic import DELTAS, MarkovGaussianAxis

OUT = Path("results")
DELTA = DELTAS[3]  # 0.08, matched to synth_dilated_cnn.json
SWITCH_PROBS = [2.442002442002442e-05, 7.326007326007326e-05,
                0.0002442002442002442, 0.0007326007326007326,
                0.0014652014652014652, 0.002442002442002442,
                0.004884004884004884]
SEEDS = (0, 1, 2)
N = 512


def main():
    rows = []
    for p in SWITCH_PROBS:
        axis = MarkovGaussianAxis(switch_prob=p)
        for seed in SEEDS:
            t = axis.eval_set(level=3, seed=seed, n=N)
            C, L = t.x.shape[1], t.x.shape[2]
            pi_bar = t.y.mean(axis=1)
            mu_S = t.x.mean(axis=(1, 2))
            pred = (DELTA / (2 * np.sqrt(C))) * (2 * pi_bar - 1)
            resid = mu_S - pred
            rows.append({
                "switch_prob": p, "seed": seed,
                "switches_per_window": float((np.diff(t.y.astype(int), axis=1)
                                              != 0).sum(axis=1).mean()),
                "channels": C, "window": L,
                "corr_mu_pred": float(np.corrcoef(mu_S, pred)[0, 1]),
                "resid_std": float(resid.std()),
                "predicted_noise_std": float(1 / np.sqrt(C * L)),
                "max_abs_resid": float(np.abs(resid).max()),
            })

    OUT.mkdir(exist_ok=True)
    (OUT / "signal_check.json").write_text(json.dumps(rows, indent=2))
    ratios = [r["resid_std"] / r["predicted_noise_std"] for r in rows]
    print(f"{len(rows)} cells over {len(SWITCH_PROBS)} switch densities")
    print(f"  resid_std / predicted_noise_std: {min(ratios):.3f} to {max(ratios):.3f}")
    print(f"  correlation with pi-bar: {min(r['corr_mu_pred'] for r in rows):.3f} "
          f"to {max(r['corr_mu_pred'] for r in rows):.3f} (falls as switches densify, "
          f"as it must: the signal term shrinks while the noise floor does not)")
    print(f"\nwrote {OUT / 'signal_check.json'}")


if __name__ == "__main__":
    main()
