"""Check Equation 1 of the paper against automatic differentiation.

The paper's scope conditions all follow from one expression for a normalisation
layer's Jacobian,

    dy_t/dx_s = (gamma / sigma_S) [ delta_ts - (1 + xhat_t xhat_s) / |S| ],

so that expression had better be right, and the claims read off it -- that
BatchNorm's off-diagonal terms vanish at evaluation, that cLN's vanish only for
the future -- had better be true of the layers as implemented rather than as
described. This builds the full Jacobian by autograd on a small input, where
that is affordable, and compares.

    python run_jacobian.py      ->  results/jacobian_check.json

Small on purpose: the Jacobian is |S| x |S|, so C=4, L=64 gives a 256 x 256
matrix. The identity being checked does not depend on the size.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from difficulty.models import CumulativeLayerNorm

OUT = Path("results")
C, L = 4, 64
S = C * L


def jacobian(layer, x):
    return torch.autograd.functional.jacobian(
        lambda v: layer(v).reshape(-1), x, vectorize=True).reshape(S, S)


def main():
    torch.manual_seed(0)
    x = torch.randn(1, C, L, dtype=torch.double, requires_grad=True)
    off = ~torch.eye(S, dtype=bool)
    out = {"channels": C, "positions": L, "set_size": S}

    # gLN: one group, statistics over channels and the whole length axis.
    gln = nn.GroupNorm(1, C, eps=0).double()
    gln.weight.data.fill_(1.0)
    gln.bias.data.zero_()
    J = jacobian(gln, x)
    sd = x.std(unbiased=False).detach()
    xh = ((x - x.mean()) / sd).reshape(-1).detach()
    closed = (torch.eye(S, dtype=torch.double)
              - (1 + torch.outer(xh, xh)) / S) / sd
    out["gln"] = {
        "max_abs_error": float((J - closed).abs().max()),
        "mean_abs_offdiag": float(J[off].abs().mean()),
        "one_over_S": 1.0 / S,
        "row_sum_offdiag": float(J[0][off[0]].sum()),
        "diagonal": float(J[0, 0]),
    }

    # BatchNorm at evaluation: running statistics are constants, so the layer is
    # pointwise and every off-diagonal term is identically zero.
    bn = nn.BatchNorm1d(C, eps=0).double()
    bn.weight.data.fill_(1.0)
    bn.bias.data.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0)
    bn.eval()
    out["batchnorm_eval"] = {
        "max_abs_offdiag": float(jacobian(bn, x)[off].abs().max())}

    # cLN: statistics over channels and time up to t. Future terms must vanish
    # exactly (it is causal); past terms must not (it is not local).
    cln = CumulativeLayerNorm(C, eps=1e-12).double()
    cln.weight.data.fill_(1.0)
    cln.bias.data.zero_()
    Jc = jacobian(cln, x).reshape(C, L, C, L)
    out["cln"] = {
        "max_abs_future": float(max(Jc[:, t, :, t + 1:].abs().max()
                                    for t in range(L - 1))),
        "max_abs_past": float(max(Jc[:, t, :, :t].abs().max()
                                  for t in range(2, L))),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "jacobian_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'jacobian_check.json'}")


if __name__ == "__main__":
    main()
