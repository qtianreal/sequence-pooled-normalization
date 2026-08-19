"""Architectures swept over the difficulty axis.

The dilated residual CNN is the one the evidence in IDEA.md comes from, and
reproducing its numbers is what validated the port. The bidirectional
selective SSM is the second arm. The transformer and U-Net are still to come.

Every model exposes `stem`, `blocks` and `head`, applies blocks through
`apply_block` so the residual convention lives in the model rather than in the
measurement code, and accepts a `skip` set so group ablation can remove blocks
without retraining. Where a block is residual, removing it is exactly the
identity map; a non-residual architecture cannot offer that and will need its
own ablation story -- which is the U-Net's problem when it arrives.

Parameter counts are matched across architectures at 226,177, the dilated
CNN's, or the comparison confounds capacity with inductive bias. Depth is the
free variable; WIDTH IS HELD AT 64 for every architecture, because d_eff
counts channels and is bounded above by the width. A d_eff of 40 out of 64 and
one of 40 out of 96 are not the same measurement, so a wider model would make
the central comparison unreadable.

Ported from lai-lowdiv/lai/methods.py and lai-lowdiv/run_ssm.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_PARAMS = 226_177  # the dilated CNN's, which everything else matches


class SequenceLabeller(nn.Module):
    """Stem, a stack of blocks, a per-position head.

    Subclasses provide the three parts and `apply_block`. Everything that
    reads a model -- forward, ablation, per-layer activations -- goes through
    this interface, so measurement code never learns which architecture it is
    looking at.
    """

    def apply_block(self, h, block):
        raise NotImplementedError

    def forward(self, x, skip=()):
        h = self.stem(x)
        for i, block in enumerate(self.blocks):
            if i in skip:
                continue
            h = self.apply_block(h, block)
        return self.head(h).squeeze(1)

    def layer_outputs(self, x, skip=()):
        """Stem output, then the output of each block. Used by measure."""
        h = self.stem(x)
        yield h
        for i, block in enumerate(self.blocks):
            if i not in skip:
                h = self.apply_block(h, block)
            yield h


class PositionwiseGroupNorm(nn.Module):
    """GroupNorm with the length axis removed from the statistics.

    GroupNorm pools over channels *and positions*, so every output position
    depends on a summary of the whole window: a global information path that no
    receptive-field calculation accounts for. This normalises over exactly the
    same channel groups at each position independently, so the only difference
    from nn.GroupNorm is which axes the statistics are taken over. That makes it
    the control for asking what the global path is worth.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups, self.eps = num_groups, eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        n, c, L = x.shape
        h = x.view(n, self.num_groups, c // self.num_groups, L)
        m = h.mean(dim=2, keepdim=True)
        v = h.var(dim=2, keepdim=True, unbiased=False)
        h = (h - m) / torch.sqrt(v + self.eps)
        return h.view(n, c, L) * self.weight + self.bias


class CumulativeLayerNorm(nn.Module):
    """Conv-TasNet's cLN: statistics over channels and time up to t.

    The causal counterpart of gLN, used when the model may not see the future.
    It still pools along the sequence, only one-sidedly, so it should still
    supply a summary -- of the past rather than of the whole utterance. Whether
    that is enough to substitute for reach is the interesting question: a
    one-sided summary can say which label has dominated so far, which is
    exactly the information a long-run-length task needs.
    """

    def __init__(self, num_channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        n, c, L = x.shape
        s = torch.cumsum(x.sum(dim=1), dim=-1)              # (n, L)
        sq = torch.cumsum((x ** 2).sum(dim=1), dim=-1)
        count = (torch.arange(1, L + 1, device=x.device, dtype=x.dtype) * c)
        mean = (s / count)[:, None, :]
        var = (sq / count)[:, None, :] - mean ** 2
        return (x - mean) / torch.sqrt(var.clamp_min(0) + self.eps) * self.weight + self.bias


# Keyed by which axes the statistics are pooled over. The hypothesis is that
# any normalisation including the length axis supplies a window-level summary
# and so substitutes for receptive field; any that excludes it does not. The
# grouping and the affine parameters are held as close to identical as the
# layers allow, so the pooling axes are what differs.
#
# gln and cln are Conv-TasNet's own layers (Luo & Mesgarani 2019), included
# because that model is a dilated TCN whose receptive field is reported and
# reasoned about, and whose recommended non-causal configuration uses gLN.
NORMS = {
    "none": lambda w: nn.Identity(),                        # --
    "positionwise": lambda w: PositionwiseGroupNorm(8, w),  # channels
    "instance": lambda w: nn.InstanceNorm1d(w, affine=True),  # length
    "group": lambda w: nn.GroupNorm(8, w),                  # channels + length
    "gln": lambda w: nn.GroupNorm(1, w),                    # channels + length, one group
    "cln": lambda w: CumulativeLayerNorm(w),                # channels + past time
    # The control for gln: same single group, same affine parameters, statistics
    # taken at each position instead of across the sequence. Differs from gln in
    # the time axis and nothing else.
    "gln_pos": lambda w: PositionwiseGroupNorm(1, w),       # channels, one group
    "batch": lambda w: nn.BatchNorm1d(w),                   # batch + length
}
POOLS_LENGTH = {"instance", "group", "gln", "cln", "batch"}


class DilatedCNN(SequenceLabeller):
    """Residual dilated 1-D convolution stack for per-position labelling.

    `norm` selects what the normalisation statistics are pooled over. The
    published architecture uses "group", which includes the length axis; the
    other settings exist because that choice, not the dilation schedule, turns
    out to determine how much receptive field is worth.
    """

    def __init__(self, in_ch=4, width=64,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256), norm="group"):
        super().__init__()
        self.dilations = tuple(dilations)
        self.norm_kind = norm
        nrm = NORMS[norm]
        self.stem = nn.Conv1d(in_ch, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nrm(width), nn.GELU(),
                nn.Conv1d(width, width, kernel_size=5, padding=2 * d, dilation=d),
                nrm(width), nn.GELU(),
                nn.Conv1d(width, width, kernel_size=1),
            )
            for d in dilations
        )
        self.head = nn.Sequential(
            nrm(width), nn.GELU(), nn.Conv1d(width, 1, kernel_size=1)
        )

    def apply_block(self, h, block):
        return h + block(h)  # residual, so skipping a block is the identity

    @property
    def receptive_field(self) -> int:
        """Positions visible to one output, in units of input positions."""
        return 1 + 4 * sum(self.dilations) + 4


def PlainCNN(in_ch=4, width=64, depth=9):
    """The control: identical stack with every dilation set to 1.

    Not an arm of the comparison. Its receptive field is ~41 positions against
    the dilated stack's 2049, and the LAI group ablation puts dilation-1-only
    accuracy at 0.52 where the d_eff peak sits -- near chance across the whole
    hard half of the axis. A representation carrying no class signal has no
    discriminant structure to measure, so this cannot answer the generality
    question either way. It runs because its predicted failure is informative:
    if the peak needs reach, that is evidence the peak tracks exploitable
    long-range evidence rather than a dilation-schedule artefact.
    """
    return DilatedCNN(in_ch=in_ch, width=width, dilations=(1,) * depth)


class PositionalStem(nn.Module):
    """Input projection plus a fixed sinusoidal positional encoding.

    Attention is permutation-equivariant, so without this the transformer
    cannot tell a switch at position 10 from one at position 3000 -- it would
    be measuring a bag-of-sites model and its null result would say nothing
    about attention. The encoding is absolute and fixed; windows are cropped at
    random offsets, so the model cannot learn anything from absolute position
    itself, only use it to form relative comparisons.
    """

    def __init__(self, in_ch, width, max_len=8192):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, width, kernel_size=5, padding=2)
        pos = torch.arange(max_len).float()[:, None]
        i = torch.arange(0, width, 2).float()[None, :]
        ang = pos / torch.pow(10_000.0, i / width)
        pe = torch.zeros(max_len, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang[:, : width // 2])
        self.register_buffer("pe", pe.t()[None])  # (1, width, max_len)

    def forward(self, x):
        return self.conv(x) + self.pe[:, :, : x.shape[-1]]


class TransformerBlock(nn.Module):
    """Pre-norm block: pooled multi-head self-attention, then a pointwise FFN.

    Attention runs on mean-pooled tokens and is upsampled back, following the
    LAI project's attn-pooled: full 4096-position attention is 16M entries per
    head per sequence, and at batch 32 it exceeds Metal's NDArray limit and
    aborts the process rather than raising. Ancestry tracts span hundreds of
    positions, so 16-fold pooling discards little -- but it does cap the
    resolution at which attention can localise a switch, which is why the
    pooling factor is reported alongside the result.
    """

    def __init__(self, width, heads=4, pool=16, ff_mult=2, norm="positionwise"):
        super().__init__()
        self.heads, self.pool = heads, pool
        self.norm1 = NORMS[norm](width)
        self.qkv = nn.Conv1d(width, 3 * width, kernel_size=1)
        self.proj = nn.Conv1d(width, width, kernel_size=1)
        self.norm2 = NORMS[norm](width)
        self.ff = nn.Sequential(
            nn.Conv1d(width, ff_mult * width, kernel_size=1), nn.GELU(),
            nn.Conv1d(ff_mult * width, width, kernel_size=1),
        )

    def forward(self, x):
        n, c, L = x.shape
        h = self.norm1(x)
        if self.pool > 1:
            h = F.avg_pool1d(h, self.pool)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        shape = lambda t: t.view(n, self.heads, c // self.heads,
                                 t.shape[-1]).transpose(2, 3)
        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v))
        a = a.transpose(2, 3).reshape(n, c, -1)
        if self.pool > 1:
            a = F.interpolate(a, size=L, mode="linear", align_corners=False)
        x = x + self.proj(a)
        return x + self.ff(self.norm2(x))


class Transformer(SequenceLabeller):
    """Standalone transformer encoder for per-position labelling.

    Defaults to per-token normalisation, as transformers are actually built.
    Reach is global at every depth here, so this architecture does not take
    part in the reach sweep; it answers the complementary question of whether
    normalisation matters at all once the architecture already supplies the
    context that normalisation would otherwise smuggle in.
    """

    def __init__(self, in_ch=4, width=64, depth=9, heads=4, pool=16, ff_mult=1,
                 norm="positionwise"):
        super().__init__()
        self.width, self.depth, self.pool = width, depth, pool
        self.norm_kind = norm
        self.stem = PositionalStem(in_ch, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads=heads, pool=pool, ff_mult=ff_mult,
                             norm=norm)
            for _ in range(depth)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(), nn.Conv1d(width, 1, kernel_size=1)
        )

    def apply_block(self, h, block):
        return block(h)  # the residuals are inside TransformerBlock


class TasNetBlock(nn.Module):
    """Conv-TasNet separator block: 1x1, depthwise dilated conv, 1x1, residual.

    Follows Luo & Mesgarani (2019): bottleneck B channels expand to H, a
    depthwise convolution at dilation d carries the temporal extent, and a 1x1
    projects back. Normalisation sits after each PReLU, which is where gLN goes
    in the published model.
    """

    def __init__(self, bottleneck, hidden, dilation, kernel=3, norm="gln"):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(bottleneck, hidden, kernel_size=1),
            nn.PReLU(), NORMS[norm](hidden),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad,
                      dilation=dilation, groups=hidden),
            nn.PReLU(), NORMS[norm](hidden),
            nn.Conv1d(hidden, bottleneck, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class TasNetTCN(SequenceLabeller):
    """The Conv-TasNet separator, as a per-position labeller.

    Real audio separation is not what this runs on; the task is the controlled
    sequence-labelling axis, where the optimum is known. What it borrows is the
    architecture and the normalisation of a deployed model whose receptive
    field is explicitly reported and reasoned about, so the question "how much
    of that reach is really normalisation" can be asked of the thing people
    build rather than of a stand-in.
    """

    def __init__(self, in_ch=4, width=64, hidden=184,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256), norm="gln"):
        super().__init__()
        self.dilations = tuple(dilations)
        self.norm_kind, self.hidden = norm, hidden
        self.stem = nn.Conv1d(in_ch, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            TasNetBlock(width, hidden, d, norm=norm) for d in dilations
        )
        self.head = nn.Sequential(
            NORMS[norm](width), nn.PReLU(), nn.Conv1d(width, 1, kernel_size=1)
        )

    def apply_block(self, h, block):
        return h + block(h)

    @property
    def receptive_field(self) -> int:
        """Kernel 3 depthwise, so each block adds 2*d rather than the dilated
        CNN's 4*d. Reported separately because the reach sweep's x-axis is
        receptive field, not block count."""
        return 1 + 2 * sum(self.dilations) + 4


def pscan(a, b):
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t along the last dim.

    Composing (A,B) at t with (A,B) at t-d gives h_t = A_t A_{t-d} h_{t-2d} +
    (B_t + A_t B_{t-d}), so B must be updated before A. h_{-1} = 0, hence the
    identity padding (1 for A, 0 for B).

    Hillis-Steele: log2(L) steps, 12 for a 4096-position window, against 4096
    sequential ones for a recurrent net. Verified against a sequential
    reference in test_pscan().
    """
    L = a.shape[-1]
    d = 1
    while d < L:
        a_sh = F.pad(a[..., :-d], (d, 0), value=1.0)
        b_sh = F.pad(b[..., :-d], (d, 0), value=0.0)
        b = b + a * b_sh
        a = a * a_sh
        d *= 2
    return b


class MambaBlock(nn.Module):
    """Minimal selective SSM (S6) block, bidirectional, residual.

    Selective is the part that matters: dt, B and C are computed from the
    input at every position, which is what separates S6 from S4-style SSMs
    where the layer collapses to a fixed long convolution.

    Bidirectional is required, not a refinement. Per-position labelling has
    informative context on both sides, and the arms this is compared against
    are all non-causal, so a causal SSM would be handicapped by construction
    and a null result from it would mean nothing.
    """

    def __init__(self, width, d_state=8, d_conv=4):
        super().__init__()
        self.d_state = d_state
        self.norm = nn.GroupNorm(8, width)
        self.in_proj = nn.Conv1d(width, 2 * width, kernel_size=1)  # x and gate z
        self.conv = nn.Conv1d(width, width, kernel_size=d_conv,
                              padding=d_conv - 1, groups=width)
        self.d_conv = d_conv
        # Selective parameters: per-timestep step size, input and output maps.
        self.x_proj = nn.Conv1d(width, 2 * d_state + 1, kernel_size=1)
        self.dt_bias = nn.Parameter(torch.zeros(width))
        # A is learned through log to keep the decay exp(dt*A) inside (0,1).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .repeat(width, 1))
        self.D = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Conv1d(width, width, kernel_size=1)
        # Zero-init the output projection: the block is exactly the identity at
        # initialisation, so a deep stack starts well-conditioned.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def scan(self, x, dt, B, C):
        """x,dt: (N,C,L); B,C: (N,S,L) -> (N,C,L)."""
        A = -torch.exp(self.A_log)[None, :, :, None]        # (1,C,S,1)
        dA = torch.exp(dt[:, :, None, :] * A)               # (N,C,S,L)
        dBx = dt[:, :, None, :] * B[:, None, :, :] * x[:, :, None, :]
        h = pscan(dA, dBx)                                  # (N,C,S,L)
        return (h * C[:, None, :, :]).sum(dim=2)            # (N,C,L)

    def forward(self, u):
        n, c, L = u.shape
        h = self.norm(u)
        x, z = self.in_proj(h).chunk(2, dim=1)
        x = F.silu(self.conv(x)[..., :L])
        p = self.x_proj(x)
        dt = F.softplus(p[:, :1] + self.dt_bias[None, :, None])  # (N,C,L)
        B, C = p[:, 1:1 + self.d_state], p[:, 1 + self.d_state:]

        y = self.scan(x, dt, B, C)
        rev = lambda t: torch.flip(t, dims=[-1])
        y = y + rev(self.scan(rev(x), rev(dt), rev(B), rev(C)))
        y = y + self.D[None, :, None] * x
        return u + self.out_proj(y * F.silu(z))


class BiSSM(SequenceLabeller):
    """Stack of bidirectional selective SSM blocks.

    Reach is unbounded by construction -- state is carried the length of the
    sequence -- so depth is chosen for parameter matching rather than for
    receptive field, and is free to differ from the CNN's nine.

    The scan materialises a (batch, width, d_state, length) tensor and autograd
    keeps one per scan step, so memory binds well before compute: at width 64,
    d_state 8, length 4096 and batch 32 that is ~270 MB per tensor and twelve
    steps per direction per block. Prefer few wide blocks with small d_state,
    and probe before committing to a long run.
    """

    def __init__(self, in_ch=4, width=64, depth=17, d_state=1, d_conv=4):
        super().__init__()
        self.width, self.depth, self.d_state = width, depth, d_state
        self.stem = nn.Conv1d(in_ch, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            MambaBlock(width, d_state=d_state, d_conv=d_conv) for _ in range(depth)
        )
        self.head = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(), nn.Conv1d(width, 1, kernel_size=1)
        )

    def apply_block(self, h, block):
        return block(h)  # the residual is inside MambaBlock


def test_pscan(device="cpu"):
    """The parallel scan must match a sequential reference."""
    torch.manual_seed(0)
    a = torch.rand(2, 3, 64, device=device) * 0.9 + 0.05
    b = torch.randn(2, 3, 64, device=device)
    ref = torch.zeros_like(b)
    h = torch.zeros(2, 3, device=device)
    for t in range(64):
        h = a[..., t] * h + b[..., t]
        ref[..., t] = h
    err = (pscan(a, b) - ref).abs().max().item()
    assert err < 1e-4, f"scan mismatch {err}"
    return err


def match_params(factory, target=TARGET_PARAMS, candidates=range(1, 65)):
    """Pick the depth whose parameter count lands closest to `target`."""
    best = min(candidates, key=lambda d: abs(n_parameters(factory(d)) - target))
    return best, n_parameters(factory(best))


def n_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def fit(model, x, y, device, epochs=15, lr=1e-3, batch=32, seed=0):
    """Train to convergence on one difficulty level.

    Held-out accuracy plateaus by epoch 3 at the settings inherited from the
    LAI sweep; 15 is kept so the comparison across architectures is not one of
    who converges fastest.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype("float32"))
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(xt.shape[0])
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            lossf(model(xt[idx].to(device)), yt[idx].to(device)).backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def accuracy(model, x, y, device, batch=64, skip=()) -> float:
    model.eval()
    correct = total = 0
    xt = torch.from_numpy(x)
    for i in range(0, xt.shape[0], batch):
        logits = model(xt[i : i + batch].to(device), skip=skip).cpu().numpy()
        correct += ((logits > 0).astype("int8") == y[i : i + batch]).sum()
        total += y[i : i + batch].size
    return float(correct / total)


class UNet(SequenceLabeller):
    """Encoder-decoder with skip connections, for the one family whose
    normalisation extent is not fixed by the architecture.

    Every other model here pools its statistics over a sequence of one length.
    A U-Net does not: at depth d the sequence has been halved d times, so a
    normalisation layer there pools over L/2^d positions, but each of those
    positions summarises 2^d input positions. The extent in *input* positions
    is therefore the same at every level, while the number of values the
    statistic is taken over falls geometrically. Section 3 says the channel's
    strength per position is O(1/|S|), so a U-Net is where the criterion and
    the magnitude come apart most, and it is the family the prior work names as
    the practical risk.

    Reach is set by `depth` rather than by a dilation schedule: each additional
    level doubles the span the convolutions cover. The receptive field is not
    derived here but measured by autograd in run_unet.py, since an off-by-one
    in a hand-derived recursion through down- and up-sampling would be silent.

    Blocks are not residual, so removing one is not the identity and this model
    has no block-ablation story. `skip` is accepted only to satisfy the shared
    interface and must be empty.
    """

    def __init__(self, in_ch=4, width=64, depth=4, norm="group"):
        super().__init__()
        self.depth, self.norm_kind = depth, norm
        nrm = NORMS[norm]
        conv = lambda: nn.Sequential(
            nrm(width), nn.GELU(), nn.Conv1d(width, width, 5, padding=2))
        self.stem = nn.Conv1d(in_ch, width, kernel_size=5, padding=2)
        self.down = nn.ModuleList(conv() for _ in range(depth))
        self.bottom = conv()
        self.up = nn.ModuleList(conv() for _ in range(depth))
        # After concatenating the skip, width doubles; project back down.
        self.fuse = nn.ModuleList(
            nn.Conv1d(2 * width, width, kernel_size=1) for _ in range(depth))
        self.head = nn.Sequential(
            nrm(width), nn.GELU(), nn.Conv1d(width, 1, kernel_size=1))
        self.blocks = nn.ModuleList()      # no residual blocks to ablate

    def forward(self, x, skip=()):
        if skip:
            raise ValueError("UNet blocks are not residual; skipping one is "
                             "not the identity, so block ablation is undefined")
        h = self.stem(x)
        saved = []
        for block in self.down:
            h = block(h)
            saved.append(h)
            h = F.avg_pool1d(h, 2)
        h = self.bottom(h)
        for block, fuse, s in zip(self.up, self.fuse, reversed(saved)):
            h = F.interpolate(h, size=s.shape[-1], mode="nearest")
            h = fuse(torch.cat([h, s], dim=1))
            h = block(h)
        return self.head(h).squeeze(1)

    def layer_outputs(self, x, skip=()):
        raise NotImplementedError("UNet is not on the d_eff path")
