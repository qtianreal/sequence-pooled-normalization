"""What a difficulty axis has to provide.

Every axis produces the same object: a batch of sequences, a per-position
binary label for each, and a difficulty value in units the axis defines. The
measurement code downstream never learns which axis it came from, which is the
point -- if the discriminant phenomenon is a property of learning under varying
difficulty rather than of one task family, it has to be measurable through an
interface that hides the task.

An axis is usable here only if it offers all four of:

    continuous     difficulty set by a scalar, not a handful of presets
    exact labels   ground truth by construction, not annotation
    unlimited data any amount at any point on the axis
    a known floor  a difficulty beyond which nothing is recoverable, so the
                   range spans easy to impossible rather than easy to annoying

The last is the one usually missing. Without it there is no way to tell a
representation that has run out of information from one that is merely
struggling.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


@dataclass
class Task:
    """One sample from a difficulty axis.

    x           (n, c, s) float32 -- n sequences, c channels, s positions
    y           (n, s)    int8    -- per-position binary label
    difficulty  scalar in the axis's own units, larger = harder
    floor       difficulty at or beyond which the axis is uninformative,
                or None where it is not known analytically
    meta        anything the axis wants to record; never read by measurement
    """

    x: np.ndarray
    y: np.ndarray
    difficulty: float
    floor: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.x.ndim != 3:
            raise ValueError(f"x must be (n, c, s), got {self.x.shape}")
        if self.y.ndim != 2:
            raise ValueError(f"y must be (n, s), got {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"x has {self.x.shape[0]} sequences, y has {self.y.shape[0]}")
        if self.x.shape[2] != self.y.shape[1]:
            raise ValueError(f"x has {self.x.shape[2]} positions, y has {self.y.shape[1]}")

    @property
    def n_channels(self) -> int:
        return self.x.shape[1]

    @property
    def base_rate(self) -> float:
        """Fraction of positions in class 1. Far from 0.5 makes accuracy a
        poor summary and the scatter estimates unstable."""
        return float(self.y.mean())


class Axis(Protocol):
    """A difficulty axis. Implementations live in difficulty.axis."""

    name: str

    def levels(self) -> list[float]:
        """The difficulty values this axis is swept over, ascending."""

    def sample(self, level: int, seed: int, n: int) -> Task:
        """Draw a Task at levels()[level].

        Two calls differing only in `seed` must be statistically independent,
        because that is how training and evaluation sets are obtained. An axis
        whose draws share latent structure -- a fixed reference panel, a fixed
        set of source items -- must make one draw span several independent
        instances of it, or a split of the result measures memorisation of the
        shared part instead of difficulty.
        """
