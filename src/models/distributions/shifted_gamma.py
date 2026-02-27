"""
Gamma distribution with additive shift.

Represents X = shift + Y, where Y ~ Gamma(concentration, rate).
Mean and mode are shifted accordingly; variance is unchanged.
"""

import torch
from torch.distributions import Gamma, TransformedDistribution, constraints
from torch.distributions.transforms import AffineTransform


class ShiftedGamma(TransformedDistribution):
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "shift": constraints.real,
    }

    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        shift: torch.Tensor,
        validate_args: bool | None = None,
    ):
        self.concentration = concentration
        self.rate = rate
        self.shift = shift

        base_dist = Gamma(
            concentration=concentration,
            rate=rate,
        )

        transform = AffineTransform(loc=shift, scale=1.0)

        super().__init__(
            base_dist,
            [transform],
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return self.shift + self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    @property
    def base_mode(self):
        shape = self.concentration
        rate = self.rate
        return torch.where(shape >= 1, (shape - 1) / rate, torch.zeros_like(shape))

    @property
    def mode(self):
        return self.shift + self.base_mode
