"""
Capped distribution: Y = min(X, cap)
No integrated with the rest of the codevase yet.
"""

import torch
import torch.distributions as dist


class CappedDistribution(dist.Distribution):
    """
    Y = min(X, cap)

    Continuous density below cap
    Discrete mass at cap equal to P(X >= cap)
    """

    has_rsample = True

    def __init__(self, base_dist: dist.Distribution, cap: float):
        self.base = base_dist
        self.cap = torch.as_tensor(
            cap,
            dtype=base_dist.mean.dtype,
            device=base_dist.mean.device,
        )

        super().__init__(
            batch_shape=base_dist.batch_shape,
            event_shape=base_dist.event_shape,
            validate_args=False,
        )

    # ============================================================
    # log_prob
    # ============================================================
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        base_logp = self.base.log_prob(value)

        # continuous region
        logp = torch.where(
            value < self.cap,
            base_logp,
            torch.full_like(value, -torch.inf),
        )

        # discrete mass at cap
        tail_mass = (1.0 - self.base.cdf(self.cap)).clamp_min(1e-12)
        log_mass = torch.log(tail_mass)

        logp = torch.where(value == self.cap, log_mass, logp)

        return logp

    # ============================================================
    # MEAN (closed form)
    # ============================================================
    @property
    def mean(self):

        # Tail probability
        tail_mass = (1.0 - self.base.cdf(self.cap)).clamp_min(1e-12)

        # If tail probability is negligible → mean ≈ base mean
        if torch.all(tail_mass < 1e-8):
            return self.base.mean

        # integrate small tail region
        n_points = 100
        grid = torch.linspace(0, 200, n_points, device=self.cap.device)

        dx = grid[1] - grid[0]

        grid_exp = grid.unsqueeze(-1)

        pdf = torch.exp(self.base.log_prob(grid_exp))

        excess = (grid_exp - self.cap) * pdf
        excess_expect = excess.sum(dim=0) * dx

        return self.base.mean - excess_expect

    # ============================================================
    # VARIANCE (approximate but stable)
    # ============================================================
    @property
    def variance(self):

        # simple stable approximation
        # Var[Y] ≈ Var[X] * P(X <= cap)

        tail_mass = 1.0 - self.base.cdf(self.cap)
        cont_mass = 1.0 - tail_mass

        return self.base.variance * cont_mass

    # ============================================================
    # SAMPLE
    # ============================================================
    def sample(self, sample_shape=torch.Size()):
        x = self.base.sample(sample_shape)
        return torch.minimum(x, self.cap)

    def rsample(self, sample_shape=torch.Size()):
        x = self.base.rsample(sample_shape)
        return torch.minimum(x, self.cap)
