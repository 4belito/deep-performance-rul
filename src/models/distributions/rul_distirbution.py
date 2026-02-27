"""
Wrapper around a base RUL distribution with capped statistics and
discrete mode estimation.

The underlying probability law (log_prob, cdf, sampling) is unchanged.
This wrapper modifies summary statistics:

- The mean is clipped to [0, cap].
- The mode is estimated via Monte Carlo sampling and integer
  histogram search within a high-quantile window.
- The estimated mode is cached for efficiency.
"""

import torch
import torch.distributions as dist


class RULDistributionWrapper(dist.Distribution):

    arg_constraints = {}
    has_rsample = False

    def __init__(
        self,
        base_dist: dist.Distribution,
        cap: int = 100,
        n_samples: int = 4096,
        quantile_search: float = 0.999,
    ):
        self.base_dist = base_dist
        self.cap = cap
        self.n_samples = n_samples
        self.quantile_search = quantile_search

        self._cached_mode = None  # <--- cache

        super().__init__(
            batch_shape=base_dist.batch_shape,
            event_shape=base_dist.event_shape,
        )

    # delegate
    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    def cdf(self, value):
        return self.base_dist.cdf(value)

    @property
    def mean(self):
        return torch.clip(self.base_dist.mean, min=0.0, max=self.cap)

    @property
    def variance(self):
        return self.base_dist.variance

    @property
    @torch.no_grad()
    def mode(self):

        if self._cached_mode is not None:
            return self._cached_mode

        samples = self.base_dist.sample((self.n_samples,))

        B = samples.shape[1]
        device = samples.device
        modes = torch.zeros(B, device=device)

        # symmetric quantile window
        q_low = torch.quantile(samples, 1.0 - self.quantile_search, dim=0)
        q_high = torch.quantile(samples, self.quantile_search, dim=0)

        min_int = torch.floor(q_low + 0.5).long()
        max_int = torch.floor(q_high + 0.5).long()

        for b in range(B):

            samples_int = torch.floor(samples[:, b] + 0.5).long()

            samples_int = samples_int.clamp(
                min=min_int[b].item(),
                max=max_int[b].item(),
            )

            # shift to zero for bincount
            shifted = samples_int - min_int[b]

            counts = torch.bincount(
                shifted,
                minlength=(max_int[b] - min_int[b] + 1).item(),
            )

            idx = counts.argmax().item()

            mode_val = min_int[b] + idx
            modes[b] = min(mode_val, self.cap)

        self._cached_mode = modes.float()
        return self._cached_mode
