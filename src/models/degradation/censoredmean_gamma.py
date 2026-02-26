import torch
import torch.distributions as dist
from torch.distributions import constraints
from torch.special import gammainc


class CensoredMeanGamma(dist.Distribution):
    """
    Distribution with censored mean:

        T* = min(T, cap)

    The distribution itself is unchanged.
    Only the reported mean is modified.
    """

    arg_constraints = {}
    has_rsample = False

    def __init__(self, base_gamma: dist.Gamma, cap: float):
        self.base_dist = base_gamma
        self.cap = cap

        super().__init__(
            batch_shape=base_gamma.batch_shape,
            event_shape=base_gamma.event_shape,
            validate_args=False,
        )

    # Delegate everything except mean

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    def cdf(self, value):
        return self.base_dist.cdf(value)

    @property
    def variance(self):
        return self.base_dist.variance

    @property
    def mean(self):
        """
        Exact censored mean:

            E[min(T, cap)]
        """

        shape = self.base_dist.concentration
        rate = self.base_dist.rate

        x = rate * self.cap

        P_alpha = gammainc(shape, x)
        P_alpha1 = gammainc(shape + 1, x)

        mean = (shape / rate) * P_alpha1 + self.cap * (1 - P_alpha)

        return mean

    @property
    def support(self):
        return self.base_dist.support
