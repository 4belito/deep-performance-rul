import abc

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from matplotlib import pyplot as plt


class DEGModel(nn.Module, abc.ABC):
    """
    Abstract base class for degradation models.

    forward(s) must return a tensor of shape [..., P],
    where parameters are encoded along the last dimension.
    """
    distribution_class: type[dist.Distribution]
    
    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        assert self.distribution_class is not None
        return self.build_distribution(self.forward(s))
    
    @abc.abstractmethod
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Return raw distribution parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_distribution(self, params: torch.Tensor) -> dist.Distribution:
        raise NotImplementedError

    @abc.abstractmethod
    def plot_distribution(
        self,
        t: np.ndarray,
        s: np.ndarray,
        func: str,
        ax: plt.Axes,
    ) -> plt.Axes:
        """Plot pdf/cdf/etc. of the predictive distribution."""
        raise NotImplementedError