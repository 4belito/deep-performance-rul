import abc

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

from src.models.sthocastic_process import StochasticProcessModel


class DegModel(StochasticProcessModel, abc.ABC):
    """
    Abstract base class for degradation models.
    """

    def __init__(self, onset: float = 0.0):
        super().__init__()
        self.onset: float
        self.register_buffer("onset", torch.tensor(float(onset)))

    # ---------- REQUIRED API ----------
    @abc.abstractmethod
    def get_raw_param_vector(self) -> torch.Tensor:
        """
        Return the raw parameter vector.
        """
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Distribution:
        """Build a torch Distribution from params."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def forward_with_raw_parameters(
        s: torch.Tensor,
        raw_params: torch.Tensor,   # [..., RP]
    ) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        raw_params: Tensor of shape [K, RP]
        returns: Tensor of shape [B, K, DP]
        """
        raise NotImplementedError


    # ---------- GENERIC METHODS ----------
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        returns: Tensor of shape [B, DP]
        """
        raw_params = self.get_raw_param_vector().unsqueeze(0)  # [1, RP]
        output = self.forward_with_raw_parameters(s, raw_params)
        return output[:, 0, :]  # [B, DP]

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        return self.build_distribution_from_params(self.forward(s))
    
    def _post_plot(self, ax: plt.Axes):
        onset = self.onset
        ax.axvline(
            x=onset,
            linestyle="--",
            color="#4CC9F0",
            label="onset"
        )
