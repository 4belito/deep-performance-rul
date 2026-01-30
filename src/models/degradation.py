import abc

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

from src.models.stochastic_process import StochasticProcessModel


class DegModel(StochasticProcessModel, abc.ABC):
    """
    Abstract base class for degradation models.
    """

    def __init__(self, onset):
        super().__init__()
        self.onset: float
        self.register_buffer("onset", torch.tensor(float(onset)))

    # ---------- REQUIRED API ----------
    @staticmethod
    @abc.abstractmethod
    def get_state_names() -> list[str]:
        """
        Names of nn.Parameter attributes that define the raw parameter vector.
        Order matters.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Distribution:
        """Build a torch Distribution from params."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def forward_with_states(
        s: torch.Tensor,
        states: torch.Tensor,  # [..., RP]
        onsets: torch.Tensor,  # [..., 1]
    ) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        states: Tensor of shape [K, RP]
        returns: Tensor of shape [B, K, DP]
        """
        raise NotImplementedError

    # ---------- GENERIC METHODS ----------
    @classmethod
    def state_dim(cls) -> int:
        return len(cls.get_state_names())

    def get_onset(self) -> float:
        return float(self.onset)

    def get_state_vector(self) -> torch.Tensor:
        return torch.stack([getattr(self, name) for name in self.get_state_names()])

    def set_state_vector(self, states: torch.Tensor) -> None:
        names = self.get_state_names()
        assert states.shape == (len(names),)

        with torch.no_grad():
            for name, value in zip(names, states):
                getattr(self, name).copy_(value)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        returns: Tensor of shape [B, DP]
        """
        states = self.get_state_vector().unsqueeze(0)  # [1, RP]
        onsets = torch.tensor([[self.onset]])  # [1, 1]
        output = self.forward_with_states(s, states, onsets)  # [B, 1, DP]
        return output[:, 0, :]  # [B, DP]

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        params_s = self.forward(s.unsqueeze(1))  # [B, DP]
        return self.build_distribution_from_params(params_s)

    def _post_plot(self, ax: plt.Axes):
        onset = self.onset
        ax.axvline(x=onset, linestyle="--", color="#4CC9F0", label="onset")
