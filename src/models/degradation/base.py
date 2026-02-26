import abc

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

from src.models.stochastic_process import StochasticProcess


class DegModel(StochasticProcess, abc.ABC):
    """
    Abstract base class for degradation models.

    onset is a fixed, data-derived attribute.
    - During training: provided explicitly
    - During loading: restored from state_dict
    """

    def __init__(self, onset: float | None = None, init_s: float | None = None):
        super().__init__()
        self.onset: float
        if onset is not None:
            self.register_buffer("onset", torch.tensor(float(onset)))
        else:
            # placeholder, will be overwritten by load_state_dict
            self.register_buffer("onset", torch.tensor(0.0))

        self.init_s: float
        if init_s is not None:
            self.register_buffer("init_s", torch.tensor(float(init_s)))
        else:
            # placeholder, will be overwritten by load_state_dict
            self.register_buffer("init_s", torch.tensor(1.0))

    # ---------- REQUIRED API ----------
    @staticmethod
    @abc.abstractmethod
    def get_state_names() -> list[str]:
        """
        Names of nn.Parameter attributes that define the raw parameter vector.
        Order matters.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def build_distribution_from_params(cls, params: torch.Tensor) -> dist.Distribution:
        """Build a torch Distribution from params."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def forward_with_states(
        cls,
        s: torch.Tensor,
        states: torch.Tensor,  # [..., RP]
        onsets: torch.Tensor,  # [..., 1]
        init_s: torch.Tensor,  # [..., 1]
    ) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        states: Tensor of shape [K, RP]
        returns: Tensor of shape [B, K, DP]
        """
        raise NotImplementedError

    # ---------- REQUIRED METADATA ----------
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """
        Short human-readable name of the degradation model.
        Example: 'gamma', 'normal', 'weibull'
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_state_semantics() -> dict[str, str]:
        """
        Provide human-readable semantics for each state parameter.
        Keys should match the names returned by get_state_names.
        Values should be short descriptions of what each parameter represents.
        """
        raise NotImplementedError

    # ---------- GENERIC METHODS ----------
    @classmethod
    def state_dim(cls) -> int:
        return len(cls.get_state_names())

    def get_onset(self) -> float:
        return float(self.onset)

    def get_init_s(self) -> float:
        return float(self.init_s)

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
        init_s = torch.tensor([self.init_s])  # [1]
        output = self.forward_with_states(s, states, onsets, init_s)  # [B, 1, DP]
        return output[:, 0, :]  # [B, DP]

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        params_s = self.forward(s.unsqueeze(1))  # [B, DP]
        return self.build_distribution_from_params(params_s)

    def _post_plot(self, ax: plt.Axes):
        onset = self.onset
        ax.axvline(x=onset, linestyle="--", color="#4CC9F0", label="onset")
