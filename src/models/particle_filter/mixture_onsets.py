from __future__ import annotations

import torch
import torch.distributions as dist

from src.models.degradation.base import DegModel
from src.models.stochastic_process import StochasticProcess


class MixtureDegModel(StochasticProcess):
    """
    Homogeneous mixture of DegModels.
    P = number of parameters per component
    K = number of mixture components
    """

    def __init__(self, components: list[DegModel], weights: torch.Tensor):
        super().__init__()

        assert len(components) > 0
        assert weights.ndim == 1
        assert len(components) == len(weights)

        states = torch.stack(
            [c.get_state_vector() for c in components],
            dim=0,
        )
        onsets = torch.tensor([c.get_onset() for c in components])

        self._init_from_tensors(
            deg_class=type(components[0]),
            states=states,
            weights=weights,
            onsets=onsets,
        )

    def _init_from_tensors(
        self,
        deg_class: type[DegModel],
        states: torch.Tensor,  # [K, RP]
        weights: torch.Tensor,  # [K]
        onsets: torch.Tensor | None = None,
    ):
        assert states.ndim == 2
        assert weights.ndim == 1
        assert states.shape[0] == weights.shape[0]

        self.n_components, self.state_dim = states.shape
        self.deg_class = deg_class
        self.states: torch.Tensor
        self.weights: torch.Tensor
        self.onsets: torch.Tensor
        self.register_buffer("states", states)
        self.register_buffer(
            "weights",
            weights / weights.sum().clamp_min(1e-12),
        )
        self.register_buffer("onsets", onsets)

    @classmethod
    def from_particles(
        cls,
        deg_class: type[DegModel],
        states: torch.Tensor,  # [K, RP]
        weights: torch.Tensor,  # [K]
        onsets: torch.Tensor,
    ) -> MixtureDegModel:
        """
        Build a mixture directly from particle tensors (PF-friendly).
        """
        obj = cls.__new__(cls)
        super(MixtureDegModel, obj).__init__()

        obj._init_from_tensors(
            deg_class=deg_class,
            states=states,
            weights=weights,
            onsets=onsets,
        )
        return obj

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        params_s = self.forward(s.unsqueeze(1))  # [B, S]
        dist_s = self.build_mixture_distribution(params_s)
        # if self.cap_value is not None:
        #     dist_s = CappedDistribution(dist_s, cap=self.cap_value)
        return dist_s

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Stack component parameters.

        Returns
        -------
        params : torch.Tensor
            Shape [B, K, DP]   (batch, component, distribution params)
        """
        onsets = self.onsets.unsqueeze(1)  # [K,1]
        return self.deg_class.forward_with_states(s, self.states, onsets)

    def build_mixture_distribution(self, params: torch.Tensor) -> dist.Distribution:
        """
        params: [B, K, DP]
        """
        # component distribution: batch [B, K]
        components_dist = self.deg_class.build_distribution_from_params(params)

        B = params.shape[0]

        # expand mixture weights to [B, K]
        mixture = dist.Categorical(probs=self.weights.expand(B, -1))

        return dist.MixtureSameFamily(mixture, components_dist, validate_args=False)

    def get_states(self) -> torch.Tensor:
        return self.states

    def get_weights(self) -> torch.Tensor:
        return self.weights

    def get_onsets(self) -> torch.Tensor:
        return self.onsets

    @torch.no_grad()
    def update(
        self,
        states: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        onsets: torch.Tensor | None = None,
    ):
        if states is not None:
            self.states.copy_(states)
        if weights is not None:
            self.weights.copy_(weights)
            self.weights /= self.weights.sum().clamp_min(1e-12)
        if onsets is not None:
            self.onsets.copy_(onsets)

    @torch.no_grad()
    def mode(self, s: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        """
        Numerical mode via grid search.

        s: [B]
        t_grid: [T]  (time grid)

        Returns:
            mode: [B]
        """
        dist_s = self.distribution(s)  # MixtureSameFamily
        log_pdf = dist_s.log_prob(t_grid.unsqueeze(-1))  # [T, B]
        idx = log_pdf.argmax(dim=0)  # [B]
        return t_grid[idx]

    @torch.no_grad()
    def mean(self, s: torch.Tensor) -> torch.Tensor:
        """
        Exact predictive mean.
        """
        return self.distribution(s).mean

    @torch.no_grad()
    def variance(self, s: torch.Tensor) -> torch.Tensor:
        return self.distribution(s).variance

    def rul_nll(self, eol: torch.Tensor):
        eol_dist = self.distribution(s=torch.tensor([0.0]))
        nll = -eol_dist.log_prob(eol.unsqueeze(0)).mean()
        return nll

    def rul_mse(self, eol: torch.Tensor):
        eol_dist = self.distribution(s=torch.tensor([0.0]))
        mse = ((eol_dist.mean - eol.unsqueeze(0)) ** 2).mean()
        return mse
