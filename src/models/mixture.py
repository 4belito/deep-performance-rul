from __future__ import annotations

import torch
import torch.distributions as dist

from src.models.degradation import DegModel, StochasticProcessModel


class MixtureDegModel(StochasticProcessModel):
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

        raw_params = torch.stack(
            [c.get_raw_param_vector() for c in components],
            dim=0,
        )
        onsets = torch.tensor([c.get_onset() for c in components])

        self._init_from_tensors(
            deg_class=type(components[0]),
            raw_params=raw_params,
            weights=weights,
            onsets=onsets,
        )

    def _init_from_tensors(
        self,
        deg_class: type[DegModel],
        raw_params: torch.Tensor,  # [K, RP]
        weights: torch.Tensor,  # [K]
        onsets: torch.Tensor | None = None,
    ):
        assert raw_params.ndim == 2
        assert weights.ndim == 1
        assert raw_params.shape[0] == weights.shape[0]

        self.K, self.RP = raw_params.shape
        self.deg_class = deg_class
        self.raw_params: torch.Tensor
        self.weights: torch.Tensor
        self.onsets: torch.Tensor
        self.register_buffer("raw_params", raw_params)
        self.register_buffer(
            "weights",
            weights / weights.sum().clamp_min(1e-12),
        )

        if onsets is not None:
            self.register_buffer("onsets", onsets)

    @classmethod
    def from_particles(
        cls,
        deg_class: type[DegModel],
        raw_params: torch.Tensor,  # [K, RP]
        weights: torch.Tensor,  # [K]
        onsets: torch.Tensor | None = None,
    ) -> MixtureDegModel:
        """
        Build a mixture directly from particle tensors (PF-friendly).
        """
        obj = cls.__new__(cls)
        super(MixtureDegModel, obj).__init__()

        obj._init_from_tensors(
            deg_class=deg_class,
            raw_params=raw_params,
            weights=weights,
            onsets=onsets,
        )
        return obj

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        return self.build_mixture_distribution(self.forward(s))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Stack component parameters.

        Returns
        -------
        params : torch.Tensor
            Shape [B, K, DP]   (batch, component, distribution params)
        """
        return self.deg_class.forward_with_raw_parameters(s, self.raw_params)

    def build_mixture_distribution(self, params: torch.Tensor) -> dist.Distribution:
        """
        params: [B, K, DP]
        """
        # component distribution: batch [B, K]
        components_dist = self.deg_class.build_distribution_from_params(params)

        B = params.shape[0]

        # expand mixture weights to [B, K]
        mixture = dist.Categorical(probs=self.weights.expand(B, -1))

        return dist.MixtureSameFamily(mixture, components_dist)

    def get_raw_params(self) -> torch.Tensor:
        return self.raw_params

    def get_weights(self) -> torch.Tensor:
        return self.weights

    def get_onsets(self) -> torch.Tensor:
        return self.onsets

    @torch.no_grad()
    def update(
        self,
        raw_params: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        onsets: torch.Tensor | None = None,
    ):
        if raw_params is not None:
            self.raw_params.copy_(raw_params)
        if weights is not None:
            self.weights.copy_(weights)
            self.weights /= self.weights.sum().clamp_min(1e-12)
        if onsets is not None:
            self.onsets.copy_(onsets)

    # ---------------------------
    # Fast Monte-Carlo quantiles
    # ---------------------------
    @torch.no_grad()
    def quantile_mc(
        self,
        s: torch.Tensor,  # [B]
        q: float,
        n_samples: int = 4096,
    ) -> torch.Tensor:
        """
        Monte-Carlo estimate of the q-quantile.

        Parameters
        ----------
        s : torch.Tensor
            Performance values, shape [B]
        q : float
            Quantile in (0, 1)
        n_samples : int
            Number of samples per s [N]

        Returns
        -------
        q_s : torch.Tensor
            Quantiles, shape [B]
        """
        dist_s = self.distribution(s)  # MixtureSameFamily
        samples = dist_s.sample((n_samples,))  # [N, B]
        return torch.quantile(samples, q, dim=0)
