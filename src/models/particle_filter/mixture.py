from __future__ import annotations

import torch
import torch.distributions as dist

from src.models.degradation.base import DegModel
from src.models.distributions.rul_distirbution import RULDistributionWrapper
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
        init_ss = torch.tensor([c.get_init_s() for c in components])

        self._init_from_tensors(
            deg_class=type(components[0]),
            states=states,
            weights=weights,
            onsets=onsets,
            init_ss=init_ss,
        )

    def _init_from_tensors(
        self,
        deg_class: type[DegModel],
        states: torch.Tensor,  # [K, RP]
        weights: torch.Tensor,  # [K]
        onsets: torch.Tensor | None = None,
        init_ss: torch.Tensor | None = None,
    ):
        assert states.ndim == 2
        assert weights.ndim == 1
        assert states.shape[0] == weights.shape[0]

        self.n_components, self.state_dim = states.shape
        self.deg_class = deg_class
        self.states: torch.Tensor
        self.weights: torch.Tensor
        self.onsets: torch.Tensor
        self.init_ss: torch.Tensor
        self.register_buffer("states", states)
        self.register_buffer(
            "weights",
            weights / weights.sum().clamp_min(1e-12),
        )
        self.register_buffer("onsets", onsets)
        self.register_buffer("init_ss", init_ss)

    @classmethod
    def from_particles(
        cls,
        deg_class: type[DegModel],
        states: torch.Tensor,  # [K, RP]
        weights: torch.Tensor,  # [K]
        onsets: torch.Tensor,
        init_ss: torch.Tensor,
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
            init_ss=init_ss,
        )
        return obj

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        params_s = self.forward(s.unsqueeze(1))  # [B, S]
        base_dist_s = self.build_mixture_distribution(params_s)
        return RULDistributionWrapper(
            base_dist=base_dist_s,
            cap=100,
            n_samples=4096,
            quantile_search=0.999,
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Stack component parameters.

        Returns
        -------
        params : torch.Tensor
            Shape [B, K, DP]   (batch, component, distribution params)
        """
        onsets = self.onsets.unsqueeze(1)  # [K,1]
        init_ss = self.init_ss.unsqueeze(1)  # [K,1]
        return self.deg_class.forward_with_states(s, self.states, onsets, init_ss)

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

    def get_states(self) -> torch.Tensor:
        return self.states

    def get_weights(self) -> torch.Tensor:
        return self.weights

    def get_onsets(self) -> torch.Tensor:
        return self.onsets

    def get_init_ss(self) -> torch.Tensor:
        return self.init_ss

    @torch.no_grad()
    def update(
        self,
        states: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        onsets: torch.Tensor | None = None,
        init_ss: torch.Tensor | None = None,
    ):
        if states is not None:
            self.states.copy_(states)
        if weights is not None:
            self.weights.copy_(weights)
            self.weights /= self.weights.sum().clamp_min(1e-12)
        if onsets is not None:
            self.onsets.copy_(onsets)
        if init_ss is not None:
            self.init_ss.copy_(init_ss)

    @torch.no_grad()
    def mode(
        self,
        s: torch.Tensor,
        cap: int = 100,
        n_samples: int = 4096,
        quantile_search: float = 0.999,
    ) -> torch.Tensor:
        dist_s = self.distribution(s)
        samples = dist_s.sample((n_samples,))  # [N, B]
        mode = self._mode_from_samples(
            samples,
            cap=cap,
            quantile_search=quantile_search,
        )
        return mode

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

    @torch.no_grad()
    def _mode_from_samples(
        self,
        samples: torch.Tensor,  # [N, B]
        cap: int = 100,
        quantile_search: float = 0.999,
    ) -> torch.Tensor:
        """
        Compute integer-cycle mode from MC samples.
        Bins centered at integers (width = 1).
        """

        B = samples.shape[1]
        device = samples.device
        modes = torch.zeros(B, device=device)

        # stable upper bound per batch element
        q_high = torch.quantile(samples, quantile_search, dim=0)
        max_int = torch.floor(q_high + 0.5).long()

        for b in range(B):

            # integer-centered bins: [k-0.5, k+0.5)
            samples_int = torch.floor(samples[:, b] + 0.5).long()

            samples_int = samples_int.clamp(min=0, max=max_int[b].item())

            counts = torch.bincount(
                samples_int,
                minlength=max_int[b].item() + 1,
            )

            idx = counts.argmax().item()

            # decision rule
            modes[b] = min(idx, cap)

        return modes.float()

    @torch.no_grad()
    def prediction_mc(
        self,
        s: torch.Tensor,
        level: float = 0.95,
        n_samples: int = 4096,
        cap: int = 100,
        quantile_search: float = 0.999,
    ):
        """
        Monte-Carlo prediction using integer-cycle mode.
        Returns lower bound, mode prediction, upper bound.
        """

        assert 0.0 < level < 1.0

        dist_s = self.distribution(s)
        samples = dist_s.sample((n_samples,))  # [N, B]

        # -------------------------
        # Uncertainty bounds
        # -------------------------
        alpha = 1.0 - level
        lower = torch.quantile(samples, alpha / 2, dim=0)
        upper = torch.quantile(samples, 1 - alpha / 2, dim=0)

        # -------------------------
        # Mode from samples
        # -------------------------
        mode = self._mode_from_samples(
            samples,
            cap=cap,
            quantile_search=quantile_search,
        )

        return lower, mode, upper
