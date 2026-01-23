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

        self.K = len(components)
        assert self.K > 0
        assert weights.ndim == 1 and len(weights) == self.K
        assert len({type(c) for c in components}) == 1

        self.model = type(components[0])

        # build everything on CPU
        raw_params = torch.stack(
            [c.get_raw_param_vector() for c in components],
            dim=0,
        )  # CPU

        onsets = torch.stack([c.onset for c in components])
        self.raw_params: torch.Tensor
        self.weights: torch.Tensor
        self.onsets: torch.Tensor
        self.register_buffer("raw_params", raw_params)
        self.register_buffer("weights", weights / weights.sum().clamp_min(1e-12))
        self.register_buffer("onsets", onsets)

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
        return self.model.forward_with_raw_parameters(s, self.raw_params)

    def build_mixture_distribution(self, params: torch.Tensor) -> dist.Distribution:
        """
        params: [B, K, DP]
        """
        components_dist = self.model.build_distribution_from_params(params)
        mixture = dist.Categorical(self.weights)
        return dist.MixtureSameFamily(mixture, components_dist)
    

    # ---------------------------
    # Fast Monte-Carlo quantiles 
    # ---------------------------
    @torch.no_grad()
    def quantile_mc(
        self,
        s: torch.Tensor,        # [B]
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
        dist_s = self.distribution(s)          # MixtureSameFamily
        samples = dist_s.sample((n_samples,)) # [N, B]
        return torch.quantile(samples, q, dim=0)
    