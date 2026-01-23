import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from src.models.deg_model import DEGModel


class MixtureDEGModel(nn.Module):
    """
    Homogeneous mixture of DEGModels.
    All components must be of the same DEGModel subclass.
    P = number of parameters per component
    K = number of mixture components
    """

    def __init__(self, components: list[DEGModel], weights: torch.Tensor):
        super().__init__()
        assert len(components) == len(weights)
        assert len({c.distribution_class for c in components}) == 1, "MixtureDEGModel must be homogeneous"
        
        # register submodules correctly
        self.components = nn.ModuleList(components)
        # fixed mixture weights
        self.weights: torch.Tensor
        self.register_buffer("weights", weights / weights.sum())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Stack component parameters.

        Returns
        -------
        params : torch.Tensor
            Shape [K, ..., P]
        """
        return torch.stack([c.forward(s) for c in self.components], dim=0)

    def build_distribution(self, params: torch.Tensor) -> dist.Distribution:
        """
        Build the mixture distribution from stacked parameters.
        """
        # params: [K, ..., P]

        # delegate distribution construction to a representative component
        ref:DEGModel = self.components[0]

        # move component axis to last batch dimension
        # [K, ..., P] -> [..., K, P]
        params = params.movedim(0, -2)

        # build component distribution (batched over K)
        components_dist = ref.build_distribution(params)

        mixture = dist.Categorical(self.weights)

        return dist.MixtureSameFamily(mixture, components_dist)
    
    def plot_distribution(
        self,
        t: np.ndarray,
        s: np.ndarray,
        func: str = "pdf",
        ax: plt.Axes | None = None,
        vmax: float | None = None,
        gamma_prob: float = 0.3,
        title: str = "Mixture degradation distribution of $T_s$",
        plot_mean: bool = True,
        mean_kwargs: dict | None = None,
    ):
        device = next(self.parameters()).device

        # build grid
        T, S = np.meshgrid(t, s)
        s_torch = torch.tensor(S.flatten(), dtype=torch.float32, device=device)
        t_torch = torch.tensor(T.flatten(), dtype=torch.float32, device=device)

        with torch.no_grad():
            # build mixture distribution
            dist_ts = self.build_distribution(self.forward(s_torch))

            if func == "pdf":
                Z = dist_ts.log_prob(t_torch).exp()
            elif func == "cdf":
                Z = dist_ts.cdf(t_torch)
            else:
                raise ValueError("func must be 'pdf' or 'cdf'")

            # ---- mixture mean curve ----
            if plot_mean:
                s_line = torch.tensor(s, dtype=torch.float32, device=device)
                dist_line = self.build_distribution(self.forward(s_line))
                mean_Ts = dist_line.mean

        Z = Z.reshape(S.shape).cpu().numpy()

        if plot_mean:
            mean_Ts = mean_Ts.cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        norm = mcolors.PowerNorm(
            gamma=gamma_prob,
            vmin=0,
            vmax=vmax if vmax is not None else np.percentile(Z, 99),
        )

        c = ax.pcolormesh(T, S, Z, shading="auto", cmap="viridis", norm=norm)
        plt.colorbar(c, ax=ax, label=func)

        # ---- plot mean ----
        if plot_mean:
            if mean_kwargs is None:
                mean_kwargs = dict(color="orange", lw=2, label="mixture mean")
            ax.plot(mean_Ts, s, **mean_kwargs)

        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim([0, t.max()])
        ax.legend()

        return ax