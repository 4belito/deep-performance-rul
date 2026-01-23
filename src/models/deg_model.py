import abc

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn


class DistributionPlot(nn.Module,abc.ABC):
    """
    Generic plotting utilities for objects that implement:

        distribution(s: torch.Tensor) -> torch.distributions.Distribution
    """
    
    @abc.abstractmethod
    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        """ Distribution at scaled performance s. """
        raise NotImplementedError

    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")


    def plot_distribution(
        self,
        t: np.ndarray,
        s: np.ndarray,
        func: str = "pdf",
        ax: plt.Axes | None = None,
        vmax: float | None = None,
        gamma_prob: float = 0.3,
        title: str = "Distribution of $T_s$",
        plot_mean: bool = True,
        mean_kwargs: dict | None = None,
    ) -> plt.Axes:

        device = self._device()

        # grid
        T, S = np.meshgrid(t, s)
        s_torch = torch.tensor(S.flatten(), dtype=torch.float32, device=device)
        t_torch = torch.tensor(T.flatten(), dtype=torch.float32, device=device)

        with torch.no_grad():
            dist_ts = self.distribution(s_torch)

            if func == "pdf":
                Z = dist_ts.log_prob(t_torch).exp()
            elif func == "cdf":
                Z = dist_ts.cdf(t_torch)
            else:
                raise ValueError("func must be 'pdf' or 'cdf'")

            if plot_mean:
                s_line = torch.tensor(s, dtype=torch.float32, device=device)
                mean_Ts = self.distribution(s_line).mean

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

        if plot_mean:
            if mean_kwargs is None:
                mean_kwargs = dict(color="orange", lw=2, label="mean")
            ax.plot(mean_Ts, s, **mean_kwargs)

        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim([0, t.max()])
        ax.legend()

        return ax




class DEGModel(DistributionPlot, abc.ABC):
    """
    Abstract base class for degradation models.
    """

    def __init__(self, onset: float = 0.0):
        super().__init__()
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
        raw_params = self.get_raw_param_vector().unsqueeze(0)
        return self.forward_with_raw_parameters(s, raw_params)

    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        return self.build_distribution_from_params(self.forward(s))