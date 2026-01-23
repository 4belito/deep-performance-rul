import abc

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn


class StochasticProcessModel(nn.Module,abc.ABC):
    """
    Generic plotting utilities for objects that implement:

        distribution(s: torch.Tensor) -> torch.distributions.Distribution
    """
    
    # --------- REQUIRED API ----------
    @abc.abstractmethod
    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        """ Distribution at scaled performance s. """
        raise NotImplementedError

    # --------- GENERIC METHODS ----------
    
    @torch.no_grad()
    def quantile_mc(
        self,
        s: torch.Tensor,
        q: float,
        n_samples: int = 4096,
    ) -> torch.Tensor:
        assert 0.0 < q < 1.0, "q must be in (0, 1)"
        
        dist_s = self.distribution(s)
        samples = dist_s.sample((n_samples,))
        return torch.quantile(samples, q, dim=0)

    @torch.no_grad()
    def uncertainty_interval(
        self,
        s: torch.Tensor,
        level: float = 0.95,
        n_samples: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert 0.0 < level < 1.0, "level must be in (0, 1)"
        
        alpha = 1.0 - level
        lower = self.quantile_mc(s, alpha / 2, n_samples)
        upper = self.quantile_mc(s, 1 - alpha / 2, n_samples)
        mean  = self.distribution(s).mean
        return lower, mean, upper
    
    
    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------
    def plot_uncertainty_band(
        self,
        s: np.ndarray,
        level: float = 0.95,
        n_samples: int = 4096,
        ax: plt.Axes | None = None,
        alpha: float = 0.4,
        title: str = "Predictive uncertainty interval",
    ) -> plt.Axes:
        """
        Plot uncertainty band over performance values s.
        """
        device = self._device()
        s_torch = torch.tensor(s, dtype=torch.float32, device=device)

        lower, mean, upper = self.uncertainty_interval(
            s_torch, level=level, n_samples=n_samples
        )

        lower = lower.cpu().numpy()
        mean  = mean.cpu().numpy()
        upper = upper.cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.fill_betweenx(
            s, lower, upper,
            color="tab:blue",
            alpha=alpha,
            label=f"{int(level * 100)}% uncertainty"
        )
        ax.plot(mean, s, color="black", lw=2, label="mean")

        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_title(title)
        ax.legend()

        return ax

    def plot_random_variable(
        self,
        t: np.ndarray,
        s: float,
        func: str = "pdf",
        level: float = 0.95,
        n_samples: int = 4096,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """
        Plot the random variable T_s at a fixed performance value s.
        """
        device = self._device()
        t_torch = torch.tensor(t, dtype=torch.float32, device=device)
        s_torch = torch.tensor([s], dtype=torch.float32, device=device)

        dist_s = self.distribution(s_torch)

        if func == "pdf":
            y = dist_s.log_prob(t_torch).exp()
        elif func == "cdf":
            y = dist_s.cdf(t_torch)
        else:
            raise ValueError("func must be 'pdf' or 'cdf'")

        y = y.cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t, y, lw=2, label=f"{func.upper()}")

        # uncertainty interval
        lower, mean, upper = self.uncertainty_interval(
            s_torch, level=level, n_samples=n_samples
        )

        lower = lower.item()
        mean  = mean.item()
        upper = upper.item()

        ax.axvline(mean, color="black", lw=2, label="mean")
        ax.axvspan(lower, upper, color="tab:blue", alpha=0.3,
                   label=f"{int(level * 100)}% uncertainty")

        ax.set_xlabel("time")
        ax.set_ylabel(func)
        ax.set_title(title or f"T_s distribution at s = {s}")
        ax.legend()

        return ax

    # def plot_distribution(
    #     self,
    #     t: np.ndarray,
    #     s: np.ndarray,
    #     func: str = "pdf",
    #     ax: plt.Axes | None = None,
    #     vmax: float | None = None,
    #     gamma_prob: float = 0.3,
    #     title: str = "Distribution of $T_s$",
    #     plot_mean: bool = True,
    #     mean_kwargs: dict | None = None,
    # ) -> plt.Axes:

    #     device = self._device()

    #     # grid
    #     T, S = np.meshgrid(t, s)
    #     s_torch = torch.tensor(S.flatten(), dtype=torch.float32, device=device)
    #     t_torch = torch.tensor(T.flatten(), dtype=torch.float32, device=device)

    #     with torch.no_grad():
    #         dist_ts = self.distribution(s_torch)

    #         if func == "pdf":
    #             Z = dist_ts.log_prob(t_torch).exp()
    #         elif func == "cdf":
    #             Z = dist_ts.cdf(t_torch)
    #         else:
    #             raise ValueError("func must be 'pdf' or 'cdf'")

    #         if plot_mean:
    #             s_line = torch.tensor(s, dtype=torch.float32, device=device)
    #             mean_Ts = self.distribution(s_line).mean

    #     Z = Z.reshape(S.shape).cpu().numpy()
    #     if plot_mean:
    #         mean_Ts = mean_Ts.cpu().numpy()

    #     if ax is None:
    #         _, ax = plt.subplots(figsize=(10, 6))

    #     norm = mcolors.PowerNorm(
    #         gamma=gamma_prob,
    #         vmin=0,
    #         vmax=vmax if vmax is not None else np.percentile(Z, 99),
    #     )

    #     c = ax.pcolormesh(T, S, Z, shading="auto", cmap="viridis", norm=norm)
    #     plt.colorbar(c, ax=ax, label=func)

    #     if plot_mean:
    #         if mean_kwargs is None:
    #             mean_kwargs = dict(color="orange", lw=2, label="mean")
    #         ax.plot(mean_Ts, s, **mean_kwargs)

    #     ax.set_title(title)
    #     ax.set_xlabel("time")
    #     ax.set_ylabel("scaled performance")
    #     ax.set_xlim([0, t.max()])
    #     ax.legend()

    #     return ax
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