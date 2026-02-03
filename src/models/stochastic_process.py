import abc

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as NDArray
import torch
import torch.distributions as dist
import torch.nn as nn


class StochasticProcessModel(nn.Module, abc.ABC):
    """
    Generic plotting utilities for objects that implement:

        distribution(s: torch.Tensor) -> torch.distributions.Distribution
    """

    uncertainty_color = "orange"
    mean_color = "blue"
    mode_color = "cyan"
    bound_color = "black"

    # --------- REQUIRED API ----------
    @abc.abstractmethod
    def distribution(self, s: torch.Tensor) -> dist.Distribution:
        """Distribution at scaled performance s."""
        raise NotImplementedError

    # --------- GENERIC METHODS ----------

    # Fast Monte-Carlo quantiles
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
        mean = self.distribution(s).mean
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
    @torch.no_grad()
    def plot_distribution(
        self,
        t: NDArray,
        s: NDArray,
        func: str = "pdf",
        ax: plt.Axes | None = None,
        vmax: float | None = None,
        gamma_prob: float = 0.3,
        title: str = "Distribution of $T_s$",
        plot_mean: bool = False,
        plot_mode: bool = False,
        legend_loc: str | None = "upper right",
    ) -> plt.Axes:

        device = self._device()

        # grid
        T, S = np.meshgrid(t, s)
        s_torch = torch.tensor(S.flatten(), dtype=torch.float32, device=device)
        t_torch = torch.tensor(T.flatten(), dtype=torch.float32, device=device)

        with torch.no_grad():
            dist_Ts = self.distribution(s_torch)

            if func == "pdf":
                Z = dist_Ts.log_prob(t_torch).exp()
            elif func == "cdf":
                Z = dist_Ts.cdf(t_torch)
            else:
                raise ValueError("func must be 'pdf' or 'cdf'")

            if plot_mean or plot_mode:
                s_line = torch.tensor(s, dtype=torch.float32, device=device)
                dist_Ts_line = self.distribution(s_line)

        Z = Z.reshape(S.shape).cpu().numpy()

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
            mean_Ts = dist_Ts_line.mean
            mean_Ts = mean_Ts.cpu().numpy()
            ax.plot(
                mean_Ts, s, color=self.mean_color, lw=1.5, linestyle="--", label="mean", alpha=0.8
            )

        if plot_mode:
            mode_Ts = dist_Ts_line.mode
            mode_Ts = mode_Ts.cpu().numpy()
            ax.plot(mode_Ts, s, color=self.mode_color, lw=2, label="mode")

        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim([0, t.max()])
        ax.set_ylim([0, 1])
        if legend_loc:
            ax.legend(
                loc=legend_loc,  # fixed → no search
                frameon=True,
                framealpha=0.9,
            )
        return ax

    @torch.no_grad()
    def plot_random_variable(
        self,
        t: NDArray,
        s: float,
        func: str = "pdf",
        ax: plt.Axes | None = None,
        title: str | None = None,
        max_prob: float = 1.0,
    ) -> plt.Axes:
        """
        Plot the random variable T_s at a fixed performance value s,
        styled to match the legacy Gamma Mixture plot.
        """
        device = self._device()
        t_torch = torch.tensor(t, dtype=torch.float32, device=device)
        s_torch = torch.tensor([s], dtype=torch.float32, device=device)

        dist_s = self.distribution(s_torch)

        # --- PDF / CDF ---
        if func == "pdf":
            y = dist_s.log_prob(t_torch).exp()
        elif func == "cdf":
            y = dist_s.cdf(t_torch)
        else:
            raise ValueError("func must be 'pdf' or 'cdf'")

        y = y.detach().cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # --- mixture curve (black dashed) ---
        ax.plot(
            t,
            y,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Mixture {func}",
        )

        # --- axes & labels ---
        ax.set_xlabel("t")
        ax.set_ylabel(func)
        ax.set_title(title or f"T_s distribution at s = {s}")
        ax.set_ylim(0, max_prob)
        ax.legend()
        return ax

    @torch.no_grad()
    def plot_uncertainty_interval(
        self,
        ax: plt.Axes,
        lower: float,
        mean: float,
        upper: float,
        ymax: float,
        label: str | None = None,
    ):
        """
        Plot an uncertainty interval as a horizontal segment near y=0
        with a short mean indicator.
        """
        # visual proportions (match original look)
        h_interval = 0.01 * ymax
        h_mean = 0.04 * ymax
        h_bounds = 0.025 * ymax

        # uncertainty rectangle
        rect = patches.Rectangle(
            (lower, 0.0),
            upper - lower,
            h_interval,
            facecolor=self.uncertainty_color,
            edgecolor="black",
            linewidth=1,
            alpha=1.0,
            label=label,
        )
        ax.add_patch(rect)

        # mean
        ax.vlines(
            mean, ymin=h_interval, ymax=h_mean, color=self.mean_color, linewidth=2, label="mean"
        )

        # bounds
        ax.vlines([lower, upper], ymin=0, ymax=h_bounds, color=self.bound_color, linewidth=2)

        ax.legend()

    @torch.no_grad()
    def plot_uncertainty_band(
        self,
        s: NDArray,
        lower: NDArray,
        mean: NDArray,
        upper: NDArray,
        level: float = 0.95,
        ax: plt.Axes | None = None,
        alpha: float = 0.5,
        title: str = "Uncertainty interval",
        legend_loc: str = "upper right",
    ) -> plt.Axes:
        """
        Plot uncertainty interval for the stochastic process over performance values s.
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # --- uncertainty band (same visual role as before) ---
        ax.fill_betweenx(
            s,
            lower,
            upper,
            color=self.uncertainty_color,
            alpha=alpha,
            label=f"{int(level * 100)}% uncertainty",
        )

        # --- mean ---
        ax.plot(
            mean,
            s,
            "-",
            color=self.mean_color,
            linewidth=2,
            label="Mean",
        )

        # --- bounds (explicit, same as original plot) ---
        ax.plot(
            lower,
            s,
            "-",
            color=self.bound_color,
            linewidth=1,
            label="Lower bound",
        )
        ax.plot(
            upper,
            s,
            "-",
            color=self.bound_color,
            linewidth=1,
            label="Upper bound",
        )

        # --- labels & layout ---
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim(left=0)
        ax.legend(loc=legend_loc)

        return ax

    @staticmethod
    @torch.no_grad()
    def plot_observations(
        ax: plt.Axes,
        t_obs: NDArray,
        s_obs: NDArray,
        current_idx: int = -1,  # just for ploting data
        legend_loc: str | None = "upper right",
    ):
        # --- Observations ---
        idx = current_idx
        if idx > 0:
            ax.plot(
                t_obs[: idx + 1],
                s_obs[: idx + 1],
                "o-",
                color="white",
                alpha=0.5,
                markersize=4,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="past obs",
            )

        # --- Future observations (reference only) ---
        if idx + 1 < len(t_obs):
            ax.plot(
                t_obs[idx + 1 :],
                s_obs[idx + 1 :],
                "o-",
                color="#9E9E9E",
                alpha=0.5,
                markersize=4,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="future (ref)",
            )

        # --- Current observation ---
        if idx >= 0:
            ax.plot(
                t_obs[idx],
                s_obs[idx],
                "o",
                color="#FF7F50",  # coral
                alpha=1.0,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=1.0,
                label="current obs",
            )

        if legend_loc:
            ax.legend(
                loc=legend_loc,  # fixed → no search
                frameon=True,
                framealpha=0.9,
            )
        return ax


#  if level > 0:
#             lower, mean, upper = self.uncertainty_interval(
#                 s_torch, level=level, n_samples=n_samples
#             )
#             self._plot_uncertainty_interval(
#                 ax=ax,
#                 lower=lower.item(),
#                 mean=mean.item(),
#                 upper=upper.item(),
#                 ymax=max_prob,
#                 label=f"{int(level * 100)}% uncertainty",
#             )
