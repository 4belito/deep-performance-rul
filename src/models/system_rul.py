from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.models.particle_filter import ParticleFilterModel

# -----------------------------
# Plot styling (reuse everywhere)
# -----------------------------


class SystemRUL:
    """
    System-level RUL estimator from multiple Particle Filters
    (one PF per performance metric).

    System RUL is defined conservatively as the minimum
    over component-level RULs.
    """

    UNCERTAINTY_COLOR = "#FF7F50"
    MEAN_COLOR = "blue"

    def __init__(
        self,
        pf_models: dict[str, ParticleFilterModel],
        conf_level: float = 0.95,
    ):
        """
        Parameters
        ----------
        pf_models : dict[str, ParticleFilterModel]
            One PF per performance metric
        conf_level : float
            Confidence level for RUL intervals
        """
        assert len(pf_models) > 0, "At least one PF required"
        assert 0.0 < conf_level < 1.0

        self.pf_models = pf_models
        self.conf_level = conf_level
        self.t_obs: list[float] = []
        self.s_obs: dict[str, list[float]] = {name: [] for name in pf_models.keys()}

        # --- history (for plotting / video) ---
        self.history_time: list[float] = []
        self.history_rul: list[torch.Tensor] = []  # each: [3] (lower, mean, upper)

    # --------------------------------------------------
    # Core stepping
    # --------------------------------------------------

    @torch.no_grad()
    def observe(
        self,
        time: float,
        observations: dict[str, float | torch.Tensor],
    ):
        for name, value in observations.items():
            self.s_obs[name].append(float(value))
        self.t_obs.append(float(time))

    @torch.no_grad()
    def step(
        self,
        noise_scales: dict[str, float | torch.Tensor],
    ):
        """
        Advance all PFs using buffered observations.
        """
        for name, pf in self.pf_models.items():
            device = pf.states.device

            t_tensor = torch.tensor(
                self.t_obs,
                dtype=torch.float32,
                device=device,
            )
            s_tensor = torch.tensor(
                self.s_obs[name],
                dtype=torch.float32,
                device=device,
            )

            pf.step(
                s_obs=s_tensor,
                t_obs=t_tensor,
                noise_scale=noise_scales[name],
            )

    # --------------------------------------------------
    # Component-level RUL
    # --------------------------------------------------

    @torch.no_grad()
    def component_rul(self, current_time: float):
        """
        Compute per-component RUL intervals.

        Returns
        -------
        dict[name] = (lower, mean, upper)
        """
        rul = {}

        q_lo = (1.0 - self.conf_level) / 2.0
        q_hi = 1.0 - q_lo

        for name, pf in self.pf_models.items():
            mixture = pf.mixture
            device = mixture.raw_params.device

            # RUL is evaluated at s = 0
            s0 = torch.tensor([0.0], device=device)

            # quantiles
            lower = mixture.quantile_mc(s0, q_lo)[0]
            upper = mixture.quantile_mc(s0, q_hi)[0]

            # mean (NOT median!)
            dist = mixture.distribution(s0)
            mean = dist.mean[0]

            # convert EOL → RUL
            rul[name] = (
                (lower - current_time).clamp_min(0.0),
                (mean - current_time).clamp_min(0.0),
                (upper - current_time).clamp_min(0.0),
            )

        return rul

    # --------------------------------------------------
    # System-level RUL
    # --------------------------------------------------

    @torch.no_grad()
    def system_rul(self, current_time: float):
        """
        Conservative system RUL = min over components.

        Returns
        -------
        (lower, mean, upper)
        """
        comp = self.component_rul(current_time)

        lowers = torch.stack([v[0] for v in comp.values()])
        means = torch.stack([v[1] for v in comp.values()])
        uppers = torch.stack([v[2] for v in comp.values()])

        return (
            lowers.min(),
            means.min(),
            uppers.min(),
        )

    # --------------------------------------------------
    # History recording
    # --------------------------------------------------

    @torch.no_grad()
    def record(self, current_time: float):
        """
        Compute and store system-level RUL at current time.
        """
        lower, mean, upper = self.system_rul(current_time)

        self.history_time.append(float(current_time))
        self.history_rul.append(torch.stack([lower, mean, upper]).cpu())

    def reset_history(self):
        """
        Clear stored RUL history.
        """
        self.history_time.clear()
        self.history_rul.clear()

    # --------------------------------------------------
    # Plotting (frame-safe)
    # --------------------------------------------------

    def plot_rul(
        self,
        ax: plt.Axes,
        eol_time: float,
        y_max: float = 100.0,
        x_max: float = 100.0,
        title: str = "System RUL prediction",
    ) -> plt.Axes:
        """
        Plot system-level RUL evolution.

        Parameters
        ----------
        ax : matplotlib axis
        eol_time : float
            End-of-life time (ground truth)
        y_max : float
            Max y-axis limit
        title : str
        """

        if len(self.history_time) == 0:
            ax.set_title(title + " (no data)")
            return

        # --- elapsed time & predictions ---
        elapsed_time = np.asarray(self.history_time)
        elapsed_preds = torch.stack(self.history_rul).numpy()
        lower, mean, upper = elapsed_preds.T

        # --- true RUL from EOL ---
        true_rul = np.maximum(eol_time - elapsed_time, 0.0)

        # --- plot true RUL ---
        ax.plot(
            elapsed_time,
            true_rul,
            "--",
            color="green",
            label="true",
        )

        # --- plot prediction ---
        ax.plot(elapsed_time, lower, "-", color="black", linewidth=0.5)
        ax.plot(elapsed_time, upper, "-", color="black", linewidth=0.5)
        ax.plot(elapsed_time, mean, "-", color=self.MEAN_COLOR, label="pred")

        ax.fill_between(
            elapsed_time,
            lower,
            upper,
            color=self.UNCERTAINTY_COLOR,
            alpha=0.5,
            label="unc",
        )

        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("RUL")
        ax.set_ylim(0, y_max)
        ax.set_xlim(0, x_max)
        ax.legend()
        return ax
