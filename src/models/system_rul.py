from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.models.particle_filter import ParticleFilterModel


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
        max_life: float = 100.0,
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
    def step(self):
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
            device = mixture.states.device

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

    def reset(self):
        """
        Reset system state, observations, and PFs.
        """
        # --- reset observation buffers ---
        self.t_obs.clear()
        self.s_obs = {name: [] for name in self.pf_models.keys()}

        # --- reset RUL history ---
        self.history_time.clear()
        self.history_rul.clear()

        # --- reset PF internal states ---
        for pf in self.pf_models.values():
            pf.reset()

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

    @torch.no_grad()
    def run_system_rul_online(
        self,
        data_t: np.ndarray,
        data_s: dict[str, np.ndarray],
        start_idx: int,
        on_step: Callable[[int, SystemRUL], None] | None = None,
    ) -> pd.DataFrame:
        """
        Run online system RUL estimation.

        Parameters
        ----------
        on_step : optional callback
            Called after record() at each step:
                on_step(k, self)
        """

        self.reset()

        for k, t_curr in enumerate(data_t):
            self.observe(
                time=float(t_curr),
                observations={name: perf[k] for name, perf in data_s.items()},
            )

            if k < start_idx:
                continue

            self.step()
            self.record(float(t_curr))

            if on_step is not None:
                on_step(k, self)

        return self.history_to_dataframe()

    def history_to_dataframe(self) -> pd.DataFrame:
        elapsed_time = np.asarray(self.history_time)
        preds = torch.stack(self.history_rul).cpu().numpy()
        lower, mean, upper = preds.T

        return pd.DataFrame(
            {
                "time": elapsed_time,
                "lower": lower,
                "mean": mean,
                "upper": upper,
            }
        )
