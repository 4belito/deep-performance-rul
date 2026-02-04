from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.models.particle_filter import ParticleFilterModel


class RULPredictor:
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
        current_obs: bool = True,
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
        self.max_life = max_life
        self.current_obs = current_obs
        self.t_obs: list[float] = []
        self.s_obs: dict[str, list[float]] = {name: [] for name in pf_models.keys()}

        # --- history (for plotting / video) ---
        self.history_time: list[float] = []
        self.history_component_eol: dict[str, list[tuple[float, float, float]]] = {
            name: [] for name in pf_models.keys()
        }
        self.history_rul: list[tuple[float, float, float]] = []

    # --------------------------------------------------
    # Core stepping
    # --------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        t_data: np.ndarray,
        s_data: dict[str, np.ndarray],
        start_idx: int,
        on_step: Callable[[int, RULPredictor], None] | None = None,
    ):
        """
        Run online system RUL estimation.

        Parameters
        ----------
        on_step : optional callback
            Called after record() at each step:
                on_step(k, self)
        """

        self.reset()

        for k, t_curr in enumerate(t_data):
            self.observe(
                time=float(t_curr),
                observations={name: perf[k] for name, perf in s_data.items()},
            )

            if k < start_idx:
                continue

            self.step()
            self.record_component_eol()
            self.record_system_rul(current_time=float(t_curr))

            if on_step is not None:
                on_step(k, self)

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
            if self.current_obs:
                t_tensor = t_tensor[[-1]]
                s_tensor = s_tensor[[-1]]

            pf.step(
                s_obs=s_tensor,
                t_obs=t_tensor,
            )

    # --------------------------------------------------
    # Component-level RUL
    # --------------------------------------------------

    @torch.no_grad()
    def record_component_eol(self):
        """
        Compute and store per-component EOL uncertainty intervals.
        """
        for name, pf in self.pf_models.items():
            device = pf.mixture.states.device
            s0 = torch.tensor([0.0], device=device)
            lower, mean, upper = pf.mixture.uncertainty_interval(s0, self.conf_level)
            self.history_component_eol[name].append((lower.item(), mean.item(), upper.item()))

    # --------------------------------------------------
    # System-level RUL
    # --------------------------------------------------

    @torch.no_grad()
    def system_rul(self, current_time: float) -> tuple[float, float, float]:
        """
        Convert system EOL to system RUL.
        """
        eol_lower, eol_mean, eol_upper = self.system_eol()

        return (
            max(eol_lower - current_time, 0.0),
            max(eol_mean - current_time, 0.0),
            max(eol_upper - current_time, 0.0),
        )

    @torch.no_grad()
    def system_eol(self) -> tuple[float, float, float]:
        """
        Conservative system EOL = min over component EOLs.
        """
        lowers = []
        means = []
        uppers = []

        for name in self.pf_models.keys():
            eol_lower, eol_mean, eol_upper = self.history_component_eol[name][-1]
            lowers.append(eol_lower)
            means.append(eol_mean)
            uppers.append(eol_upper)

        # Conservative aggregation in EOL-space
        eol_lower, eol_mean, eol_upper = self.eol_aggregation(lowers, means, uppers)

        # Physical bounds
        eol_lower = float(np.clip(eol_lower, 0.0, self.max_life))
        eol_mean = float(np.clip(eol_mean, 0.0, self.max_life))
        eol_upper = float(np.clip(eol_upper, 0.0, self.max_life))

        return eol_lower, eol_mean, eol_upper

    @torch.no_grad()
    def eol_aggregation(
        self,
        lowers: list[float],
        means: list[float],
        uppers: list[float],
    ) -> tuple[float, float, float]:
        return min(lowers), min(means), min(uppers)

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
        self.history_component_eol = {name: [] for name in self.pf_models.keys()}

        # --- reset PF internal states ---
        for pf in self.pf_models.values():
            pf.reset()

    # --------------------------------------------------
    # History recording
    # --------------------------------------------------

    @torch.no_grad()
    def record_system_rul(self, current_time: float):
        """
        Compute and store system-level RUL at current time.
        """
        lower, mean, upper = self.system_rul(current_time)

        self.history_time.append(float(current_time))
        self.history_rul.append((lower, mean, upper))

    def history_to_dataframe(self) -> pd.DataFrame:
        elapsed_time = np.asarray(self.history_time)
        preds = np.asarray(self.history_rul)
        lower, mean, upper = preds.T

        return pd.DataFrame(
            {
                "time": elapsed_time,
                "lower": lower,
                "mean": mean,
                "upper": upper,
            }
        )
