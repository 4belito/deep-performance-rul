"""Particle-Filter MLP"""

from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Particle-Filter MLP
# ============================================================
class ParticleFilterMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...],
        activation: Callable[[], nn.Module] = lambda: nn.ReLU(),
    ):
        super().__init__()

        self.state_dim = state_dim
        output_dim = 2 * state_dim + 2  # noise vector + correction vector

        layers = []
        dims = (2, *hidden_dims, output_dim)
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.net(x))

    def tuple_out(self, x: torch.Tensor):
        noise = x[..., : self.state_dim]
        correction = x[..., self.state_dim :]
        return noise, correction

    def correction_tuple(self, x: torch.Tensor):
        correct_prior = x[..., : self.state_dim]
        correct_lik = x[..., [-2]]
        forget_lik = x[..., [-1]]
        return correct_prior, correct_lik, forget_lik

    def tuple_forward(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        x = self.tuple_in(t_obs, s_obs)
        out = self.forward(x)
        return self.tuple_out(out)

    def tuple_forward_mean(
        self,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
    ):
        x = self.tuple_in(
            t_obs,
            s_obs,
        )
        out = self.forward(x)
        out_mean = out.mean(dim=0)
        return self.tuple_out(out_mean)

    @staticmethod
    def tuple_in(
        t_obs,
        s_obs,
    ):
        t_scaled = t_obs / 100.0
        s_scaled = s_obs

        return torch.cat(
            [
                t_scaled.unsqueeze(-1),
                s_scaled.unsqueeze(-1),
            ],
            dim=-1,
        )

    @torch.no_grad()
    def plot_output(
        self,
        t: np.ndarray,
        s: np.ndarray,
        dim: int,
        ax: plt.Axes | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gamma: float = 0.5,
        title: str | None = None,
    ) -> plt.Axes:
        """
        Plot NN output over (t, s).

        Parameters
        ----------
        output : {"noise", "correct"}
            Which NN head to visualize.
        """

        device = next(self.parameters()).device

        # ---- grid ----
        T, S = np.meshgrid(t, s)
        t_torch = torch.tensor(T.flatten(), dtype=torch.float32, device=device)
        s_torch = torch.tensor(S.flatten(), dtype=torch.float32, device=device)

        x = torch.cat(
            [t_torch.unsqueeze(-1), s_torch.unsqueeze(-1)],
            dim=-1,
        )

        # ---- forward (positive outputs) ----
        out = self.forward(x)  # exp to ensure positivity

        Z = out[..., dim]
        label = f"dim: {dim}"

        Z = Z.reshape(S.shape).cpu().numpy()

        # ---- plot ----
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        norm = mcolors.PowerNorm(
            gamma=gamma,
            vmin=np.min(Z) if vmin is None else vmin,
            vmax=np.max(Z) if vmax is None else vmax,
        )

        c = ax.pcolormesh(
            T,
            S,
            Z,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )
        plt.colorbar(c, ax=ax, label=label)

        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim([s.min(), s.max()])

        if title is None:
            title = f"NN output: {label}"
        ax.set_title(title)

        return ax
