from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.math import inv_softplus
from src.models.degradation import DegModel


class NormalDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mean_Ts = y_pred[..., 0]
        var_Ts  = y_pred[..., 1]
        std = torch.sqrt(var_Ts.clamp_min(1e-6))
        dist_s = dist.Normal(mean_Ts, std)
        loss = -(dist_s.log_prob(y_true)).mean()
        return loss


class NormalDegradationModel(DegModel):

    def __init__(self, onset: float = 0.0):
        super().__init__(onset=onset)

        self.m0_raw = nn.Parameter(torch.logit(torch.tensor(0.9999)))
        self.m1_raw = nn.Parameter(inv_softplus(torch.tensor(100.0)))
        self.p_raw  = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.v0_raw = nn.Parameter(inv_softplus(torch.tensor(0.0001)))
        self.v1_raw = nn.Parameter(inv_softplus(torch.tensor(1.0)))

    def get_raw_param_vector(self) -> torch.Tensor:
        return torch.stack([
            self.m0_raw,
            self.m1_raw,
            self.p_raw,
            self.v0_raw,
            self.v1_raw,
        ])

    def tuple_forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean, variance) as separate tensors.
        Convenience wrapper around forward().
        """
        params = self.forward(s)
        mean = params[..., 0]
        var  = params[..., 1]
        return mean, var
                
        
    @staticmethod
    def forward_with_raw_parameters(
        s: torch.Tensor,          # [B]
        raw_params: torch.Tensor, # [K, RP]
    ) -> torch.Tensor:
        """
            s: Tensor of shape [B]
            raw_params: Tensor of shape [K, RP] RP = 5
            returns: Tensor of shape [B, K, DP] RP = 2
        """
        # unpack raw params
        m0_raw, m1_raw, p_raw, v0_raw, v1_raw = raw_params.unbind(-1)

        # constrain
        m0 = torch.sigmoid(m0_raw)      # [K]
        m1 = F.softplus(m1_raw)         # [K]
        p  = F.softplus(p_raw)          # [K]
        v0 = F.softplus(v0_raw)         # [K]
        v1 = F.softplus(v1_raw)         # [K]

        # add batch dim ONCE → [1, K]
        m0, m1, p, v0, v1 = [x.unsqueeze(0) for x in (m0, m1, p, v0, v1)]

        # add component dim ONCE → [B, 1]
        s = s.unsqueeze(1)
        
        # degradation law
        mean = m1 * torch.clamp(1.0 - s / m0, min=0.0).pow(p)   # [B, K]
        var  = 0.25 + (v0 + v1 * s).pow(2)                      # [B, K]

        return torch.stack([mean, var], dim=-1)  # [B, K, 2]

    @staticmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Normal:
        """
        params: [..., 2] = (mean, variance)
        """
        mean = params[..., 0]
        var  = params[..., 1]
        std  = torch.sqrt(var.clamp_min(1e-6))
        return dist.Normal(mean, std)

    def plot_distribution(
        self,
        t: np.ndarray,
        s: np.ndarray,
        func: str = "pdf",
        ax: plt.Axes = None,
        vmax: float = None,
        gamma_prob: float = 0.3,
        title: str = "Normal degradation PDF of $T_s$",
        plot_mean: bool = True,
        mean_kwargs: dict | None = None,
    ):
        device = next(self.parameters()).device

        #t = t[t>=self.onset]
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

            # ---- mean curve ----
            mean_Ts,_ = self.tuple_forward(torch.tensor(s, dtype=torch.float32, device=device))

        Z = Z.reshape(S.shape).cpu().numpy()
        mean_Ts = mean_Ts.cpu().numpy()

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        onset = self.onset.cpu().item()
        ax.axvline(x=onset, linestyle="--",color="#4CC9F0", label="onset")
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
                mean_kwargs = dict(color="orange", lw=2, label="mean")
            ax.plot(mean_Ts, s, **mean_kwargs)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim([0, t.max()])

        
        ax.legend()

        return ax
    
