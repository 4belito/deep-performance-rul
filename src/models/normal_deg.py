from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(x, eps=1e-6):
    x = torch.as_tensor(x)
    return torch.where(
        x > 20.0,
        x,
        torch.log(torch.expm1(torch.clamp(x, min=eps)))
    )

class NormalDegradationModel(nn.Module):
    MEAN_PARAMS = ["m0_raw", "m1_raw", "p_raw"]
    VAR_PARAMS  = ["v0", "v1"]
    
    def __init__(self,onset: float = 0.0):
        super().__init__()

        self.register_buffer("onset", torch.tensor(float(onset)))
        self.m0_raw = nn.Parameter(torch.logit(torch.tensor(0.9999)))
        self.m1_raw  = nn.Parameter(inv_softplus(torch.tensor(100)))
        self.p_raw  = nn.Parameter(inv_softplus(torch.tensor(1.0)))

        # variance (simple)
        self.v0_raw = nn.Parameter(inv_softplus(torch.tensor(0.0001)))
        self.v1_raw = nn.Parameter(inv_softplus(torch.tensor(1.0)))
    
        
    @property
    def m0(self):
        return F.sigmoid(self.m0_raw)
    
    @property
    def m1(self):
        return F.softplus(self.m1_raw)
    
    
    @property
    def p(self):
        return F.softplus(self.p_raw)
    
    @property
    def v0(self):
        return F.softplus(self.v0_raw)
    
    @property
    def v1(self):
        return F.softplus(self.v1_raw)
    
    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_Ts = self.m1 * torch.clamp(1-s/self.m0,min=0).pow(self.p)
        var_Ts  = 0.25 + (self.v0 + self.v1 * s).pow(2)
        return mean_Ts, var_Ts

    def distribution(self, s: torch.Tensor) -> dist.Normal:
        m_s, v_s = self(s)
        return dist.Normal(m_s, torch.sqrt(v_s.clamp_min(1e-6)))
    
    def freeze_params(self, names):
        for n, p in self.named_parameters():
            if any(n.startswith(name) for name in names):
                p.requires_grad_(False)

    def unfreeze_params(self, names):
        for n, p in self.named_parameters():
            if any(n.startswith(name) for name in names):
                p.requires_grad_(True)
                
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
            mean_Ts, _ = self(torch.tensor(s, dtype=torch.float32, device=device))

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
    

class NormalDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mean_Ts, var_Ts = y_pred
        t = y_true
        dist_s = dist.Normal(mean_Ts, torch.sqrt(var_Ts.clamp_min(1e-6)))
        loss = -(dist_s.log_prob(t)).mean()
        return loss
    
