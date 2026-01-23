from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.math import inv_softplus
from src.models.deg_model import DEGModel


class NormalDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mean_Ts = y_pred[..., 0]
        var_Ts  = y_pred[..., 1]
        std = torch.sqrt(var_Ts.clamp_min(1e-6))
        dist_s = dist.Normal(mean_Ts, std)
        loss = -(dist_s.log_prob(y_true)).mean()
        return loss


class NormalDegradationModel(DEGModel):

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
        # unpack raw params
        m0_raw, m1_raw, p_raw, v0_raw, v1_raw = raw_params.unbind(-1)

        # constrain
        m0 = torch.sigmoid(m0_raw)      # [K]
        m1 = F.softplus(m1_raw)         # [K]
        p  = F.softplus(p_raw)          # [K]
        v0 = F.softplus(v0_raw)         # [K]
        v1 = F.softplus(v1_raw)         # [K]

        # reshape for broadcasting
        s  = s[:, None]   # [B, 1]
        m0 = m0[None, :]  # [1, K]
        m1 = m1[None, :]
        p  = p[None, :]
        v0 = v0[None, :]
        v1 = v1[None, :]

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

    
