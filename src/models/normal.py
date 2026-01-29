from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.math import inv_softplus
from src.models.degradation import DegModel


class NormalDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mean_Ts = y_pred[..., 0]
        var_Ts = y_pred[..., 1]
        std = torch.sqrt(var_Ts.clamp_min(1e-6))
        dist_s = dist.Normal(mean_Ts, std)
        loss = -(dist_s.log_prob(y_true)).mean()
        return loss


class NormalDegradationModel(DegModel):

    def __init__(self, onset: float = 0.0):
        super().__init__(onset=onset)

        self.m0_raw = nn.Parameter(torch.logit(torch.tensor(0.9999)))
        self.m1_raw = nn.Parameter(inv_softplus(torch.tensor(100.0)))
        self.p_raw = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.v0_raw = nn.Parameter(inv_softplus(torch.tensor(0.0001)))
        self.v1_raw = nn.Parameter(inv_softplus(torch.tensor(1.0)))

    def get_state_vector(self) -> torch.Tensor:
        return torch.stack(
            [
                self.m0_raw,
                self.m1_raw,
                self.p_raw,
                self.v0_raw,
                self.v1_raw,
            ]
        )

    @classmethod
    def get_state_names(self) -> list[str]:
        return ["m0_raw", "m1_raw", "p_raw", "v0_raw", "v1_raw"]

    def set_state_vector(self, states: torch.Tensor) -> None:
        """
        Set raw parameters from an external estimator.

        Parameters
        ----------
        states : torch.Tensor
            Shape [5] = (m0_raw, m1_raw, p_raw, v0_raw, v1_raw)
        """
        assert states.shape == (5,), f"Expected shape (5,), got {states.shape}"

        with torch.no_grad():
            self.m0_raw.copy_(states[0])
            self.m1_raw.copy_(states[1])
            self.p_raw.copy_(states[2])
            self.v0_raw.copy_(states[3])
            self.v1_raw.copy_(states[4])

    def tuple_forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean, variance) as separate tensors.
        Convenience wrapper around forward().
        """
        params = self.forward(s)
        mean = params[..., 0]
        var = params[..., 1]
        return mean, var

    @staticmethod
    def forward_with_stateeters(
        s: torch.Tensor,  # [B]
        states: torch.Tensor,  # [K, RP]
    ) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        states: Tensor of shape [K, RP] RP = 5
        returns: Tensor of shape [B, K, DP] RP = 2
        """
        # unpack raw params
        m0_raw, m1_raw, p_raw, v0_raw, v1_raw = states.unbind(-1)

        # constrain
        m0 = torch.sigmoid(m0_raw)  # [K]
        m1 = F.softplus(m1_raw)  # [K]
        p = F.softplus(p_raw)  # [K]
        v0 = F.softplus(v0_raw)  # [K]
        v1 = F.softplus(v1_raw)  # [K]

        # add batch dim ONCE → [1, K]
        m0, m1, p, v0, v1 = [x.unsqueeze(0) for x in (m0, m1, p, v0, v1)]

        # add component dim ONCE → [B, 1]
        s = s.unsqueeze(1)

        # degradation law
        mean = m1 * torch.clamp(1.0 - s / m0, min=0.0).pow(p)  # [B, K]
        var = 0.25 + (v0 + v1 * s).pow(2)  # [B, K]

        return torch.stack([mean, var], dim=-1)  # [B, K, 2]

    @staticmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Normal:
        """
        params: [..., 2] = (mean, variance)
        """
        mean = params[..., 0]
        var = params[..., 1]
        std = torch.sqrt(var.clamp_min(1e-6))
        return dist.Normal(mean, std)
