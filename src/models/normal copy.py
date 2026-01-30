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

        # raw (unconstrained) state parameters
        self.m0_raw = nn.Parameter(torch.logit(torch.tensor(0.9999)))
        self.m1_raw = nn.Parameter(inv_softplus(torch.tensor(100.0)))
        self.mp_raw = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.v0_raw = nn.Parameter(inv_softplus(torch.tensor(0.0001)))
        self.v1_raw = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.mn_raw = nn.Parameter(torch.logit(torch.tensor(0.9)))

    # ------------------------------------------------------------------
    # State definition
    # ------------------------------------------------------------------
    @staticmethod
    def get_state_names() -> list[str]:
        return [
            "m0_raw",
            "m1_raw",
            "mp_raw",
            "v0_raw",
            "v1_raw",
            "mn_raw",
        ]

    # ------------------------------------------------------------------
    # Core forward with masking (NO torch.where)
    # ------------------------------------------------------------------
    @staticmethod
    def forward_with_states(
        s: torch.Tensor,  # [B,1]
        states: torch.Tensor,  # [K, S]
        onsets: torch.Tensor,  # [K, 1]
    ) -> torch.Tensor:
        """
        s: Tensor of shape [B,1]
        states: Tensor of shape [K, RP]
        returns: Tensor of shape [B, K, 2]
        """
        # --------------------------------------------------
        # unpack raw params
        # --------------------------------------------------
        m0_raw, m1_raw, mp_raw, v0_raw, v1_raw, mn_raw = [x.unsqueeze(0) for x in states.unbind(-1)]

        # --------------------------------------------------
        # constrain
        # --------------------------------------------------
        m0 = torch.sigmoid(m0_raw)  # [1,K]
        m1 = F.softplus(m1_raw)  # [1,K]
        mp = F.softplus(mp_raw)  # [1,K]
        v0 = F.softplus(v0_raw)  # [1,K]
        v1 = F.softplus(v1_raw)  # [1,K]
        mn = torch.sigmoid(mn_raw)  # [K]

        t_onsets = torch.tensor(onsets.view(1, -1).item())  # [1, K]
        # --------------------------------------------------
        # onset (guarded)
        # --------------------------------------------------
        ratio = torch.clamp(t_onsets / m1, max=1.0 - 1e-6)
        s_onset = m0 * (1.0 - ratio.pow(1.0 / mp))  # [1, K]

        # --------------------------------------------------
        # allocate outputs
        # --------------------------------------------------
        B = s.shape[0]
        K = states.shape[0]
        s = s.expand(B, K).unsqueeze(1)  # [B, 1]
        mean = torch.zeros_like(s)
        var = 0.25 + (v0 + v1 * s).pow(2)

        # --------------------------------------------------
        # masks
        # --------------------------------------------------
        mask_nom = s > s_onset
        mask_deg = ~mask_nom

        # --------------------------------------------------
        # degradation branch (s <= s_onset)
        # --------------------------------------------------
        if mask_deg.any():
            mean[mask_deg] = m1.expand_as(mean)[mask_deg] * torch.clamp(
                1.0 - s.expand_as(mean)[mask_deg] / m0.expand_as(mean)[mask_deg], min=0.0
            ).pow(mp.expand_as(mean)[mask_deg])

        # --------------------------------------------------
        # nominal branch (s > s_onset)
        # --------------------------------------------------
        if mask_nom.any():
            den = torch.clamp(
                mn.expand_as(mean)[mask_nom] - s_onset.expand_as(mean)[mask_nom], min=1e-6
            )
            mean[mask_nom] = (
                t_onsets * (mn.expand_as(mean)[mask_nom] - s.expand_as(mean)[mask_nom]) / den
            )
        # --------------------------------------------------
        return torch.stack([mean, var], dim=-1)

    # ------------------------------------------------------------------
    # Distribution builder
    # ------------------------------------------------------------------
    @staticmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Normal:
        mean = params[..., 0]
        var = params[..., 1]
        std = torch.sqrt(var.clamp_min(1e-6))
        return dist.Normal(mean, std)
