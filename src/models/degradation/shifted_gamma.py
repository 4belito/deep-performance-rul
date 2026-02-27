"""
Shifted Gamma degradation model. The degradation mean follows a shifted power law, and the variance grows quadratically with time/performance. The nominal regime has a mean that decreases linearly and a variance that grows quadratically with time/performance.

This is in an experimental stage, and is not part of the results of the paper.
"""

from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.math import inv_softplus
from src.models.degradation.base import DegModel
from src.models.distributions.shifted_gamma import ShiftedGamma


class ShiftedGammaDegradation(DegModel):
    max_nvy = 100.0  # avoid numerical issues with (dvx + dvs * so + nvy * (s - so))^2
    min_so_gab = 1e-3  # avoid numerical issues with (nmy - s) / (nmy - so)
    min_to_gab = 1  # mean EOL 1 cycle after degradation onset
    min_dmc = 0.05  # minimal curvature to avoid numerical issues with (1 - (to/dmx)^(1/dmc))

    def __init__(self, onset: float | None = None):
        super().__init__(onset=onset)

        # --------------------------------------------------
        # Raw (unconstrained) parameters
        # t : time (x-axis)
        # s : performance (y-axis):
        # ---------------------------------------
        #
        # Degradation mean parameters
        self.raw_dmy = nn.Parameter(torch.logit(torch.tensor(0.9999)))  # mean on y-axis
        self.raw_dmx = nn.Parameter(inv_softplus(torch.tensor(100.0)))  # mean on x-axis
        self.raw_dmc = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # mean curvature

        # Degradation variance parameters
        self.raw_dvx = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # variance intercept
        self.raw_dvs = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # variance slope

        # Nominal regime parameters
        self.raw_nmy = nn.Parameter(torch.logit(torch.tensor(0.9)))  # nominal mean on y-axis
        self.raw_nvy = nn.Parameter(
            torch.logit(torch.tensor(1.0 / self.max_nvy))
        )  # nominal variance slope

        # Distribution shift
        self.raw_loc = nn.Parameter(inv_softplus(torch.tensor(0.1)))  # gamma shift

    # ------------------------------------------------------------------
    # State definition
    # ------------------------------------------------------------------
    @staticmethod
    def get_state_names() -> list[str]:
        return [
            "raw_dmy",
            "raw_dmx",
            "raw_dmc",
            "raw_dvx",
            "raw_dvs",
            "raw_loc",
            "raw_nmy",
            "raw_nvy",
        ]

    @staticmethod
    def get_state_semantics() -> dict[str, str]:
        return {
            "raw_dmy": "Degradation mean on performance/y-axis)",
            "raw_dmx": "Degradation mean on time/x-axis",
            "raw_dmc": "Degradation mean curvature exponent",
            "raw_dvx": "Degradation variance on time/x-axis",
            "raw_dvs": "Degradation variance slope",
            "raw_loc": "Gamma time shift",
            "raw_nmy": "Nominal mean on performance/y-axis",
            "raw_nvy": "Nominal variance on performance/y-axis",
        }

    @classmethod
    def name(cls) -> str:
        return "shifted_gamma"

    # ------------------------------------------------------------------
    # Core forward with masking
    # ------------------------------------------------------------------
    @classmethod
    def forward_with_states(
        cls,
        s: torch.Tensor,  # [B,1]
        states: torch.Tensor,  # [K, RP]
        onsets: torch.Tensor,  # [K, 1]
    ) -> torch.Tensor:
        """
        s: Tensor of shape [B]
        states: Tensor of shape [K, RP]
        returns: Tensor of shape [B, K, 2]
        """
        # --------------------------------------------------
        # unpack raw params
        # --------------------------------------------------
        raw_dmy, raw_dmx, raw_dmc, raw_dvx, raw_dvs, raw_loc, raw_nmy, raw_nvy = (
            x.unsqueeze(0) for x in states.unbind(-1)
        )
        to = onsets.view(1, -1)  # [1, K]

        # --------------------------------------------------
        # constrain
        # --------------------------------------------------

        loc = -F.softplus(raw_loc)  # [1,K]
        dmx = to - loc + cls.min_to_gab + F.softplus(raw_dmx)  # [1,K]
        dmc = cls.min_dmc + F.softplus(raw_dmc)  # [1,K]
        dmy = torch.sigmoid(raw_dmy)  # [1,K]

        ratio = (to - loc) / dmx
        so = dmy * (1.0 - ratio.pow(1.0 / dmc))  # [1, K]

        dvx = F.softplus(raw_dvx)  # [1,K]
        dvs = F.softplus(raw_dvs)  # [1,K]

        nmy_min = so + cls.min_so_gab  # ensure nmy > so for numerical stability
        nmy = nmy_min + (1 - nmy_min) * torch.sigmoid(raw_nmy)  # [1,K]
        nvy = cls.max_nvy * torch.sigmoid(raw_nvy)  # [1,K]

        # --------------------------------------------------
        # compute s_onset
        # --------------------------------------------------

        # --------------------------------------------------
        # allocate outputs
        # --------------------------------------------------
        B = s.shape[0]
        K = states.shape[0]
        s = s.expand(B, K)
        mean = torch.zeros(B, K, device=s.device)
        var = torch.zeros(B, K, device=s.device)
        # --------------------------------------------------
        # masks
        # --------------------------------------------------
        mask_nom = s > so
        mask_deg = ~mask_nom

        # ==================================================
        # Nominal branch: s > s_onset
        # ==================================================
        if mask_nom.any():
            s_ = s[mask_nom]
            so_ = so.expand(B, K)[mask_nom]
            to_ = to.expand(B, K)[mask_nom]
            loc_ = loc.expand(B, K)[mask_nom]

            # parameters for nominal branch
            nmy_ = nmy.expand(B, K)[mask_nom]
            nvy_ = nvy.expand(B, K)[mask_nom]

            dvx_ = dvx.expand(B, K)[mask_nom]
            dvs_ = dvs.expand(B, K)[mask_nom]

            den = (nmy_ - so_).clamp_min(1e-6)
            num = (nmy_ - s_).clamp_min(1e-6)
            mean[mask_nom] = torch.clamp((to_ - loc_) * num / den + loc_, min=1e-6)
            var[mask_nom] = 0.25 + (dvx_ + dvs_ * so_ + nvy_ * (s_ - so_)).pow(2)

        # ==================================================
        # Degradation branch: s <= s_onset
        # ==================================================
        if mask_deg.any():
            s_ = s[mask_deg]
            dmy_ = dmy.expand(B, K)[mask_deg]
            dmx_ = dmx.expand(B, K)[mask_deg]
            dmc_ = dmc.expand(B, K)[mask_deg]
            dvx_ = dvx.expand(B, K)[mask_deg]
            dvs_ = dvs.expand(B, K)[mask_deg]

            mean[mask_deg] = dmx_ * torch.clamp(1.0 - s_ / dmy_, min=1e-6).pow(dmc_)
            var[mask_deg] = 0.25 + (dvx_ + dvs_ * s_).pow(2)

        shape = mean.pow(2) / var.clamp_min(1e-6)
        rate = mean / var.clamp_min(1e-6)
        shift = loc.expand(B, K)

        return torch.stack([shape, rate, shift], dim=-1)

    # ------------------------------------------------------------------
    # Distribution builder
    # ------------------------------------------------------------------
    @staticmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Normal:
        shape = params[..., 0]
        rate = params[..., 1]
        shift = params[..., 2]
        return ShiftedGamma(shape, rate, shift)


class ShiftedGammaDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        shape_Ts = y_pred[:, [0]]
        rate_Ts = y_pred[:, [1]]
        shift_Ts = y_pred[:, [2]]
        dist_s = ShiftedGamma(shape_Ts, rate_Ts, shift_Ts)
        loss = -(dist_s.log_prob(y_true)).mean()
        return loss
