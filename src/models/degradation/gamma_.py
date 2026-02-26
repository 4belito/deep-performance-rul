from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from src.helpers.math import inv_softplus
from src.models.degradation.base import DegModel


class GammaDegradation(DegModel):
    max_nvy = 100.0  # avoid numerical issues with (dvx + dvs * so + nvy * (s - so))^2 (100)
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
            "raw_nmy": "Nominal mean on performance/y-axis",
            "raw_nvy": "Nominal variance on performance/y-axis",
        }

    @classmethod
    def name(cls) -> str:
        return "gamma"

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
        raw_dmy, raw_dmx, raw_dmc, raw_dvx, raw_dvs, raw_nmy, raw_nvy = (
            x.unsqueeze(0) for x in states.unbind(-1)
        )
        to = onsets.view(1, -1)  # [1, K]

        # --------------------------------------------------
        # constrain
        # --------------------------------------------------

        dmx = to + cls.min_to_gab + F.softplus(raw_dmx)  # [1,K]
        dmc = cls.min_dmc + F.softplus(raw_dmc)  # [1,K]
        dmy = torch.sigmoid(raw_dmy)  # [1,K]

        ratio = to / dmx
        so = dmy * (1.0 - ratio.pow(1.0 / dmc))  # [1, K]

        dvx = F.softplus(raw_dvx)  # [1,K]
        dvs = F.softplus(raw_dvs)  # [1,K]

        nmy_min = so + cls.min_so_gab  # ensure nmy > so for numerical stability
        nmy = nmy_min + (1 - nmy_min) * torch.sigmoid(raw_nmy)  # [1,K]
        nvy = cls.max_nvy * torch.sigmoid(raw_nvy)  # [1,K]

        print(f"dmy: {dmy.min().item():.10f} - {dmy.max().item():.16f}")
        print(f"dmx: {dmx.min().item():.16f} - {dmx.max().item():.16f}")
        print(f"dmc: {dmc.min().item():.16f} - {dmc.max().item():.16f}")
        print(f"dvx: {dvx.min().item():.16f} - {dvx.max().item():.16f}")
        print(f"dvs: {dvs.min().item():.16f} - {dvs.max().item():.16f}")
        print(f"to: {to.min().item():.16f} - {to.max().item():.16f}")
        print(f"so: {so.min().item():.16f} - {so.max().item():.16f}")
        print(f"nmy: {nmy.min().item():.16f} - {nmy.max().item():.16f}")
        print(f"nvy: {nvy.min().item():.16f} - {nvy.max().item():.16f}")
        diff = 1.0 - (to / dmx).pow(1.0 / dmc)
        print(f"(1 - (to/dmx)^(1/dmc)): {diff.min().item():.16f} - {diff.max().item():.16f}")
        print(f"to/dmx: {(to/dmx).min().item():.16f} - {(to/dmx).max().item():.16f}")

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

            # parameters for nominal branch
            nmy_ = nmy.expand(B, K)[mask_nom]
            nvy_ = nvy.expand(B, K)[mask_nom]

            dvx_ = dvx.expand(B, K)[mask_nom]
            dvs_ = dvs.expand(B, K)[mask_nom]

            den = (nmy_ - so_).clamp_min(1e-6)
            num = (nmy_ - s_).clamp_min(1e-6)
            mean[mask_nom] = torch.clamp(to_ * num / den, min=1e-6)
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

        return torch.stack([shape, rate], dim=-1)

    # ------------------------------------------------------------------
    # Distribution builder
    # ------------------------------------------------------------------
    @staticmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Normal:
        shape = params[..., 0]
        rate = params[..., 1]
        return Gamma(shape, rate)


class GammaDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        shape_Ts = y_pred[:, [0]]
        rate_Ts = y_pred[:, [1]]
        dist_s = Gamma(shape_Ts, rate_Ts)
        loss = -(dist_s.log_prob(y_true)).mean()
        return loss
