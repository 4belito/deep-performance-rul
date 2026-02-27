"""
Gamma degradation model with a censored mean.

The censored mean limits the influence of non-terminated units in mixture
models, preventing large expected RUL values from biasing the mixture mean.
The likelihood remains unchanged.

Experimental variant, not used in the paper results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.math import inv_softplus
from src.models.degradation.base import DegModel
from src.models.distributions.censoredmean_gamma import CensoredMeanGamma


class CensoredMeanGammaDegradation(DegModel):
    min_so_gab = 1e-2
    min_to_gab = 1.0
    min_dmc = 0.001
    onset_left = 0.2
    onset_right = 0.2
    max_life = 100.0
    null_mean_value = 1e-6
    null_var_value = 1e-6

    def __init__(self, onset: float | None = None, init_s: float | None = None):
        super().__init__(onset=onset, init_s=init_s)

        # -------------------------
        # Learn s0 directly in (0,1)
        # -------------------------
        self.raw_so = nn.Parameter(torch.logit(torch.tensor(0.99)))

        # -------------------------
        # Degradation geometry
        # -------------------------
        self.raw_dmx = nn.Parameter(inv_softplus(torch.tensor(100.0)))
        self.raw_dmc = nn.Parameter(inv_softplus(torch.tensor(1.0)))

        # -------------------------
        # Variance
        # -------------------------
        self.raw_dvx = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.raw_dvs = nn.Parameter(inv_softplus(torch.tensor(1.0)))

        # -------------------------
        # Nominal mean
        # -------------------------
        self.raw_nmy = nn.Parameter(torch.logit(torch.tensor(0.9)))

        # -------------------------
        # Small displacement of to
        # -------------------------
        self.raw_to = nn.Parameter(torch.logit(torch.tensor(0.5)))

    @staticmethod
    def get_state_names() -> list[str]:
        return [
            "raw_so",
            "raw_dmx",
            "raw_dmc",
            "raw_dvx",
            "raw_dvs",
            "raw_nmy",
            "raw_to",
        ]

    @staticmethod
    def get_state_semantics() -> dict[str, str]:
        return {
            "raw_so": "Degradation regime boundary s0 in (0,1)",
            "raw_dmx": "Degradation x-scale",
            "raw_dmc": "Degradation curvature exponent",
            "raw_dvx": "Variance intercept",
            "raw_dvs": "Variance slope",
            "raw_nmy": "Nominal mean level",
            "raw_to": "Onset displacement",
        }

    @classmethod
    def name(cls) -> str:
        return f"gamma_censoredmean_onset{cls.onset_left}-{cls.onset_right}"

    @classmethod
    def forward_with_states(
        cls,
        s: torch.Tensor,
        states: torch.Tensor,
        onsets: torch.Tensor,
        init_s: torch.Tensor,
    ) -> torch.Tensor:

        (
            raw_so,
            raw_dmx,
            raw_dmc,
            raw_dvx,
            raw_dvs,
            raw_nmy,
            raw_to,
        ) = (x.unsqueeze(0) for x in states.unbind(-1))

        # -------------------------
        # Learn s0 in (0,1)
        # -------------------------

        # -------------------------
        # Learned onset
        # -------------------------
        onsets = onsets.view(1, -1)
        init_s = init_s.view(1, -1)
        lower_to = onsets * (1 - cls.onset_left)
        higher_to = onsets + (cls.max_life - onsets) * cls.onset_right
        to = lower_to + (higher_to - lower_to) * torch.sigmoid(raw_to)

        # -------------------------
        # Degradation geometry
        # -------------------------
        dmx = to + cls.min_to_gab + F.softplus(raw_dmx)
        dmc = cls.min_dmc + F.softplus(raw_dmc)

        ratio = to / dmx

        A = 1.0 - ratio.pow(1.0 / dmc)
        so = torch.min(init_s, A) * torch.sigmoid(raw_so)
        # 🔥 compute dmy from so
        dmy = so / A.clamp_min(1e-8)

        # -------------------------
        # Variances
        # -------------------------
        dvx = F.softplus(raw_dvx)
        dvs = F.softplus(raw_dvs)

        nmy_min = so + cls.min_so_gab
        nmy = nmy_min + (1 - nmy_min) * torch.sigmoid(raw_nmy)

        # print(f"dmy: {dmy.min().item():.10f} - {dmy.max().item():.16f}")
        # print(f"dmx: {dmx.min().item():.16f} - {dmx.max().item():.16f}")
        # print(f"dmc: {dmc.min().item():.16f} - {dmc.max().item():.16f}")
        # print(f"dvx: {dvx.min().item():.16f} - {dvx.max().item():.16f}")
        # print(f"dvs: {dvs.min().item():.16f} - {dvs.max().item():.16f}")
        # print(f"to: {to.min().item():.16f} - {to.max().item():.16f}")
        # print(f"so: {so.min().item():.16f} - {so.max().item():.16f}")
        # print(f"nmy: {nmy.min().item():.16f} - {nmy.max().item():.16f}")
        # print(f"ratio: {ratio.min().item():.16f} - {ratio.max().item():.16f}")
        # diff = 1.0 - (to / dmx).pow(1.0 / dmc)
        # print(f"(1 - (to/dmx)^(1/dmc)): {diff.min().item():.16f} - {diff.max().item():.16f}")
        # print(f"to/dmx: {(to/dmx).min().item():.16f} - {(to/dmx).max().item():.16f}")

        # -------------------------
        # Allocate
        # -------------------------
        B = s.shape[0]
        K = states.shape[0]

        s = s.expand(B, K)
        mean = torch.zeros(B, K, device=s.device)
        var = torch.zeros(B, K, device=s.device)

        mask_null = s > init_s
        mask_nom = (s > so) & (s <= init_s)
        mask_deg = s <= so

        # -------------------------
        # Nominal branch
        # -------------------------
        if mask_nom.any():
            s_ = s[mask_nom]
            so_ = so.expand(B, K)[mask_nom]
            to_ = to.expand(B, K)[mask_nom]
            nmy_ = nmy.expand(B, K)[mask_nom]

            den = (nmy_ - so_).clamp_min(1e-6)
            num = (nmy_ - s_).clamp_min(1e-6)

            mean[mask_nom] = to_ * num / den

            dvx_ = dvx.expand(B, K)[mask_nom]
            dvs_ = dvs.expand(B, K)[mask_nom]

            var[mask_nom] = (0.25 + dvx_ + dvs_ * so_).pow(2)

        # -------------------------
        # Degradation branch
        # -------------------------
        if mask_deg.any():
            s_ = s[mask_deg]
            dmy_ = dmy.expand(B, K)[mask_deg]
            dmx_ = dmx.expand(B, K)[mask_deg]
            dmc_ = dmc.expand(B, K)[mask_deg]

            base = (1.0 - s_ / dmy_).clamp_min(1e-8)
            mean[mask_deg] = dmx_ * base.pow(dmc_)

            dvx_ = dvx.expand(B, K)[mask_deg]
            dvs_ = dvs.expand(B, K)[mask_deg]

            var[mask_deg] = (0.25 + dvx_ + dvs_ * s_).pow(2)

        mean = torch.where(mask_null, cls.null_mean_value, mean)
        var = torch.where(mask_null, cls.null_var_value, var)

        shape = mean.pow(2) / var
        rate = mean / var

        return torch.stack([shape, rate], dim=-1)

    @classmethod
    def build_distribution_from_params(cls, params: torch.Tensor) -> CensoredMeanGamma:
        shape = params[..., 0]
        rate = params[..., 1]
        return CensoredMeanGamma(shape, rate, cap=cls.max_life)


class CensoredMeanGammaDegradationNLL(nn.Module):
    max_life = CensoredMeanGammaDegradation.max_life

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        shape_Ts = y_pred[:, [0]]
        rate_Ts = y_pred[:, [1]]
        dist_s = CensoredMeanGamma(shape_Ts, rate_Ts, cap=self.max_life)
        return -(dist_s.log_prob(y_true)).mean()
