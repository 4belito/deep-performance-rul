from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from src.helpers.math import inv_softplus
from src.models.degradation.base import DegModel


class GammaDegradation(DegModel):
    min_so_gab = 1e-2
    min_to_gab = 1.0
    min_dmc = 0.001
    onset_left = 0.3
    onset_right = 0.3
    max_life = 100.0

    def __init__(self, onset: float | None = None):
        super().__init__(onset=onset)

        # -------------------------
        # Degradation mean
        # -------------------------
        # dmy > 1 (singularity outside domain)
        self.raw_dmy = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.raw_dmx = nn.Parameter(inv_softplus(torch.tensor(100.0)))
        self.raw_dmc = nn.Parameter(inv_softplus(torch.tensor(1.0)))

        # -------------------------
        # Degradation variance
        # -------------------------
        self.raw_dvx = nn.Parameter(inv_softplus(torch.tensor(1.0)))
        self.raw_dvs = nn.Parameter(inv_softplus(torch.tensor(1.0)))

        # -------------------------
        # Nominal regime
        # -------------------------
        self.raw_nmy = nn.Parameter(torch.logit(torch.tensor(0.9)))

        # -------------------------
        # Small displacement of to
        # -------------------------
        self.raw_to = nn.Parameter(torch.logit(torch.tensor(0.5)))

    # --------------------------------------------------
    # State definition
    # --------------------------------------------------
    @staticmethod
    def get_state_names() -> list[str]:
        return [
            "raw_dmy",
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
            "raw_dmy": "Degradation mean y-asymptote (>1)",
            "raw_dmx": "Degradation mean x-scale",
            "raw_dmc": "Degradation curvature exponent",
            "raw_dvx": "Variance intercept",
            "raw_dvs": "Variance slope",
            "raw_nmy": "Nominal mean level",
            "raw_to": "Onset displacement",
        }

    @classmethod
    def name(cls) -> str:
        return "gamma_flexible"

    # --------------------------------------------------
    # Freeze helpers
    # --------------------------------------------------
    def freeze_to(self):
        self.raw_to.requires_grad = False

    def unfreeze_to(self):
        self.raw_to.requires_grad = True

    # --------------------------------------------------
    # Core forward
    # --------------------------------------------------
    @classmethod
    def forward_with_states(
        cls,
        s: torch.Tensor,
        states: torch.Tensor,
        onsets: torch.Tensor,
    ) -> torch.Tensor:

        (
            raw_dmy,
            raw_dmx,
            raw_dmc,
            raw_dvx,
            raw_dvs,
            raw_nmy,
            raw_to,
        ) = (x.unsqueeze(0) for x in states.unbind(-1))

        # ---- learned onset ----
        onsets = onsets.view(1, -1)
        lower_to = onsets * (1 - cls.onset_left)
        higher_to = onsets + (cls.max_life - onsets) * cls.onset_right
        to = lower_to + (higher_to - lower_to) * torch.sigmoid(raw_to)

        # ---- degradation geometry ----
        dmx = to + cls.min_to_gab + F.softplus(raw_dmx)
        dmc = cls.min_dmc + F.softplus(raw_dmc)

        # 🔥 dmy strictly > 1
        dmy = F.softplus(raw_dmy)

        ratio = to / dmx
        so = dmy * (1.0 - ratio.pow(1.0 / dmc))

        # ---- variances ----
        dvx = F.softplus(raw_dvx)
        dvs = F.softplus(raw_dvs)

        nmy_min = so + cls.min_so_gab
        nmy = nmy_min + (1 - nmy_min) * torch.sigmoid(raw_nmy)

        print(f"dmy: {dmy.min().item():.10f} - {dmy.max().item():.16f}")
        print(f"dmx: {dmx.min().item():.16f} - {dmx.max().item():.16f}")
        print(f"dmc: {dmc.min().item():.16f} - {dmc.max().item():.16f}")
        print(f"dvx: {dvx.min().item():.16f} - {dvx.max().item():.16f}")
        print(f"dvs: {dvs.min().item():.16f} - {dvs.max().item():.16f}")
        print(f"to: {to.min().item():.16f} - {to.max().item():.16f}")
        print(f"so: {so.min().item():.16f} - {so.max().item():.16f}")
        print(f"nmy: {nmy.min().item():.16f} - {nmy.max().item():.16f}")
        print(f"ratio: {ratio.min().item():.16f} - {ratio.max().item():.16f}")
        diff = 1.0 - (to / dmx).pow(1.0 / dmc)
        print(f"(1 - (to/dmx)^(1/dmc)): {diff.min().item():.16f} - {diff.max().item():.16f}")
        print(f"to/dmx: {(to/dmx).min().item():.16f} - {(to/dmx).max().item():.16f}")

        # ---- allocate ----
        B = s.shape[0]
        K = states.shape[0]

        s = s.expand(B, K)
        mean = torch.zeros(B, K, device=s.device)

        mask_nom = s > so
        mask_deg = ~mask_nom

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

        # -------------------------
        # Degradation branch
        # -------------------------
        if mask_deg.any():
            s_ = s[mask_deg]
            dmy_ = dmy.expand(B, K)[mask_deg]
            dmx_ = dmx.expand(B, K)[mask_deg]
            dmc_ = dmc.expand(B, K)[mask_deg]

            # Extra safety (should never activate now)
            base = (1.0 - s_ / dmy_).clamp_min(1e-8)
            mean[mask_deg] = dmx_ * base.pow(dmc_)

        # -------------------------
        # Variance
        # -------------------------
        dvx_ = dvx.expand(B, K)
        dvs_ = dvs.expand(B, K)

        var = (0.25 + dvx_ + dvs_ * s).pow(2)

        shape = mean.pow(2) / var
        rate = mean / var

        return torch.stack([shape, rate], dim=-1)

    # --------------------------------------------------
    # Distribution builder
    # --------------------------------------------------
    @staticmethod
    def build_distribution_from_params(params: torch.Tensor) -> dist.Gamma:
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
