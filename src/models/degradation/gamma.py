from __future__ import annotations

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, TransformedDistribution, constraints
from torch.distributions.transforms import AffineTransform

from src.helpers.math import inv_softplus
from src.models.degradation.base import DegModel


class ShiftedGamma(TransformedDistribution):
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
        "shift": constraints.real,
    }

    def __init__(
        self,
        concentration: torch.Tensor,
        rate: torch.Tensor,
        shift: torch.Tensor,
        validate_args: bool | None = None,
    ):
        self.concentration = concentration
        self.rate = rate
        self.shift = shift

        base_dist = Gamma(
            concentration=concentration,
            rate=rate,
        )

        transform = AffineTransform(loc=shift, scale=1.0)

        super().__init__(
            base_dist,
            [transform],
            validate_args=validate_args,
        )

    @property
    def mean(self):
        return self.shift + self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    @property
    def base_mode(self):
        shape = self.concentration
        rate = self.rate
        return torch.where(shape >= 1, (shape - 1) / rate, torch.zeros_like(shape))

    @property
    def mode(self):
        return self.shift + self.base_mode


class GammaDegradation(DegModel):

    def __init__(self, onset: float | None = None):
        super().__init__(onset=onset)

        # --------------------------------------------------
        # Raw (unconstrained) parameters
        # t : time (x-axis)
        # s : performance (y-axis):
        # --------------------------------------------------

        # Degradation mean parameters
        self.raw_dmy = nn.Parameter(torch.logit(torch.tensor(0.9999)))  # mean on y axis
        self.raw_dmx = nn.Parameter(inv_softplus(torch.tensor(100.0)))  # mean on x axis
        self.raw_dmc = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # mean curvature

        # Degradation variance parameters
        self.raw_dvx = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # variance on x axis
        self.raw_dvs = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # variance slope

        # Distribution shift
        self.raw_loc = nn.Parameter(inv_softplus(torch.tensor(0.1)))  # gamma shift

        # Nominal regime parameters
        self.raw_nmy = nn.Parameter(torch.logit(torch.tensor(0.999)))  # nominal mean on y axis
        self.raw_nvy = nn.Parameter(inv_softplus(torch.tensor(1.0)))  # nominal variance on y axis

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
        return "gamma"

    # ------------------------------------------------------------------
    # Core forward with masking
    # ------------------------------------------------------------------
    @staticmethod
    def forward_with_states(
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

        # --------------------------------------------------
        # constrain
        # --------------------------------------------------
        dmy = 1e-6 + torch.sigmoid(raw_dmy)  # [1,K]
        dmx = 1e-6 + F.softplus(raw_dmx)  # [1,K]
        dmc = F.softplus(raw_dmc)  # [1,K]
        dvx = F.softplus(raw_dvx)  # [1,K]
        dvs = F.softplus(raw_dvs)  # [1,K]
        loc = -F.softplus(raw_loc)  # [1,K]
        nmy = torch.sigmoid(raw_nmy)  # [1,K]
        nvy = F.softplus(raw_nvy)  # [1,K]

        # --------------------------------------------------
        # compute s_onset
        # --------------------------------------------------

        to = onsets.view(1, -1)  # [1, K]
        ratio = ((to - loc) / dmx).clamp_max(1.0 - 1e-6)
        so = dmy * (1.0 - ratio.pow(1.0 / dmc))  # [1, K]

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


class GammaDegradationNLL(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        shape_Ts = y_pred[:, [0]]
        rate_Ts = y_pred[:, [1]]
        shift_Ts = y_pred[:, [2]]
        dist_s = ShiftedGamma(shape_Ts, rate_Ts, shift_Ts)
        loss = -(dist_s.log_prob(y_true)).mean()
        return loss
