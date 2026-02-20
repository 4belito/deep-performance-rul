from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.degradation.base import DegModel
from src.models.particle_filter.mixture import MixtureDegModel


# ============================================================
# Particle-Filter MLP
# ============================================================
class ParticleFilterMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: tuple[int, ...],
        activation: Callable[[], nn.Module] = lambda: nn.ReLU(),
    ):
        super().__init__()

        self.state_dim = state_dim
        output_dim = 2 * state_dim + 2  # noise vector + correction vector

        layers = []
        dims = (5, *hidden_dims, output_dim)
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.net(x))

    def tuple_out(self, x: torch.Tensor):
        noise = x[..., : self.state_dim]
        correction = x[..., self.state_dim :]
        return noise, correction

    def correction_tuple(self, x: torch.Tensor):
        correct_prior = x[..., : self.state_dim]
        correct_lik = x[..., [-2]]
        forget_lik = x[..., [-1]]
        return correct_prior, correct_lik, forget_lik

    def tuple_forward(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        x = self.tuple_in(t_obs, s_obs)
        out = self.forward(x)
        return self.tuple_out(out)

    def tuple_forward_mean(
        self,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
        mean_loglik: torch.Tensor,
        std_loglik: torch.Tensor,
        ess: torch.Tensor,
    ):
        x = self.tuple_in(
            t_obs,
            s_obs,
            mean_loglik,
            std_loglik,
            ess,
        )
        out = self.forward(x)
        out_mean = out.mean(dim=0)
        return self.tuple_out(out_mean)

    @staticmethod
    def tuple_in(
        t_obs,
        s_obs,
        mean_loglik,
        std_loglik,
        ess,
    ):
        t_scaled = t_obs / 100.0
        s_scaled = s_obs
        mean_scaled = torch.tanh(mean_loglik / 50.0)
        std_scaled = torch.tanh(std_loglik / 10.0)
        ess_scaled = ess

        return torch.cat(
            [
                t_scaled.unsqueeze(-1),
                s_scaled.unsqueeze(-1),
                mean_scaled.unsqueeze(-1),
                std_scaled.unsqueeze(-1),
                ess_scaled.unsqueeze(-1),
            ],
            dim=-1,
        )

    @torch.no_grad()
    def plot_output(
        self,
        t: np.ndarray,
        s: np.ndarray,
        dim: int,
        ax: plt.Axes | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        gamma: float = 0.5,
        title: str | None = None,
    ) -> plt.Axes:
        """
        Plot NN output over (t, s).

        Parameters
        ----------
        output : {"noise", "correct"}
            Which NN head to visualize.
        """

        device = next(self.parameters()).device

        # ---- grid ----
        T, S = np.meshgrid(t, s)
        t_torch = torch.tensor(T.flatten(), dtype=torch.float32, device=device)
        s_torch = torch.tensor(S.flatten(), dtype=torch.float32, device=device)

        x = torch.cat(
            [t_torch.unsqueeze(-1), s_torch.unsqueeze(-1)],
            dim=-1,
        )

        # ---- forward (positive outputs) ----
        out = self.forward(x)  # exp to ensure positivity

        Z = out[..., dim]
        label = f"dim: {dim}"

        Z = Z.reshape(S.shape).cpu().numpy()

        # ---- plot ----
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        norm = mcolors.PowerNorm(
            gamma=gamma,
            vmin=np.min(Z) if vmin is None else vmin,
            vmax=np.max(Z) if vmax is None else vmax,
        )

        c = ax.pcolormesh(
            T,
            S,
            Z,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )
        plt.colorbar(c, ax=ax, label=label)

        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim([s.min(), s.max()])

        if title is None:
            title = f"NN output: {label}"
        ax.set_title(title)

        return ax


# ============================================================
# Diagonal Mahalanobis Noise (PF-safe)
# ============================================================
class DiagonalMahalanobisNoise(nn.Module):
    def __init__(self, eps: float = 1e-6, max_scale: float = 0.1):
        super().__init__()
        self.eps = eps
        self.register_buffer("_sigma", None)

    @torch.no_grad()
    def fit(self, states: torch.Tensor):
        assert states.ndim == 2
        sigma = states.std(dim=0).clamp_min(self.eps)

        if self._sigma is None:
            self._sigma = sigma
        else:
            self._sigma.copy_(sigma)

    def forward(
        self,
        states: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply diagonal Mahalanobis roughening.

        states: Tensor [N, d]
        scale: scalar or Tensor [N] or [N, 1]
        """
        assert self._sigma is not None, "Call fit(states) before using noise"

        noise = torch.randn_like(states)

        return states + noise * self._sigma * scale


# ============================================================
# Particle Filter Model
# ============================================================


class ParticleFilterModel(nn.Module):
    """
    Particle Filter over raw degradation-model parameters.
    """

    def __init__(
        self,
        base_models: list[DegModel],
        net: ParticleFilterMLP,
        n_particles: int,
    ):
        super().__init__()

        assert len(base_models) > 0, "At least one base model required"
        assert len({type(m) for m in base_models}) == 1, "All base models must be of the same class"

        self.net = net
        base_states, base_onsets = self._extract_parameters(base_models)

        # --- noise model ---
        self.noise = DiagonalMahalanobisNoise()
        self.noise.fit(base_states)

        # --- mixture model ---
        self.mixture = self._init_mixture(
            deg_class=type(base_models[0]),
            base_states=base_states,
            base_onsets=base_onsets,
            n_particles=n_particles,
        )

        device = base_states.device

        self.loglik_mean: torch.Tensor
        self.loglik_std: torch.Tensor
        self.ess: torch.Tensor
        self.register_buffer("loglik_mean", torch.tensor(0.0, device=device))
        self.register_buffer("loglik_std", torch.tensor(0.0, device=device))
        self.register_buffer("ess", torch.tensor(1.0, device=device))

    # --------------------------------------------------------
    # Core PF steps
    # --------------------------------------------------------

    @torch.no_grad()
    def reset(self):
        """
        Reset particle filter to its initial prior.
        """
        self.prior_states = self.base_states.clone()
        weights = torch.full(
            (self.n_particles,),
            1.0 / self.n_particles,
            device=self.base_states.device,
        )

        self.mixture.update(
            states=self.base_states,
            weights=weights,
            onsets=self.onsets,
        )

        self.loglik_mean.zero_()
        self.loglik_std.zero_()
        self.ess.fill_(1.0)

    def step(self, t_obs: torch.Tensor, s_obs: torch.Tensor) -> MixtureDegModel:
        if self.training:
            new_s, new_w, mix = self._core_train(t_obs, s_obs)
        else:
            new_s, new_w, mix = self._core_eval(t_obs, s_obs)
        self.mixture.update(
            states=new_s,
            weights=new_w,
        )
        self.resample()
        return mix

    def _core_train(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        old_states = self.states.detach()
        onsets = self.onsets.clone().detach()
        new_states, new_weights = self._core_step(old_states, onsets, t_obs, s_obs)

        loss_mixture = MixtureDegModel.from_particles(
            deg_class=self.deg_class,
            states=new_states,
            weights=new_weights,
            onsets=onsets,
        )
        return new_states, new_weights, loss_mixture

    @torch.no_grad()
    def _core_eval(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        new_states, new_weights = self._core_step(self.states, self.onsets, t_obs, s_obs)
        return new_states, new_weights, self.mixture

    def _core_step(
        self,
        old_states: torch.Tensor,
        onsets: torch.Tensor,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B = t_obs.shape[0]

        mean_loglik = self.loglik_mean.expand(B)
        std_loglik = self.loglik_std.expand(B)
        ess = self.ess.expand(B)

        noise, correction = self.net.tuple_forward_mean(
            t_obs,
            s_obs,
            mean_loglik,
            std_loglik,
            ess,
        )

        new_states = self.predict(old_states, noise)
        if torch.isnan(new_states).any():
            print("NaN after predict()")
            raise RuntimeError("NaN in states")
        new_weights = self.correct(new_states, onsets, t_obs, s_obs, correction)
        return new_states, new_weights

    @torch.no_grad()
    def resample(self):
        """
        Multinomial resampling.
        """
        n = self.n_particles
        idx = torch.multinomial(self.weights, n, replacement=True)

        self.mixture.update(
            states=self.states[idx],
            weights=torch.full((n,), 1.0 / n, device=self.weights.device),
            onsets=self.onsets[idx],
        )
        self.prior_states = self.prior_states[idx]

    def predict(self, states: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        PURE prediction (no mutation, differentiable).
        """
        return self.noise(states, scale=noise)

    def correct(
        self,
        states: torch.Tensor,
        onsets: torch.Tensor,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
        correction: torch.Tensor,
    ):
        """
        PURE correction (no mutation, differentiable).
        """
        s_obs = s_obs.unsqueeze(1)  # [B, 1]
        onsets = onsets.unsqueeze(1)
        correct_prior, correct_lik, forget_lik = self.net.correction_tuple(correction)
        params = self.deg_class.forward_with_states(s_obs, states, onsets)
        comp_dist = self.deg_class.build_distribution_from_params(params)

        log_probs = comp_dist.log_prob(t_obs.unsqueeze(1))
        self.record_controller_statistics(log_probs)
        log_lik = self.weighted_log_likelihood(
            log_probs=log_probs,
            alpha=forget_lik[0],  # scalar, stable
        )
        # log_lik = comp_dist.log_prob(t_obs.unsqueeze(1)).mean(dim=0)
        log_prior = self.trajectory_log_prior(states)

        log_w = correct_lik[0] * log_lik + (correct_prior * log_prior).sum(dim=1)
        return torch.softmax(log_w, dim=0)

    @torch.no_grad()
    def record_controller_statistics(self, log_probs: torch.Tensor):
        loglik_mean = log_probs.detach().mean(dim=0)  # [N]
        loglik_std = log_probs.detach().std(dim=0, unbiased=False)  # [N]

        # Effective Sample Size
        w = self.weights.detach()
        ess = 1.0 / (w.pow(2).sum().clamp_min(1e-12))
        ess = ess / self.n_particles  # normalize to [0,1]

        # expand to match batch shape
        self.loglik_mean.copy_(loglik_mean.mean().detach())
        self.loglik_std.copy_(loglik_std.mean().detach())
        self.ess.copy_(ess.detach())

    def weighted_log_likelihood(
        self,
        log_probs: torch.Tensor,  # [B, N]
        alpha: torch.Tensor,  # [1]
    ):
        """
        Exponentially weighted average over time.
        More recent observations get higher weight.
        """
        B = log_probs.shape[0]
        device = log_probs.device

        # time indices: [-B+1, ..., 0]
        idx = torch.arange(B, device=device) - (B - 1)

        # positive decay
        alpha = F.softplus(alpha)

        weights = torch.exp(alpha * idx)  # [B]
        weights = weights / weights.sum()  # normalize

        return (weights[:, None] * log_probs).sum(dim=0)

    def trajectory_log_prior(self, states: torch.Tensor):
        """
        Gaussian prior around initial particle states.
        """
        diff = states - self.prior_states
        # diagonal Mahalanobis (reuse sigma!)
        inv_var = 1.0 / (self.noise._sigma**2)
        log_prior = -0.5 * (diff**2 * inv_var)
        return log_prior

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    @property
    def deg_class(self) -> type[DegModel]:
        return self.mixture.deg_class

    @property
    def n_particles(self) -> int:
        return self.mixture.n_components

    @property
    def state_dim(self) -> int:
        return self.mixture.state_dim

    @property
    def states(self) -> torch.Tensor:
        return self.mixture.states

    @property
    def weights(self) -> torch.Tensor:
        return self.mixture.weights

    @property
    def onsets(self) -> torch.Tensor:
        return self.mixture.onsets

    @staticmethod
    def _extract_parameters(base_models: list[DegModel]):
        states = []
        onsets = []
        for m in base_models:
            states.append(m.get_state_vector())
            onsets.append(m.get_onset())
        return torch.stack(states, dim=0), torch.tensor(onsets)

    def _init_mixture(
        self,
        deg_class: type[DegModel],
        base_states: torch.Tensor,
        base_onsets: torch.Tensor,
        n_particles: int,
    ):
        base_n = base_states.shape[0]
        assert (
            n_particles % base_n == 0
        ), "n_particles must be a multiple of the number of base models"

        # --- initialize particles ---
        repeat = n_particles // base_n
        base_states = base_states.repeat(repeat, 1).clone()
        onsets = base_onsets.repeat(repeat)
        self.base_states: torch.Tensor
        self.register_buffer("base_states", base_states.clone())
        self.prior_states = base_states.clone()

        weights = torch.full((n_particles,), 1.0 / n_particles)

        # --- mixture is the SINGLE source of truth ---
        return MixtureDegModel.from_particles(
            deg_class=deg_class,
            states=base_states,
            weights=weights,
            onsets=onsets,
        )
