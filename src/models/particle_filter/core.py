import torch
import torch.nn as nn

from src.models.degradation.base import DegModel
from src.models.networks.pf_mlp import ParticleFilterMLP
from src.models.particle_filter.mixture import MixtureDegModel


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


class ParticleFilter(nn.Module):
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
        self._init_base(base_models, n_particles)

        # --- noise model ---
        self.noise = DiagonalMahalanobisNoise()
        self.noise.fit(self.base_states)

        # --- mixture model ---
        self.prior_states = self.base_states.clone()
        self._init_weights(n_particles)
        self.mixture = MixtureDegModel.from_particles(
            deg_class=type(base_models[0]),
            states=self.base_states.clone(),
            weights=self.init_weights.clone(),
            onsets=self.base_onsets.clone(),
            init_ss=self.base_init_ss.clone(),
        )

    # --------------------------------------------------------
    # Core PF steps
    # --------------------------------------------------------

    @torch.no_grad()
    def reset(self):
        """
        Reset particle filter to its initial prior.
        """
        self.prior_states = self.base_states.clone()

        self.mixture.update(
            states=self.base_states.clone(),
            weights=self.init_weights.clone(),
            onsets=self.base_onsets.clone(),
            init_ss=self.base_init_ss.clone(),
        )

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
        init_ss = self.init_ss.clone().detach()
        new_states, new_weights = self._core_step(old_states, onsets, init_ss, t_obs, s_obs)

        loss_mixture = MixtureDegModel.from_particles(
            deg_class=self.deg_class,
            states=new_states,
            weights=new_weights,
            onsets=onsets,
            init_ss=init_ss,
        )
        return new_states, new_weights, loss_mixture

    @torch.no_grad()
    def _core_eval(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        new_states, new_weights = self._core_step(
            self.states, self.onsets, self.init_ss, t_obs, s_obs
        )
        return new_states, new_weights, self.mixture

    def _core_step(
        self,
        old_states: torch.Tensor,
        onsets: torch.Tensor,
        init_ss: torch.Tensor,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        noise, correction = self.net.tuple_forward_mean(
            t_obs,
            s_obs,
        )

        new_states = self.predict(old_states, noise)
        if torch.isnan(new_states).any():
            print("NaN after predict()")
            raise RuntimeError("NaN in states")
        new_weights = self.correct(new_states, onsets, init_ss, t_obs, s_obs, correction)
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
            init_ss=self.init_ss[idx],
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
        init_ss: torch.Tensor,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
        correction: torch.Tensor,
    ):
        """
        PURE correction (no mutation, differentiable).
        """
        s_obs = s_obs.unsqueeze(1)  # [B, 1]
        onsets = onsets.unsqueeze(1)
        init_ss = init_ss.unsqueeze(1)
        correct_prior, correct_lik, forget_lik = self.net.correction_tuple(correction)
        params = self.deg_class.forward_with_states(s_obs, states, onsets, init_ss)
        comp_dist = self.deg_class.build_distribution_from_params(params)

        log_probs = comp_dist.log_prob(t_obs.unsqueeze(1))
        log_lik = self.weighted_log_likelihood(
            log_probs=log_probs,
            alpha=forget_lik[0],  # scalar, stable
        )
        # log_lik = comp_dist.log_prob(t_obs.unsqueeze(1)).mean(dim=0)
        log_prior = self.trajectory_log_prior(states)

        log_w = correct_lik[0] * log_lik + (correct_prior * log_prior).sum(dim=1)
        return torch.softmax(log_w, dim=0)

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

    @property
    def init_ss(self) -> torch.Tensor:
        return self.mixture.init_ss

    def _init_weights(self, n_particles: int):
        uniform = torch.full(
            (n_particles,),
            1.0 / n_particles,
            device=self.base_states.device,
        )
        self.init_weights: torch.Tensor
        self.register_buffer("init_weights", uniform)

    def _init_base(
        self,
        base_models: list[DegModel],
        n_particles: int,
    ):
        base_n = len(base_models)
        assert (
            n_particles % base_n == 0
        ), "n_particles must be a multiple of the number of base models"
        repeat = n_particles // base_n
        states = []
        onsets = []
        init_ss = []
        for m in base_models:
            states.append(m.get_state_vector())
            onsets.append(m.get_onset())
            init_ss.append(m.get_init_s())
        base_states = torch.stack(states, dim=0).repeat(repeat, 1)
        base_onsets = torch.tensor(onsets).repeat(repeat)
        base_init_ss = torch.tensor(init_ss).repeat(repeat)

        self.base_states: torch.Tensor
        self.base_onsets: torch.Tensor
        self.base_init_ss: torch.Tensor
        self.register_buffer("base_states", base_states)
        self.register_buffer("base_onsets", base_onsets)
        self.register_buffer("base_init_ss", base_init_ss)
        self.register_buffer("base_init_ss", base_init_ss)
        self.register_buffer("base_init_ss", base_init_ss)
        self.register_buffer("base_init_ss", base_init_ss)
        self.register_buffer("base_init_ss", base_init_ss)
        self.register_buffer("base_init_ss", base_init_ss)
        self.register_buffer("base_init_ss", base_init_ss)
