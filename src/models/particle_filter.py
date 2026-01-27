import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.degradation import DegModel
from src.models.mixture import MixtureDegModel

# ============================================================
# Particle-Filter MLP
# ============================================================


class ParticleFilterMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1

        layers = []
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.net(x))

    def tuple_forward(self, x):
        out = self.forward(x)
        noise_scale = out[..., :-1]
        correct_scale = out[..., -1]
        return noise_scale, correct_scale


# ============================================================
# Diagonal Mahalanobis Noise (PF-safe)
# ============================================================


class DiagonalMahalanobisNoise(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("_sigma", None)

    def fit(self, states: torch.Tensor):
        assert states.ndim == 2
        sigma = states.std(dim=0).clamp_min(self.eps)

        if self._sigma is None:
            self._sigma = sigma
        else:
            self._sigma.copy_(sigma)

    @torch.no_grad()
    def forward(
        self,
        states: torch.Tensor,
        scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """
        Apply diagonal Mahalanobis roughening.

        states: Tensor [N, d]
        scale: scalar or Tensor [N] or [N, 1]
        """
        assert self._sigma is not None, "Call fit(states) before using noise"

        noise = torch.randn_like(states)

        if isinstance(scale, torch.Tensor):
            scale = scale.view(-1, 1)

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
        max_life: float,
        n_particles: int,
        multiply_scale: float = 1e-3,
        name: str = "perform_name",
    ):
        super().__init__()

        assert len(base_models) > 0, "At least one base model required"
        assert len({type(m) for m in base_models}) == 1, "All base models must be of the same class"

        self.net = ParticleFilterMLP(layer_dims=[2, 32, 32, 16, 2])
        self.max_life = max_life
        self.name = name
        self.multiply_scale = float(multiply_scale)

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

    # --------------------------------------------------------
    # Core PF steps
    # --------------------------------------------------------

    @torch.no_grad()
    def resample(self):
        """
        Multinomial resampling.
        """
        n = self.n_particles
        idx = torch.multinomial(self.weights, n, replacement=True)

        self.mixture.update(
            raw_params=self.states[idx],
            weights=torch.full((n,), 1.0 / n, device=self.weights.device),
            onsets=self.onsets[idx],
        )

    def prediction(self, noise_scale: float | torch.Tensor):
        """
        PF prediction: resample + roughening.
        """
        new_states = self.noise(self.states, scale=noise_scale)

        self.mixture.update(raw_params=new_states)

    def correction(self, t_obs: torch.Tensor, s_obs: torch.Tensor, correct_scale: torch.Tensor):
        """
        Particle-wise measurement update.

        s_obs : [B]
        t_obs : [B]
        """

        # params: [B, K, DP]
        params = self.mixture.forward(s_obs)

        # component distributions (NO mixture)
        comp_dist = self.deg_class.build_distribution_from_params(params)

        # log-likelihoods per particle
        log_lik = comp_dist.log_prob(t_obs.unsqueeze(1))  # [B, K]

        # aggregate over batch (system-level obs)
        log_lik = log_lik.mean(dim=0)  # [K]

        # Bayesian update
        log_w = correct_scale * log_lik  # torch.log(self.mixture.weights + 1e-12) + log_lik
        new_weights = torch.softmax(log_w, dim=0)

        self.mixture.update(weights=new_weights)

    def step(self, t_obs: torch.Tensor, s_obs: torch.Tensor, eol: torch.Tensor):
        # 1. NN controls
        x = torch.cat([t_obs.unsqueeze(-1), s_obs.unsqueeze(-1)], dim=-1)
        noise_scale, correct_scale = self.net.tuple_forward(x)
        noise_scale = noise_scale.mean()
        correct_scale = correct_scale.mean()

        # 2. Resample (DATA)
        with torch.no_grad():
            self.resample()

        # 3. Predict (DIFFERENTIABLE)
        old_states = self.states.detach()
        pred_states = self.noise(old_states, scale=noise_scale)

        # 4. Temporary belief (NO mutation)
        params = self.deg_class.forward_with_raw_parameters(s_obs, pred_states)
        comp_dist = self.deg_class.build_distribution_from_params(params)
        log_lik = comp_dist.log_prob(t_obs.unsqueeze(1)).mean(dim=0)

        log_w = correct_scale * log_lik
        weights = torch.softmax(log_w, dim=0)

        temp_mixture = MixtureDegModel.from_particles(
            deg_class=self.deg_class,
            raw_params=pred_states,
            weights=weights,
            onsets=self.onsets,
        )

        # 5. Loss on belief
        eol_dist = temp_mixture.distribution(s=torch.tensor([0.0], device=pred_states.device))
        loss = -eol_dist.log_prob(eol.unsqueeze(0)).mean()

        # 6. Commit belief (STOP GRAD)
        self.mixture.update(
            raw_params=pred_states.detach(),
            weights=weights.detach(),
        )

        return loss

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    @property
    def deg_class(self) -> type[DegModel]:
        return self.mixture.deg_class

    @property
    def n_particles(self) -> int:
        return self.mixture.K

    @property
    def state_dim(self) -> int:
        return self.mixture.RP

    @property
    def states(self) -> torch.Tensor:
        return self.mixture.raw_params

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
            states.append(m.get_raw_param_vector())
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
        states = base_states.repeat(repeat, 1)
        onsets = base_onsets.repeat(repeat)

        with torch.no_grad():
            states = self.noise(states, scale=self.multiply_scale)

        weights = torch.full((n_particles,), 1.0 / n_particles)

        # --- mixture is the SINGLE source of truth ---
        return MixtureDegModel.from_particles(
            deg_class=deg_class,
            raw_params=states,
            weights=weights,
            onsets=onsets,
        )

    @torch.no_grad()
    def plot(
        self,
        ax: plt.Axes,
        t_grid: np.ndarray,
        s_grid: np.ndarray,
        t_obs: np.ndarray,
        s_obs: np.ndarray,
        title: str | None = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """
        Plot PF state on a given axis.
        (Safe to compose with other plots.)
        """

        # --- Mixture PDF ---
        self.mixture.plot_distribution(
            t_grid,
            s_grid,
            func="pdf",
            ax=ax,
            plot_mean=True,
        )

        # --- Observations ---
        ax.plot(
            t_obs,
            s_obs,
            "o-",
            color="white",
            alpha=0.8,
            markersize=4,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="obs",
        )

        ax.set_xlim([0, t_grid.max()])
        ax.set_ylim([0, 1])

        if title is not None:
            ax.set_title(title)

        if show_legend:
            ax.legend()

        return ax
