import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.degradation import DegModel
from src.models.mixture import MixtureDegModel

# ============================================================
# Particle-Filter MLP
# ============================================================


class ParticleFilterMLP(nn.Module):
    def __init__(self, state_dim: int, hidden_dims=(128, 128, 32)):
        super().__init__()

        self.state_dim = state_dim
        output_dim = state_dim + 1  # noise vector + correction scalar

        layers = []
        dims = (2, *hidden_dims, output_dim)
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

        self.apply(self._init_identity)

    def _init_identity(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.constant_(m.bias, 0.0)

    def tuple_logforward(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        """
        Returns:
            log_noise_vec : [B, state_dim]
            log_correct   : [B]
        """
        x = torch.cat([t_obs.unsqueeze(-1), s_obs.unsqueeze(-1)], dim=-1)
        out = self.net(x)

        log_noise_vec = out[..., : self.state_dim]
        log_correct = out[..., self.state_dim]

        return log_noise_vec, log_correct

    def tuple_forward(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        """
        Returns:
            noise_vec : [B, state_dim]
            correct   : [B]
        """
        log_noise_vec, log_correct = self.tuple_logforward(t_obs, s_obs)
        noise_vec = F.softplus(log_noise_vec)
        correct = F.softplus(log_correct)
        return noise_vec, correct

    def tuple_forward_mean(self, t_obs: torch.Tensor, s_obs: torch.Tensor):
        """
        Returns:
            noise_vec : [state_dim]
            correct   : scalar
        """
        log_noise_vec, log_correct = self.tuple_logforward(t_obs, s_obs)
        noise_vec = F.softplus(log_noise_vec).mean(dim=0)
        correct = F.softplus(log_correct).mean()
        return noise_vec, correct

    def forward(self, x):
        return F.softplus(self.net(x))

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
        max_life: float,
        n_particles: int,
        multiply_scale: float = 1e-3,
        name: str = "perform_name",
    ):
        super().__init__()

        assert len(base_models) > 0, "At least one base model required"
        assert len({type(m) for m in base_models}) == 1, "All base models must be of the same class"

        self.net = net
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

    def predict_states(self, states: torch.Tensor, noise_scale: torch.Tensor):
        """
        PURE prediction (no mutation, differentiable).
        """
        return self.noise(states, scale=noise_scale)

    def compute_weights(
        self,
        states: torch.Tensor,
        t_obs: torch.Tensor,
        s_obs: torch.Tensor,
        correct_scale: torch.Tensor,
    ):
        """
        PURE correction (no mutation, differentiable).
        """
        params = self.deg_class.forward_with_raw_parameters(s_obs, states)
        comp_dist = self.deg_class.build_distribution_from_params(params)
        log_lik = comp_dist.log_prob(t_obs.unsqueeze(1)).mean(dim=0)

        log_w = correct_scale * log_lik
        return torch.softmax(log_w, dim=0)

    def prediction(self, noise_scale):
        new_states = self.predict_states(self.states, noise_scale)
        self.mixture.update(raw_params=new_states)

    def correction(self, t_obs, s_obs, correct_scale):
        weights = self.compute_weights(self.states, t_obs, s_obs, correct_scale)
        self.mixture.update(weights=weights)

    def step(self, t_obs: torch.Tensor, s_obs: torch.Tensor) -> MixtureDegModel:
        if self.training:
            return self._step_train(t_obs, s_obs)
        else:
            self._step_eval(t_obs, s_obs)
            return self.mixture

    def _step_train(self, t_obs: torch.Tensor, s_obs: torch.Tensor) -> MixtureDegModel:
        noise_vec, correct_scale = self.net.tuple_forward_mean(t_obs, s_obs)

        # resample (data only)
        self.resample()

        old_states = self.states.detach()

        # PURE steps
        pred_states = self.predict_states(old_states, noise_vec)
        weights = self.compute_weights(pred_states, t_obs, s_obs, correct_scale)

        temp_mixture = MixtureDegModel.from_particles(
            deg_class=self.deg_class,
            raw_params=pred_states,
            weights=weights,
            onsets=self.onsets,
        )

        # commit belief (NO grad)
        self.mixture.update(
            raw_params=pred_states.detach(),
            weights=weights.detach(),
        )
        return temp_mixture

    @torch.no_grad()
    def _step_eval(self, t_obs, s_obs):
        noise_vec, correct_scale = self.net.tuple_forward_mean(t_obs, s_obs)

        self.resample()
        self.prediction(noise_vec)
        self.correction(t_obs, s_obs, correct_scale)

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
