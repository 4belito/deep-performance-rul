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
        raw_out = self.net(x)
        return F.softplus(raw_out)

    def tuple_forward(self, x):
        """
        Returns (spatial, selection) as separate tensors.
        """
        output = self.forward(x)
        spatial = output[..., :-1]
        selection = output[..., -1]
        return spatial, selection


# ============================================================
# Diagonal Mahalanobis Noise (PF-safe)
# ============================================================


class DiagonalMahalanobisNoise(nn.Module):
    """
    PF-safe diagonal Mahalanobis roughening.
    States == raw parameters.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._sigma: torch.Tensor | None
        self.register_buffer("_sigma", None)

    def fit(self, states: torch.Tensor):
        """
        Estimate per-parameter scale from particle cloud.

        states: Tensor [N, d]
        """
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
    def __init__(
        self,
        base_models: list[DegModel],  # [N, d] raw parameters
        net: ParticleFilterMLP,
        max_life: float,
        n_particles: int | None = None,
        multiply_scale: float = 1e-3,
        name: str = "perform_name",
    ):
        super().__init__()

        assert len(base_models) > 0, "At least one base model is required"
        base_states, base_onsets = self.extract_parameters(base_models)

        self.N, self.d = base_states.shape
        self.max_life = max_life
        self.net = net
        self.name = name
        self.multiply_scale = float(multiply_scale)

        n_particles = self._resolve_n_particles(n_particles)

        self.noise_model = DiagonalMahalanobisNoise()
        self.noise_model.fit(base_states)

        self._init_particles(base_states, base_onsets, n_particles)
        self._init_weights(n_particles)
        self.mixture_model = MixtureDegModel.from_particles(
            deg_model_class=type(base_models[0]),
            raw_params=self.states,
            weights=self.weights,
        )

    # --------------------------------------------------------
    # Initialization helpers
    # --------------------------------------------------------

    def _resolve_n_particles(self, n_particles):
        n_particles = n_particles or self.N
        assert (
            n_particles % self.N == 0
        ), f"n_particles ({n_particles}) must be a multiple of base_states ({self.N})"
        return n_particles

    def _init_particles(
        self, base_states: torch.Tensor, base_onsets: torch.Tensor, n_particles: int
    ):
        """
        Replicate base particles and apply PF roughening.
        """
        repeat_factor = n_particles // self.N

        states = self._multiply_particles(
            base_states,
            repeat_factor,
            self.multiply_scale,
        )
        onsets = base_onsets.repeat(repeat_factor)

        self.register_buffer("states", states)
        self.register_buffer("onsets", onsets)
        self.states: torch.Tensor
        self.onsets: torch.Tensor

    def _init_weights(self, n_particles):
        """
        Initialize uniform particle weights.
        """
        probs = torch.full((n_particles,), 1.0 / n_particles)
        self.register_buffer("weights", probs)
        self.weights: torch.Tensor

    def extract_parameters(self, base_models: list[DegModel]):
        base_states = []
        base_onsets = []
        for model in base_models:
            base_states.append(model.get_raw_param_vector())
            base_onsets.append(model.get_onset())
        base_states = torch.stack(base_states, dim=0)  # [N, d]
        base_onsets = torch.tensor(base_onsets)  # [N]
        return base_states, base_onsets

    # --------------------------------------------------------
    # PF utilities
    # --------------------------------------------------------

    def _multiply_particles(
        self,
        base_states: torch.Tensor,
        repeat_factor: int,
        scale: float,
    ) -> torch.Tensor:
        """
        Replicate base states and apply diagonal Mahalanobis roughening.
        """
        states = base_states.repeat(repeat_factor, 1)

        with torch.no_grad():
            states = self.noise_model(states, scale=scale)

        return states
