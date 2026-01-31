import matplotlib.pyplot as plt
import torch

from src.models.particle_filter import ParticleFilterModel


def create_pf_prediciton_freame(
    ax: plt.Axes,
    pf: ParticleFilterModel,
    t_grid: torch.Tensor,
    s_grid: torch.Tensor,
    t_data: torch.Tensor,
    s_data: torch.Tensor,
    uncertainty_level: float,
    current_step: int,
) -> plt.Axes:
    t_data_np = t_data.cpu().numpy()
    s_data_np = s_data.cpu().numpy()
    pf.step(
        t_obs=t_data[: current_step + 1],
        s_obs=s_data[: current_step + 1],
    )

    # --- render frame --

    # distribution
    pf.mixture.plot_distribution(
        ax=ax,
        t=t_grid,
        s=s_grid,
        title=f"PF prediction | step {current_step}",
        vmax=0.25,
        plot_mean=True,
    )

    # data
    pf.mixture.plot_observations(
        ax=ax,
        t_obs=t_data_np,
        s_obs=s_data_np,
        current_idx=current_step,
    )

    # uncertainty interval
    device = next(pf.parameters()).device
    s0 = torch.zeros(1, dtype=torch.float32, device=device)
    lower, mean, upper = pf.mixture.uncertainty_interval(s=s0, level=uncertainty_level)
    pf.mixture._plot_uncertainty_interval(
        ax=ax,
        lower=lower.item(),
        mean=mean.item(),
        upper=upper.item(),
        ymax=1.0,
        label=f"{int(uncertainty_level * 100)}% uncertainty",
    )
    return ax
