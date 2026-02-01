import matplotlib.pyplot as plt
import numpy as np

from src.models.particle_filter import ParticleFilterModel
from src.models.system_rul import SystemRUL


def create_pf_prediciton_freame(
    ax: plt.Axes,
    pf: ParticleFilterModel,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    t_data: np.ndarray,
    s_data: np.ndarray,
    uncertainty_level: float,
    current_step: int,
    title: str = "",
    dist_vmax: float = 0.25,
    dist_plot_mean: bool = True,
    dist_legend_loc="lower left",
) -> plt.Axes:
    # --- render frame --
    # distribution
    pf.mixture.plot_distribution(
        ax=ax,
        t=t_grid,
        s=s_grid,
        title=title or f"PF prediction | step {current_step}",
        vmax=dist_vmax,
        plot_mean=dist_plot_mean,
        legend_loc=dist_legend_loc,
    )

    # data
    pf.mixture.plot_observations(
        ax=ax,
        t_obs=t_data,
        s_obs=s_data,
        current_idx=current_step,
        legend_loc=dist_legend_loc,
    )

    # uncertainty interva
    lower, mean, upper = pf.mixture.uncertainty_interval(s=np.zeros(1), level=uncertainty_level)
    pf.mixture._plot_uncertainty_interval(
        ax=ax,
        lower=lower.item(),
        mean=mean.item(),
        upper=upper.item(),
        ymax=1.0,
        label=f"{int(uncertainty_level * 100)}% uncertainty",
    )
    return ax


def create_rul_prediciton_frame(
    sys_rul: SystemRUL,
    step: int,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    t_data_np: np.ndarray,
    s_data_np: dict[str, np.ndarray],
    uncertainty_level: float,
    eol_time: float,
    test_unit: int,
    dist_vmax: float = 0.25,
    dist_plot_mean: bool = True,
    dist_legend_loc="lower left",
):
    # --- number of performances ---
    n_perf = len(sys_rul.pf_models)

    # --- layout ---
    fig, ax_rul, ax_pf = make_rul_pf_layout(n_perf)

    # --- fill PF axes left → right, top → bottom ---
    for ax, (name, pf) in zip(ax_pf, sys_rul.pf_models.items()):
        create_pf_prediciton_freame(
            ax=ax,
            pf=pf,
            t_grid=t_grid,
            s_grid=s_grid,
            t_data=t_data_np,
            s_data=s_data_np[name],
            uncertainty_level=uncertainty_level,
            current_step=step,
            title=f"{name} | unit {test_unit} | step {step}",
            dist_vmax=dist_vmax,
            dist_plot_mean=dist_plot_mean,
            dist_legend_loc=dist_legend_loc,
        )

    # --- disable unused PF axes (odd case) ---
    for ax in ax_pf[n_perf:]:
        ax.axis("off")

    # --- system RUL ---
    sys_rul.plot_rul(
        ax=ax_rul,
        eol_time=eol_time,
        y_max=100,
        title=f"System RUL – unit {test_unit}",
    )

    # --- render frame ---
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return frame


def make_rul_pf_layout(n_perf: int, n_cols: int = 2):
    """
    Create a layout with:
      - Left column: RUL (spans all rows)
      - Right columns: PF plots filled row-wise

    Returns
    -------
    fig : Figure
    ax_rul : Axes
    ax_pf : list[Axes]  # flat list, length = n_rows * n_cols
    """
    # ceil division
    n_rows = (n_perf + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 4.5 * n_rows))

    gs = fig.add_gridspec(
        n_rows,
        1 + n_cols,  # 1 for RUL + PF columns
        width_ratios=[2.6] + [2.2] * n_cols,
        wspace=0.15,
        hspace=0.20,
    )

    # --- Main RUL axis (spans all rows) ---
    ax_rul = fig.add_subplot(gs[:, 0])

    # --- PF axes (flat list, fill order) ---
    ax_pf = [fig.add_subplot(gs[r, c + 1]) for r in range(n_rows) for c in range(n_cols)]

    return fig, ax_rul, ax_pf
