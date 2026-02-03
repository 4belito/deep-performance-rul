import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.models.particle_filter import ParticleFilterModel
from src.models.rul_predictor import RULPredictor


def create_pf_prediciton_frame(
    ax: plt.Axes,
    pf: ParticleFilterModel,
    t_grid: NDArray,
    s_grid: NDArray,
    t_data: NDArray,
    s_data: NDArray,
    pred_interval: tuple[float, float, float],
    conf_level: float,
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

    pf.mixture.plot_uncertainty_interval(
        ax=ax,
        lower=pred_interval[0],
        mean=pred_interval[1],
        upper=pred_interval[2],
        ymax=1.0,
        unc_label=f"{int(conf_level * 100)}% unc",
        legend_loc=dist_legend_loc,
    )
    return ax


def create_rul_prediction_frame(
    rulpred: RULPredictor,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    t_data_np: np.ndarray,
    s_data_np: dict[str, np.ndarray],
    step: int,
    eol_time: float,
    unit: int,
    dist_vmax: float = 0.25,
    dist_plot_mean: bool = True,
    dist_legend_loc="lower left",
):
    assert set(s_data_np) == set(rulpred.pf_models), "PF models and data keys do not match."
    # --- number of performances ---
    n_perf = len(rulpred.pf_models)

    # --- layout ---
    fig, ax_rul, ax_pf = make_rul_pf_layout(n_perf)

    # --- fill PF axes left → right, top → bottom ---
    for ax, (name, pf) in zip(ax_pf, rulpred.pf_models.items()):
        create_pf_prediciton_frame(
            ax=ax,
            pf=pf,
            t_grid=t_grid,
            s_grid=s_grid,
            t_data=t_data_np,
            s_data=s_data_np[name],
            pred_interval=rulpred.history_component_eol[name][-1],
            conf_level=rulpred.conf_level,
            current_step=step,
            title=f"{name} | unit {unit} | step {step}",
            dist_vmax=dist_vmax,
            dist_plot_mean=dist_plot_mean,
            dist_legend_loc=dist_legend_loc,
        )

    # --- disable unused PF axes (odd case) ---
    for ax in ax_pf[n_perf:]:
        ax.axis("off")

    df = rulpred.history_to_dataframe()
    df["true_rul"] = np.maximum(eol_time - df["time"], 0.0)

    # --- system RUL ---
    plot_rul_from_dataframe(
        ax=ax_rul,
        df=df,
        y_max=100,
        t_max=eol_time,
        title=f"System RUL – unit {unit}",
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


def plot_rul_from_dataframe(
    ax: plt.Axes,
    df: pd.DataFrame,
    t_max: float = 100,
    y_max: float = 100,
    title: str = "RUL Prediction",
    unc_label: str = "unc",
):
    ax.plot(
        df["time"],
        df["true_rul"],
        "--",
        color="green",
        label="true",
    )

    ax.plot(
        df["time"],
        df["mean"],
        "-",
        color="blue",
        label="pred",
    )

    ax.plot(df["time"], df["lower"], "-", color="black", linewidth=0.5)
    ax.plot(df["time"], df["upper"], "-", color="black", linewidth=0.5)

    ax.fill_between(
        df["time"],
        df["lower"],
        df["upper"],
        color="#FF7F50",
        alpha=0.4,
        label=unc_label,
    )

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("RUL")
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, t_max)
    ax.legend()
