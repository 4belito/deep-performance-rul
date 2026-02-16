import copy
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

from src.models.degradation.base import DegModel


class BestModelTracker(Callback):
    def __init__(
        self,
        monitor: str = "train_loss",
        min_delta: float = 0.0,
        save_dir: Path | str = ".",
        f_params: str = "best_model.pt",
        load_best: bool = False,  # optional
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.save_dir = Path(save_dir)
        self.f_params = f_params
        self.load_best = load_best

    def initialize(self):
        self.best_loss = float("inf")
        self.best_state_dict = None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        return self

    def on_epoch_end(self, net: NeuralNetRegressor, **kwargs):
        current_loss = float(net.history[-1, self.monitor])

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_state_dict = copy.deepcopy(net.module_.state_dict())

    def on_train_end(self, net: NeuralNetRegressor, **kwargs):
        if self.best_state_dict is None:
            return

        if self.load_best:
            net.module_.load_state_dict(self.best_state_dict)

        torch.save(
            self.best_state_dict,
            self.save_dir / self.f_params,
        )


class PlotDegModelWithData(Callback):
    def __init__(
        self,
        t_grid: np.ndarray,
        s_grid: np.ndarray,
        time_data=None,
        perform_data=None,
        plot_every: int | None = None,  # None → no periodic plots
        func: str = "pdf",
        title: str = "Normal PDF of $T_s$",
        plot_at_end: bool = True,
        show: bool = False,  # ← show interactively?
        save_dir: str | Path | None = None,  # ← save figures here if not None
        legend_loc: str | None = "lower left",
        dpi: int = 150,
    ):
        self.t_grid = t_grid
        self.s_grid = s_grid
        self.time_data = time_data
        self.perform_data = perform_data
        self.plot_every = plot_every
        self.func = func
        self.title = title
        self.plot_at_end = plot_at_end
        self.show = show
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.dpi = dpi
        self.legend_loc = legend_loc

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_save_dir(self, save_dir):
        if save_dir is None:
            self.save_dir = None
            return

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # -------- shared plotting logic --------
    def _plot(self, model: DegModel, title: str, suffix: str):
        _, ax = plt.subplots(figsize=(10, 6))

        model.plot_distribution(
            t=self.t_grid,
            s=self.s_grid,
            func=self.func,
            ax=ax,
            title=title,
            legend_loc=self.legend_loc,
        )

        if self.time_data is not None and self.perform_data is not None:
            ax.plot(
                self.time_data,
                self.perform_data,
                "o-",
                color="white",
                alpha=0.3,
                markersize=4,
                markeredgecolor="black",
                markeredgewidth=0.8,
                label="data",
            )
            ax.legend(loc=self.legend_loc)

        plt.tight_layout()

        # ---- save if requested ----
        if self.save_dir is not None:
            fname = f"{self.title.replace(' ', '_')}_{suffix}.png"
            plt.savefig(self.save_dir / fname, dpi=self.dpi, bbox_inches="tight")

        # ---- show or close ----
        if self.show:
            plt.show()
        else:
            plt.close()

    # -------- periodic plotting --------
    def on_epoch_end(self, net: NeuralNetRegressor, **kwargs):
        if self.plot_every is None:
            return

        epoch = net.history[-1]["epoch"]
        if epoch % self.plot_every != 0:
            return

        model = net.module_.cpu()
        self._plot(
            model,
            title=f"{self.title} (epoch {epoch})",
            suffix=f"epoch_{epoch}",
        )

    # -------- final plotting --------
    def on_train_end(self, net: NeuralNetRegressor, **kwargs):
        if not self.plot_at_end:
            return

        model = net.module_.cpu()
        self._plot(
            model,
            title=f"{self.title} (final)",
            suffix="final",
        )
