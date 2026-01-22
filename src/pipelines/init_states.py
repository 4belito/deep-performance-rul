

import copy

import numpy as np
from matplotlib import pyplot as plt
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback, Checkpoint

from src.models.normal_deg import NormalDegradationModel


class PlotNormalDistWithData(Callback):
    def __init__(
        self,
        t_grid: np.ndarray,
        s_grid: np.ndarray,
        time_data,
        perform_data,
        plot_every: int = 500,
        func: str = "pdf",
        title: str = "Normal PDF of $T_s$",
    ):
        self.t_grid = t_grid
        self.s_grid = s_grid
        self.time_data = time_data
        self.perform_data = perform_data
        self.plot_every = plot_every
        self.func = func
        self.title = title

    def on_epoch_end(self, net: NeuralNetRegressor, **kwargs):
        epoch = net.history[-1]["epoch"]

        if epoch % self.plot_every != 0:
            return

        model:NormalDegradationModel = copy.deepcopy(net.module_).cpu()

        _ , ax = plt.subplots(figsize=(10, 6))

        # --- distribution ---
        model.plot_distribution(
            t=self.t_grid,
            s=self.s_grid,
            func=self.func,
            ax=ax,
            title=f"{self.title} (epoch {epoch})",
        )

        # --- overlay data ---
        ax.plot(self.time_data,self.perform_data,'o-',
			color='white',alpha=0.3,markersize=4,markeredgecolor='black',markeredgewidth=0.8,
			label='data')

        ax.legend()
        plt.tight_layout()
        plt.show()
        

class ThresholdCheckpoint(Callback):
    def __init__(
        self,
        checkpoint: Checkpoint,
        monitor: str = "train_loss",
        min_delta: float = 0.01,
        tol:float=1e-6,
    ):
        self.checkpoint = checkpoint
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.tol = tol
        self.activated = False
        self.best_saved_loss = np.inf

    def on_epoch_end(self, net:NeuralNetRegressor, **kwargs):
        current_loss = net.history[-1][self.monitor]

        # save only if improvement over LAST SAVED model is significant
        if not self.activated:
            if current_loss < self.best_loss - self.tol:
                self.best_loss = current_loss
            else:
                # first non-improving epoch → activate
                self.activated = True
        
        if self.activated and (current_loss < self.best_saved_loss - self.min_delta):
            self.best_saved_loss = current_loss
            self.checkpoint.on_epoch_end(net, **kwargs)

class DelayedCheckpoint(Callback):
    def __init__(self, checkpoint:Checkpoint, start_epoch: int):
        self.checkpoint = checkpoint
        self.start_epoch = start_epoch

    def on_epoch_end(self, net:NeuralNetRegressor, **kwargs):
        epoch = net.history[-1]["epoch"]
        if epoch >= self.start_epoch:
            self.checkpoint.on_epoch_end(net, **kwargs)



class LossTriggeredCheckpoint(Callback):
    def __init__(self, checkpoint: Checkpoint, monitor: str = "train_loss",tol:float = 1e-6):
        self.checkpoint = checkpoint
        self.monitor = monitor
        self.tol = tol
        self.best_loss = np.inf
        self.activated = False

    def on_epoch_end(self, net, **kwargs):
        current_loss = net.history[-1][self.monitor]

        # before activation: watch for first non-improvement
        if not self.activated:
            if current_loss < self.best_loss - self.tol:
                self.best_loss = current_loss
            else:
                # first non-improving epoch → activate
                self.activated = True

        # after activation: delegate to checkpoint
        if self.activated:
            self.checkpoint.on_epoch_end(net, **kwargs)