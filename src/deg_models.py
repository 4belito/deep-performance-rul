import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNet

from src.config import EPS_POS
from src.helpers.math import gamma_stats2params as stats2params
from src.particle_filter import ParticleFilter


class DistNet(NeuralNet):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return self.criterion_(y_pred, y_true)


def get_time_weights(seq_len, scale=0.0, device=None):
    x = torch.linspace(0.0, 1.0, seq_len, device=device)
    w = torch.exp(scale * x)
    return w / w.sum()

class GammaNLLLoss(nn.Module):
    def __init__(self, seq_len:int, scale:float=0.0):
        super().__init__()
        weights = get_time_weights(seq_len, scale=scale)
        self.weights: torch.Tensor
        self.register_buffer("weights", weights)

    def forward(self, params: dict[str,torch.Tensor], t: torch.Tensor):
        beta = params["beta"]
        lamb = params["lambda"]
        mu = params["mu"]

        lamb = torch.clamp(lamb, min=1e-8, max=1e6)

        t_shifted = t + mu

        term1 = torch.lgamma(beta)
        term2 = (beta - 1.0) * torch.log(t_shifted)
        term3 = beta * torch.log(lamb)
        term4 = lamb * t_shifted

        nll = term1 - term2 - term3 + term4

        return torch.mean(self.weights * nll)
    
class MSE(nn.Module):
    def __init__(self, seq_len, scale=0.0):
        super().__init__()
        weights = get_time_weights(seq_len, scale=scale)
        self.weights: torch.Tensor
        self.register_buffer("weights", weights)

    def forward(self, params: dict, t: torch.Tensor):
        beta = params["beta"]
        lamb = params["lambda"]
        mu = params["mu"]

        Ts_mean = beta / lamb - mu
        error = Ts_mean - t

        return torch.mean(self.weights * error ** 2)
    
class CombinedGammaLoss(nn.Module):
    def __init__(self, seq_len, nll_scale=0.0, mse_scale=0.0, alpha=0.5):
        super().__init__()
        self.gamma_nll = GammaNLLLoss(seq_len, scale=nll_scale)
        self.mse = MSE(seq_len, scale=mse_scale)
        self.alpha = alpha

    def forward(self, params: dict, t: torch.Tensor):
        loss_nll = self.gamma_nll(params, t)
        loss_mse = self.mse(params, t)
        return (1.0 - self.alpha) * loss_nll + self.alpha * loss_mse


class DistModel(nn.Module):
    """
    Gamma distribution with time-dependent rate:
        X ~ Gamma(beta, lambda(s), mu)

    lambda(s) = lambda / (1 - s / a)^(1/p)
    """

    def __init__(self, max_s: float):
        super().__init__()

        # ---- Raw (unconstrained) parameters ----
        self.raw_mean = nn.Parameter(torch.tensor(229.0626))
        self.raw_var  = nn.Parameter(torch.tensor(44.6767))

        self.raw_mu = nn.Parameter(torch.tensor(145.1253))
        self.raw_p  = nn.Parameter(torch.tensor(10.8037))
        self.raw_a  = nn.Parameter(torch.tensor(np.log(np.exp(EPS_POS) - 1)))

        self.max_s = max_s

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------
    @staticmethod
    def _positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def _a_param(self) -> torch.Tensor:
        # ensure a >= max_s
        return self.max_s + F.softplus(self.raw_a)

    # ------------------------------------------------------------------
    # Public parameter access (NO ARGUMENTS)
    # ------------------------------------------------------------------
    def get_parameters(self) -> dict:
        """
        Return constrained *static* model parameters as tensors.
        """
        mean = self._positive(self.raw_mean)
        var  = self._positive(self.raw_var)
        
        p  = self._positive(self.raw_p)
        mu = self._positive(self.raw_mu)
        a  = self._a_param()


        return {
            "mean": mean,
            "variance": var,
            "mu": mu,
            "p": p,
            "a": a,
        }

    # ------------------------------------------------------------------
    # Time deformation
    # ------------------------------------------------------------------
    @staticmethod
    def lambda_denominator(s: torch.Tensor, a: torch.Tensor, p: torch.Tensor):
        return torch.clamp(a - s, min=EPS_POS) ** (1.0 / p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, s: torch.Tensor) -> dict:
        # skorch safety: (N,1) → (N,)
        if s.ndim == 2 and s.shape[1] == 1:
            s = s.squeeze(1)

        params = self.get_parameters()

        mean = params["mean"]
        var  = params["variance"]
        mu   = params["mu"]
        p    = params["p"]
        a    = params["a"]

        beta, lamb = stats2params(mean, var)
        denom = self.lambda_denominator(s, a, p)
        lamb_s = lamb / denom

        # explicit broadcasting
        beta = beta.expand_as(lamb_s)
        mu   = mu.expand_as(lamb_s)

        return {
            "beta": beta,
            "lambda": lamb_s,
            "mu": mu,
        }

class Stopper():
    def __init__(self,change_tol,patiance):
        self.best_loss = float('inf')
        self.prev_best_loss = float('inf')
        self.prev_best_epoch = -2
        self.best_epoch = -1
        self.change_tol = change_tol
        self.stagnant_best = 0
        self.patiance = patiance
    
    
    def get_best_loss_derivative(self):
        return (self.best_loss-self.prev_best_loss)/(self.best_epoch - self.prev_best_epoch)
    
    def update_stagnant(self):
        if self.get_best_loss_derivative() > -self.change_tol:
            self.stagnant_epochs += 1
            if self.stagnant_epochs > self.patiance/2:
                print(f"Best Loss change below tolerance → {self.stagnant_epochs}/{self.patiance}")
        else:
            self.stagnant_epochs = 0  #
    
    def update_best(self,current_loss,current_epoch):
        self.prev_best_loss = self.best_loss 
        self.best_loss = current_loss
        self.prev_best_epoch = self.best_epoch
        self.best_epoch = current_epoch
    
    def check_best_loss(self,current_loss):
        return current_loss < self.best_loss

    def check_stagnant(self):
        return self.stagnant_epochs >= self.patiance


def fit(model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module = nn.MSELoss(),
    lr: float = 0.01,
    n_epochs: int = 100000,
    device: torch.device = torch.device("cpu")
    ) -> None:
    """
    Trains the given model using the provided data.
    """
    optimizer = optim.Adam(model.parameters(), lr)
    

    # Get data
    t,s=data
    x=torch.tensor(s, dtype=torch.float32).to(device)
    y=torch.tensor(t, dtype=torch.float32).to(device)
    model.to(device)
    
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(n_epochs):
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        y_hat = model(x)

        # Compute the loss
        loss = criterion(y_hat, y)
        current_loss=loss.item()
        if current_loss < best_loss:
            best_model=copy.deepcopy(model)
            best_loss = current_loss
            
        loss.backward()
        optimizer.step()

        # Print progress every 100 epochs
        if epoch>0 and (epoch % int(n_epochs/100) == 0):
            print(f"Epoch {epoch}, Best Loss: {best_loss}")
            # Debugging Plot
            if (epoch % int(n_epochs/5) == 0) and isinstance(model,DistModel):
                params = model.get_parameters()
                mean = params["mean"]
                var  = params["variance"]
                p    = params["p"]
                mu   = params["mu"]
                a    = params["a"]
                state = np.array([mean.item(),var.item(),p.item(),mu.item(),a.item()]) 
                ParticleFilter.plot_states(
                        np.stack(state[np.newaxis,:],axis=0),
                        150,
                        data[0],
                        data[1],
                        multiply_level=None,
                        n_particles=None,
                        title = 'Debugging Plot',
                        resolution=1024)
                beta = y_hat['beta']
                lamb_s = y_hat['lambda']
                mu_pos = y_hat['mu']
                Ts_mean = beta/lamb_s-mu_pos
                plt.plot(Ts_mean.detach().numpy(),data[1],'-',color='orange',alpha=0.6,label='mean')
                
                plt.xlim([0,None])
                
                plt.show()
    print(f'Best value: {best_loss}')
    print(f"Best parameters: {best_model.get_parameters()}")
    return best_model