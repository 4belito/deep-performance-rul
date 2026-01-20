import copy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.config import EPS_POS
from src.helpers.math import gamma_stats2params as stats2params
from src.particle_filter import ParticleFilter


def get_time_weights(seq_len, scale=1.0, device=None):
    """
    Returns a normalized weight vector of shape [seq_len],
    where weights increase over time (toward the end).

    Args:
        seq_len (int): length of the sequence
        power (float): how strongly to weight later steps
                    (1 = linear, 2 = quadratic, etc.)
        device (torch.device, optional): to match model/device
    """
    x = torch.linspace(0, 1, seq_len, device=device)
    weights = torch.exp(scale * x)  # larger scale = more emphasis on the end
    return weights / weights.sum()

class GammaNLLLoss(nn.Module):
    def __init__(self,seq_len, scale=1.0):
        super().__init__()
        weights=get_time_weights(seq_len, scale=scale)
        self.register_buffer('weights',weights)

    def forward(self, params, t):
        beta, lamb, mu_pos = params
        lamb = torch.clamp(lamb, min=1e-8, max=1e6)
        # Compute each term of the NLL
        t = t + mu_pos
        term1 = torch.lgamma(beta)
        term2 = (beta - 1) * torch.log(t) 
        term3 = beta * torch.log(lamb)
        term4 = lamb * t   # lgamma computes the log of the gamma function

        # Combine terms to compute the NLL
        nll = term1 - term2 - term3 + term4
        
        # Return the mean NLL over the batch

        return torch.mean(self.weights * nll)
    
class MSE(nn.Module):
    def __init__(self,seq_len, scale=0):
        super().__init__()
        weights=get_time_weights(seq_len, scale=scale)
        self.register_buffer('weights',weights)

    def forward(self, params, t):
        beta, lamb_s, mu_pos = params
        Ts_mean = beta/lamb_s-mu_pos
        error=Ts_mean-t
        return torch.sqrt(torch.mean(self.weights*error**2))
    
class CombinedGammaLoss(nn.Module):
    def __init__(self, seq_len, nll_scale=0,mse_scale=0,mse_alpha=0.5):  # alpha ∈ [0, 1]
        super(CombinedGammaLoss, self).__init__()
        self.gamma_nll = GammaNLLLoss(seq_len, scale=nll_scale)
        self.mse = MSE(seq_len, scale=mse_scale)
        self.alpha = mse_alpha

    def forward(self, params, t):
        loss_nll = self.gamma_nll(params, t)
        loss_mse = self.mse(params, t)
        return (1 - self.alpha) * loss_nll + self.alpha * loss_mse

class DegModel(nn.Module):
    def __init__(self):
        super(DegModel, self).__init__()
        # Learnable parameters
        self.raw_p = nn.Parameter(torch.tensor(1.0, requires_grad=True)) 
        self.raw_mu = nn.Parameter(torch.tensor(1.0, requires_grad=True)) 
        self.raw_a = nn.Parameter(torch.tensor(1.0, requires_grad=True)) 
    
        self.p_bounds = (0, None)
        self.mu_bounds = (0, None)
        self.a_bounds = (0, None)
    
    def get_parameters(self,as_dict=False,as_float=False):
        p = self.apply_lower_bound(self.raw_p,self.p_bounds[0])    # p > 0
        mu = self.apply_lower_bound(self.raw_mu,self.mu_bounds[0])  # mu < 0
        #a = self.apply_2bounds(self.raw_a,self.a_bounds) 
        a = self.apply_lower_bound(self.raw_a,self.a_bounds[0]) 
        if as_float:
            p=p.item()
            mu=mu.item()
            a=a.item()
        if as_dict:
            return {'p':p,'mu':mu,'a':a}
        else:
            return p,mu,a
    
    def apply_2bounds(self,x,bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(x)
    
    def apply_lower_bound(self,x,lower_bound):
        return lower_bound+F.softplus(x)
    
    def apply_upper_bound(self,x,upper_bound):
        return upper_bound-F.softplus(x)
    
    @staticmethod
    def inverse_apply_2bounds(y, bounds):
        a, b = bounds
        eps = 1e-8  # avoid division by 0
        y = torch.clamp(y, a + eps, b - eps)
        return torch.log((y - a) / (b - y))
    
    @staticmethod
    def inverse_apply_lower_bound(y, upper_bound, eps=1e-8):
        shifted = upper_bound - y
        return torch.where(
            shifted > 20,  # avoid computing exp(-x) when x is very large
            shifted,
            torch.log(torch.expm1(shifted) + eps)
        )
        
    @staticmethod
    def inverse_apply_upper_bound(y, lower_bound, eps=1e-8):
        shifted = y - lower_bound
        return torch.where(
            shifted > 20,  # avoid computing exp(-x) when x is very large
            shifted,
            torch.log(torch.expm1(shifted) + eps)
        )
    
    def forward(self, x):
        pass

# Define the custom neural network
class MeanModel(DegModel):
    def __init__(self):
        super(MeanModel, self).__init__()
        # Learnable parameters
        self.raw_b = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.b_bounds = (0.0, 0.05)
    
    def get_parameters(self,as_dict=False,as_float=False):
        last_param=super().get_parameters(as_dict,as_float)
        b = self.apply_2bounds(self.raw_b,self.b_bounds)  
        if as_float:
            b=b.item()
        if as_dict:
            out={'b':b}
            out.update(last_param)
            return out
        else:
            return (b,)+last_param
    
    def compute_function(self, t, a, b, mu, p):
        """Centralized function computation."""
        return a   - (b * (t + mu)) ** p
    
    def get_function(self):
        b,p,mu,a=self.get_parameters(as_float=True)
        
        def learned_function(t):
            return self.compute_function(t, a, b, mu, p)
        
        return learned_function

    def forward(self, t):
        b,p,mu,a=self.get_parameters()
        # Compute the custom function
        return self.compute_function(t, a, b, mu, p)

# Define the custom neural network
class DistModel(DegModel):
    def __init__(self,max_s):
        super(DistModel, self).__init__()
        # Learnable parameters
        self.raw_p = nn.Parameter(torch.tensor(10.8037, requires_grad=True))
        self.raw_mu = nn.Parameter(torch.tensor(145.1253, requires_grad=True))
        self.raw_a = nn.Parameter(torch.tensor(np.log(np.exp(EPS_POS)-1), requires_grad=True))
        self.raw_mean = nn.Parameter(torch.tensor(229.0626, requires_grad=True))
        self.raw_var = nn.Parameter(torch.tensor(44.6767, requires_grad=True))
        
        # invgamma bounds
        self.a_bounds=(max_s,None)
        self.mean_bounds = (0, None) #(0, 0.1) #
        self.var_bounds = (0, None) #(0, 1e-8)   #
    
    def get_parameters(self,as_dict=False,as_float=False):
        last_param=super().get_parameters(as_dict,as_float)
        mean = self.apply_lower_bound(self.raw_mean,self.mean_bounds[0]) 
        var = self.apply_lower_bound(self.raw_var,self.var_bounds[0])
        a = self.apply_lower_bound(self.raw_a,self.a_bounds[0])
        if as_float:
            mean=mean.item()
            var=var.item()
            a=a.item()
        if as_dict:
            last_param.update({'mean':mean,'var':var,'a':a})
            return last_param
        else:
            return (mean,var) + last_param[:2] +(a,) 
    
    @staticmethod
    def get_lambda_denominator(s,a,p):
        return torch.clamp(1-s/a, min=EPS_POS, max=None)**(1 / p)

    def forward(self, s):
        mean,var,p,mu,a=self.get_parameters()
        beta,lamb = stats2params(mean,var)
        lamb_s=lamb/DistModel.get_lambda_denominator(s,a,p)
        return beta,lamb_s,mu

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
    if isinstance(model,MeanModel):
        x=torch.tensor(t, dtype=torch.float32).to(device)
        y=torch.tensor(s, dtype=torch.float32).to(device)
    elif isinstance(model,DistModel):
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
                state = np.array(list(best_model.get_parameters(as_float=True))) 
                ParticleFilter.plot_states(
                        np.stack(state[np.newaxis,:],axis=0),
                        150,
                        data[0],
                        data[1],
                        multiply_level=None,
                        n_particles=None,
                        title = 'Debugging Plot',
                        resolution=1024)
                beta, lamb_s, mu_pos  =y_hat
                Ts_mean = beta/lamb_s-mu_pos
                plt.plot(Ts_mean.detach().numpy(),data[1],'-',color='orange',alpha=0.6,label='mean')
                
                plt.xlim([0,None])
                
                plt.show()
    print(f'Best value: {best_loss}')
    print(f"Best parameters: {best_model.get_parameters(as_dict=True,as_float=True)}")
    return best_model