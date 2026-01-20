
import math
from typing import List

# Visualization
import matplotlib.pyplot as plt
import numpy as np
import optuna

from src.gmp import GammaMixtureProcess
from src.helpers.math import gamma_stats2params as stats2params

# custom
from src.helpers.math import normalize
from src.helpers.math import stable_softplus as softplus
from src.helpers.optuna import get_muliply_scale
from src.helpers.visualization import plot_data

# Fixed Neural Network Class

class NN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1  # Number of weight matrices (excluding input layer)
        self.params = None
        
    def __call__(self, x):
        """ Forward pass (assuming parameters are always loaded) """
        for i in range(self.n_layers-1):
            W = self.params[f"W{i+1}"]
            b = self.params[f"b{i+1}"]

            x = np.tanh(np.dot(W, x) + b)  # Linear transformation + activation
            
        W = self.params[f"W{self.n_layers}"]
        b = self.params[f"b{self.n_layers}"]
        output = softplus(np.dot(W, x) + b)
        spatial = np.clip(output[:-1],0,2)
        selection = np.clip(output[[-1]],0,1) 
        return spatial,selection
    
    def load(self,params):
        lower=0
        self.params = {} 
        for i in range(self.n_layers):
            W_dim=(self.layer_dims[i+1],self.layer_dims[i])
            b_dim = (self.layer_dims[i+1],1)
            upper_w=lower+math.prod(W_dim)
            upper_b=upper_w+self.layer_dims[i+1]
            W_values=[params[f'net_params_{j}'] for j in range(lower,upper_w)]
            self.params[f"W{i+1}"] = np.array(W_values).reshape(W_dim)
            b_values=[params[f'net_params_{j}'] for j in range(upper_w,upper_b)]
            self.params[f"b{i+1}"] = np.array(b_values).reshape(b_dim)
            lower=upper_b
        
    @staticmethod
    def get_n_parameters(layer_dims):
        layer_dims = np.array(layer_dims)
        return int(np.sum(layer_dims[:-1] * layer_dims[1:] + layer_dims[1:]))





class Bounds:
    """
    This class is designed for visualization purposes, where it tracks and updates bounds 
    on state and PDF values to support visual representations.
    """
    onpdf=0
    def __init__(self, states: np.ndarray):
        self.onstate=Bounds.compute_onstate(states)
    
    # Public Method
    def update_onstate(self, states: np.ndarray):
        """
        Update the bounds on state values by including new state values.
        """
        extended_states = np.concatenate([states, self.onstate],axis=0)
        state_b=Bounds.compute_onstate(extended_states)
        self.onstate=state_b
    
    def update_onpdf(self, pdfs: np.ndarray):
        """
        Update the upper bound on PDF values by including new PDF values.
        """
        extended_pdf= np.append(pdfs.flatten(), self.onpdf) 
        pdf_b=Bounds.compute_onpdf(extended_pdf)
        self.onpdf=pdf_b
    
    def get_onstate(self) -> np.ndarray:
        """
        Get the current bounds on state values.
        """
        return self.onstate
    
    def get_onpdf(self) -> float:
        """
        Get the current upper bound on PDF values.
        """
        return self.onpdf
    
    # Private methods
    @staticmethod
    def compute_onstate(states: np.ndarray) -> np.ndarray:
        """
        Compute the lower and upper bounds of state values.
        """
        min_b = np.min(states, axis=0)
        max_b = np.max(states, axis=0)
        return np.stack([min_b,max_b])
    
    @staticmethod
    def compute_onpdf(pdfs: np.ndarray|list) -> float:
        """
        Compute the upper bound of PDF values.

        Args:
            pdfs (Union[np.ndarray, list]): Array or list of PDF values.

        Returns:
            float: The maximum value in the provided PDF values.
        """
        upper_pdf = np.max(pdfs, axis=0)
        return upper_pdf



class HiddenNoise:
    """
    A class for applying log-space noise to state data as part of the Particle Filter method.

    The Particle Filter method operates on states transformed to log-space (`log(states)`), 
    where noise is applied for state estimation. However, to maintain better explainability 
    of the state parameters (e.g., mean and standard deviation), this log transformation is 
    hidden by applying a log-noise. Then the class instead operates on the original coordinates 
    while ensuring that particles remain inside bounds (e.g., positive values for parameters) after noise is applied. 

    This class computes log-space standard deviations from initial states and supports 
    rescaling of the noise level. Noise is generated using independent Gaussian random variables.
    """


    def __init__(self, state_dim:int,hidden_std: np.ndarray, seed: int = None):
        self.d = state_dim
        self.hidden_std = hidden_std 
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
    # Public instance methods
    def apply(self, states: np.ndarray, scale:float) -> np.ndarray: #, a_max: float = None
        """
        Apply additive log-noise to the given states while keeping the last feature within bounds.
        """
        N = states.shape[0]
        hidden_noise = self.rng.normal(loc=0.0, scale=self.hidden_std*scale, size=(N, self.d))
        noise = HiddenNoise._hidden2visible(hidden_noise)
        noisy_states=HiddenNoise._noisy_states(states,noise)#,a_max=a_max
        return noisy_states

    # # Private methods
    @staticmethod
    def _compute_hidden_std(states: np.ndarray) -> np.ndarray:
        """
        Compute the standard deviations for the hidden states.
        """
        hidden_states=HiddenNoise._visible2hidden(states)
        hidden_stds = np.std(hidden_states, axis=0)
        return hidden_stds
    
    @staticmethod
    def _noisy_states(states,noise):#,a_max=1
        return states*noise
    
    @staticmethod
    def _visible2hidden(states):
        hidden_states = np.log(states) 
        return hidden_states
    
    @staticmethod    
    def _hidden2visible(hidden_states):
        states = np.exp(hidden_states) 
        return states

class ParticleFilter:
    def __init__(self, 
                base_states:np.ndarray,
                max_life:float,
                net: NN, 
                multiply_scale: np.ndarray,
                hidden_noise_std:np.ndarray=None,
                n_particles: int=None,
                seed: int = None,
                name: str='perform_name'):
    
        
        assert base_states.ndim==2, 'States must be a Nxm np.array where N is the number of samples and m the state dimension'
        self.N,self.d = base_states.shape
        n_particles = n_particles or self.N
        assert n_particles % self.N == 0, f'Number of particles ({n_particles}) must be a multiple of the number of initial states ({self.N}). This is for code simplicity'
        # random number generator for reproducibility
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        if hidden_noise_std is None:
            hidden_noise_std = np.ones(self.d) 
        self.noise=HiddenNoise(state_dim=self.d,hidden_std=hidden_noise_std,seed=self.rng.randint(0, 1e6))
        self.states=base_states
        self.prob=self._uniform_prob()
        self.multiply_particles(n_particles//self.N,scale=multiply_scale)
        self.gmp=GammaMixtureProcess(max_life=max_life)
        self.update_GMP()
        self.name=name
        
        self.bounds = None
        self.eol = None
        
        # Learnable function
        self.net=net

    
    # Public Methods
    def multiply_particles(self,n,scale):
        if n>1:
            new_states=[]
            for _ in range(n-1):
                new_states.append(self.noise.apply(self.states,scale))#,a_max=1
            new_states = np.concatenate(new_states,axis=0)
            self.states=np.concatenate((self.states,new_states),axis=0)
            self.prob=np.concatenate(n*[self.prob],axis=0)/n
            self.N*=n   
            
    def update_GMP(self):
        shapes,scales,p,loc,a = ParticleFilter.state2gmp_params(self.states)
        self.gmp.update(shapes,scales,p,loc,a, prob=self.prob)
        
    ## Particle Filter
    def prediction(self,noise_scale):
        self._resample()
        self._add_noise(scale=noise_scale)
        

    def correction(self,obs,sharpness=1):
        pdf_obs=self.gmp.component_dist(*obs.T,func='pdf',log=False,grid=False)
        obs_prob = self._smooth_normalize(pdf_obs,sharpness=sharpness)
        self.prob = np.mean(obs_prob,axis=1)
        
        
    def step(self,obs):
        input=self._compute_net_input(obs)
        spatial,selection= self.net(input)
        self.prediction(noise_scale=spatial[:,0])
        self.update_GMP()
        self.correction(obs,sharpness = selection[:,0]) 
        
    ## Prognostic methods
    def pred_eol(self,current_time,conf_level):
        lower,pred,upper=self.gmp.conf_interval(s=np.array([0]),level=conf_level)
        eol= np.concatenate((lower,pred,upper))
        return np.clip(eol,a_min=current_time,a_max=self.gmp.max_life)
    
    def set_eol(self,current_time,conf_level):
        self.eol=self.pred_eol(current_time,conf_level)
    
    def pred_prob(self,data,log=False):
        dist=self.gmp.dist(*data.T,grid=False)
        if log: 
            dist = np.clip(dist,a_min=1e-323,a_max=None)
            dist=np.log(dist)
        return dist

    ## Visualization
    def show(self,gamma_prob=0.3,title=None,ax=None):
        if ax is None: 
            fig, ax = plt.subplots(figsize=(10, 6))
        if title: 
            fig.suptitle(title, fontsize=16)
        pdfs=self.gmp.dist(grid=True)
        if self.bounds is not None:
            self.bounds.update_onpdf(pdfs)
            vmax=self.bounds.get_onpdf()
        else:
            vmax = None
        self.gmp.plot_dist(func='pdf', show_prob=False, ax=ax,mix_dist=pdfs,vmax=vmax,gamma_prob=gamma_prob,title=f"PF Predictive Distribution({self.name})")
        return ax
            
    
    # Private methods    
    def _initiate_bounds(self,bounds=None):
        if not bounds:
            bounds=Bounds(self.states)
        self.bounds=bounds
    
    def _compute_net_input(self,obs):
        pred_probability=self.pred_prob(obs,log=True)
        present_cert=np.exp(np.mean(pred_probability))
        current_obs=obs[-1]
        t=current_obs[0]/self.gmp.max_life
        s=current_obs[1]
        return np.array([[t],[s],[present_cert]]) 
        
    def _resample(self):
        sampled_indexes=self.rng.choice(np.arange(self.N), size=self.N, p=self.prob)
        self.states=self.states[sampled_indexes]
    
    def _add_noise(self,scale):
        self.states = self.noise.apply(self.states,scale=scale)
        
    def _uniform_prob(self):
        return np.ones(self.N)/self.N
    
    def _smooth_normalize(self,x,sharpness=1):
        x_smooth=x**sharpness
        mask_0=np.all(x_smooth==0,axis=0)
        x_smooth[:, mask_0] = 1/self.N 
        x_smooth[:, ~mask_0]=normalize(x_smooth[:, ~mask_0])
        return x_smooth
    
    
    # Static methods
    @staticmethod
    def state2gmp_params(states):
        mean,var,p,pos_mu,a=states.T
        beta,lamb=stats2params(mean,var) 
        shapes = beta
        scales=1/lamb
        loc=-pos_mu
        return shapes,scales,p,loc,a
    
    
    @staticmethod
    def plot_states(states,max_life,time_data,perform_data,multiply_level=0.3,n_particles=None,title=None,alpha_data=0.3,gamma_prob=1,resolution=1024):
        GammaMixtureProcess.resolution=resolution
        pf=ParticleFilter(states,max_life,net=None,multiply_scale=multiply_level,n_particles=n_particles)
        # Multiply particles
        ax = pf.show(gamma_prob=gamma_prob,title=title)
        plot_data(time_data,perform_data,alpha=alpha_data,ax=ax)

        
    @staticmethod
    def create_particle_filter(study_attr:dict,
                            trial:optuna.trial.FrozenTrial,
                            states:np.ndarray,
                            hidden_noise_std, 
                            name:str = 'perform_name',
                            resolution: None|int=None,
                            bounds: List[int]=[],
                            seed: None|int = None
                            ): 
        multiply_scale= get_muliply_scale(trial,attr=study_attr)
        n_particles,max_life,layer_dims=[study_attr.get(k) for k in ('n_particles','max_life','layer_dims')]
        net=NN(layer_dims)
        net.load(trial.params)
        if resolution:
            GammaMixtureProcess.resolution = resolution
        pf=ParticleFilter(states,max_life,net,multiply_scale,hidden_noise_std,n_particles,seed,name)
        pf._initiate_bounds(bounds)
        return pf
