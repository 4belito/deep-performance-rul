from functools import partial
from typing import Union

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_dist

from src.config import EPS_POS
from src.helpers.math import safe_brentq


class GammaMixtureProcess:
    """ 
    A class to model and analyze gamma mixture processes with statistical distributions 
    and plotting capabilities. This class supports operations like computing mixture 
    distributions, quantiles, confidence intervals, and generating visualizations.
    """   
    uncertainty_color= 'orange' 
    mean_color = 'blue' 
    bound_color = 'black'
    resolution = 32 #picture:1024, video: 128, fast video: 64
    
    # Constructor
    def __init__(self, shape:np.ndarray=None, 
                scale:np.ndarray=None, 
                p:np.ndarray=None, 
                loc:np.ndarray=None, 
                a:np.ndarray=None, 
                prob:np.ndarray=None,
                max_life: int=100):
        """
        Initialize the Gamma Mixture Process with component parameters.

        Args:
            shape (np.ndarray): Shape parameters of the gamma components (dimension: n_components).
            scale (np.ndarray): Scale parameters of the gamma components (dimension: n_components).
            p (np.ndarray): Time concavity parameter for each component (dimension: n_components).
            loc (np.ndarray): Time translation for each component (dimension: n_components).
            a (np.ndarray): Performance scaling parameter for each component (dimension: n_components).
            prob (np.ndarray): Component weights (dimension: n_components), must sum to 1.
        """
        self.shape=shape
        self.scale=scale
        self.p = p
        self.loc=loc
        self.a = a
        self.prob=prob
        self.max_life = max_life
        self.s = np.linspace(0,1,self.resolution)
        self.t = np.linspace(0,max_life,self.resolution)
    
    def update(self,shape:np.ndarray, 
                scale:np.ndarray, 
                p:np.ndarray, 
                loc:np.ndarray, 
                a:np.ndarray, 
                prob:np.ndarray):
        self.shape=shape
        self.scale=scale
        self.p = p
        self.loc=loc
        self.a = a
        self.prob=prob
    
    
    def dist(self, 
            t:np.ndarray=None, 
            s:np.ndarray=None, 
            grid:bool=True,
            func:str ='pdf',
            return_comp:bool=False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the gamma mixture distribution for the given time and shape parameters.
        """
        dists = self.component_dist(t, s, func=func, log=False, grid=grid)
        mixture = np.dot(self.prob, dists).T
        return (mixture, dists.T) if return_comp else mixture


    def component_dist(self,
                    t:np.ndarray = None, 
                    s:np.ndarray = None,
                    func: str = 'pdf',
                    log: bool = False,
                    grid: bool = True) -> np.ndarray:
        """
        Calculate the gamma component distributions for the given time(t) and performance(s) values.
        """
        if t is None: 
            t=self.t
        if s is None: 
            s=self.s   
        s,shape,scale_s,_,loc,_=self._indexed_parameters(s)
        t = np.atleast_1d(t)
        if grid:
            t = t.reshape(-1, 1,1)
        else:
            assert s.shape[1]==len(t), 'If you want the distribution on the sequence (s,t) they must have same length, otherwise make a sxt grid'
            t =t.reshape(1,-1)  
        
        dists=GammaMixtureProcess.safe_gamma_dist(t, shape, scale_s, loc,func,log)
        return dists
    
    def mean(self,s: np.ndarray) -> np.ndarray:
        """
        Calculate the mean of the Gamma Mixture Process for given performance values.
        """
        s,shape,scale_s,_,loc,_=self._indexed_parameters(s)
        mean = np.dot(self.prob,shape * scale_s+loc)
        return mean
    
    def single_quantile(self, s: float, q: float, upper_bound: float = 1e6) -> float:
        """
        Calculate a single quantile value for the Gamma Mixture Process at a given performance value s.
        """
        cdf = partial(self.dist, s=np.array([s]), func='cdf', return_comp=False)
        return safe_brentq(cdf, q , np.min(self.loc), upper_bound)
    
    def quantile(self, s: np.ndarray, q: float) -> np.ndarray:
        """
        Calculate quantiles for a range of shape parameter values.
        """
        quantile = np.vectorize(lambda s: self.single_quantile(s, q))
        return quantile(s)
    
    def conf_interval(self, s: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the confidence interval for the gamma mixture process along the performance values (s).
        """
        mean = self.mean(s)
        alpha = 1 - level
        lower = self.quantile(s,alpha/2)
        upper = self.quantile(s,1-alpha/2)
        return lower,mean,upper
    
    ## This is temporal sofar
    def filter_mask(self, s: np.ndarray) -> np.ndarray:
        """
        Generate a boolean mask to filter out performance values that exceed scaling parameters.
        """
        return np.all(s[:,None]<self.a[None,:],axis=1)
    
    ## This is temporal sofar
    def filter(self, s: np.ndarray) -> np.ndarray:
        """
        Filter performance values based on the scaling parameters.
        """
        return s[self.filter_mask(s)]
    
    
    # Plots
    def plot_dist(self, t: np.ndarray = None, s: np.ndarray = None, func: str = 'pdf', show_prob: bool = False, 
                ax: plt.Axes = None, mix_dist: np.ndarray = None, vmax: float = None, gamma_prob: float = 0.3,title='Gamma Mixture Process') -> plt.Axes:
        """
        Plot the gamma mixture distribution over time and performance where the distribution value is the color dimension.
        """
        # Build the grid
        if t is None: 
            t=self.t
        if s is None: 
            s=self.s   
        X,Y = np.meshgrid(t,s)
        if mix_dist is None: 
            mix_dist = self.dist(t,s,func=func)
        
        # Plot
        if ax is None: 
            _, ax = plt.subplots(figsize=(10, 6))
        norm = mcolors.PowerNorm(gamma=gamma_prob, vmin=0, vmax= 0.5)  # Adjust gamma for non-linearity # TDO improve vmax
        c = ax.pcolormesh(X, Y, np.clip(mix_dist, 0, 1) , shading='auto', cmap='viridis', norm=norm)
        plt.colorbar(c, ax=ax, label=f'{func}')

        ## Plot Information
        if show_prob: 
            self._add_prob_legend(ax)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("scaled performance")
        return ax  
    
    def plot_confidence_interval_segment(self, ax: plt.Axes, lower: float, mean: float, upper: float,  
                                        ymax: float = 1, label: str = None):
        """
        Plot a confidence interval segment and the mean

        Args:
            ax (plt.Axes): Axis to plot on.
            lower (float): Lower bound of the confidence interval.
            mean (float): Mean value within the interval.
            upper (float): Upper bound of the confidence interval.
            ymax (float): Maximum value on the y-axis for scaling.
            label (str): Label for the confidence interval(it should include the confident level). Default is None
        """
        # Plot the confidence interval lines
        line_height_mean =  0.035 * ymax  # 5% of the y-range for the mean line
        line_height_bounds = 0.02 * ymax  # 3% of the y-range for bounds
        hight= 0.01 * ymax
        width=upper-lower
        confidence = patches.Rectangle(
                (lower, 0),        # Bottom-left corner (x, y)
                width,             # Width
                hight,             # Height
                linewidth=1,   # Border thickness
                edgecolor='black',  # Black border color
                facecolor=self.uncertainty_color,  # Fill color
                alpha=1 ,     # Transparency
                label= label)
        ax.add_patch(confidence)
        ax.vlines([mean], ymin=hight, ymax=line_height_mean, color=self.mean_color, linewidth=2, label='mean')
        ax.vlines([lower, upper], ymin=0, ymax=line_height_bounds, color=self.bound_color, linewidth=2)

    def plot_rv_dist(self, t: np.ndarray, s: float, func: str = 'pdf', conf_level: float = 0.95, show_prob: bool = False, 
                show_comp: bool = True, max_prob: float = 1, ax: plt.Axes = None) -> plt.Axes:
        """
        Plot the Gamma Mixture Process at a specific performance value (s) (random variable Ts).

        Args:
            t (np.ndarray): Time values.
            s (float): Performance value.
            func (str): The type of distribution function to plot ('pdf' or 'cdf').
            conf_level (float): Confidence level for the interval (e.g., 0.95 for 95%).
            show_prob (bool): Whether to display component probabilities on the legend.
            show_comp (bool): Whether to show the individual component distributions.
            max_prob (float): Maximum probability value for the y-axis limit.
            ax (plt.Axes): Axis to plot on. A new axis is created if not provided.

        Returns:
            plt.Axes: The axis containing the plot.
        """
        mix_cdf, cdfs = self.dist(t=t, s=np.array([s]), func=func, return_comp=True)

        if ax is None: 
            _, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t, mix_cdf[0], color='black', linestyle='--', label=f"Mixture {func}")
        if show_comp:
            cdfs = cdfs[0]
            if show_prob:
                for i, cdf in enumerate(cdfs):
                    ax.plot(t, cdf, label=f'prob: {self.prob[i]:.2f}')
            else:
                ax.plot(t, cdfs.T)

        if conf_level:
            lower, mean, upper = self.conf_interval(np.array([s]), level=conf_level)
            self.plot_confidence_interval_segment(ax, lower[0], mean[0], upper[0],ymax=max_prob,label=f'{(100*conf_level):.15g}% confidence')

        ax.set_title(f'Gamma Mixture Process at s={s}')
        ax.set_xlabel("t")
        ax.set_ylabel(f'{func}')
        ax.legend()
        ax.set_ylim(0, max_prob)

        return ax  # Return the axis with the plot
    
    def plot_conf_interval(self, s: np.ndarray, level: float = 0.95, ax: plt.Axes = None, alpha: float = 0.5) -> plt.Axes:
        """
        Plot the confidence interval for the gamma mixture process over a range of performance values (s).

        Args:
            s (np.ndarray): Performance values.
            level (float): Confidence level (e.g., 0.95 for 95%).
            ax (plt.Axes): Axis to plot on. A new axis is created if not provided.
            alpha (float): Transparency of the confidence interval fill.

        Returns:
            plt.Axes: The axis containing the plot.
        """
        
        s=self.filter(s)
        lower, mean, upper = self.conf_interval(s, level)
        if ax is None: 
            _ , ax = plt.subplots(figsize=(10, 6))

        # Plot the confidence interval
        ax.fill_betweenx(s, lower, upper, color=self.uncertainty_color, alpha=alpha, label=f'{(100*level):.15g}% confidence')
        ax.plot(mean, s, '-', color=self.mean_color, markersize=4, linewidth=2, label='Mean')
        ax.plot(lower, s, '-', color=self.bound_color, markersize=4, linewidth=1, label='Lower bound')
        ax.plot(upper, s, '-', color=self.bound_color, markersize=4, linewidth=1, label='Upper bound')

        # Add labels and title
        ax.set_title('Confidence Interval')
        ax.set_xlabel('Value')
        ax.set_ylabel('s')
        ax.set_xlim(left=0)
        ax.legend()

        return ax 
        
    @staticmethod
    def get_lambda_denominator(s,a,p):
        return np.maximum(a - s ,EPS_POS) ** np.clip((1 / p), a_min=None, a_max=128) 
        
        
    ## Private Methods
    def _add_prob_legend(self, ax: plt.Axes, title: str = 'Prob', loc: str = 'lower left'):
        """
        Add a legend to an axis for component probabilities.

        Args:
            ax (plt.Axes): Axis to add the legend to.
            title (str): Title of the legend.
            loc (str): Location of the legend on the plot.
        """
        for prob in self.prob: 
            ax.plot([], [], color='none', marker='o', markersize=3, markerfacecolor='black', label=f'{prob:.2f}')

        # Add the legend on the axis
        ax.legend(loc=loc, title=title, handlelength=0.5, handletextpad=0.4, 
                borderaxespad=0.3, labelspacing=0.2, framealpha=0.8)
        
    def _adjust_dim(self, s: np.ndarray, shape: np.ndarray, scale: np.ndarray, 
                    p: np.ndarray, loc: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Adjust the dimensions of input arrays for broadcasting.

        Args:
            s (np.ndarray): Performance (shape) values.
            shape (np.ndarray): Shape parameters of the gamma components.
            scale (np.ndarray): Scale parameters of the gamma components.
            p (np.ndarray): Concavity parameter values.
            loc (np.ndarray): Time translation values for the components.
            a (np.ndarray): Performance scaling parameter values.

        Returns:
            tuple[np.ndarray, ...]: The adjusted arrays with consistent dimensions.
        """
        s=s[None,:]
        shape=shape[:,None]
        scale=scale[:,None]
        p = p[:,None]  
        loc = loc[:,None]
        a = a[:,None]
        return s,shape,scale,p,loc,a
    
    def _indexed_scale(self, s: np.ndarray, scale: np.ndarray, p: np.ndarray, a: np.ndarray,epsilon:float=1e-8) -> np.ndarray:
        """
        Calculate the scale parameter indexed by the performance values s

        Args:
            s (np.ndarray): Performance values to index the scales.
            scale (np.ndarray): Scale parameters for each component.
            p (np.ndarray): Concavity parameter values.
            a (np.ndarray): Performance scaling parameters.

        Returns:
            np.ndarray: The indexed scale parameters.
        """
        return  scale * GammaMixtureProcess.get_lambda_denominator(s,a,p)

    def _indexed_parameters(self, s: np.ndarray) -> tuple[np.ndarray, ...]:
        """
        Retrieve and adjust indexed parameters for the gamma mixture process.

        Args:
            s (np.ndarray): Performance component values.

        Returns:
            tuple[np.ndarray, ...]: The adjusted parameters including `s`, `shape`, 'scaled values', `p`, `loc`, and `a`.
        """  
        s,shape,scale,p,loc,a = self._adjust_dim(s,self.shape,self.scale,self.p,self.loc,self.a)
        scale_s = self._indexed_scale(s,scale,p,a)
        return s,shape,scale_s,p,loc,a

    @staticmethod
    def safe_gamma_dist(x: np.ndarray, shape: np.ndarray, scale: np.ndarray, loc: float = 0.0, 
                        func: str = 'pdf', log: bool = False) -> np.ndarray:
        """
        Safely compute the gamma distribution for given parameters, returning 0 when the scale is zero.

        Args:
            x (np.ndarray): Input time values.
            shape (np.ndarray): Shape parameters for the gamma distribution.
            scale (np.ndarray): Scale parameters for the gamma distribution.
            loc (float): Location parameter for the Scholastic Process. Defaults to 0.
            func (str): The distribution function to evaluate ('pdf' or 'cdf').
            log (bool): Whether to compute logarithmic values.

        Returns:
            np.ndarray: The computed distribution values. If `scale` is zero, values are set to 0 (or `-inf` for log).
        """
        x, shape, scale, loc = np.broadcast_arrays(x, shape, scale, loc)

        # Initialize the result array with zeros
        if log:
            result = np.full_like(x,-np.inf, dtype=float)
        else:
            result = np.zeros_like(x, dtype=float)

        # Create a valid mask for entries where scale > 0
        valid_mask = np.logical_and(scale > 1e-200, shape<1e16)

        # Apply the gamma distribution function
        #scale_safe = np.clip(scale[valid_mask],a_min=1e-300,a_max=None)
        match (func, log):
            case ('pdf', False):
                result[valid_mask] = gamma_dist.pdf(x[valid_mask], shape[valid_mask], loc=loc[valid_mask], scale=scale[valid_mask])
            case ('cdf', False):
                result[valid_mask] = gamma_dist.cdf(x[valid_mask], shape[valid_mask], loc=loc[valid_mask], scale=scale[valid_mask])
            case ('pdf', True):
                result[valid_mask] = gamma_dist.logpdf(x[valid_mask], shape[valid_mask], loc=loc[valid_mask], scale=scale[valid_mask])
            case ('cdf', True):
                result[valid_mask] = gamma_dist.logcdf(x[valid_mask], shape[valid_mask], loc=loc[valid_mask], scale=scale[valid_mask])
            case _:
                raise ValueError("func argument must be 'pdf' or 'cdf' and log a boolean")
        return result






