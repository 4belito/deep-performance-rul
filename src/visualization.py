# data
from typing import Union

import matplotlib.cm as cm

# Visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def get_color(i, total):
    """Generate a visually distinct color for index i among total items."""
    cmap = cm.get_cmap('tab20', total)  # Or 'hsv', 'nipy_spectral', etc.
    return cmap(i)

def plot_trajs(time_cycles, units_perform, alias, functions=None, n_train=0, y_norm=False, plot_units='all',add_legend=True,threshold=None):
    plt.figure(figsize=(10, 8))
    if functions: 
        t = np.linspace(0, 100, 1001)
    if plot_units == 'all':
        plot_units = list(range(len(time_cycles)))
    total_colors = len(time_cycles)
    for i, (time, perform) in enumerate(zip(time_cycles, units_perform)):
        if i in plot_units:
            color = get_color(i, total_colors)
            if i < n_train:
                plt.plot(time, perform, 'o-', markersize=4, linewidth=1, color='gray', label=f'unit {i}')
            else:
                plt.plot(time, perform, 'o-', markersize=4, linewidth=1, color=color, label=f'Unit {i+1}')
            if functions and (i in functions.keys()):
                func = functions[i]
                plt.plot(t, func(t), '--', markersize=4, linewidth=1, color=color)
    if threshold is not None:
        plt.axhline(y=threshold, color='black', linestyle='--', label='Threshold')
    plt.title(alias)
    plt.ylabel(alias)
    if y_norm: 
        plt.ylim([0, 1])
    plt.xlabel('time (cycle)')
    if add_legend:
        plt.legend()
    plt.grid(True)


def plot_df_single_color(data, variables, labels, size=12, labelsize=17, name=None):
    """
    """
    plt.clf()        
    input_dim = len(variables)
    cols = min(np.floor(input_dim**0.5).astype(int),4)
    rows = (np.ceil(input_dim / cols)).astype(int)
    gs   = gridspec.GridSpec(rows, cols)    
    fig  = plt.figure(figsize=(size,max(size,rows*2))) 
    
    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        ax.plot(data[variables[n]], marker='.', markerfacecolor='none', alpha = 0.7)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)
        plt.ylabel(labels[n], fontsize=labelsize)
        plt.xlabel('Time [s]', fontsize=labelsize)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, format='png', dpi=300)   
    plt.show()
    plt.close()


def plot_data(time_data: Union[np.ndarray, list[np.ndarray]], 
            perform_data: Union[np.ndarray, list[np.ndarray]], 
            color: str = 'white', edge_color: str = 'black', alpha: float = 0.3, 
            add_legend: bool = False, ax: plt.Axes = None) -> plt.Axes:
    """
    Plot performance data over time with customizable appearance and optional legend.

    Args:
        time_data (Union[np.ndarray, list[np.ndarray]]): Array or list of arrays containing time values.
        perform_data (Union[np.ndarray, list[np.ndarray]]): Array or list of arrays containing performance values.
        color (str): Color of the markers. Defaults to 'white'.
        edge_color (str): Color of the marker edges. Defaults to 'black'.
        alpha (float): Transparency of the markers. Defaults to 0.3.
        add_legend (bool): Whether to add a legend to the plot. Defaults to False.
        ax (plt.Axes, optional): Axis to plot on. A new axis is created if not provided.

    Returns:
        plt.Axes: The axis containing the plot.
    """
    
    if ax is None: 
        _, ax = plt.subplots(figsize=(10, 6))
    if add_legend: 
        ax.plot([], [],'o-', color=color, alpha=alpha, markersize=4, markeredgecolor=edge_color, markeredgewidth=0.8,label='data')  
    if isinstance(time_data,list) and isinstance(perform_data,list):
        for time,perform in zip(time_data, perform_data):
            ax.plot(time, perform,'o-', color=color, alpha=alpha, markersize=4, markeredgecolor=edge_color, markeredgewidth=0.8) 
    elif isinstance(time_data,np.ndarray) and isinstance(perform_data,np.ndarray):
        ax.plot(time_data, perform_data,'o-', color=color, alpha=alpha, markersize=4, markeredgecolor=edge_color, markeredgewidth=0.8) 
    return ax