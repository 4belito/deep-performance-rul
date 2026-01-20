from typing import Union

import imageio
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

uncertainty_color= '#FF7F50' # light orange
mean_color = 'blue' #'#CC5E3B' # dark orange

def plot_particle_param(ax,pf,color='green',title='Parameter Space - Random Walk'):
    mean,var,p,mu_pos,a=pf.states.T#.states.T
    state_bounds=pf.bounds.get_onstate()
    min_b,max_b=state_bounds
    mean_min,var_min,p_min,mu_pos_min,a_min=min_b
    mean_max,var_max,p_max,mu_pos_max,a_max=max_b
    color_param=mu_pos/a
    color_min=mu_pos_min/a_max #3.6e2/1 #=mu_pos_min/a_max
    color_max=mu_pos_max/a_min #2e3/9.4e-1 #=mu_pos_max/a_min
    mu_pos_normalized = (color_param-color_min)/ (color_max-color_min)
    ax.scatter(mean,var, p, c=mu_pos_normalized, cmap='magma', alpha=0.1)#0.01
    ax.scatter([],[],[],color=color,alpha=0.5,label='particles')
    ax.set_title(title, pad=20)
    ax.set_xlabel(r"$mean$", labelpad=10)
    ax.set_ylabel(r'$var$', labelpad=10)
    ax.set_zlabel("p", rotation=90, labelpad=10)
    ax.set_xlim([mean_min,mean_max])   #[5.1e-4, 2.7e-3]set by using print_state_box argument in predict_dataset
    ax.set_ylim([var_min,var_max])    #[1e-14, 4.03e-10]
    ax.set_zlim([p_min,p_max])  #[2.27e1, 9.59e1]
    ax.legend()

def plot_rul(ax,time,elapsed_time,elapsed_preds,true_rul,y_max=100,title='RUL'):
    lower,pred,upper=elapsed_preds.T
    ax.plot(time, true_rul, '--', color='green', label='true')  # Adjusted the color for distinction
    ax.plot(elapsed_time, lower, '-', color='black',linewidth=0.5)
    ax.plot(elapsed_time, upper, '-', color='black',linewidth=0.5)
    ax.plot(elapsed_time, pred, '-', color=mean_color, label='pred')
    ax.fill_between(elapsed_time, lower, upper,  color=uncertainty_color, alpha=0.5, label='unc')
    ax.set_title('RUL prediction')
    ax.set_xlabel("time")
    ax.set_ylabel("RUL")

    ax.set_ylim(0, y_max)  # Add margin to y-axis limits
    ax.legend()


## modularize this 
def create_frame(pf_list,performs_list,obs_win_list,time,left_plot_arg,video,t_init,t_step):
    n_performs = len(performs_list)
    if video=='param':
        fig, axes = plt.subplots(n_performs, 2,figsize=(10, n_performs*5)) 
        for j,pf in enumerate(pf_list):
            pos=axes[j,0].get_position()
            ax_3d = fig.add_subplot([pos.x0 - 0.07, pos.y0, pos.width, pos.height], projection='3d')
            fig.delaxes(axes[j,0]) 
            plot_particle_param(ax_3d,pf,**left_plot_arg)
    
    if video=='RUL':
        fig, axes = plt.subplots(1, n_performs +1,figsize=(5*(n_performs+1), 5)) 
        plot_rul(axes[0],time[t_init:],**left_plot_arg)
        
        
    
    for j,pf in enumerate(pf_list):
        ax=axes[j+1] if video=='RUL' else axes[j,1]
        perform=performs_list[j]
        obs_win=obs_win_list[j]
        pf.show(ax=ax)
        plot_data(time[:t_step], perform[:t_step],alpha=0.5,color = 'yellow',ax=ax)
        plot_data(time[t_step:], perform[t_step:],alpha=0.5,color = 'white',ax=ax)
        ax.plot(*obs_win.T,'o-', color='#FF7F50', alpha=1, markersize=4, markeredgecolor='black', markeredgewidth=0.8, label='true') 
        pf.gmp.plot_confidence_interval_segment(ax, *pf.eol,ymax=1)
        # ax.text(0.8, -0.12, f'obs eval = {np.sum(pf.prob==0)}',
        #     transform=ax.transAxes,  # Apply transformation to the current axis
        #     fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8))
    
    if video=='RUL': 
        plt.tight_layout()
    

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img

def create_video_from_frames(frames, video_filename, fps=10):
    with imageio.get_writer(video_filename, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved as {video_filename}")


def plot_2d_color(X,Y,C,z_val,z_step,nn_name='nn_name',perform_name='perform_name',data=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    i=int(z_val/z_step)
    # Create a 2D heatmap where Z is represented as color
    norm = mcolors.PowerNorm(gamma=0.2, vmin=0, vmax=np.max(C))  # Adjust gamma for non-linearity
    c = ax.pcolormesh(X[:,:,i], Y[:,:,i], C[:,:,i], cmap='viridis', shading='auto', norm=norm)
    if data:
        plot_data(data['time'],data['performs'],alpha=data['alpha'],ax=ax)
    # Add colorbar
    fig.colorbar(c, ax=ax, label=nn_name)

    # Labels and title
    ax.set_xlabel('future_cert_t')
    ax.set_ylabel('future_cert_s')
    ax.set_title(f'{perform_name}-{nn_name} Neural Network')

    # Show plot
    plt.show()

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