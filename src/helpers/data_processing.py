from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DataConfig:
    path: str
    name: str
    max_life: Optional[int] = 100
    deg_performs_id: Optional[List[int]] = None

def get_units_list(df,column_names,group_by='unit'):
    return [group[column_names].to_numpy() for _, group in df.groupby(group_by)]


def tilted_linspace_end(x_start, x_end, n_interp, tilt):
    base = np.linspace(0, 1, n_interp)
    tilted = 1 - (1 - base)**tilt
    result = x_start + (x_end - x_start) * tilted
    return result

def interpolate(xs,ys,n_points,tilt=5):
    xs_interp=tilted_linspace_end(xs[0], xs[-1],n_points,tilt)
    ys_interp=np.interp(xs_interp,xs,ys)
    return xs_interp,ys_interp


def norm_ROM(performs,u_new=False,th_w=False):
    '''perform is assumed decreasing and non positive'''
    if not u_new: 
        u_new = max([unit_perform.max() for unit_perform in performs])  # Find the maximum value in the dataframe
        #u_new = min(u_new,0)
    if not th_w: 
        th_w = min([unit_perform.min() for unit_perform in performs])
    norm_performs=[]
    for unit_perform in performs:
        #unit_perform = np.clip(unit_perform,a_min=th_w,a_max=None)
        norm_perform = 1-(unit_perform - u_new) /th_w
        norm_performs.append(norm_perform) 
    return norm_performs

def norm_perform(performs, u_new=None, th_w= None,monot='decreasing'):
    """
    Normalize performance data based on specified parameters.
    """
    if monot =='increasing':
        performs = [-perform for perform in performs]  # Invert the performance data for increasing normalization
        if th_w: 
            th_w=-th_w
    if not u_new: 
        u_new = max([unit_perform.max() for unit_perform in performs]) 
    if not th_w: 
        th_w = min([unit_perform.min() for unit_perform in performs])
    norm_performs=[]
    for unit_perform in performs:
        norm_perform = (unit_perform - th_w) / (u_new - th_w)
        norm_performs.append(norm_perform) 
    return norm_performs
    
    
def merge_performances(performs):
    n_units=len(next(iter(performs.values())))
    units_performs=[]
    for i in range(n_units):
        unit_performs = np.vstack([units_perform[i] for units_perform in performs.values()])
        units_performs.append(unit_performs)
    return units_performs

def sep_performances(merged_performs,keys):
    performs={}
    for j,key in enumerate(keys):
        performs[key]=[unit_performs[j] for unit_performs in merged_performs]
    return performs


def prune_obs(time,performs,threshed=0.8):
    mask=np.all(performs < threshed, axis=0)
    time=time[mask]
    performs=performs[:,mask]
    return time,performs
        
        

def hi_computation(unit_performances):
    return np.min(unit_performances,axis=0)


def prune_data(
    time: List[np.ndarray], 
    performs: Dict[str, List[np.ndarray]], 
    threshold: float
) -> None:
    """
    Prune time and performance data based on a performance threshold.

    Args:
        time (List[np.ndarray]): A list of numpy arrays, where each array represents the time-series data for a specific unit.
        performs (Dict[str, List[np.ndarray]]): A dictionary where keys represent performance names, 
                                                and values are lists of numpy arrays containing performance data 
                                                for corresponding units.
        threshold (float): A threshold value used to prune performance data. Observations above this threshold are removed.
    
    Modifies:
        - Updates `time` and `performs` in-place by pruning entries based on the given threshold.

    Note:
        This function modifies the input data structures `time` and `performs` in-place. Ensure to pass mutable objects.
    """
    merged_performs = merge_performances(performs)
    pruned_times = []
    pruned_performs =[]
    for unit_time, unit_performs in zip(time, merged_performs):
        pruned_time, pruned_perform = prune_obs(unit_time, unit_performs, threshed=threshold)
        pruned_times.append(pruned_time)
        pruned_performs.append(pruned_perform)
    pruned_performs = sep_performances(pruned_performs, performs.keys())
    return pruned_times, pruned_performs