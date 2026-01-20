import numpy as np


### Evaluation
def compute_rul_data(times):
    ruls = []
    for time in times:
        rul = compute_rul(eol=time[-1],current_time=time)
        ruls.append(rul)
    return ruls

def compute_rul(eol,current_time):
    return eol-current_time

def get_observation_window(t_step,time,perform,window_size):
    # Get Observation window
    t_step+=1
    win_init_idx=max(t_step - window_size, 0)
    time_window=time[win_init_idx:t_step]
    perform_window=perform[win_init_idx:t_step]
    obs_win = np.stack((time_window,perform_window), axis=1)
    return obs_win


def get_future_obs(t_step,time,perform):
    ''' it also includes current time for convenience at the end'''
    fut_time=time[t_step:]
    fut_perform=perform[t_step:]
    fut_obs = np.stack((fut_time,fut_perform), axis=1)
    return fut_obs
