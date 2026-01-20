
from typing import List

import optuna

from src.helpers.math import mult_round


def print_dict(dictio,name):
    print(f"{name}:")
    for key, value in dictio.items():
        print(f" {key}: {value}")


def find_failed_trials(trials,last=1):
    n_trials=len(trials)
    failed_trials=[]
    n=0
    for i in range(n_trials-1,0,-1):
        if trials[i].state == optuna.trial.TrialState.FAIL:
            failed_trials.append(i)
            n+=1
        if n==last:
            break
    return failed_trials

def get_attributes(study,*keys):
    return unpack_if_unitary([study.user_attrs.get(key) for key in keys])

def get_parameters(trial,*keys):
    return unpack_if_unitary([trial.params.get(key) for key in keys])

def unpack_if_unitary(lst):
    return lst[0] if len(lst) == 1 else lst

def check_study_attributes(study,attrs):
    for key, value in attrs.items():
        existing_value = study.user_attrs.get(key)
        
        if existing_value is not None and existing_value != value:
            print(f"⚠️ WARNING: Study attribute '{key}' differs! Existing: {existing_value}, New: {value}")
            return False
    return True

def check_equal_attributes(study1,study2,except_list):
    for (key1, value1), value2 in zip(study1.user_attrs.items(),study2.user_attrs.values()):
        if key1 in except_list:
            continue
        if value1 != value2:
            print(f"⚠️ WARNING: Study attribute '{key1}' differs! Study 1: {value1}, Study 2: {value2}")
            return False
    return True

def generate_trial_seed(trial,study=None):
    if not study:
        study = trial.study
    exp_seed,n_train,n_rep=get_attributes(study,'exp_seed','n_train','n_repetitions')
    return exp_seed + n_rep * n_train * trial._trial_id

#Experiment
def set_params(trial):    
    #param_bounds,n_net_params,state_dim=get_attributes(trial.study,'param_bounds','n_net_params','state_dim')
    param_bounds,n_net_params=get_attributes(trial.study,'param_bounds','n_net_params')
    trial.suggest_int('window',*param_bounds['window']) #1, 30)
    state_dim=get_attributes(trial.study,'state_dim')
    for i in range(state_dim):
        trial.suggest_float(f"multiply_scale_{i}", *param_bounds['multiply_scale'])   
    # for i in range(state_dim):
    #     trial.suggest_float(f"spatial_{i}", *param_bounds['spatial'])
    for i in range(n_net_params):
        trial.suggest_float(f"net_params_{i}", *param_bounds['net_params'])


def get_muliply_scale(trial,attr=None):
    if attr is None:
        attr = trial.study.user_attrs
    state_dim=attr.get('state_dim')
    return unpack_if_unitary([trial.params.get(f"multiply_scale_{i}") for i in range(state_dim)])


def set_attrs(trial):
    trial_seed=generate_trial_seed(trial)
    trial.set_user_attr("seed",trial_seed)#generate_trial_seed(trial)



def get_experiment(exp:str,
                trials_id:List[int],
                new:dict={},
                print_info:bool=False,
                data_split:str='test'):
    trials=[]
    study_attrs=[]
    for j,trial_id in enumerate(trials_id):
        study_name=f'hyp_opt_perform{j}' 
        study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{exp}/{study_name}.db')
        
        if trial_id == 'best':
            trial = study.best_trial
        else:
            trial = study.trials[trial_id]
        attr = study.user_attrs
        
        attr.update(new)  
        n_train = attr['n_train']
        if data_split == 'train':
            n_train-=attr['fold_size']
        attr['n_particles']=mult_round(attr['n_particles'],n_train)
        
        
        print(f"Trial #: {trial.number}")
        print(f"Values: {trial.values}")
        if print_info:
            print_dict(attr,name = "Study Attributes")
            print_dict(trial.params,name = "Parameters")
            print_dict(trial.user_attrs,name = "Tiral Attributes")
            
        trials.append(trial)
        study_attrs.append(attr)
    return study_attrs,trials
