import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

def normalize(x):
    return x/np.sum(x, axis=0, keepdims=True)

def invgamma_stats2params(mean,var):
    ratio = mean**2/var
    beta = ratio + 2
    lamb = (ratio + 1)*mean
    return beta,lamb

def gamma_stats2params(mean,var):
    lamb=mean/var
    beta = lamb*mean
    return beta,lamb

def mult_round(n_particles:int,n_train:int):
    return n_train*(n_particles//n_train+1)

def stable_softplus(x):
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))

def safe_brentq(f, q, lower_bound, upper_bound):
    per=brentq(lambda x: f(x)[0][0] - q, lower_bound, upper_bound)
    return per

def confidence_to_std(conf_level):
    return norm.ppf(0.5 + conf_level / 2)
