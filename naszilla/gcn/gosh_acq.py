import numpy as np
from scipy.stats import norm
import sys

# Different acquisition functions
def gosh_acq(prediction, std, explore_type='ucb'):

    # Upper confidence bound (UCB) acquisition function
    if explore_type == 'ucb':
        explore_factor = 0.5
        obj = prediction - explore_factor * std

    return obj