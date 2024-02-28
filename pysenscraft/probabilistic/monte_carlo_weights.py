# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ..criteria import random_distribution as dist
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def monte_carlo_weights(n: int, distribution: str, params: dict, num_samples: int =1000):
    """
    Generate criteria weights probabilistically using Monte Carlo simulation.

    Parameters
    ----------
    n : int
        Number of weights to generate.

    distribution : str
        Probability distribution for weight modification.
        Options: 'chisquare', 'laplace', 'normal', 'random', 'triangular', 'uniform'.

    params : dict
        Parameters for the chosen distribution. Check NumPy documentation for details.

    num_samples : int, optional, default=1000
        Number of samples to generate in the Monte Carlo simulation.

    Returns
    -------
    ndarray
        Array of modified criteria weights based on Monte Carlo simulation.

    Examples
    --------
    >>> n = 3
    >>> modified_weights = monte_carlo_weights(n, num_samples=1000, distribution='normal', params={'loc': 0.5, 'scale': 0.1})
    >>> print(modified_weights)
    """

    Validator.is_type_valid(n, int)
    Validator.is_type_valid(distribution, str)
    allowed_distributions = ['chisquare', 'laplace', 'normal', 'random', 'triangular', 'uniform']
    Validator.is_in_list(distribution, allowed_distributions)
    Validator.is_type_valid(num_samples, int)
    Validator.is_type_valid(params, dict)

    modified_weights = []

    for _ in range(num_samples):
        
        try:
            method = getattr(dist, f'{distribution}_distribution')
            weights = method(**params, size=n)
        except Exception as err:
            raise ValueError(err)

        modified_weights.append(weights)

    return np.array(modified_weights)
