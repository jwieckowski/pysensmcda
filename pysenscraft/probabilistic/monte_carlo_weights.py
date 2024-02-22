# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ..criteria import random_distribution as dist

def monte_carlo_weights(n: int, distribution='uniform', num_samples=1000, params=None):
    """
    Generate criteria weights probabilistically using Monte Carlo simulation.

    Parameters
    ----------
    n : int
        Number of weights to generate.

    num_samples : int, optional, default=1000
        Number of samples to generate in the Monte Carlo simulation.

    distribution : str, optional, default='uniform'
        Probability distribution for weight modification.
        Options: 'chisquare', 'laplace', 'normal', 'random', 'triangular', 'uniform'.

    params : dict, optional
        Parameters for the chosen distribution. Check NumPy documentation for details.

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

    allowed_distributions = ['chisquare', 'laplace', 'normal', 'random', 'triangular', 'uniform']
    if distribution not in allowed_distributions:
        raise ValueError(f'Invalid distribution. Choose from: {allowed_distributions}')

    if params is None:
        params = {}

    modified_weights = []

    for _ in range(num_samples):
        
        try:
            method = getattr(dist, f'{distribution}_distribution')
            weights = method(**params, size=n)
        except Exception as err:
            raise ValueError(err)

        modified_weights.append(weights)

    return np.array(modified_weights)
