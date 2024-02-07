# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def uniform_distribution(low: float = 0.0, high: float = 1.0, size: int = 3):
    """
    Generate a set of normalized weights sampled from a uniform distribution.

    Parameters
    ----------
    low : float, optional, default=0.0
        Lower bound of the uniform distribution.

    high : float, optional, default=1.0
        Upper bound of the uniform distribution.

    size : int, optional, default=3
        Number of weights to generate.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a uniform distribution.

    ## Example
    --------
    TODO
    """

    weights = np.abs(np.random.uniform(low, high, size=size))
    return np.array(weights) / np.sum(weights)

if __name__ == '__main__':
    weights = uniform_distribution()
    print(weights)   
