# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def normal_distribution(loc: float = 0.0, scale: float = 1.0, size: int = 3):
    """
    Generate a set of normalized weights sampled from a normal distribution.

    Parameters
    ----------
    loc : float, optional, default=0.0
        Mean of the normal distribution.

    scale : float, optional, default=1.0
        Standard deviation of the normal distribution.

    size : int, optional, default=3
        Number of weights to generate.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a normal distribution.

    ## Example
    --------
    TODO
    """

    if scale < 0:
        raise ValueError('Standard deviation (scale) must be non-negative')


    weights = np.abs(np.random.normal(loc, scale, size=size))
    return np.array(weights) / np.sum(weights)

if __name__ == '__main__':
    weights = normal_distribution()
    print(weights)   
