# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def laplace_distribution(loc: float = 0.0, scale: float = 1.0, size: int = 3):
    """
    Generate a set of normalized weights sampled from a normal distribution.

    Parameters
    ----------
    loc : float, optional, default=0.0
        The position of distribution peak
    
    scale : float, optional, default=1.0
        The exponential decay. Must be non-negative

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
        raise ValueError('The exponential decay (scale) must be non-negative')

    weights = np.abs(np.random.laplace(loc, scale, size=size))
    return np.array(weights) / np.sum(weights)

if __name__ == '__main__':
    weights = laplace_distribution()
    print(weights)   
