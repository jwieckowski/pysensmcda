# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def normal_distribution(size: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """
    Generate a set of normalized weights sampled from a normal distribution.

    Parameters
    ----------
    size : int
        Number of weights to generate.

    loc : float, optional, default=0.0
        Mean of the normal distribution.

    scale : float, optional, default=1.0
        Standard deviation of the normal distribution.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a normal distribution.

    ## Example
    --------
    ### Example 1: Generate normalized weights from a normal distribution with default parameters
    >>> weights = normal_distribution(3)
    >>> print(weights)

    ### Example 2: Generate normalized weights from a normal distribution with explicit parameters
    >>> weights = normal_distribution(3, 5, 2)
    >>> print(weights)
    """

    if scale < 0:
        raise ValueError('Standard deviation (scale) must be non-negative')


    weights = np.abs(np.random.normal(loc, scale, size=size))
    return np.array(weights) / np.sum(weights)

