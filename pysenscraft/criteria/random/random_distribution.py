# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def random_distribution(size: int):
    """
    Generate a set of normalized weights sampled from a random distribution ( from half-open interval [0.0, 1.0) ).

    Parameters
    ----------
    size : int
        Number of weights to generate.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a random distribution.

    ## Example
    --------
    ### Example 1: Generate normalized weights from a random distribution with default parameters
    >>> weights = random_distribution(3)
    >>> print(weights)
    """

    weights = np.abs(np.random.random(size=size))
    return np.array(weights) / np.sum(weights)
