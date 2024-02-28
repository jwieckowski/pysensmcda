# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def chisquare_distribution(size: int, df: float = 1.0) -> np.ndarray:
    """
    Generate a set of normalized weights sampled from a normal distribution.

    Parameters
    ----------
    size : int
        Number of weights to generate.

    df : float, optional, default=1.0
        Number of degrees of freedom. Must be > 0.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a normal distribution.

    ## Example
    --------
    ### Example: Generate normalized weights from a chi-square distribution with default parameters
    >>> weights = chisquare_distribution(3)
    >>> print(weights)

    ### Example 2: Generate normalized weights from a chi-square distribution with explicit parameters
    >>> weights = chisquare_distribution(3, 5)
    >>> print(weights)
    """

    if df <= 0:
        raise ValueError('Number of degrees of freedom must be greater than 0')

    weights = np.abs(np.random.chisquare(df, size=size))
    return np.array(weights) / np.sum(weights)
