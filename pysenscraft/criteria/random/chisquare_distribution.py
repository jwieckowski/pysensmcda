# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def chisquare_distribution(df: float = 1.0, size: int = 3):
    """
    Generate a set of normalized weights sampled from a normal distribution.

    Parameters
    ----------
    df : float, optional, default=1.0
        Number of degrees of freedom. Must be > 0.

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

    if df <= 0:
        raise ValueError('Number of degrees of freedom must be greater than 0')

    weights = np.abs(np.random.chisquare(df, size=size))
    return np.array(weights) / np.sum(weights)

if __name__ == '__main__':
    weights = chisquare_distribution()
    print(weights)   
