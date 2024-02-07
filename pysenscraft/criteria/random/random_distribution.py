# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def random_distribution(size: int = 3):
    """
    Generate a set of normalized weights sampled from a random distribution ( from half-open interval [0.0, 1.0) ).

    Parameters
    ----------
    size : int, optional, default=3
        Number of weights to generate.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a random distribution.

    ## Example
    --------
    TODO
    """

    weights = np.abs(np.random.random(size=size))
    return np.array(weights) / np.sum(weights)

if __name__ == '__main__':
    weights = random_distribution()
    print(weights)   
