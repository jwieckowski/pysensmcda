# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def triangular_distribution(left: float = 0.0, mode: float = 0.5, right: float = 1.0, size: int = 3):
    """
    Generate a set of normalized weights sampled from a triangular distribution.

    Parameters
    ----------
    left : float, optional, default=0.0
        The lower bound of the triangular distribution.

    mode : float, optional, default=0.5
        The mode of the triangular distribution.

    right : float, optional, default=1.0
        The upper bound of the triangular distribution.

    size : int, optional, default=3
        Number of weights to generate.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a triangular distribution.

    ## Example
    --------
    TODO
    """

    if left > mode or mode > right or left > right:
        raise ValueError('Parameters should follow the condition left <= mode <= right')


    weights = np.abs(np.random.triangular(left, mode, right, size=size))
    return np.array(weights) / np.sum(weights)

if __name__ == '__main__':
    weights = triangular_distribution()
    print(weights)   
