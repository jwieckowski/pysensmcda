# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ...validator import Validator

def uniform_distribution(size: int, low: float = 0.0, high: float = 1.0):
    """
    Generate a set of normalized weights sampled from a uniform distribution.

    Parameters
    ----------
    size : int
        Number of weights to generate.

    low : float, optional, default=0.0
        Lower bound of the uniform distribution.

    high : float, optional, default=1.0
        Upper bound of the uniform distribution.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a uniform distribution.

    ## Example
    --------
    ### Example 1: Generate normalized weights from a uniform distribution with default parameters
    >>> weights = uniform_distribution(3)
    >>> print(weights)

    ### Example 2: Generate normalized weights from a uniform distribution with explicit parameters
    >>> weights = uniform_distribution(3, 2, 5)
    >>> print(weights)
    """

    Validator.is_type_valid(size, int)
    Validator.is_positive_value(size)
    
    if low < high:
        raise ValueError('Parameters should follow the condition low < high')

    weights = np.abs(np.random.uniform(low, high, size=size))
    return np.array(weights) / np.sum(weights)
