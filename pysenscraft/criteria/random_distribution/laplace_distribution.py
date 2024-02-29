# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ...validator import Validator
from ...utils import memory_guard

@memory_guard
def laplace_distribution(size: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """
    Generate a set of normalized weights sampled from a laplace distribution.

    Parameters
    ----------
    size : int
        Number of weights to generate.

    loc : float, optional, default=0.0
        The position of distribution peak
    
    scale : float, optional, default=1.0
        The exponential decay. Must be non-negative

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a laplace distribution.

    ## Example
    --------
    ### Example: Generate normalized weights from a laplace distribution with default parameters
    >>> weights = laplace_distribution(3)
    >>> print(weights)

    ### Example 2: Generate normalized weights from a laplace distribution with explicit parameters
    >>> weights = laplace_distribution(3, 5, 2)
    >>> print(weights)
    """

    Validator.is_type_valid(size, int)
    Validator.is_positive_value(size)
    Validator.is_type_valid(loc, (int, float))
    Validator.is_type_valid(scale, (int, float))
    Validator.is_positive_value(scale)

    weights = np.abs(np.random.laplace(loc, scale, size=size))
    return np.array(weights) / np.sum(weights)
