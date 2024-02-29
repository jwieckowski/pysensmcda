# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ...validator import Validator
from ...utils import memory_guard

@memory_guard
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

    Validator.is_type_valid(size, int)
    Validator.is_positive_value(size)
    Validator.is_type_valid(loc, (int, float))
    Validator.is_type_valid(scale, (int, float))
    Validator.is_positive_value(scale)

    weights = np.abs(np.random.normal(loc, scale, size=size))
    return np.array(weights) / np.sum(weights)

