# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ...validator import Validator
from ...utils import memory_guard

@memory_guard
def triangular_distribution(size: int, left: float = 0.0, mode: float = 0.5, right: float = 1.0) -> np.ndarray:
    """
    Generate a set of normalized weights sampled from a triangular distribution.

    Parameters
    ----------
    size : int
        Number of weights to generate.

    left : float, optional, default=0.0
        The lower bound of the triangular distribution.

    mode : float, optional, default=0.5
        The mode of the triangular distribution.

    right : float, optional, default=1.0
        The upper bound of the triangular distribution.

    Returns
    -------
    ndarray
        Array of normalized weights sampled from a triangular distribution.

    ## Example
    --------
    ### Example 1: Generate normalized weights from a triangular distribution with default parameters
    >>> weights = triangular_distribution(3)
    >>> print(weights)
    
    ### Example 2: Generate normalized weights from a triangular distribution with explicit parameters
    >>> weights = triangular_distribution(3, 2, 5, 6)
    >>> print(weights)
    """

    
    Validator.is_type_valid(size, int)
    Validator.is_positive_value(size)

    if left > mode or mode > right or left > right:
        raise ValueError('Parameters should follow the condition left <= mode <= right')

    weights = np.abs(np.random.triangular(left, mode, right, size=size))
    return np.array(weights) / np.sum(weights)
