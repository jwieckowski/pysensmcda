# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ...validator import Validator
from ...utils import memory_guard

@memory_guard
def chisquare_distribution(size: int, df: float = 1.0) -> np.ndarray:
    """
    Generate a set of normalized weights sampled from a normal distribution.

    Parameters:
    ------------
    size : int
        Number of weights to generate.

    df : float, optional, default=1.0
        Number of degrees of freedom. Must be > 0.

    Returns:
    ---------
    ndarray
        Array of normalized weights sampled from a normal distribution.

    Examples:
    ----------
    Example 1: Generate normalized weights from a chi-square distribution with default parameters
    
    >>> weights = chisquare_distribution(3)
    >>> print(weights)

    Example 2: Generate normalized weights from a chi-square distribution with explicit parameters
    
    >>> weights = chisquare_distribution(3, 5)
    >>> print(weights)
    """

    Validator.is_type_valid(size, (int, np.integer), 'size')
    Validator.is_positive_value(size, var_name='size')
    Validator.is_type_valid(df, (int, np.integer, float, np.floating), 'df')
    Validator.is_positive_value(df, var_name='df')

    weights = np.abs(np.random.chisquare(df, size=size))
    return np.array(weights) / np.sum(weights)
