# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import numpy as np
from scipy.stats import rankdata
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def rank_position(rankings: np.ndarray) -> np.ndarray:
    """
    Calculates compromised ranking using rank position method.

    Parameters:
    ------------
        rankings : ndarray
            Two-dimensional matrix containing different rankings in columns.

    Returns:
    ---------
        ndarray
            Vector including compromise ranking.
    
    Example:
    ---------
        >>> rankings = np.array([[3, 2, 3],
        >>>                     [4, 4, 4],
        >>>                     [2, 3, 2],
        >>>                     [1, 1, 1]])
        >>> rank = rank_position_method(rankings)

    """
    Validator.is_type_valid(rankings, np.ndarray, 'rankings')

    preference = 1 / (np.sum((1 / rankings), axis = 1))
    
    return rankdata(preference)