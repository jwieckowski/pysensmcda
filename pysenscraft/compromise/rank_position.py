# Copyright (C) 2024 Bartosz Paradowski

import numpy as np
from scipy.stats import rankdata

def rank_position(rankings: np.ndarray):
    """
    Calculates compromised ranking using rank position method.

    Parameters
    ----------
        rankings : ndarray
            Two-dimensional matrix containing different rankings in columns.

    Example
    -------
        >>> rankings = np.array([[3, 2, 3],
        >>>                     [4, 4, 4],
        >>>                     [2, 3, 2],
        >>>                     [1, 1, 1]])
        >>> rank = rank_position_method(matrix)

    Returns
    -------
        ndarray
            Vector including compromise ranking.
    """

    preference = 1 / (np.sum((1 / rankings), axis = 1))
    
    return rankdata(preference)