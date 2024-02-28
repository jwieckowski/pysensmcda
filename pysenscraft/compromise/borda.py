# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import numpy as np
from scipy.stats import rankdata
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def borda(rankings: np.ndarray) -> np.ndarray:
    """
    Calculates compromised ranking using borda voting rule.

    Parameters
    ----------
        rankings: ndarray
            Two-dimensional matrix containing different rankings in columns.

    Example
    ----------
        >>> rankings = np.array([[3, 2, 3],
        >>>                     [4, 4, 4],
        >>>                     [2, 3, 2],
        >>>                     [1, 1, 1]])
        >>> compromised_ranking = borda(rankings)

    Returns
    --------
        ndarray
            Numpy array containing compromised ranking.
    """

    Validator.is_type_valid(rankings, np.ndarray)

    alt_num = rankings.shape[0]
    count = np.sum((alt_num + 1) - rankings, axis=1)

    return rankdata(-count)