# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import pandas as pd
from ..validator import Validator

def fuzzy_ranking(rankings: np.ndarray, normalization_axis: None | int = None):
    """
    Generate fuzzy ranking matrix based on positional rankings.

    Parameters
    ----------
    rankings : np.ndarray
        2D array with positional rankings, where each row represents a separate positional ranking of alternatives.

    normalization_axis : int, optional
        Specifies the type of fuzzy ranking representation. 
        If 1, it normalizes the obtained fuzzy rankings regarding values in rows (by positions in rankings).
        If 0, it normalizes the obtained fuzzy rankings regarding values in columns (by distribution of positions for alternatives).
        If None or not specified, it returns the default fuzzy ranking matrix without data normalization.

    Returns
    -------
    np.ndarray
        Fuzzy ranking matrix based on the specified normalization_axis. Each element represents the membership degree of alternatives and ranking positions.

    Examples
    --------
    >>> rankings = np.array([
    ...     [1, 2, 3, 4, 5],
    ...     [2, 1, 5, 3, 4],
    ...     [4, 3, 2, 5, 1],
    ...     [3, 2, 1, 4, 5],
    ... ])
    >>> fuzzy_rank = fuzzy_ranking(rankings, normalization_axis=0)
    >>> print(fuzzy_rank)
    """

    Validator.is_type_valid(rankings, np.ndarray)
    # if not isinstance(rankings, np.ndarray):
    #     raise TypeError('Rankings should be given as numpy array type')
    
    Validator.is_dimension_valid(rankings, 2)
    # if rankings.ndim != 2:
    #     raise ValueError('Rankings should be given as 2D array')

    Validator.is_type_valid(normalization_axis, (None, int))
    if normalization_axis is not None:
        Validator.is_in_list(normalization_axis, [0, 1])
        # if normalization_axis not in [0, 1]:
        #     raise ValueError('Normalization axis should equal 1 or 0')

    ALT = len(rankings[0])

    columns_labels = [f'A{i+1}' for i in range(ALT)]

    pd_rank = pd.DataFrame(rankings, columns=columns_labels)
    rank_prob = np.zeros((ALT, ALT))  

    for row, col in enumerate(pd_rank.columns):
        for pos in range(ALT):
            rank_prob[pos, row] = len(pd_rank[pd_rank[col] == pos+1])

    rank_prob = np.round(rank_prob / len(pd_rank), 4)
    rank_prob = pd.DataFrame(rank_prob, columns=columns_labels)

    if normalization_axis is None:
        return rank_prob.to_numpy()
    else:
        M = rank_prob.to_numpy()
        return np.round(M / np.max(M, axis=normalization_axis), 4)
