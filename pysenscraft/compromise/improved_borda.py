# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import numpy as np
from scipy.stats import rankdata
from ..validator import Validator
from ..utils import memory_guard

def vector_normalization(x: np.ndarray, cost: bool=True) -> np.ndarray:
    """
    Parameters
    ----------
    x: ndarray
        Vector of numbers to be normalized.
    cost: bool, optional, cost=True
        Type of normalization. If True normalize as cost criterion, if False normalize as profit criterion.

    Example
    -------
    >>> normalized_matrix = matrix.copy()
    >>> for i in range(criteria_number):
    >>>     cost = True if types[i] == -1 else False
    >>>     normalized_matrix[:, i] = normalization(matrix[:, i], cost)

    Returns
    -------
    ndarray
        Normalized vector.
    """
    if cost:
        return 1 - (x / np.sqrt(sum(x ** 2)))
    return x / np.sqrt(np.sum(x ** 2))

@memory_guard
def improved_borda(preferences: np.ndarray, preference_types: np.ndarray | list= [], normalization: callable = vector_normalization, utility_funcs: list[callable] = [], norm_types: np.ndarray | list = []) -> np.ndarray:
    """
    Improved borda was presented along Probabilistic Linguistic MULTIMOORA, where authors used specific utility functions. This implementation relyes on the concept proposed by author, however it does provide freedom for the user.

    Parameters
    ----------
    preferences: ndarray
        Preferences for alternatives in rows that will be further compromised. Columns designates methods / criteria.
    preference_types: list | ndarray, optional, default=[]
        List of types of methods, changes direction of evaluation: -1 for the ascending ranking, 1 for the descending ranking. Defaults to descending for all.
    normalization: callable, optional, default=vector_normalization
        Function to normalize utility functions results. `See vector_normalization` for further information.
    utility_funcs: list[callable], optional, default=[]
        List of utility functions for each of criterion. If provided, must align with number of criteria in preference matrix.
    norm_types: list | ndarray, optional, default=[]
        Changes type of normalization if needed, -1 for cost, 1 for profit.

    Example
    --------
    ### Example 1: no utility functions
    >>> matrix = np.random.random((8,5))
    >>> criteria_num = matrix.shape[1]
    >>> weights = np.ones(criteria_num)/criteria_num
    >>> types = np.ones(criteria_num)
    >>> preferences = np.array([topsis(matrix, weights, types), vikor(matrix, weights, types)]).T
    >>> 
    >>> compromise_ranking = improved_borda(preferences, [1, -1])

    Returns
    -------
    ndarray
        Compromised ranking.

    """
    
    Validator.is_type_valid(preferences, np.ndarray)
    Validator.is_type_valid(preference_types, (list, np.ndarray))
    Validator.is_callable(normalization)
    Validator.is_callable(utility_funcs)
    Validator.is_type_valid(norm_types, (list, np.ndarray))
    Validator.is_in_list(norm_types, [-1, 1])

    alternatives_num, methods_num = preferences.shape

    if not preference_types:
        preference_types = np.ones(methods_num)

    if not norm_types:
        norm_types = np.ones(methods_num)

    Validator.is_shape_equal(len(preference_types), methods_num, custom_message="Number of columns in 'preferences' and length of 'preference_types' are different")
    Validator.is_shape_equal(len(norm_types), methods_num, custom_message="Number of columns in 'preferences' and length of 'norm_types' are different")
    
    if utility_funcs:
        Validator.is_shape_equal(len(utility_funcs), methods_num, custom_message="Number of columns in 'preferences' and length of 'utility_funcs' are different")

    util_prefs = preferences.copy()
    for idx, util_func in enumerate(utility_funcs):
        util_prefs[:, idx] = util_func(preferences[:, idx])

    norm_prefs = util_prefs.copy()
    for i in range(methods_num):
        cost = True if norm_types[i] == -1 else False
        norm_prefs[:, i] = normalization(util_prefs[:, i], cost)

    rankings = rankdata(preferences * -1 * preference_types, axis=0)

    IBS = np.zeros(alternatives_num)
    an = alternatives_num
    for i in range(methods_num):
        if preference_types[i] == -1:
            IBS -= norm_prefs[:, i] * ((rankings[:, i])/((an*(an+1))/2))
        else:
            IBS += norm_prefs[:, i] * ((an - rankings[:, i] + 1)/((an*(an+1))/2))

    compromise_ranking = rankdata(IBS * -1)
    return compromise_ranking
