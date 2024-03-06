# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import pymcdm
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def ranking_alteration(weights: np.ndarray, 
                        initial_ranking: np.ndarray, 
                        method: callable, 
                        call_kwargs: dict, 
                        ranking_descending: bool, 
                        step: float = 0.01) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Identify the minimal change in criteria weights that causes an alteration in the ranking of alternatives.

    Parameters:
    ------------
    weights : ndarray
        1D vector with criteria weights used for multi-criteria evaluation.

    initial_ranking : ndarray
        1D vector representing the initial ranking of alternatives.

    method : callable
        The evaluation function to be used for preference and ranking calculation.
        Should include `matrix` and `weights` as one of the parameters to pass the decision matrix to the assessment process.

    call_kwargs : dict
        Dictionary with keyword arguments to be passed to the evaluation function.
        Should include `matrix` and `weights` as one of the parameters to pass the modified decision matrix to the assessment process.

    ranking_descending: bool
        Flag determining the direction of alternatives ordering in ranking.
        By setting the flag to True, greater values will have better positions in ranking.

    step : float, optional, default = 0.01
        Step size for the weights modification.

    Returns:
    ---------
    list of tuples
        Each tuple contains the index of the modified weight, the new set of weights, and the resulting ranking.

    Examples:
    ----------
    Example 1: Ranking alteration analysis with default parameters 
    
    >>> weights = np.array([0.4, 0.5, 0.1])
    >>> matrix = np.array([
    ...     [4, 2, 6],
    ...     [7, 3, 2],
    ...     [9, 6, 8]
    ... ])
    >>> types = np.array([-1, 1, -1])
    >>> aras = pymcdm.methods.ARAS()
    >>> pref = aras(matrix, weights, types)
    >>> initial_ranking = aras.rank(pref)
    >>> call_kwargs = {
    ...     "matrix": matrix,
    ...     "weights": weights,
    ...     "types": types
    ... }
    >>> ranking_descending = True
    >>> results = ranking_alteration(weights, initial_ranking, aras, call_kwargs, ranking_descending)
    >>> for r in results:
    ...     print(r)
    
    Example 2: Ranking alteration analysis with default parameters with explicitly defined step
    
    >>> weights = np.array([0.4, 0.5, 0.1])
    >>> matrix = np.array([
    ...     [4, 2, 6],
    ...     [7, 3, 2],
    ...     [9, 6, 8]
    ... ])
    >>> types = np.array([-1, 1, -1])
    >>> aras = pymcdm.methods.ARAS()
    >>> pref = aras(matrix, weights, types)
    >>> initial_ranking = aras.rank(pref)
    >>> call_kwargs = {
    ...     "matrix": matrix,
    ...     "weights": weights,
    ...     "types": types
    ... }
    >>> ranking_descending = True
    >>> step = 0.05
    >>> results = ranking_alteration(weights, initial_ranking, aras, call_kwargs, ranking_descending, step)
    >>> for r in results:
    ...     print(r)
    
    """
    
    Validator.is_callable(method, 'method')
    Validator.is_type_valid(weights, np.ndarray, 'weights')
    Validator.is_dimension_valid(weights, 1, 'weights')
    Validator.is_sum_valid(weights, 1)
    Validator.is_type_valid(initial_ranking, np.ndarray, 'initial_ranking')
    Validator.is_type_valid(call_kwargs, dict, 'call_kwargs')
    Validator.is_key_in_dict(['matrix', 'weights'], call_kwargs, 'call_kwargs')
    Validator.is_type_valid(ranking_descending, bool, 'ranking_descending')
    Validator.is_type_valid(step, (float, np.floating), 'step')
    Validator.is_positive_value(step, var_name='step')

    results = []

    for crit_idx in range(weights.shape[0]):
        
        flag = True
        change_index = 1

        while flag:
            for val in [-1, 1]:
                new_weights = weights.copy()
                new_val = new_weights[crit_idx] + (step*change_index) * val
                
                # no allowed change in weights that cause ranking alteration
                if new_val <= 0 or new_val >= 1:
                    results.append((crit_idx, weights, initial_ranking))
                    flag = False
                    break

                # change weights
                new_weights[crit_idx] = new_val

                equal_diff = np.abs(weights[crit_idx] - new_val) / (weights.shape[0] - 1) * -val
                
                # adjust rest of the weights
                for idx in range(weights.shape[0]):
                    if idx != crit_idx:
                        new_weights[idx] += equal_diff
                
                # no allowed change in weights that cause ranking alteration
                if any([w >=1 or w <= 0 for w in new_weights]):
                    results.append((crit_idx, weights, initial_ranking))
                    flag = False
                    break

                call_kwargs['weights'] = new_weights
                try:
                    new_preferences = method(**call_kwargs)
                    new_ranking = pymcdm.helpers.rankdata(new_preferences, ranking_descending)
                except Exception as err:
                    raise ValueError(err)
                

                if not np.array_equal(initial_ranking, new_ranking):
                    results.append((crit_idx, new_weights, new_ranking))
                    flag = False
                    break

            change_index += 1

    return results
