# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

def remove_single(matrix: np.ndarray, weights: np.ndarray, combinations: np.ndarray = None) -> dict:
    """
    Generate scenarios after removing a single criterion from the matrix and adjusting criteria weights.

    Parameters
    ----------
    matrix : ndarray
        2D array with decision matrix containing multiple criteria and alternatives.

    weights : ndarray
        Vector of initial criteria weights.

    combinations : ndarray, optional, default=None
        Vector of indices of the criteria to be removed. If not provided, all criteria will be considered subsequently.

    Returns
    -------
    dict
        Dictionary of scenarios containing modified decision matrices and adjusted criteria weights.
        Dictionary contains key of 'C[n]' where n represents the index of removed criterion. 
        The value corresponding to the key is represented by dictionary {'matrix': [...], 'weights': [...]}, where both matrix and weights contain data without the removed criterion.

    Notes
    -----
    This function generates multiple scenarios by iteratively removing a single criterion from the decision matrix and adjusting the corresponding criteria weights. Each scenario includes a new decision matrix and updated criteria weights.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    # Example 1: Generate scenarios based on given criteria indexes
    >>> matrix = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
    >>> weights = np.array([0.4, 0.3, 0.3])
    >>> combinations = np.array([1, 2])
    >>> scenarios = remove_single(matrix, weights, combinations)
    >>> print(scenarios)
    >>> {'C[1]': {'matrix': array([[0.2, 0.3],
                                   [0.6, 0.1]]),
                'weights': array([0.550, 0.450])},
         'C[2]': {'matrix': array([[0.2, 0.5],
                                   [0.6, 0.3]]),
                'weights': array([0.45, 0.55])}}
    
    # Example 2: Generate scenarios based on default criteria indexes
    >>> matrix = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
    >>> weights = np.array([0.4, 0.3, 0.3])
    >>> scenarios = remove_single(matrix, weights)
    >>> print(scenarios)
    >>> {'C[0]': {'matrix': array([[0.2, 0.5],
                                   [0.6, 0.3]]),
                'weights': array([0.550, 0.450])},
         'C[1]': {'matrix': array([[0.2, 0.3],
                                   [0.6, 0.1]]),
                'weights': array([0.550, 0.450])},
         'C[2]': {'matrix': array([[0.2, 0.5],
                                   [0.6, 0.3]]),
                'weights': array([0.45, 0.55])}}
    """

    # Validator.check_type(matrix, 'matrix', np.ndarray)
    # Validator.check_type(weights, 'weights', np.ndarray)

    # Validator.matrix_extension(matrix)
    # Validator.weights_extension(weights)
    # Validator.weights_sum(weights)
    # Validator.check_length(weights, 'weights', matrix.shape[1])

    # if combinations is not None:
        # Validator.check_type(combinations, 'combinations', np.ndarray)
        # Validator.check_minimum_length(combinations, 'combinations', 0)
        # Validator.check_maximum_length(combinations, 'combinations', weights.shape[0])
        # Validator.check_minimum_value(combinations, 'combinations', 0)
        # Validator.check_maximum_value(combinations, 'combinations', weights.shape[0]-1)
        # Validator.check_unique(combinations, 'combinations')

    scenarios = {}

    # generate vector of subsequent criteria indexes to remove
    criteria_combs = combinations if combinations is not None else range(0, matrix.shape[1])

    # remove column in decision matrix and adjust criteria weights values
    for crit in criteria_combs:
        new_matrix = np.delete(matrix, crit, axis=1)
        deleted_weight = weights[crit]
        new_weights = np.delete(weights, crit)
        new_weights += deleted_weight / new_weights.shape[0]

        scenarios[f'C[{crit}]'] = {
            'matrix': new_matrix,
            'weights': new_weights
        } 

    return scenarios

def remove_multiple(matrix: np.ndarray, weights: np.ndarray, combinations: np.ndarray = None, n: int = 3) -> dict:
    """
    Generate scenarios after removing multiple criteria simultaneously from the matrix and adjusting criteria weights.

    Parameters
    ----------
    matrix : ndarray
        2D array with decision matrix containing multiple criteria and alternatives.

    weights : ndarray
        Vector of initial criteria weights.

    combinations : ndarray, optional, default=None
        2D array of combinations of criteria indices to be removed. If not provided, combinations will be generated automatically based on the specified `n` value.

    n : int, optional, default=3
        Number of minimum number of criteria to keep in the decision matrix. In calculations used only when `combinations` is not provided.

    Returns
    -------
    dict
        Dictionary of scenarios containing modified decision matrices and adjusted criteria weights.
        Dictionary contains key of 'C[n1-n2-...-nk]', where 'n1', 'n2', etc., represents the indexes of removed criteria. 
        The value corresponding to the key is represented by dictionary {'matrix': [...], 'weights': [...]}, where both matrix and weights contain data without the removed criteria.

    Notes
    -----
    This function generates multiple scenarios by iteratively removing multiple criteria from the decision matrix and adjusting the corresponding criteria weights. Each scenario includes a new decision matrix and updated criteria weights.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    # Example 1: Generate scenarios with custom combinations
    >>> matrix = np.array([
            [1, 2, 3, 4, 4],
            [1, 2, 3, 4, 4],
            [4, 3, 2, 1, 4]
        ])
    >>> weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    >>> combinations = np.array([[1, 2], [2, 3], [1, 3], [0, 2]])
    >>> scenarios = remove_multiple(matrix, weights, combinations)
    >>> print(scenarios)
    >>> {'C[1-2]': {'matrix': array([[1, 4, 4],
                                   [1, 4, 4],
                                   [4, 1, 4]]),
                'weights': array([0.4 , 0.35, 0.25])},
         'C[2-3]': {'matrix': array([[1, 2, 4],
                                   [1, 2, 4],
                                   [4, 3, 4]]),
                'weights': array([0.38333333, 0.38333333, 0.23333333])}, ...}
    
    # Example 2: Generate scenarios with default combinations
    >>> matrix = np.array([
            [1, 2, 3, 4, 4],
            [1, 2, 3, 4, 4],
            [4, 3, 2, 1, 4]
        ])
    >>> weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    >>> scenarios = remove_multiple(matrix, weights)
    >>> print(scenarios)
    >>> {'C[0-1]': {'matrix': array([[3, 4, 4],
                                   [3, 4, 4],
                                   [2, 1, 4]]),
                'weights': array([0.36666667, 0.36666667, 0.26666667])},
         'C[0-2]': {'matrix': array([[2, 4, 4],
                                   [2, 4, 4],
                                   [3, 1, 4]]),
                'weights': array([0.4 , 0.35, 0.25])}, ...}
    """

    # Validator.check_type(matrix, 'matrix', np.ndarray)
    # Validator.check_type(weights, 'weights', np.ndarray)
    # Validator.matrix_extension(matrix)
    # Validator.weights_extension(weights)
    # Validator.weights_sum(weights)
    # Validator.check_length(weights, 'weights', matrix.shape[1])

    # if combinations is not None:
        # Validator.check_dimension(combinations)
        # Validator.check_type(combinations, 'combinations', np.ndarray)
        # Validator.check_minimum_length(combinations, 'combinations', 0)
        # Validator.check_maximum_length(combinations, 'combinations', weights.shape[0])
        # Validator.check_minimum_value(combinations, 'combinations', 0)
        # Validator.check_maximum_value(combinations, 'combinations', weights.shape[0]-1)
        # Validator.check_unique(combinations, 'combinations')

    criteria_combs = []

    if combinations is None:
        criteria_set = np.arange(0, weights.shape[0], 1)
        for i in range(2, weights.shape[0]-n+1):
            c_comb = itertools.combinations(criteria_set, i)
            criteria_combs.append(list(c_comb))
        
        criteria_combs = np.array([item for sublist in criteria_combs for item in sublist], dtype='object')
    else:
        criteria_combs = combinations

    scenarios = {}

    for crit in criteria_combs:
        crit = np.array(crit, dtype='int')

        new_matrix = np.delete(matrix, crit, axis=1)
        deleted_weights = np.sum(weights[crit])
        new_weights = np.delete(weights, crit)
        new_weights += deleted_weights / new_weights.shape[0]

        scenarios[f'C[{"-".join(map(str, crit))}]'] = {
            'matrix': new_matrix,
            'weights': new_weights
        } 

    return scenarios

if __name__ == '__main__':
    # matrix = np.array([
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [4, 3, 2, 1]
    # ])
    # weights = np.array([0.25, 0.25, 0.2, 0.3])

    # # single criterion
    # criteria = np.array([1, 2])

    # r = remove_single(matrix, weights)
    # print(r)

    # multiple criteria simultaneously

    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    criteria = np.array([[1, 2], [2, 3], [1, 3], [0, 2]])
    r = remove_multiple(matrix, weights, criteria)
    # r = remove_multiple(matrix, weights)
    print(r)