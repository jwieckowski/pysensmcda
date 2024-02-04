# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

# @memory_guard()
def discrete_single(matrix: np.ndarray, discrete_values: (dict | np.ndarray)) -> dict:
    """
    Generate scenarios by modifying elements of the given matrix with discrete values.

    Parameters
    ----------
    matrix : ndarray
        2D array with the input matrix to modify.

    discrete_values : dict or ndarray
        Discrete values for modifying matrix elements.
        If a dict, it should provide discrete values for selected or (all) criteria for all alternatives.
        If an ndarray, it should provide discrete values for each matrix element.

    Returns
    -------
    dict
        Dictionary of modified matrix scenarios.
        The key of the dictionary is presented as 'A[n]-C[m]-S[k]', where:
        - 'n' represents the number of the modified alternative.
        - 'm' represents the index of the modified criterion.
        - 'k' indicates the number of the scenario generated for a specific element in the decision matrix with indexes 'n' and 'm'.

    Notes
    -----
    This function generates multiple scenarios by modifying the elements of the input matrix with discrete values. 
    Each modified scenario is stored in the dictionary, where the keys represent the modification details.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    # Example 1: Generate scenarios with the same discrete values of criteria for each alternative
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> discrete_values = {
    >>>     0: [1.2, 2, 3, 3.4, 4.2],
    >>>     1: [1, 1.8, 2.4, 3.5],
    >>>     2: [2.5, 2.7, 2.9, 2.95, 3.1],
    >>>     3: [1, 2, 3, 4]
    >>> }
    >>> scenarios = discrete_single(matrix, discrete_values)
    >>> print(scenarios)
    {'A[0]-C[0]-S[0]': array([[1.2, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[0]-S[1]': array([[2, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), ...'}

    # Example 2: Generate scenarios with different discrete values for each matrix element
    >>> matrix = np.array([[1, 3, 3], [1, 2, 4]])
    >>> discrete_values = np.array([[[1.1, 2.5], [1.9, 2.3, 2.4], [3.1, 3.2, 3.4]], [[0.9, 1.2, 1.3], [2], [2.95, 3.1, 3.2]]])
    >>> scenarios = discrete_single(matrix, discrete_values)
    >>> print(scenarios)
    {'A[0]-C[0]-S[0]': array([[1.1, 3, 3], [1, 2, 4]]), 'A[0]-C[0]-S[1]': array([[2.5, 3, 3], [1, 2, 4]]), 'A[0]-C[1]-S[0]': array([[1, 1.9, 3], [1, 2, 4]]), ...'}
    """

    # Validator.check_type(matrix, 'matrix', np.ndarray)
    # Validator.check_type(discrete_values, 'discrete_values', dict) dict or np.ndarray
    # Validator.dict_keys(discrete_values, matrix.shape[1], int)
    # Validator.matrix_extension(matrix)
    # if isinstance(discrete_values, dict):
        # Validator.check_dimension(discrete_values, 'discrete_values', matrix.ndim)

    if isinstance(discrete_values, dict):
        thresholds = np.array([[discrete_values[col] if col in list(discrete_values.keys()) else np.array([]) for col in range(matrix.shape[1])]for row in range(matrix.shape[0])])
    else: 
        thresholds = discrete_values
    
    scenarios = {}

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if len(thresholds[i, j]) > 0:
                for scenario_idx, new_value in enumerate(thresholds[i, j]):
                    new_matrix = matrix.copy().astype(float)
                    new_matrix[i,j] = new_value

                    scenarios[f'A[{i}]-C[{j}]-S[{scenario_idx}]'] = new_matrix

    return scenarios

# @memory_guard()
def discrete_multiple(matrix: np.ndarray, discrete_values: (dict | np.ndarray), combinations: (dict | np.ndarray) = None) -> dict:
    """
    Generate multiple scenarios by modifying elements of the given matrix with discrete values.
    Scenarios are generated with modifications for one alternative at a time.

    Parameters
    ----------
    matrix : ndarray
        2D array with the input matrix to modify.

    discrete_values : dict or ndarray
        Discrete values for modifying matrix elements. 
        If a dict, it should provide discrete values for selected (or all) criteria for all alternatives.
        If an ndarray, it should provide discrete values for each matrix element.

    combinations : dict or ndarray, optional, default=None
        Combinations of alternatives and criteria indices for modification. If not provided, all possible combinations of criteria for all alternatives are considered.
        If a dict, it should provide combinations for selected (or all) alternative.
        If an ndarray, the same combinations for criteria indexes are applied to all alternatives.

    Returns
    -------
    dict
        Dictionary of modified matrix scenarios.
        The key of the dictionary is presented as 'A[n]-C[m]-S[k]', where:
        - 'n' represents the number of the modified alternative.
        - 'm' represents the index of the modified criterion.
        - 'k' indicates the number of the scenario generated for a specific element in the decision matrix with indexes 'n' and 'm'.

    Notes
    -----
    This function generates multiple scenarios by modifying the elements of the input matrix with discrete values. 
    Each modified scenario is stored in the dictionary, where the keys represent the modification details.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    # Example 1: Generate scenarios with the individual discrete values of criteria for individual alternative
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> discrete_values = np.array([
    >>>     [[1, 1.2, 1.4], [2, 4], [3.2, 3.5], [3.95, 3.99, 4.06]],
    >>>     [[1, 2, 3], [2, 3], [2.8, 3.1, 3.2], [4.2, 4.4]],
    >>>     [[3.8, 4.1, 4.2], [3, 3.5], [1.9, 2.15, 2.25], [1.4, 1.7, 1.99]],
    >>> ], dtype='object')
    >>> scenarios = discrete_multiple(matrix, discrete_values)
    >>> print(scenarios)
    {'A[0]-C[0-1]-S[0]': array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[0-1]-S[1]': array([[1, 4, 3, 4],[1, 2, 3, 4],[4, 3, 2, 1]]), ...}

    # Example 2: Generate scenarios with selected criteria indexes for all alternatives in the same manner
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> discrete_values = np.array([
    >>>     [[1, 1.2, 1.4], [2, 4], [3.2, 3.5], [3.95, 3.99, 4.06]],
    >>>     [[1, 2, 3], [2, 3], [2.8, 3.1, 3.2], [4.2, 4.4]],
    >>>     [[3.8, 4.1, 4.2], [3, 3.5], [1.9, 2.15, 2.25], [1.4, 1.7, 1.99]],
    >>> ], dtype='object')
    >>> combinations = np.array([[1, 3], [1, 2], [2, 3]])
    >>> scenarios = discrete_multiple(matrix, discrete_values, combinations)
    >>> print(scenarios)
    {'A[0]-C[1-3]-S[0]': array([[1, 2, 3, 3.95], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[1-3]-S[1]': array([[1, 2, 3, 3.99], [1, 2, 3, 4], [4, 3, 2, 1]]) ...}
    
    # Example 3: Generate scenarios with individual combinations of criteria indexes for each alternative separately
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> discrete_values = np.array([
    >>>     [[1, 1.2, 1.4], [2, 4], [3.2, 3.5], [3.95, 3.99, 4.06]],
    >>>     [[1, 2, 3], [2, 3], [2.8, 3.1, 3.2], [4.2, 4.4]],
    >>>     [[3.8, 4.1, 4.2], [3, 3.5], [1.9, 2.15, 2.25], [1.4, 1.7, 1.99]],
    >>> ], dtype='object')
    >>> combinations = {
    >>>     0: [[0, 2], [1, 3]],
    >>>     1: [[1, 2], [0, 3]],
    >>>     2: [[0, 1], [2, 3]],
    >>> }
    >>> scenarios = discrete_multiple(matrix, discrete_values, combinations)
    >>> print(scenarios)
    {'A[0]-C[0-2]-S[0]': array([[1, 2, 3.2, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[0-2]-S[1]': array([[1, 2, 3.5, 4], [1, 2, 3, 4], [4, 3, 2, 1]]) ...}
    
    """
    
    # for one alternative at a time

    criteria_combs = []
    alternatives_set = []

    # generation of possible values combinations for alternatives and criteria
    if combinations is None:
        criteria_set = np.arange(0, matrix.shape[1], 1)
        for i in range(2, matrix.shape[1]):
            c_comb = itertools.combinations(criteria_set, i)
            criteria_combs.append(list(c_comb))

        criteria_combs = np.array([[item for sublist in criteria_combs for item in sublist] for alt in range(matrix.shape[0])], dtype='object')
        alternatives_set = np.arange(0, matrix.shape[0])

    else:
        if isinstance(combinations, np.ndarray):
            criteria_combs = np.array([combinations for alt in range(matrix.shape[0])])
            alternatives_set = np.arange(0, matrix.shape[0])
        else:
            criteria_combs = np.array([combinations[alt] if alt in list(combinations.keys()) else np.array([]) for alt in range(matrix.shape[0])], dtype='object')
            alternatives_set = np.array(list(combinations.keys()))

    if isinstance(discrete_values, dict):
        thresholds = np.array([[discrete_values[col] if col in list(discrete_values.keys()) else np.array([]) for col in range(matrix.shape[1])]for row in range(matrix.shape[0])])
    else: 
        thresholds = discrete_values

    scenarios = {}

    for alt in alternatives_set:
        for criteria in criteria_combs[alt]:
            # generate pairs of possible combinations for given criteria indexes
            values_combs = itertools.product(*[thresholds[alt, c] for c in criteria])
            for scenario_idx, values in enumerate(values_combs):
                new_matrix = matrix.copy().astype(float)
                
                for value_idx, new_value in enumerate(values):
                    
                    new_matrix[alt, criteria[value_idx]] = new_value

                    scenarios[f'A[{alt}]-C[{"-".join(map(str, criteria))}]-S[{scenario_idx}]'] = new_matrix

    return scenarios

if __name__ == '__main__':
    # single

    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ])

    discrete_values = {
        0: [1, 2, 3, 3.4, 4.2],
        1: [1, 1.8, 2.4, 3.5],
        2: [2.5, 2.7, 2.9, 2.95, 3.1],
        3: [1, 2, 3, 4]
    }

    # scenarios = discrete_single(matrix, discrete_values)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])

    # discrete_values = np.array([
    #     [[1, 1.2, 1.4], [2, 4], [3.2, 3.5], [3.95, 3.99, 4.06]],
    #     [[1, 2, 3], [2, 3], [2.8, 3.1, 3.2], [4.2, 4.4]],
    #     [[3.8, 4.1, 4.2], [3, 3.5], [1.9, 2.15, 2.25], [1.4, 1.7, 1.99]],
    # ])

    # scenarios = discrete_single(matrix, discrete_values)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])


    # multiple
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ])
    discrete_values = np.array([
        [[1, 1.2, 1.4], [2, 4], [3.2, 3.5], [3.95, 3.99, 4.06]],
        [[1, 2, 3], [2, 3], [2.8, 3.1, 3.2], [4.2, 4.4]],
        [[3.8, 4.1, 4.2], [3, 3.5], [1.9, 2.15, 2.25], [1.4, 1.7, 1.99]],
    ], dtype='object')

    # np.ndarray - for all alternatives the same combinations
    combinations = np.array([[1, 3], [1, 2], [2, 3]])

    scenarios = discrete_multiple(matrix, discrete_values)
    scenarios = discrete_multiple(matrix, discrete_values, combinations)

    # dict - for each alternative individual combinations
    combinations = {
        0: [[0, 2], [1, 3]],
        1: [[1, 2], [0, 3]],
        2: [[0, 1], [2, 3]],
    }
    scenarios = discrete_multiple(matrix, discrete_values, combinations)
    print(scenarios)


