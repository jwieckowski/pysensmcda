# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

# @memory_guard()
def range_single(matrix: np.ndarray, ranges: (dict | np.ndarray), steps: np.ndarray) -> dict:
    """
    Generate scenarios by modifying elements of the given matrix within specified value ranges and steps.

    Parameters
    ----------
    matrix : ndarray
        2D array with the input matrix to modify.

    ranges : dict or ndarray
        Value ranges for modifying matrix elements. 
        If a dict, it should provide ranges for selected columns. Key should represent the index of criterion in the matrix (index of column), and the value should be given as list with range to modify the values in the decision matrix for given criterion and each alternative.
        If an ndarray, it should provide ranges for each decision matrix element.

    steps : ndarray
        Steps for the modification of matrix elements corresponding to each range.
        If 1D array, all ranges vectors for criteria will be generated with the same step within given criterion for each alternative.
        If 2D array, each alternative will have its own step in the process of ranges vectors generation.

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
    This function generates multiple scenarios by modifying the elements of the input matrix based on the specified value ranges and steps. Each modified scenario is stored in the dictionary, where the keys represent the modification details.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    # Example 1: Generate scenarios with same ranges of values of criteria for each alternative
    >>> matrix = np.array([
                        [1, 2, 3, 4], 
                        [1, 2, 3, 4], 
                        [4, 3, 2, 1]
                    ])
    >>> ranges = {0: [1, 6], 1: [1.5, 5], 2: [2, 4], 3: [1, 6]}
    >>> steps = np.array([0.4, 0.5, 0.2, 1])
    >>> scenarios = range_single(matrix, ranges, steps)
    >>> print(scenarios)
    {'A[0]-C[0]-S[0]': array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[0]-S[1]': array([[1.41, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), ...}

    # Example 2: Generate scenarios with individual ranges of values of criteria for each alternative with individual steps
    >>> matrix = np.array([
                        [1, 2, 3], 
                        [1, 2, 3], 
                        [4, 3, 2]
                    ])
    >>> ranges = np.array([
                            [[1, 4], [2, 5], [3, 4]],
                            [[1, 4], [2, 3.5], [2.5, 3.3]],
                            [[2, 4], [2, 3], [2, 4]]
                    ])
    >>> steps = np.array([[0.2, 0.5, 0.4], 
                        [0.4, 0.3, 0.1], 
                        [0.1, 0.8, 0.2]
                    ])
    >>> scenarios = range_single(matrix, ranges, steps)
    >>> print(scenarios)
    {'A[0]-C[0]-S[0]': array([[1, 2, 3], [1, 2, 3], [4, 3, 2]]), 'A[0]-C[0]-S[1]': array([[1.2, 2, 3], [1, 2, 3], [4, 3, 2]]), ...}
    """

    # Validator.check_type(matrix, 'matrix', np.ndarray)
    # Validator.check_type(ranges, 'ranges', dict)
    # Validator.dict_keys(ranges, weights.shape[0])
    # Validator.ranges_bound(ranges)
    # Validator.check_type(steps, 'steps', np.ndarray)
    # Validator.check_length(steps, 'steps', len(ranges.keys()))

    # Validator.matrix_extension(matrix)


    # generation of vector with subsequent values of elements in decision matrix
    if isinstance(ranges, dict):
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(ranges[col][0], ranges[col][1], int((ranges[col][1] - ranges[col][0])/steps[col])+1) if col in list(ranges.keys()) else np.array([]) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(ranges[col][0], ranges[col][1], int((ranges[col][1] - ranges[col][0])/steps[row][col])+1) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
    else:
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(ranges[row][col][0], ranges[row][col][1], int((ranges[row][col][1] - ranges[row][col][0])/steps[col])+1) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(ranges[row][col][0], ranges[row][col][1], int((ranges[row][col][1] - ranges[row][col][0])/steps[row][col])+1) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')

    scenarios = {}

    # for each element in matrix modify the matrix values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            if len(thresholds[i, j]) != 0:
                for scenario_idx, new_value in enumerate(thresholds[i, j]):
                    new_matrix = matrix.copy().astype(float)

                    new_matrix[i,j] = new_value

                    scenarios[f'A[{i}]-C[{j}]-S[{scenario_idx}]'] = new_matrix

    return scenarios


# @memory_guard()
def range_multiple(matrix: np.ndarray, ranges: (dict | np.ndarray), steps: np.ndarray, combinations: (dict | np.ndarray) = None) -> dict:
    """
    Generate scenarios by modifying elements of the given matrix within specified value ranges.

    Parameters
    ----------
    matrix : ndarray
        The 2D array representing the input matrix to be modified.

    ranges : dict or ndarray
        The value ranges for modifying matrix elements.
        If a dict, it should provide value ranges for criteria for all alternatives.
        If an ndarray, it should provide value ranges for each matrix element.

    steps : ndarray
        The step sizes used for the generation fo the vectors between the value ranges.

    combinations : dict or ndarray, optional
        Combinations of criteria to be modified for each alternative. If not provided, all combinations are considered.

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
    This function generates multiple scenarios by modifying the elements of the input matrix within specified value ranges.
    Each modified scenario is stored in the dictionary, where the keys represent the modification details.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    # Example 1: Generate scenarios with the same value ranges for criteria for each alternative
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> ranges = {
    >>>     0: [1, 6],
    >>>     1: [1.5, 5],
    >>>     2: [2, 4],
    >>>     3: [1, 6]
    >>> }
    >>> steps = np.array([0.4, 0.5, 0.2, 1])
    >>> scenarios = range_multiple(matrix, ranges, steps)
    >>> print(scenarios)
    {'A[0]-C[0-1]-S[0]': array([[1, 1.5, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[0-1]-S[1]': array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), ...}

    # Example 2: Generate scenarios with different value ranges for criteria for each alternative with individual steps for each element in decision matrix
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> ranges = np.array([
    >>>     [[1, 4], [2, 5], [3, 4], [2, 5]],
    >>>     [[1, 4], [2, 3.5], [2.5, 3.3], [4, 6]],
    >>>     [[2, 4], [2, 3], [2, 4], [1, 1.5]],
    >>> ])
    >>> steps = np.array([
    >>>     [0.2, 0.5, 0.2, 1],
    >>>     [0.4, 0.5, 0.2, 1],
    >>>     [0.4, 0.5, 0.2, 0.1],
    >>> ])
    >>> # Generate scenarios with all combinations for each alternative
    >>> scenarios = range_multiple(matrix, ranges, steps)
    >>> print(scenarios)
    {'A[0]-C[0-1]-S[0]': array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[0-1]-S[1]': array([[1, 2.5, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), ...}

    # Example 3: Generate scenarios with the individual combinations for selected alternatives
    >>> matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    >>> ranges = np.array([
    >>>     [[1, 4], [2, 5], [3, 4], [2, 5]],
    >>>     [[1, 4], [2, 3.5], [2.5, 3.3], [4, 6]],
    >>>     [[2, 4], [2, 3], [2, 4], [1, 1.5]],
    >>> ])
    >>> steps = np.array([
    >>>     [0.2, 0.5, 0.2, 1],
    >>>     [0.4, 0.5, 0.2, 1],
    >>>     [0.4, 0.5, 0.2, 0.1],
    >>> ])
    >>> combinations = {
    >>>     0: [[0, 1], [1, 3]],
    >>>     1: [[1, 2], [0, 3]],
    >>>     2: [[0, 2], [2, 3]],
    >>> }
    >>> scenarios = range_multiple(matrix, ranges, steps, combinations)
    >>> print(scenarios)
    { 'A[0]-C[1-2]-S[0]': array([[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), 'A[0]-C[1-2]-S[1]': array([[1, 2, 3.2, 4], [1, 2, 3, 4], [4, 3, 2, 1]]), ...}
    
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

    if isinstance(ranges, dict):
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(ranges[col][0], ranges[col][1], int((ranges[col][1] - ranges[col][0])/steps[col])+1) if col in list(ranges.keys()) else np.array([]) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(ranges[col][0], ranges[col][1], int((ranges[col][1] - ranges[col][0])/steps[row][col])+1) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
    else:
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(ranges[row][col][0], ranges[row][col][1], int((ranges[row][col][1] - ranges[row][col][0])/steps[col])+1) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(ranges[row][col][0], ranges[row][col][1], int((ranges[row][col][1] - ranges[row][col][0])/steps[row][col])+1) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')

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
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ])
    ranges = {
        0: [1, 6],
        1: [1.5, 5],
        2: [2, 4],
        3: [1, 6]
    }
    steps = np.array([0.4, 0.5, 0.2, 1])

    # scenarios = range_single(matrix, ranges, steps)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])

    ranges2 = np.array([
        [[1, 4], [2, 5], [3, 4], [2, 5]],
        [[1, 4], [2, 3.5], [2.5, 3.3], [4, 6]],
        [[2, 4], [2, 3], [2, 4], [1, 1.5]],
    ])
    # scenarios = range_single(matrix, ranges2, steps)

    steps2 = np.array([
        [0.2, 0.5, 0.2, 1],
        [0.4, 0.5, 0.2, 1],
        [0.4, 0.5, 0.2, 0.1],
    ])
    # scenarios = range_single(matrix, ranges2, steps2)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])


    # scenarios = range_single(matrix, ranges, steps2)

    # multiple
    # v1
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ])
    ranges = {
        0: [1, 6],
        1: [1.5, 5],
        2: [2, 4],
        3: [1, 6]
    }
    # steps = np.array([0.4, 0.5, 0.2, 1])
    # scenarios = range_multiple(matrix, ranges, steps)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])

    # v2
    ranges = np.array([
        [[1, 4], [2, 5], [3, 4], [2, 5]],
        [[1, 4], [2, 3.5], [2.5, 3.3], [4, 6]],
        [[2, 4], [2, 3], [2, 4], [1, 1.5]],
    ])

    # scenarios = range_multiple(matrix, ranges, steps)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])

    # v3
    steps = np.array([
        [0.2, 0.5, 0.2, 1],
        [0.4, 0.5, 0.2, 1],
        [0.4, 0.5, 0.2, 0.1],
    ])
    # scenarios = range_multiple(matrix, ranges, steps)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])

    # v4
    combinations = np.array([[1, 3], [0, 1, 2]], dtype='object')
    # scenarios = range_multiple(matrix, ranges, steps, combinations)
    # keys = list(scenarios.keys())
    # values = list(scenarios.values())
    # for i in range(10):
    #     print(keys[i], values[i])
   
    # v5
    combinations = {
        0: [[1, 2], [1, 3]],
        1: [[1, 2], [0, 3]],
        2: [[0, 1], [2, 3]],
    }
    scenarios = range_multiple(matrix, ranges, steps, combinations)
    keys = list(scenarios.keys())
    values = list(scenarios.values())
    for i in range(10):
        print(keys[i], values[i])

