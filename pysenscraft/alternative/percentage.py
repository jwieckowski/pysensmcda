# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

# @memory_guard()
def single_percentage(matrix: np.ndarray, percentages: dict, steps: np.ndarray, direction: str = 'both') -> dict:
    """
    Generate scenarios by modifying elements of the given matrix based on specified percentages values and steps used for generating vectors of percentage changes.

    Parameters
    ----------
    matrix : ndarray
        The 2D array representing the input matrix to be modified.

    percentages : dict
        The percentages for modifying matrix elements. 
        If a dict, it should provide percentage ranges for criteria for selected alternatives.
        If an ndarray, it should provide percentage ranges for each matrix element.

    steps : ndarray
        The step sizes used for the generation of the percentage values between the specified ranges.

    direction : str, optional, default=both
        The direction of modifications. It can take one of the following values:
        - 'increases': Generate scenarios with increased values only.
        - 'decreases': Generate scenarios with decreased values only.
        - 'both': Generate scenarios with both increased and decreased values (default).

    Returns
    -------
    dict
        Dictionary of modified matrix scenarios based on the specified direction.
        If 'direction' is 'both', the dictionary will have two keys: 'increases' and 'decreases'.

    Notes
    -----
    This function generates multiple scenarios by modifying the elements of the input matrix based on specified percentages and steps. 
    The percentage change can either increase or decrease the values based on the 'direction' parameter.

    If 'direction' is set to 'increases', the modifications will increase the values in the matrix by the specified percentage. 
    If 'direction' is set to 'decreases', the modifications will decrease the values by the specified percentage, with a check to ensure that values do not go below zero.
    If 'direction' is set to 'both', scenarios with both increased and decreased values will be generated.

    Examples
    --------
    # Example 1: Generate scenarios with increased and decreased matrix values
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> percentages = {
    >>>     0: [10, 20],
    >>>     1: [30, 40]
    >>> }
    >>> steps = np.array([2, 3])
    >>> direction = 'both'
    >>> scenarios = single_percentage(matrix, percentages, steps, direction)
    >>> print(scenarios['increases'])
    {'A[0]-C[0]-S[0]': array([[1.1, 2], [3, 4]]), 'A[0]-C[0]-S[1]': array([[1.12, 2], [3, 4]]), ...}
    >>> print(scenarios['decreases'])
    {'A[0]-C[0]-S[0]': array([[0.9, 2], [3, 4]]), 'A[0]-C[0]-S[1]': array([[0.88, 2], [3, 4]]), ...}

    # Example 2: Generate scenarios with increased matrix values only
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> percentages = np.array([
    >>>     [10, 20],
    >>>     [30, 40]
    >>> ])
    >>> steps = np.array([2, 3])
    >>> direction = 'increases'
    >>> scenarios = single_percentage(matrix, percentages, steps, direction)
    >>> print(scenarios)
    {'A[0]-C[0]-S[0]': array([[1.1, 2], [3, 4]]), 'A[0]-C[0]-S[1]': array([[1.12, 2], [3, 4]]), ...}

    # Example 3: Generate scenarios with decreased matrix values only
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> percentages = np.array([
    >>>     [10, 20],
    >>>     [30, 40]
    >>> ])
    >>> steps = np.array([2, 3])
    >>> direction = 'decreases'
    >>> scenarios = single_percentage(matrix, percentages, steps, direction)
    >>> print(scenarios)
    {'A[0]-C[0]-S[0]': array([[0.9, 2], [3, 4]]), 'A[0]-C[0]-S[1]': array([[0.88, 2], [3, 4]]), ...}
    """

    def _generate_matrix(matrix: np.ndarray, thresholds: np.ndarray, increase: bool) -> dict:
        """
        Generate scenarios by modifying elements of the given matrix based on specified thresholds.

        Parameters
        ----------
        matrix : ndarray
            The 2D array representing the input matrix to be modified.

        thresholds : ndarray
            The thresholds for modifying matrix elements. 
            This should be a 2D array where each element represents a percentage change to apply to the corresponding element in the input matrix.

        increase : bool
            A flag indicating whether the modifications should increase (True) or decrease (False) the values in the matrix.

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
        This function generates multiple scenarios by modifying the elements of the input matrix based on the specified thresholds. The percentage change can either increase or decrease the values based on the 'increase' parameter.

        If 'increase' is set to True, the modifications will increase the values in the matrix by the specified percentage. If 'increase' is set to False, the modifications will decrease the values by the specified percentage, with a check to ensure that values do not go below zero.

        Examples
        --------
        # Example 1: Generate scenarios by increasing matrix values
        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> thresholds = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> increase = True
        >>> scenarios = _generate_matrix(matrix, thresholds, increase)
        >>> print(scenarios)
        {'A[0]-C[0]-S[0]': array([[1.1, 2. ], [3. , 4. ]]), 'A[0]-C[0]-S[1]': array([[1.2, 2. ], [3. , 4. ]]), ...}

        # Example 2: Generate scenarios by decreasing matrix values
        >>> matrix = np.array([[1, 2], [3, 4]])
        >>> thresholds = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> increase = False
        >>> scenarios = _generate_matrix(matrix, thresholds, increase)
        >>> print(scenarios)
        {'A[0]-C[0]-S[0]': array([[0.9, 2. ], [3. , 4. ]]), 'A[0]-C[0]-S[1]': array([[0.8, 2. ], [3. , 4. ]]), ...}
        """
        matrix_scenarios = {}
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                if len(thresholds[i, j]) != 0:
                    for scenario_idx, percent in enumerate(thresholds[i, j]):
                        new_matrix = matrix.copy().astype(float)

                        if increase: 
                            new_value = matrix[i, j] + matrix[i, j] * percent 
                        else:
                            new_value = matrix[i, j] - matrix[i, j] * percent 
                            if new_value < 0:
                                break

                        new_matrix[i,j] = new_value

                        matrix_scenarios[f'A[{i}]-C[{j}]-S[{scenario_idx}]'] = new_matrix
        return matrix_scenarios


    if isinstance(percentages, dict):
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(percentages[col][0], percentages[col][1], int((percentages[col][1] - percentages[col][0])/steps[col])+1)/100 if col in list(percentages.keys()) else np.array([]) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(percentages[col][0], percentages[col][1], int((percentages[col][1] - percentages[col][0])/steps[row][col])+1)/100 for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
    else:
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(percentages[row][col][0], percentages[row][col][1], int((percentages[row][col][1] - percentages[row][col][0])/steps[col])+1)/100 for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(percentages[row][col][0], percentages[row][col][1], int((percentages[row][col][1] - percentages[row][col][0])/steps[row][col])+1)/100 for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')

    scenarios = {}

    if direction == 'increases' or direction == 'both':
        scenarios['increases'] = _generate_matrix(matrix, thresholds, True)
    if direction == 'decreases' or direction == 'both':
        scenarios['decreases'] = _generate_matrix(matrix, thresholds, False)

    if direction == 'both':
        return scenarios

    return scenarios[direction]

# @memory_guard()
def multiple_percentage(matrix: np.ndarray, percentages: (dict | np.ndarray), steps: np.ndarray, combinations: (dict | np.ndarray) = None, direction: str = 'both') -> dict:
    """
    Generate scenarios by modifying elements of the given matrix based on specified percentage thresholds for one alternative at a time.

    Parameters
    ----------
    matrix : ndarray
        The 2D array representing the input matrix to be modified.

    percentages : dict or ndarray
        If a dict, it represents the percentage range for modifying matrix elements. 
        If an ndarray, it should have the same shape as the 'matrix' parameter, and each element represents a percentage change to apply to the corresponding element in the input matrix.

    steps : ndarray
        An ndarray representing the step size for generating percentage values within the specified range. It should have the same shape as the 'matrix' parameter.

    combinations : dict or ndarray, optional, default=None
        An optional parameter that allows specifying combinations of alternatives and criteria to modify. If None, all possible combinations will be considered.

    direction : str, optional, default='both'
        A flag indicating which modification directions to consider: 'increases', 'decreases', or 'both'. Default is 'both'.

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
    This function generates multiple scenarios by modifying the elements of the input matrix based on the specified percentage thresholds. The percentage change can either increase or decrease the values based on the 'direction' parameter.

    If 'direction' is set to 'increases', the modifications will increase the values in the matrix by the specified percentage. If 'direction' is set to 'decreases', the modifications will decrease the values by the specified percentage, with a check to ensure that values do not go below zero. If 'direction' is set to 'both', both increase and decrease scenarios will be generated.

    Examples
    --------
    # Example 1: Generate scenarios by increasing matrix values with same step for all alternatives
    >>> matrix = np.array([[1, 2, 3], [3, 4, 2]])
    >>> percentages = {
        0: [10, 30],
        1: [20, 40],
    }
    >>> steps = np.array([5, 10])
    >>> scenarios = multiple_percentage(matrix, percentages, steps, direction='increases')
    >>> print(scenarios)
    {'increases': {'A[0]-C[0]-S[0]': array([[1.1, 2.4, 3], [3 , 4, 2]]), 'A[0]-C[0]-S[1]': array([[1.1, 2.6, 3], [3, 4, 2]]), ...}}

    # Example 2: Generate scenarios by decreasing matrix values with same step for all alternatives
    >>> matrix = np.array([[1, 2, 3], [3, 4, 2]])
    >>> percentages = {
        0: [10, 30],
        1: [20, 40],
    }
    >>> steps = np.array([5, 10])
    >>> scenarios = multiple_percentage(matrix, percentages, steps, direction='decreases')
    >>> print(scenarios)
    {'decreases': {'A[0]-C[0]-S[0]': array([[0.9, 1.6, 3], [3, 4, 2]]), 'A[0]-C[0]-S[1]': array([[0.9, 1.4, 2], [3, 4, 2]]), ...}}
    """

    def _generate_matrix(matrix: np.ndarray, alternatives_set: np.ndarray, criteria_combs: np.ndarray, thresholds: np.ndarray, increase: bool) -> dict:
        """
        Generate scenarios by modifying elements of the given matrix based on specified percentage thresholds.

        Parameters
        ----------
        matrix : ndarray
            The 2D array representing the input matrix to be modified.

        alternatives_set : ndarray
            An array containing the indices of alternatives to be modified.

        criteria_combs : ndarray
            An array of arrays representing the combinations of criteria to be modified for each alternative.

        thresholds : ndarray
            The thresholds for modifying matrix elements. 
            This should be a 2D array where each element represents a percentage change to apply to the corresponding element in the input matrix.

        increase : bool
            A flag indicating whether the modifications should increase (True) or decrease (False) the values in the matrix.

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
        This function generates multiple scenarios by modifying the elements of the input matrix based on the specified thresholds. The percentage change can either increase or decrease the values based on the 'increase' parameter.

        If 'increase' is set to True, the modifications will increase the values in the matrix by the specified percentage. If 'increase' is set to False, the modifications will decrease the values by the specified percentage, with a check to ensure that values do not go below zero.

        Examples
        --------
        # Example 1: Generate scenarios by increasing matrix values
        >>> matrix = np.array([[1, 2, 3], [3, 4, 2]])
        >>> alternatives_set = np.array([0, 1])
        >>> criteria_combs = np.array([[0, 1], [0, 2]])
        >>> thresholds = np.array([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]])
        >>> increase = True
        >>> scenarios = _generate_matrix(matrix, alternatives_set, criteria_combs, thresholds, increase)
        >>> print(scenarios)
        {'A[0]-C[0]-S[0]': array([[1.1, 2.4, 3], [3, 4, 2]]), 'A[1]-C[0]-S[1]': array([[1.1, 2.6, 3], [3, 4, 2]]), ...}

        # Example 2: Generate scenarios by decreasing matrix values
        >>> matrix = np.array([[1, 2, 3], [3, 4, 2]])
        >>> alternatives_set = np.array([0, 1])
        >>> criteria_combs = np.array([[0, 1], [0, 1]])
        >>> thresholds = np.array([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]])
        >>> increase = False
        >>> scenarios = _generate_matrix(matrix, alternatives_set, criteria_combs, thresholds, increase)
        >>> print(scenarios)
        {'A[0]-C[0]-S[0]': array([[0.9, 1.8, 3], [3, 4, 2]]), 'A[1]-C[0]-S[1]': array([[0.9, 1.6, 3], [3, 4, 2]]), ...}
        """
        matrix_scenarios = {}

        for alt in alternatives_set:
            for criteria in criteria_combs[alt]:
                # generate pairs of possible combinations for given criteria indexes
                percent_combs = itertools.product(*[thresholds[alt, c] for c in criteria])
                for scenario_idx, percents in enumerate(percent_combs):
                    new_matrix = matrix.copy().astype(float)
                    
                    for value_idx, percent in enumerate(percents):

                        if increase: 
                            new_value = matrix[alt, criteria[value_idx]] + matrix[alt, criteria[value_idx]] * percent 
                        else:
                            new_value = matrix[alt, criteria[value_idx]] - matrix[alt, criteria[value_idx]] * percent 
                            if new_value < 0:
                                break
                        
                        new_matrix[alt, criteria[value_idx]] = new_value

                        matrix_scenarios[f'A[{alt}]-C[{"-".join(map(str, criteria))}]-S[{scenario_idx}]'] = new_matrix

        return matrix_scenarios

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

    if isinstance(percentages, dict):
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(percentages[col][0], percentages[col][1], int((percentages[col][1] - percentages[col][0])/steps[col])+1)/100 if col in list(percentages.keys()) else np.array([]) for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(percentages[col][0], percentages[col][1], int((percentages[col][1] - percentages[col][0])/steps[row][col])+1)/100 for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
    else:
        if steps.ndim == 1:
            thresholds = np.array([[np.linspace(percentages[row][col][0], percentages[row][col][1], int((percentages[row][col][1] - percentages[row][col][0])/steps[col])+1)/100 for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')
        else: 
            thresholds = np.array([[np.linspace(percentages[row][col][0], percentages[row][col][1], int((percentages[row][col][1] - percentages[row][col][0])/steps[row][col])+1)/100 for col in range(matrix.shape[1])] for row in range(matrix.shape[0])], dtype='object')

    scenarios = {}

    if direction == 'increases' or direction == 'both':
        scenarios['increases'] = _generate_matrix(matrix, alternatives_set, criteria_combs, thresholds, True)
    if direction == 'decreases' or direction == 'both':
        scenarios['decreases'] = _generate_matrix(matrix, alternatives_set, criteria_combs, thresholds, False)

    if direction == 'both':
        return scenarios

    return scenarios[direction]


if __name__ == '__main__':
    # matrix = np.array([
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [4, 3, 2, 1]
    # ])

    # percentages = {
    #     0: [1, 5],
    #     1: [2, 7], 
    #     2: [5, 10],
    # }
    # steps = np.array([1, 1, 1])
    # direction = 'both'

    # scenarios = single_percentage(matrix, percentages, steps)
    # keys = list(scenarios['increases'].keys())
    # values = list(scenarios['increases'].values())
    # for i in range(0, 10):
    #     print(keys[i], values[i])
    
    
    # multiple
    # matrix = np.array([
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [4, 3, 2, 1]
    # ])

    # percentages = {
    #     0: [1, 5],
    #     1: [2, 7], 
    #     2: [5, 10],
    # }
    # steps = np.array([1, 1, 1])
    # direction = 'both'
    # scenarios = multiple_percentage(matrix, percentages, steps)
    # keys = list(scenarios['increases'].keys())
    # values = list(scenarios['increases'].values())
    # for i in range(0, 10):
    #     print(keys[i], values[i])

    matrix = np.array([[1, 2, 3], [3, 4, 2]])
    percentages = {
        0: [10, 30],
        1: [20, 40],
    }
    steps = np.array([5, 10])
    scenarios = multiple_percentage(matrix, percentages, steps, direction='both')
    print(scenarios)



