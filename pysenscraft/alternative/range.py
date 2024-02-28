# Copyright (C) 2023 - 2024 Jakub Więckowski

import numpy as np
from itertools import product
from ..validator import Validator

def range_modification(matrix: np.ndarray, range_values: np.ndarray, indexes: None | np.ndarray = None, step: int | float | np.ndarray = 1):
    """
    Modify a decision matrix based on specified range values, indexes representing the combination of columns to be modified, and steps of range modifications.

    Parameters
    ----------
    matrix : ndarray
        2D array representing the initial decision matrix.

    range_values : ndarray
        Range of values for each value in the decision matrix specifying the allowed changes.
        If 2D array, each element represents the lower and upper bounds of the allowed range for each column in the matrix.
        If 3D array, each element represents the lower and upper bounds of the allowed range for each value in the decision matrix separately.

    indexes : None | ndarray, optional, default=None
        Indexes of the columns from the matrix to be modified. If None, all columns are considered subsequently.
        If ndarray, it specifies the indexes or combinations of indexes for the columns to be modified.

    step : int | float | np.ndarray, optional, default=1
        Step size for the change in given range. If int, all changes for columns are made with the same step.
        If ndarray, the modification step is adjusted for each column separately.

    Returns
    -------
    List[Tuple[int, int | tuple, tuple, ndarray]]
        A list of tuples containing information about the modified alternative index, criteria index, range change,
        and the resulting decision matrix.

    ## Examples
    --------
    ### Example 1: Modify the decision matrix with a 2D range change, same range for row in matrix for a given column
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    >>> results = range_modification(matrix, range_values)
    >>> for r in results:
    ...     print(r)

    ### Example 2: Modify the decision matrix with a 3D range change array, given range for every element in matrix
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>>  range_values = np.array([
    ...  [[3, 5], [1, 3], [4, 6]],
    ...  [[2, 5], [5, 7], [2, 4]],
    ...  [[8, 11], [4, 7], [7, 9]],
    ... ])
    >>> results = range_modification(matrix, range_values)
    >>> for r in results:
    ...     print(r)

    ### Example 3: Modify the decision matrix with specific column indexes
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    >>> indexes = np.array([[0, 2], 1], dtype='object')
    >>> results = range_modification(matrix, range_values, indexes)
    >>> for r in results:
    ...     print(r)

    ### Example 4: Modify the decision matrix with a specified step size
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    >>> step = 0.5
    >>> results = range_modification(matrix, range_values, step=step)
    >>> for r in results:
    ...     print(r)

    ### Example 5: Modify the decision matrix with individual step sizes for each column
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    >>> step = np.array([0.25, 0.4, 0.5])
    >>> results = range_modification(matrix, range_values, step=step)
    >>> for r in results:
    ...     print(r)
    """

    def modify_matrix(matrix, alt_idx, crit_idx, change):
        new_matrix = matrix.copy().astype(float)

        new_matrix[alt_idx, crit_idx] = change

        return new_matrix

    Validator.is_type_valid(matrix, np.ndarray)
    Validator.is_dimension_valid(matrix, 2)
    Validator.is_type_valid(range_values, np.ndarray)
    Validator.is_type_valid(step, (int, float, np.ndarray))

    if range_values.ndim == 2:
        Validator.is_shape_equal(matrix.shape[1], range_values.shape[0], custom_message="Number of columns in 'matrix' and length of 'range_values' are different")
    elif range_values.ndim == 3:
        Validator.is_shape_equal(matrix.shape, range_values.shape, custom_message="Shapes 'matrix' and 'range_values' are different")

    Validator.is_type_valid(indexes, (None, np.ndarray))
    if indexes is not None:
        Validator.are_indexes_valid(indexes, matrix.shape[1])

    if isinstance(step, np.ndarray):
        # check if matrix and step have the same length
        Validator.is_shape_equal(matrix.shape[1], step.shape[0], custom_message="Number of columns in 'matrix' and length of 'step' are different")

    results = []

    # criteria indexes to modify matrix values
    indexes_values = None
    if indexes is None:
        indexes_values = np.arange(0, matrix.shape[1], dtype=int)
    else:
        indexes_values = indexes

    if isinstance(step, (int, float)):
        change_steps = np.array([step] * matrix.shape[1])
    else:
        change_steps = step

    # generation of vector with subsequent values of weights for criteria
    if range_values.ndim == 2:
        range_changes = np.array([np.arange(range_values[i][0], range_values[i][1]+change_steps[i], change_steps[i]) for i in range(matrix.shape[1])], dtype='object')
        range_changes = np.array([[val for val in rc if val >= range_values[idx][0] and val <= range_values[idx][1]] for idx, rc in enumerate(range_changes)], dtype='object')
    elif range_values.ndim == 3:
        range_changes = np.array([[np.arange(range_values[i][j][0], range_values[i][j][1]+change_steps[j], change_steps[j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])], dtype='object')
        range_changes = np.array([[[val for val in range_changes[i][j] if val >= range_values[i][j][0] and val <= range_values[i][j][1]] for j in range(matrix.shape[1])] for i in range(matrix.shape[0])], dtype='object')

    alt_indexes = np.arange(0, matrix.shape[0], dtype=int)

    for alt_idx in alt_indexes:
        for crit_idx in indexes_values:

            if range_values.ndim == 2:
                if isinstance(crit_idx, (int, np.integer)):
                    changes = range_changes[crit_idx]
                else:
                    changes = list(product(*range_changes[crit_idx]))
            elif range_values.ndim == 3:
                if isinstance(crit_idx, (int, np.integer)):
                    changes = range_changes[alt_idx][crit_idx]
                else:
                    changes = list(product(*range_changes[alt_idx][crit_idx]))

            for change in changes:
                change_val = np.round(change, 6) if isinstance(change, (int, np.integer, float)) else tuple(np.round(change, 6).tolist())

                new_matrix = modify_matrix(matrix, alt_idx, crit_idx, change)

                criteria_idx = crit_idx if isinstance(crit_idx, (int, np.integer)) else tuple(crit_idx)
                results.append((alt_idx, criteria_idx, change_val, new_matrix))

    return results
