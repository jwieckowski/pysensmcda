# Copyright (C) 2023 - 2024  Jakub Więckowski

import numpy as np
from itertools import product

def percentage_modification(matrix: np.ndarray, percentages: int | np.ndarray, direction: None | np.ndarray = None, indexes: None | np.ndarray = None, step: int | np.ndarray = 1) -> list[tuple[int, int | tuple, tuple, np.ndarray]]:
    """
    Modify a decision matrix based on specified percentage changes, directions, indexes, and steps of percentage modifications.

    Parameters
    ----------
    matrix : ndarray
        2D array representing the initial decision matrix.

    percentages : int | ndarray
        Percentage changes to be applied to the values in decision matrix. 
        If int, the same percentage change is applied to all values.
        If ndarray, it specifies the percentage change for each column from matrix individually.

    direction : None | ndarray, optional, default=None
        Direction of the modification for each column in the matrix. If None, both increase and decrease directions are considered.
        If ndarray, it specifies the direction (1 for increase, -1 for decrease) for each criterion individually.

    indexes : None | ndarray, optional, default=None
        Indexes of the columns from matrix to be modified. If None, all columns are considered subsequently.
        If ndarray, it specifies the indexes or combinations of indexes for the columns to be modified.

    step : int | np.ndarray, optional, default=1
        Step size for the percentage change. If int, all changes for columns are made with the same step.
        If ndarray, the modification step is adjusted for each column separately.

    Returns
    -------
    List[Tuple[int, int | tuple, tuple, ndarray]]
        A list of tuples containing information about the modified alternative index, criteria index, percentage change,
        and the resulting decision matrix.

    ## Examples
    --------
    ### Example 1: Modify decision matrix with a single percentage change
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> percentages = 5
    >>> results = percentage_modification(matrix, percentages)
    >>> for r in results:
    ...     print(r)

    # Example 2: Modify matrix with percentages list
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> percentages = np.array([3, 5, 8])
    >>> results = percentage_modification(matrix, percentages)
    >>> for r in results:
    ...     print(r)

    ### Example 3: Modify matrix with percentages list and specific direction for each column
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> percentages = np.array([2, 4, 6])
    >>> direction = np.array([-1, 1, -1])
    >>> results = percentage_modification(matrix, percentages, direction)
    >>> for r in results:
    ...     print(r)

    ### Example 4: Modify matrix with percentages list, and specific column indexes
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> percentages = np.array([1, 4, 2])
    >>> indexes = np.array([[0, 2], 1], dtype='object')
    >>> results = percentage_modification(matrix, percentages, indexes)
    >>> for r in results:
    ...     print(r)

    ### Example 5: Modify matrix with percentages list, and specific modification step
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> percentages = np.array([2, 4, 9])
    >>> step = np.array([2, 2, 3])
    >>> results = percentage_modification(matrix, percentages, step=step)
    >>> for r in results:
    ...     print(r)
    """

    def modify_matrix(matrix: np.ndarray, alt_idx: int, crit_idx: int, diff: float, direction_val: int) -> np.ndarray:
        new_matrix = matrix.copy().astype(float)

        new_matrix[alt_idx, crit_idx] = matrix[alt_idx, crit_idx] + diff * direction_val

        return new_matrix

    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix should be given as numpy array')

    if matrix.ndim != 2:
        raise ValueError('Matrix should be given as 2D array')
    
    if isinstance(percentages, np.ndarray):
        # check if matrix and percentages have the same length
        if matrix.shape[1] != percentages.shape[0]:
            raise ValueError('Matrix columns and percentages have different length')

    if direction is not None:
        # check if matrix and direction have the same length
        if matrix.shape[1] != direction.shape[0]:
            raise ValueError('Matrix columns and direction have different length')

    if isinstance(step, np.ndarray):
        # check if matrix and step have the same length
        if matrix.shape[1] != step.shape[0]:
            raise ValueError('Matrix columns and step have different length')

    if indexes is not None:
        for c_idx in indexes:
            if isinstance(c_idx, (int, np.integer)):
                if c_idx < 0 or c_idx >= matrix.shape[1]:
                    raise IndexError(f'Given index ({c_idx}) out of range')
            elif isinstance(c_idx, list):
                if any([idx < 0 or idx >= matrix.shape[1] for idx in c_idx]):
                    raise IndexError(f'Given indexes ({c_idx}) out of range')

    results = []

    # size of changes of matrix values
    percentages_values = None
    if isinstance(percentages, (int, np.integer)):
        percentages_values = np.array([percentages] * matrix.shape[1])
    if isinstance(percentages, np.ndarray):
        percentages_values = percentages

    # vectors with subsequent changes for criteria in matrix
    if isinstance(step, int):
        percentages_changes = np.array([np.arange(step, p+step, step) / 100 for p in percentages_values], dtype='object')
    else:
        percentages_changes = np.array([np.arange(step[idx], p+step[idx], step[idx]) / 100 for idx, p in enumerate(percentages_values)], dtype='object')

    # increasing or decreasing matrix values
    direction_values = None
    if direction is None:
        direction_values = np.array([[-1, 1]] * matrix.shape[1])
    else:
        direction_values = np.array([[val] for val in direction])

    # criteria indexes to modify matrix values
    indexes_values = None
    if indexes is None:
        indexes_values = np.arange(0, matrix.shape[1], dtype=int)
    else:
        indexes_values = indexes

    alt_indexes = np.arange(0, matrix.shape[0], dtype=int)

    for alt_idx in alt_indexes:
        for crit_idx in indexes_values:
            if isinstance(crit_idx, (int, np.integer)):
                changes = percentages_changes[crit_idx]
            else:
                changes = list(product(*percentages_changes[crit_idx]))
            
            for change in changes:
                diff = matrix[alt_idx, crit_idx] * change
                
                change_direction = direction_values[crit_idx]
                if not isinstance(crit_idx, (int, np.integer)):
                    change_direction = [direction_values[crit_idx][0]]
                
                for val in change_direction:
                    if isinstance(val, (int, np.integer)):
                        new_matrix = modify_matrix(matrix, alt_idx, crit_idx, diff, val)
                        results.append((alt_idx, crit_idx, change * val, new_matrix))
                    else:
                        for v in val:
                            change_val = tuple(c * v for c in change)
                            new_matrix = modify_matrix(matrix, alt_idx, crit_idx, diff, v)
                            results.append((alt_idx, tuple(crit_idx), change_val, new_matrix))
    
    return results
