# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from itertools import product
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def discrete_modification(matrix: np.ndarray, discrete_values: np.ndarray, indexes: None | np.ndarray = None) -> list[tuple[int, int | tuple, tuple, np.ndarray]]:
    """
    Modify a decision matrix based on specified discrete values and indexes combinations representing the columns modified at the time.

    Parameters
    ----------
    matrix : ndarray
        2D array representing the initial decision matrix.

    discrete_values : ndarray
        Discrete values for each value in the decision matrix specifying the allowed changes.
        If 2D array, each element represents the discrete values that will be put in each column in the matrix.
        If 3D array, each element represents the discrete values that will be put for each value in the decision matrix separately.

    indexes : None | ndarray, optional, default=None
        Indexes of the columns from matrix to be modified. If None, all columns are considered subsequently.
        If ndarray, it specifies the indexes or combinations of indexes for the columns to be modified.

    Returns
    -------
    List[Tuple[int, int | tuple, tuple, ndarray]]
        A list of tuples containing information about the modified alternative index, criteria index, discrete change,
        and the resulting decision matrix.

    ## Examples
    --------
    ### Example 1: Modify decision matrix with discrete values
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> discrete_values = np.array([[2, 3, 4], [1, 5, 6], [3, 4]], dtype='object)
    >>> results = discrete_modification(matrix, discrete_values)
    >>> for r in results:
    ...     print(r)

    ### Example 2: Modify matrix with discrete values list and specified indexes
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> discrete_values = np.array([[2, 3, 4], [1, 5, 6], [3, 4]], dtype='object')
    >>> indexes = np.array([[0, 2], 1], dtype='object')
    >>> results = discrete_modification(matrix, discrete_values, indexes)
    >>> for r in results:
    ...     print(r)

    ### Example 3: Modify matrix with 3D discrete values array
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> discrete_values = np.array([
    ...     [[5, 6], [2, 4], [5, 8]],
    ...     [[3, 5.5], [4], [3.5, 4.5]],
    ...     [[7, 8], [6], [8, 9]],
    ... ], dtype='object')
    >>> results = discrete_modification(matrix, discrete_values)
    >>> for r in results:
    ...     print(r)

    ### Example 4: Modify matrix with 3D discrete values array and specified indexes
    >>> matrix = np.array([
    ... [4, 1, 6],
    ... [2, 6, 3],
    ... [9, 5, 7],
    ... ])
    >>> discrete_values = np.array([
    ...     [[5, 6], [2, 4], [5, 8]],
    ...     [[3, 5.5], [4], [3.5, 4.5]],
    ...     [[7, 8], [6], [8, 9]],
    ... ], dtype='object')
    >>> indexes = np.array([[0, 2], 1], dtype='object')
    >>> results = discrete_modification(matrix, discrete_values, indexes)
    >>> for r in results:
    ...     print(r)
    """

    def modify_matrix(matrix: np.ndarray, alt_idx: int, crit_idx: int, change: float) -> np.ndarray:
        new_matrix = matrix.copy().astype(float)

        new_matrix[alt_idx, crit_idx] = change

        return new_matrix

    Validator.is_type_valid(matrix, np.ndarray)
    Validator.is_dimension_valid(matrix, 2)
    Validator.is_type_valid(discrete_values, np.ndarray)

    dv_dim = 0
    # check if matrix and discrete values have the same length
    if discrete_values.dtype == 'object':
        Validator.is_array_2D_3D(discrete_values, matrix)
    else:
        if discrete_values.ndim == 2:
            Validator.is_shape_equal(matrix.shape[1], discrete_values.shape[0], custom_message="Number of columns in 'matrix' and length of 'discrete_values' are different")
        elif discrete_values.ndim == 3:
            Validator.is_shape_equal(matrix.shape, discrete_values.shape, custom_message="Shapes of 'matrix' and 'discrete_values' are different")
        dv_dim = discrete_values.ndim

    Validator.is_type_valid(indexes, (None, np.ndarray))
    if indexes is not None:
        Validator.are_indexes_valid(indexes, matrix.shape[1])

    results = []
    
    # criteria indexes to modify matrix values
    indexes_values = None
    if indexes is None:
        indexes_values = np.arange(0, matrix.shape[1], dtype=int)
    else:
        indexes_values = indexes

    alt_indexes = np.arange(0, matrix.shape[0], dtype=int)

    for alt_idx in alt_indexes:
        for crit_idx in indexes_values:
            
            if dv_dim == 2:
                if isinstance(crit_idx, (int, np.integer)):
                    changes = discrete_values[crit_idx]
                else:
                    changes = list(product(*discrete_values[crit_idx]))
            elif dv_dim == 3:
                if isinstance(crit_idx, (int, np.integer)):
                    changes = discrete_values[alt_idx][crit_idx]
                else:
                    changes = list(product(*discrete_values[alt_idx][crit_idx]))
                
            for change in changes:
                change_val = np.round(change, 6) if isinstance(change, (int, np.integer, float)) else tuple(np.round(change, 6).tolist())

                new_matrix = modify_matrix(matrix, alt_idx, crit_idx, change)

                criteria_idx = crit_idx if isinstance(crit_idx, (int, np.integer)) else tuple(crit_idx)
                results.append((alt_idx, criteria_idx, change_val, new_matrix))

    return results
