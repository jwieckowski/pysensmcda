# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from itertools import product

def discrete_modification(matrix: np.ndarray, discrete_values: np.ndarray, indexes: None | np.ndarray = None):
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

    def modify_matrix(matrix, alt_idx, crit_idx, change):
        new_matrix = matrix.copy().astype(float)

        new_matrix[alt_idx, crit_idx] = change

        return new_matrix

    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix should be given as numpy array')

    if matrix.ndim != 2:
        raise ValueError('Matrix should be given as 2D array')

    if not isinstance(discrete_values, np.ndarray):
        raise TypeError('Discrete values should be given as numpy array')
    
    dv_dim = 0
    # check if matrix and discrete values have the same length
    if discrete_values.dtype == 'object':
        try:
            # 2D
            if isinstance(discrete_values[0][0], (int, np.integer, float)):
                shapes = tuple(len(dv) for dv in discrete_values)
                if matrix.shape[1] != len(shapes):
                    raise ValueError('Matrix and discrete values have different length')
                dv_dim = 2
            # 3D
            else:
                dv_shape = [len(tuple(len(vals) for vals in dv)) for dv in discrete_values]
                if len(np.unique(dv_shape)) != 1 or matrix.shape[0] != dv_shape[0] and matrix.shape[1] != dv_shape[1]:
                    raise ValueError('Matrix and discrete values have different length')
                dv_dim = 3
        except TypeError: 
            raise TypeError('Discrete values should be given as 2D or 3D array')
    else:
        if discrete_values.ndim == 2:
            if matrix.shape[1] != discrete_values.shape[0]:
                raise ValueError('Matrix and discrete values have different length')
        elif discrete_values.ndim == 3:
            if matrix.shape[0] != discrete_values.shape[0] and matrix.shape[1] != discrete_values.shape[1]:
                raise ValueError('Matrix and discrete values have different length')
        dv_dim = discrete_values.ndim

    if indexes is not None:
        for c_idx in indexes:
            if isinstance(c_idx, (int, np.integer)):
                if c_idx < 0 or c_idx >= matrix.shape[1]:
                    raise IndexError(f'Given index ({c_idx}) out of range')
            elif isinstance(c_idx, (list, np.ndarray)):
                if any([idx < 0 or idx >= matrix.shape[1] for idx in c_idx]):
                    raise IndexError(f'Given indexes ({c_idx}) out of range')

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
