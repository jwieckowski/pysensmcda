# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def remove_criteria(matrix: np.ndarray, weights: np.ndarray, indexes: None | int | np.ndarray = None) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Remove one or more criteria from a decision matrix and adjust corresponding criteria weights.

    Parameters:
    ------------
    matrix : ndarray
        2D array with decision matrix containing multiple criteria and alternatives.

    weights : ndarray
        1D vector of initial criteria weights.

    indexes : None | int | ndarray, optional, default=None
        Index or array of indexes specifying which criteria to remove. 
        If None, one criterion will be removed by default

    Returns:
    ---------
    List[Tuple[int, ndarray, ndarray]]
        A list of tuples containing information about the removed criteria, new decision matrix,
        and adjusted criteria weights.

    Examples:
    ----------
    Example 1: no indexes given
    
    >>> matrix = np.array([
    ...     [1, 2, 3, 4, 4],
    ...     [1, 2, 3, 4, 4],
    ...     [4, 3, 2, 1, 4]
    ... ])
    >>> weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    >>> results = remove_criteria(matrix, weights)
    >>> for result in results:
    ...     print(result)

    Example 2: int index given
    
    >>> matrix = np.array([
    ...     [1, 2, 3, 4, 4],
    ...     [1, 2, 3, 4, 4],
    ...     [4, 3, 2, 1, 4]
    ... ])
    >>> weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    >>> results = remove_criteria(matrix, weights, 3)
    >>> for result in results:
    ...     print(result)

    Example 3: array indexes given, one-dimensional
    
    >>> matrix = np.array([
    ...     [1, 2, 3, 4, 4],
    ...     [1, 2, 3, 4, 4],
    ...     [4, 3, 2, 1, 4]
    ... ])
    >>> weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    >>> results = remove_criteria(matrix, weights, np.array([1, 2, 3]))
    >>> for result in results:
    ...     print(result)

    Example 4: array indexes given, elements of array as list
    
    >>> matrix = np.array([
    ...     [1, 2, 3, 4, 4],
    ...     [1, 2, 3, 4, 4],
    ...     [4, 3, 2, 1, 4]
    ... ])
    >>> weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    >>> results = remove_criteria(matrix, weights, np.array([[0, 4], 2, 3], dtype='object'))
    >>> for result in results:
    ...     print(result)
    """
    
    Validator.is_type_valid(matrix, np.ndarray, 'matrix')
    Validator.is_type_valid(weights, np.ndarray, 'weights')
    Validator.is_dimension_valid(matrix, 2, 'matrix')
    Validator.is_dimension_valid(weights, 1, 'weights')
    Validator.is_shape_equal(matrix.shape[1], weights.shape[0], custom_message="Number of columns in 'matrix' and length of 'weights' are different")
    if indexes is not None:
        Validator.is_type_valid(indexes, (int, np.integer, np.ndarray), 'indexes')
        Validator.are_indexes_valid(indexes, weights.shape[0])

    crit_indexes = None
    if indexes is None:
        crit_indexes = np.arange(0, matrix.shape[1])
    elif isinstance(indexes, int):
        crit_indexes = np.array([indexes])
    else:
        crit_indexes = indexes

    data = []
    # remove column in decision matrix and adjust criteria weights values
    for c_idx in crit_indexes:
        try:
            new_matrix = np.delete(matrix, c_idx, axis=1)
            # adjust criteria weights
            deleted_weight = weights[c_idx]
            new_weights = np.delete(weights, c_idx)
            if isinstance(c_idx, int):
                new_weights += deleted_weight / new_weights.shape[0]
            elif isinstance(c_idx, list):
                new_weights += np.sum(deleted_weight) / new_weights.shape[0]
            new_weights = new_weights / np.sum(new_weights)

            data.append((c_idx, new_matrix, new_weights))
        except:
            raise ValueError(f'Calculation error. Check elements in {c_idx} index')

    return data
    
