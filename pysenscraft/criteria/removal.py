# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def remove_criteria(matrix: np.ndarray, weights: np.ndarray, indexes: None | int | np.ndarray = None):
    """
    TODO

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
        TODO

    ## Examples
    --------
    ### Example 1: Generate scenarios with custom combinations
    TODO
    ### Example 2: Generate scenarios with default combinations
    TODO
    """
    
    matrix = np.array(matrix)
    weights = np.array(weights)

    # check if matrix and weights the same length
    if matrix.shape[1] != weights.shape[0]:
        raise ValueError('Matrix and weights have different length')

    # weights dimension
    if weights.ndim != 1:
        raise ValueError('Weights should be given as one dimensional vector')

    # matrix dimension
    if matrix.ndim != 2:
        raise ValueError('Matrix should be given as at two dimensional vector')

    crit_indexes = None

    if indexes is None:
        # generate vector of subsequent criteria indexes to remove
        crit_indexes = np.arange(0, matrix.shape[1])
    
    if isinstance(indexes, int):
        if indexes >= weights.shape[0] or indexes < 0:
            raise IndexError(f'Given index ({indexes}) out of range')
        crit_indexes = np.array([indexes])

    if isinstance(indexes, np.ndarray):
        for c_idx in indexes:
            if isinstance(c_idx, int):
                if c_idx < 0 or c_idx >= weights.shape[0]:
                    raise IndexError(f'Given index ({indexes}) out of range')
            elif isinstance(c_idx, list):
                if any([idx < 0 or idx >= weights.shape[0] for idx in c_idx]):
                    raise IndexError(f'Given indexes ({c_idx}) out of range')

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

            data.append((c_idx, new_matrix, new_weights))
        except:
            raise ValueError(f'Calculation error. Check elements in {c_idx} index')

    return data
    

if __name__ == '__main__':
    # Example 1, no indexes given
    # by default one criterion removed from data subsequently 
    print('Example 1 ---------------------')
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights)
    for result in results:
        print(result)

    # Example 2 int index given
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    print('Example 2 ---------------------')
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights, 3)
    for result in results:
        print(result)

    # Example 3 np array indexes given, one-dimensional
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    print('Example 3 ---------------------')
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights, np.array([1, 2, 3]))
    for result in results:
        print(result)

    # Example 4 np array indexes given, elements of array as list
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    print('Example 4 ---------------------')
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights, np.array([[0, 5], 2, 3], dtype='object'))
    for result in results:
        print(result)

