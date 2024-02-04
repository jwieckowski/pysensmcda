# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

# @memory_guard()
def remove_single(matrix: np.ndarray, combinations: np.ndarray = None) -> dict:
    """
    Generate scenarios by removing one alternative from the given decision matrix.

    Parameters
    ----------
    matrix : ndarray
        The 2D array representing the decision matrix.

    combinations : ndarray, optional, default=None
        An array of alternative indexes to be removed, one at a time. If not provided, all alternatives are considered for removal subsequently.

    Returns
    -------
    dict
        Dictionary of modified decision matrix scenarios.
        The key of the dictionary is presented as 'A[n]', where 'n' represents the removed alternative index.

    Notes
    -----
    This function generates multiple scenarios by removing one or more alternatives from the input decision matrix.
    Each modified scenario is stored in the dictionary, where the keys represent the removed alternatives.

    Examples
    --------
    # Example 1: Remove all alternatives one by one
    >>> matrix = np.array([
    >>>     [1, 2, 3, 4],
    >>>     [1, 2, 3, 4],
    >>>     [4, 3, 2, 1],
    >>>     [3, 5, 3, 2],
    >>>     [4, 2, 5, 5],
    >>> ])
    >>> scenarios = remove_single(matrix)
    >>> print(scenarios)
    {'A[0]': array([[1, 2, 3, 4],
                    [4, 3, 2, 1],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[1]': array([[1, 2, 3, 4],
                    [4, 3, 2, 1],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[2]': array([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[3]': array([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [4, 3, 2, 1],
                    [4, 2, 5, 5]]),
    'A[4]': array([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [4, 3, 2, 1],
                    [3, 5, 3, 2]])}

    # Example 2: Remove specific alternatives by providing alternative indexes, one at a time]
    >>> matrix = np.array([
    >>>     [1, 2, 3, 4],
    >>>     [1, 2, 3, 4],
    >>>     [4, 3, 2, 1],
    >>>     [3, 5, 3, 2],
    >>>     [4, 2, 5, 5],
    >>> ])
    >>> combinations = np.array([0, 1])
    >>> scenarios = remove_single(matrix, combinations=combinations)
    >>> print(scenarios)
    {'A[0]': array([[4, 3, 2, 1],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[1]': array([[4, 3, 2, 1],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]])}
    """

    scenarios = {}

    # generate vector of subsequent alternative indexes to remove
    alternative_combs = combinations if combinations is not None else range(0, matrix.shape[0])

    # remove row in decision matrix
    for alt in alternative_combs:
        new_matrix = np.delete(matrix, alt, axis=0)

        scenarios[f'A[{alt}]'] = new_matrix

    return scenarios

# @memory_guard()
def remove_multiple(matrix: np.ndarray, combinations: np.ndarray = None, n: int = 3) -> dict:
    """
    Generate scenarios by removing multiple alternatives at a time from the given decision matrix.

    Parameters
    ----------
    matrix : ndarray
        The 2D array representing the decision matrix.

    combinations : ndarray, optional, default=None
        An array of arrays, each containing alternative indexes to be removed simultaneously.
        If not provided, combinations of alternatives to be removed are generated automatically.

    n : int, optional, default=3
        The minimum number of alternatives to keep in the decision matrix in each scenario.
        This parameter is only used when `combinations` is not provided.

    Returns
    -------
    dict
        Dictionary of modified decision matrix scenarios.
        The key of the dictionary is presented as 'A[n1-n2-...-nk]', where 'n1', 'n2', etc., represent the removed alternative indexes.

    Notes
    -----
    This function generates multiple scenarios by removing multiple alternatives from the input decision matrix.
    Each modified scenario is stored in the dictionary, where the keys represent the removed alternatives.

    Examples
    --------
    # Example 1: Remove multiple alternatives automatically
    >>> matrix = np.array([
    >>>     [1, 2, 3, 4],
    >>>     [1, 2, 3, 4],
    >>>     [4, 3, 2, 1],
    >>>     [3, 5, 3, 2],
    >>>     [4, 2, 5, 5],
    >>> ])
    >>> scenarios = remove_multiple(matrix)
    >>> print(scenarios)
    {'A[0-1]': array([[4, 3, 2, 1],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[0-2]': array([[1, 2, 3, 4],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[1-2]': array([[1, 2, 3, 4],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    ...}

    # Example 2: Remove specific combinations of alternatives
    >>> combinations = np.array([[0, 1], [1, 3]])
    >>> scenarios = remove_multiple(matrix, combinations=combinations)
    >>> print(scenarios)
    {'A[0-1]': array([[4, 3, 2, 1],
                    [3, 5, 3, 2],
                    [4, 2, 5, 5]]),
    'A[1-3]': array([[1, 2, 3, 4],
                    [4, 3, 2, 1],
                    [4, 2, 5, 5]])}
    """

    alternatives_combs = []

    if combinations is None:
        alternative_set = np.arange(0, matrix.shape[0], 1)
        for i in range(2, matrix.shape[0]-n+1):
            c_comb = itertools.combinations(alternative_set, i)
            alternatives_combs.append(list(c_comb))
        
        alternatives_combs = np.array([item for sublist in alternatives_combs for item in sublist], dtype='object')
    else:
        alternatives_combs = combinations

    scenarios = {}

    for alts in alternatives_combs:
        alts = np.array(alts, dtype='int')
        new_matrix = np.delete(matrix, alts, axis=0)

        scenarios[f'A[{"-".join(map(str, alts))}]'] = new_matrix

    return scenarios

if __name__ == '__main__':
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    r = remove_single(matrix)
    print(r)

    combinations = np.array([0, 1])
    r = remove_single(matrix, combinations=combinations)
    print(r)


    # multiple alternatives simultaneously
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])

    r = remove_multiple(matrix)
    print(r)

    combinations = np.array([[0, 1], [1, 3]])
    r = remove_multiple(matrix, combinations=combinations)
    print(r)

