# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from itertools import product

def range_modification(weights: np.ndarray, range_values: np.ndarray, indexes: None | np.ndarray = None, step: float = 0.01):
    """
    Modify a set of criteria weights based on specified percentage changes, directions, and indexes.

    Parameters
    ----------
    weights : ndarray
        1D array representing the initial criteria weights. Should sum up to 1.
    range_values : ndarray
        Range of values for each criterion specifying the allowed changes.
        Should be given as a two-dimensional array where each row represents a criterion, 
        and the columns represent the lower and upper bounds of the allowed range.
    indexes : None | ndarray, optional, default=None
        Indexes of the criteria to be modified. If None, all criteria are considered subsequently.
        If ndarray, it specifies the indexes or combinations of indexes for the criteria to be modified.
    step : float, optional, default=0.01
        Step size for the percentage change.

    Returns
    -------
    List[Tuple[int, Union[float, Tuple[float, ...]], ndarray]]
        A list of tuples containing information about the modified criteria index, 
        percentage change, and the resulting criteria weights.

    ## Examples
    --------
    ### Example 1: Modify weights with a single percentage change
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    >>> results = range_modification(weights, range_values)
    >>> for r in results:
    ...     print(r)

    ### Example 2: Modify weights with range values, specific indexes, and step size
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> indexes = np.array([[0, 1], 2], dtype='object')
    >>> range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    >>> results = range_modification(weights, range_values, indexes=indexes)
    >>> for r in results:
    ...     print(r)

    ### Example 3: Modify weights with range values and individual step sizes
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    >>> step = 0.02
    >>> results = range_modification(weights, range_values, step=step)
    >>> for r in results:
    ...     print(r)
    """

    def modify_weights(weights, crit_idx, change):
        new_weights = weights.copy()

        modified_criteria = 1
        if isinstance(crit_idx, (int, np.integer)):
            new_weights[crit_idx] = change
        else:
            if np.sum(change) >= 1:
                return None

            modified_criteria = len(crit_idx)
            new_weights[crit_idx] = change

        new_sum = np.sum(new_weights)
        adjust_direction = -1 if new_sum > 1 else 1
        equal_diff = np.abs(1 - new_sum) / (weights.shape[0] - modified_criteria)
        # adjust weights to sum up to 1
        for idx, w in enumerate(weights):
            if isinstance(crit_idx, (int, np.integer)):
                if crit_idx != idx:
                    new_weights[idx] = w + equal_diff * adjust_direction
            else:
                if idx not in crit_idx:
                    new_weights[idx] = w + equal_diff * adjust_direction

        return new_weights / np.sum(new_weights)

    # weights dimension
    if weights.ndim != 1:
        raise ValueError('Weights should be given as one dimensional array')

    # weights sum to 1
    if np.round(np.sum(weights), 3) != 1:
        raise ValueError('Weights vector should sum up to 1')

    # check if weights and percentages have the same length
    if weights.shape[0] != range_values.shape[0]:
        raise ValueError('Weights and range values have different length')

    if not isinstance(range_values, np.ndarray):
        raise TypeError('Range values should be given as np array')

    # check if all range values given as list
    if range_values.ndim != 2:
        raise ValueError('Range values should be given as two dimensional array')

    if indexes is not None:
        for c_idx in indexes:
            if isinstance(c_idx, (int, np.integer)):
                if c_idx < 0 or c_idx >= weights.shape[0]:
                    raise IndexError(f'Given index ({c_idx}) out of range')
            elif isinstance(c_idx, list):
                if any([idx < 0 or idx >= weights.shape[0] for idx in c_idx]):
                    raise IndexError(f'Given indexes ({c_idx}) out of range')

    results = []

    # generation of vector with subsequent values of weights for criteria
    range_changes = np.array([np.arange(range_values[i][0], range_values[i][1]+step, step) for i in range(weights.shape[0])], dtype='object')
    range_changes = np.array([[val for val in rc if val >= range_values[idx][0] and val <= range_values[idx][1]] for idx, rc in enumerate(range_changes)], dtype='object')

    # criteria indexes to modify weights values
    indexes_values = None
    if indexes is None:
        indexes_values = np.arange(0, weights.shape[0], dtype=int)
    else:
        indexes_values = indexes

    for crit_idx in indexes_values:
        if isinstance(crit_idx, (int, np.integer)):
            changes = range_changes[crit_idx]
        else:
            changes = list(product(*range_changes[crit_idx]))

        for change in changes:
            change_val = np.round(change, 6) if isinstance(change, float) else tuple(np.round(change, 6).tolist())
            new_weights = modify_weights(weights, crit_idx, change)
            if new_weights is not None:
                results.append((crit_idx, change_val, new_weights))

    return results
