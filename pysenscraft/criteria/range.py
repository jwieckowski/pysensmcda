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
        Percentage changes to be applied to the criteria weights. 
        If ndarray, it specifies the percentage change for each criterion individually.
    indexes : None | ndarray, optional, default=None
        Indexes of the criteria to be modified. If None, all criteria are considered subsequently.
        If ndarray, it specifies the indexes or combinations of indexes for the criteria to be modified.
    step : float, optional, default=0.01
        Step size for the percentage change.
    Returns
    -------
    List[Tuple[int, ndarray, ndarray]]
        A list of tuples containing information about the modified criteria index, percentage change,
        and the resulting criteria weights.
    ## Examples
    --------
    ### Example 1: Modify weights with a single percentage change
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentage = 5
    >>> results = percentage_modification(weights, percentage)
    >>> for r in results:
    ...     print(r)
    # Example 2: Modify weights with percentages, specific indexes, and step size
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentages = np.array([5, 5, 5])
    >>> indexes = np.array([[0, 1], 2], dtype='object')
    >>> results = percentage_modification(weights, percentages, indexes=indexes)
    >>> for r in results:
    ...     print(r)
    ### Example 3: Modify weights with percentages and specific direction for each criterion
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentages = np.array([6, 4, 5])
    >>> direction = np.array([-1, 1, -1])
    >>> results = percentage_modification(weights, percentages, direction=direction)
    >>> for r in results:
    ...     print(r)
    ### Example 4: Modify weights with percentages, specific indexes, and individual step sizes
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentages = np.array([6, 4, 8])
    >>> indexes = np.array([0, 2])
    >>> step = 2
    >>> results = percentage_modification(weights, percentages, indexes=indexes, step=step)
    >>> for r in results:
    ...     print(r)
    """

    def modify_weights(weights, crit_idx, diff):
        new_weights = weights.copy()

        modified_criteria = 1
        if isinstance(crit_idx, (int, np.integer)):
            new_weights[crit_idx] = weights[crit_idx] + diff
        else:
            modified_criteria = len(crit_idx)
            new_weights[crit_idx] = weights[crit_idx] + diff

        equal_diff = np.sum(diff) / (weights.shape[0] - modified_criteria)
        # adjust weights to sum up to 1
        for idx, w in enumerate(weights):
            if isinstance(crit_idx, (int, np.integer)):
                if crit_idx != idx:
                    new_weights[idx] = w + equal_diff
            else:
                if idx not in crit_idx:
                    new_weights[idx] = w + equal_diff

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
    range_changes = np.array([np.linspace(range_values[i][0], range_values[i][1], int((range_values[i][1] - range_values[i][0])/step)+1) for i in range(weights.shape[0])], dtype='object')

    # criteria indexes to modify weights values
    indexes_values = None
    if indexes is None:
        indexes_values = np.arange(0, weights.shape[0], dtype=int)
    else:
        indexes_values = indexes
    
    print(range_changes)
    print(indexes_values)

    # for crit_idx in indexes_values:
    #     if isinstance(crit_idx, (int, np.integer)):
    #         changes = percentages_changes[crit_idx]
    #     else:
    #         changes = list(product(*percentages_changes[crit_idx]))

    #     for change in changes:
    #         diff = weights[crit_idx] * change

    #         for val in direction_values[crit_idx]:
    #             if isinstance(val, (int, np.integer)):
    #                 new_weights = modify_weights(weights, crit_idx, diff, val)
    #                 results.append((crit_idx, change * val, new_weights))
    #             else:
    #                 for v in [-1, 1]:
    #                     new_weights = modify_weights(weights, crit_idx, diff, v)
    #                     results.append((crit_idx, np.asarray(change) * v, new_weights))

    return results

# Example 1
weights = np.array([0.3, 0.3, 0.4])
range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
results = range_modification(weights, range_values)
for r in results:
    print(r)