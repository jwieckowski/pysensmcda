# Copyright (C) 2023 - 2024 Jakub Więckowski

import numpy as np
from itertools import product
from ..validator import Validator
from ..utils import memory_guard

@memory_guard
def percentage_modification(weights: np.ndarray, percentages: int | np.ndarray, direction: None | np.ndarray = None, indexes: None | np.ndarray = None, step: int | float = 1):
    """
    Modify a set of criteria weights based on specified percentage changes, directions, and indexes.

    Parameters
    ----------
    weights : ndarray
        1D array representing the initial criteria weights. Should sum up to 1.

    percentages : int | ndarray
        Percentage changes to be applied to the criteria weights. 
        If int, the same percentage change is applied to all criteria.
        If ndarray, it specifies the percentage change for each criterion individually.

    direction : None | ndarray, optional, default=None
        Direction of the modification for each criterion. If None, both increase and decrease directions are considered.
        If ndarray, it specifies the direction (1 for increase, -1 for decrease) for each criterion individually.

    indexes : None | ndarray, optional, default=None
        Indexes of the criteria to be modified. If None, all criteria are considered subsequently.
        If ndarray, it specifies the indexes or combinations of indexes for the criteria to be modified.

    step : int | float, optional, default=1
        Step size for the percentage change.

    Returns
    -------
    List[Tuple[int | tuple, tuple, ndarray]]
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

    def modify_weights(weights, crit_idx, diff, direction_val):
        new_weights = weights.copy()

        modified_criteria = 1
        if isinstance(crit_idx, (int, np.integer)):
            new_weights[crit_idx] = weights[crit_idx] + diff * direction_val
        else:
            modified_criteria = len(crit_idx)
            new_weights[crit_idx] = weights[crit_idx] + diff * direction_val
        
        equal_diff = np.sum(diff) / (weights.shape[0] - modified_criteria)
        # adjust weights to sum up to 1
        for idx, w in enumerate(weights):
            if isinstance(crit_idx, (int, np.integer)):
                if crit_idx != idx:
                    new_weights[idx] = w + equal_diff * (direction_val * -1)
            else:
                if idx not in crit_idx:
                    new_weights[idx] = w + equal_diff * (direction_val * -1)
        
        return new_weights / np.sum(new_weights)

    Validator.is_type_valid(weights, np.ndarray)
    Validator.is_dimension_valid(weights, 1)
    Validator.is_sum_valid(weights, 1)
    Validator.is_type_valid(percentages, np.ndarray)
    Validator.is_shape_equal(weights.shape, percentages.shape, custom_message="Shapes of 'weights' and 'percentages' are different")
    Validator.is_type_valid(direction, (None, np.ndarray))
    Validator.is_shape_equal(weights.shape, direction.shape, custom_message="Shapes of 'weights' and 'direction' are different")
    
    if direction is not None:
        Validator.is_in_list(direction, [-1, 1])

    Validator.is_type_valid(indexes, (None, np.ndarray))
    Validator.are_indexes_valid(indexes, weights.shape[0])

    results = []

    # size of changes of criteria weights
    percentages_values = None
    if isinstance(percentages, (int, np.integer)):
        percentages_values = np.array([percentages] * weights.shape[0])
    if isinstance(percentages, np.ndarray):
        percentages_values = percentages

    # vectors with subsequent changes for criteria
    percentages_changes = np.array([np.arange(step, p+step, step) / 100 for p in percentages_values], dtype='object')

    # increasing or decreasing weights
    direction_values = None
    if direction is None:
        direction_values = np.array([[-1, 1]] * weights.shape[0])
    else:
        direction_values = np.array([[val] for val in direction])

    # criteria indexes to modify weights values
    indexes_values = None
    if indexes is None:
        indexes_values = np.arange(0, weights.shape[0], dtype=int)
    else:
        indexes_values = indexes

    for crit_idx in indexes_values:
        if isinstance(crit_idx, (int, np.integer)):
            changes = percentages_changes[crit_idx]
        else:
            changes = list(product(*percentages_changes[crit_idx]))
        
        for change in changes:
            diff = weights[crit_idx] * change

            change_direction = direction_values[crit_idx]
            if not isinstance(crit_idx, (int, np.integer)):
                change_direction = [direction_values[crit_idx][0]]
                
            for val in change_direction:
                if isinstance(val, (int, np.integer)):
                    new_weights = modify_weights(weights, crit_idx, diff, val)
                    results.append((crit_idx, change * val, new_weights))
                else:
                    for v in val:
                        change_val = tuple(c * v for c in change)
                        new_weights = modify_weights(weights, crit_idx, diff, v)
                        results.append((tuple(crit_idx), change_val, new_weights))

    return results

