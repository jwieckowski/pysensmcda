# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

# @memory_guard
def single_range(weights: np.ndarray, ranges: dict, step: float = 0.01) -> dict:
    """
    Generates weights vector modifications based on a given range of values for particular criterion.

    Parameters
    ----------
    weights : ndarray
        Vector of criteria weights in a crisp form.
    ranges : dict
        Vectors for criteria to generate weights scenarios with specific values from given ranges.
    step : float, optional, default=0.01
        Step used in the iteration through the ranges of criteria weights.

    Returns
    -------
    dict
        Dictionary containing scenarios of modified weight vectors within the given ranges.
        The key of the dictionary is presented as 'C[n]-S[m]', where:
            - 'n' represents the index of the modified criterion.
            - 'm' indicates the number of the scenario generated for a specific element in the criteria weights with index 'n'.

    Notes
    -----
    This function generates various scenarios of modified weight vectors based on the specified ranges and step. For each criterion, scenarios are generated with weights modified while keeping one criterion within a specified range and adjusting other criteria accordingly.

    Examples
    --------
    >>> weights = np.array([0.2, 0.3, 0.5])
    >>> ranges = {0: [0.1, 0.4], 2: [0.4, 0.7]}
    >>> scenarios = single_range(weights, ranges, step=0.05)
    >>> print(scenarios)
    {'C[0]-S[0]': array([0.1, 0.35, 0.55]), 'C[0]-S[1]': array([0.15, 0.325, 0.575]), ...}
    """


    # Validator.check_type(ranges, 'ranges', dict)
    # Validator.dict_keys(ranges, weights.shape[0])
    # Validator.check_type(step, 'step', float)
    # Validator.step_bound(step, float, np.array([0, 1]))
    # Validator.weights_extension(weights)
    # Validator.weights_sum(weights)
    # Validator.range_weights_size(ranges, weights)
    # Validator.ranges_bound(ranges)

    scenarios = {}

    # generation of vector with subsequent values of weights for criteria
    thresholds = np.array([np.linspace(ranges[i][0], ranges[i][1], int((ranges[i][1] - ranges[i][0])/step)+1) if i in list(ranges.keys()) else np.array([]) for i in range(weights.shape[0])], dtype='object')

    for crit_idx, threshold_ranges in enumerate(thresholds):

        if len(threshold_ranges) > 0:
            for scenario_idx, threshold in enumerate(threshold_ranges):

                new_weights = weights.copy()
                diff = abs(weights[crit_idx] - threshold) 

                new_weights[crit_idx] = threshold

                equal_diff = diff / (len(weights)-1)
                for i, w in enumerate(new_weights):
                    if i != crit_idx:
                        if weights[crit_idx] < threshold:
                            new_weights[i] = w - equal_diff 
                        elif weights[crit_idx] > threshold:
                            new_weights[i] = w + equal_diff 
                scenarios[f'C[{crit_idx}]-S[{scenario_idx}]'] = np.array(new_weights)

    return scenarios


# @memory_guard
def multiple_range(weights: np.ndarray, ranges: dict, combinations: np.ndarray = None, step: float = 0.01) -> dict:
    """
    Generates multiple scenarios of weights vectors modification based on specified criteria values combinations and ranges.
    
    Parameters
    ----------
    weights : ndarray
        Vector of initial criteria weights.
        
    ranges: dict
        Dictionary of vectors specifying the ranges for each criteria weight.
        
    combinations: ndarray, optional, default=None
        2D array of criteria combinations to generate scenarios for (default is None). If None is given, all possible combinations will generated.
        
    step: float, optional, default=0.01
        Step used in the iteration through the ranges of criteria weights (default is 0.01).
        
    Returns
    -------
    dict
        Dictionary of weight vectors modified within given ranges and combinations.
        The key of the dictionary is presented as 'C[n1-n2-n3-...]-S[m]', where:
            - 'n1, n2, n3' represents the indexes of the simultaneously modified criteria.
            - 'm' indicates the number of the scenario generated for a specific element in the criteria weights with index 'n'.
    Notes
    -----
    This function generates a set of weight vectors, each representing a scenario where specific criteria weights are modified based on given combinations and ranges.
    
    The function validates input data and ensures that the generated scenarios adhere to specific constraints (between 0 and 1 for each weight).
    
    Examples
    --------
    # Example 1: Generate scenarios with default combinations and step
    >>> weights = np.array([0.3, 0.5, 0.2])
    >>> ranges = {0: (0.2, 0.5), 1: (0.4, 0.7), 2: (0.1, 0.3)}
    >>> scenarios = multiple_range(weights, ranges)
    >>> print(scenarios)
    {'C[0-1]-S[0]': array([0.2, 0.40, 0.40]), 'C[0-1]-S[1]': array([0.2, 0.41, 0.39]), ..., 'C[1-2]-S[0]: array([0.5, 0.4, 0.1]), ...}
    
    # Example 2: Generate scenarios with custom combinations and step
    >>> weights = np.array([0.3, 0.5, 0.2])
    >>> ranges = {0: (0.2, 0.5), 1: (0.4, 0.7), 2: (0.1, 0.3)}
    >>> custom_combinations = np.array([[0, 1], [0, 2]])
    >>> custom_step = 0.05
    >>> scenarios_custom = multiple_range(weights, ranges, combinations=custom_combinations, step=custom_step)
    >>> print(scenarios_custom)
    {'C[0-1]-S[0]': array([0.2, 0.40, 0.40]), 'C[0-1]-S[1]': array([0.2, 0.41, 0.39]), ..., 'C[0-2]-S[0]: array([0.2, 0.5, 0.3]), ...}
    """

    # Validator.check_type(ranges, 'ranges', dict)
    # Validator.dict_keys(ranges, weights.shape[0])
    # Validator.check_type(step, 'step', float)
    # Validator.step_bound(step, float, np.array([0, 1]))
    # Validator.weights_extension(weights)
    # Validator.weights_sum(weights)
    # Validator.range_weights_size(ranges, weights)
    # Validator.ranges_bound(ranges)
    # if combinations is not None:
        # Validator.check_combinations(combinations, weights.shape[0]-1)


    scenarios = {}

    ranges_combs = []
    criteria_combs = []

    # generation of possible criteria weights values combinations
    if combinations is None:
        criteria_set = np.arange(0, weights.shape[0], 1)
        for i in range(2, weights.shape[0]):
            c_comb = itertools.combinations(criteria_set, i)
            criteria_combs.append(list(c_comb))

        criteria_combs = np.array([item for sublist in criteria_combs for item in sublist], dtype='object')
    else:
        criteria_combs = combinations


    # generation of vector with subsequent values of weights for criteria
    thresholds = np.array([np.linspace(ranges[i][0], ranges[i][1], int((ranges[i][1] - ranges[i][0])/step)+1) if i in list(ranges.keys()) else np.array([]) for i in range(weights.shape[0])], dtype='object')

    # generation of criteria weights vectors based on the possible values combinations
    for criteria in criteria_combs:
        temp_combs = []
        for cc in criteria:
            temp_combs.append(thresholds[cc])
        ranges_combs.append(np.array(list(itertools.product(*temp_combs))))

    ranges_combs = np.array(ranges_combs, dtype='object')

    for criteria, ranges_comb in zip(criteria_combs, ranges_combs):
        for scenario_idx, ranges in enumerate(ranges_comb):
                new_weights = weights.copy()

                diff = 1
                sum_new = 0
                for idx_c, criterion in enumerate(criteria):
                    new_weights[criterion] = ranges[idx_c]
                    sum_new += ranges[idx_c]
                    diff -= ranges[idx_c]

                current_sum = 0
                for idx_w, w in enumerate(weights):
                    if idx_w not in list(criteria):
                        current_sum += w
                
                new_diff = diff - current_sum
                equal_diff = new_diff / (weights.shape[0] - len(criteria))


                for idx_w, w in enumerate(weights):
                    if idx_w not in list(criteria):
                        new_weights[idx_w] = w + equal_diff 

                if any(new_weights < 0) or any(new_weights > 1):
                    continue
                                    
                scenarios[f'C[{"-".join(map(str, criteria))}]-S[{scenario_idx}]'] = new_weights

    return scenarios 

if __name__ == '__main__':
    # single range
    weights = np.array([0.3, 0.3, 0.2, 0.2])
    ranges = {
        0: [0.3, 0.4],
        1: [0.3, 0.45],
        2: [0.15, 0.25],
        3: [0.1, 0.2]
    }
    step = 0.01
    # scenarios = single_range(weights, ranges, step)
    # for k, v in scenarios.items():
    #     print(k, v)

    # multiple range
    # scenarios = multiple_range(weights, ranges)
    # for k, v in scenarios.items():
    #     print(k, v)

    combinations = np.array([[0, 1], [1, 2, 3]])

    scenarios = multiple_range(weights, ranges, combinations=combinations)
    
    keys = list(scenarios.keys())
    values = list(scenarios.values())
    for i in range(10):
        print(keys[i], values[i])

        