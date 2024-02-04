# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import itertools

# @memory_guard
def single_percentage(weights: np.ndarray, percentages: dict, direction: str = 'both', step: (int | float) = 1) -> dict:
    """
    Generates weight scenarios based on specified percentage thresholds and direction.

    Parameters
    ----------
    weights : ndarray
        Vector of initial criteria weights.

    percentages: dict
        Dictionary containing percentage thresholds for each criteria weight modification. Percentages should be placed in range [1, 300].

    direction: str, optional, default='both'
        Direction of weight modification scenarios. Options: 'increases', 'decreases', 'both'.

    step: int or float, optional, default=1
        Step size for percentage thresholds generation.

    Returns
    -------
    dict
        Dictionary of weight scenarios modified based on percentage thresholds and direction. Based on the determined direction, dictionary include 'increases', 'decreases' or both of the keys, each of them containing dictionary with generated scenarios.

    Notes
    -----
    This function generates a set of weight vectors, each representing a scenario where specific criteria weights are modified based on given percentage thresholds and direction.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints (between 0 and 1 for each weight).

    Examples
    --------
    # Example 1: Generate weight scenarios with increasing weights
    >>> weights = np.array([0.3, 0.5, 0.2])
    >>> percentages = np.array([5, 4, 3])
    >>> direction = 'increases'
    >>> scenarios_increase = single_percentage(weights, percentages, direction)
    >>> print(scenarios_increase)
    >>> {'increases': {'C[0]-S[0]-P[0.01]': [0.3030, 0.4975, 0.1975], 'C[0]-S[1]-P[0.02]': [0.3060, 0.497, 0.197], ...}}

    # Example 2: Generate weight scenarios with both increasing and decreasing weights
    >>> weights = np.array([0.3, 0.5, 0.2])
    >>> percentages = 10
    >>> direction = 'both'
    >>> scenarios_both = single_percentage(weights, percentages, direction)
    >>> print(scenarios_both)
    >>> {'increases': {'C[0]-S[0]-P[0.01]': [0.3030, 0.4975, 0.1975], 'C[0]-S[1]-P[0.02]': [0.3060, 0.497, 0.197], ...},
         'decreases': {'C[0]-S[0]-P[0.01]': [0.2970, 0.5015, 0.2015], 'C[0]-S[1]-P[0.02]': [0.2940, 0.5030, 0.2030], ...}}
    """

    # change validation from array to dict
    # Validator.check_type(weights, 'weights', dict)
    # Validator.check_length(percentages, 'percentages', weights.shape[0])
    # Validator.percentage_direction(direction)
    # Validator.step_bound(step, type(step), np.array([0.01, 100]))
    # Validator.percentage_bounds(percentages, np.array([1, 300]))

    def _generate_weights(weights: np.ndarray, thresholds: np.ndarray, increase: bool) -> dict:
        """
        Generates weight scenarios based on the specified percentage thresholds and the direction of change.

        Parameters
        ----------
        weights : ndarray
            Vector of initial criteria weights.

        thresholds: ndarray
            2D array of percentages thresholds for each criteria weight. Percentages should be placed in range [1, 300]

        increase: bool
            Flag indicating whether to increase (True) or decrease (False) the weights.

        Returns
        -------
        dict
            Dictionary of weight vectors modified based on percentage thresholds and increase direction.

        Notes
        -----
        This function generates a set of weight vectors, each representing a scenario where specific criteria weights are modified based on given percentage thresholds and direction of change.

        The function validates input data and ensures that the generated scenarios adhere to specific constraints (between 0 and 1 for each weight).

        Examples
        --------
        # Example 1: Generate weight scenarios with increasing weights
        >>> weights = np.array([0.3, 0.5, 0.2])
        >>> thresholds = np.array([[0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.02, 0.03, 0.04], [0.01, 0.02, 0.03]])
        >>> increase = True
        >>> scenarios_increase = _generate_weights(weights, thresholds, increase)
        >>> print(scenarios_increase)
        >>> {'C[0]-S[0]-P[0.01]': [0.3030, 0.4975, 0.1975], 'C[0]-S[1]-P[0.02]': [0.3060, 0.497, 0.197], ...}

        # Example 2: Generate weight scenarios with decreasing weights
        >>> weights = np.array([0.3, 0.5, 0.2])
        >>> thresholds = np.array([[0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.02, 0.03, 0.04], [0.01, 0.02, 0.03]])
        >>> decrease = False
        >>> scenarios_decrease = _generate_weights(weights, thresholds, decrease)
        >>> print(scenarios_decrease)
        >>> {'C[0]-S[0]-P[0.01]': [0.2970, 0.5015, 0.2015], 'C[0]-S[1]-P[0.02]': [0.2940, 0.5030, 0.2030], ...}
        """
        weight_scenarios = {}

        for c_idx in range(weights.shape[0]):
            for p_idx, percent in enumerate(thresholds[c_idx]):
                new_weights = weights.copy()
                diff = weights[c_idx] * percent

                if increase: 
                    new_weights[c_idx] = new_weights[c_idx] + diff
                else:
                    new_weights[c_idx] = new_weights[c_idx] - diff

                equal_diff = diff / (len(weights)-1)
                for i, w in enumerate(new_weights):
                    if i != c_idx:
                        if increase:
                            new_weights[i] = w - equal_diff 
                        else:
                            new_weights[i] = w + equal_diff 

                if any(new_weights < 0) or any(new_weights > 1):
                    break
                
                weight_scenarios[f'C[{c_idx}]-S[{p_idx}]-P[{percent}]'] = new_weights
        return weight_scenarios

    scenarios = {}

    thresholds = np.array([np.arange(step, p+step, step) / 100 for p in list(percentages.values())], dtype='object')

    if direction == 'increases' or direction == 'both':
        scenarios['increases'] = _generate_weights(weights, thresholds, True)
    if direction == 'decreases' or direction == 'both':
        scenarios['decreases'] = _generate_weights(weights, thresholds, False)

    if direction == 'both':
        return scenarios

    return scenarios[direction]

# @memory_guard
def multiple_percentage(weights: np.ndarray, percentages: dict, combinations: np.ndarray = None, direction: str = 'both', step: (int | float) = 1) -> dict:
    """
    Generate weight scenarios based on the specified combinations of criteria, percentage thresholds, and direction.

    Parameters
    ----------
    weights : ndarray
        Vector of initial criteria weights.

    percentages: dict
        Dictionary containing percentage thresholds for each criteria weight modification. Percentages should be placed in range [1, 300].

    combinations: ndarray, optional, default=None
        2D array of combinations of criteria indices for each scenario. If not provided, generated automatically.

    direction: str, optional, default='both
        Specifies the direction of weight modification. Can be 'increases', 'decreases', or 'both'.

    step: int or float, optional, default=1
        Step used for percentage thresholds.

    Returns
    -------
    dict
        Dictionary of weight vectors modified based on criteria combinations, percentage thresholds, and direction.

    Notes
    -----
    This function generates weight vectors for different scenarios where specific criteria weights are modified based on given combinations of criteria indices, corresponding percentage thresholds, and direction.

    The function ensures that the generated scenarios adhere to specific constraints (between 0 and 1 for each weight).

    Examples
    --------
    # Example 1: Generate weight scenarios with increasing weights
    >>> weights = np.array([0.3, 0.5, 0.2])
    >>> percentages = {0: 100, 1: 50, 2: 70}
    >>> combinations = np.array([[0, 1], [1, 2]])
    >>> direction = 'increases'
    >>> step = 0.01
    >>> scenarios_increase = multiple_percentage(weights, percentages, combinations, direction, step)
    >>> print(scenarios_increase)
    >>> {'C[0-1]-S[0]-P[0.01, 0.01]': [0.303, 0.505, 0.192], 'C[0-1]-S[1]-P[0.01-0.02]': [0.303, 0.510, 0.187], ...}

    # Example 2: Generate weight scenarios with both directions
    >>> weights = np.array([0.3, 0.5, 0.2])
    >>> percentages = {0: 100, 1: 50, 2: 70}
    >>> combinations = np.array([[0, 2], [0, 1, 2]])
    >>> direction = 'both'
    >>> step = 0.01
    >>> scenarios_both = multiple_percentage(weights, percentages, combinations, direction, step)
    >>> print(scenarios_both)
    >>> {'increases': {'C[0-1]-S[0]-P[0.01, 0.01]': [0.303, 0.505, 0.192], 'C[0-1]-S[1]-P[0.01-0.02]': [0.303, 0.510, 0.187],, ...}, 'decreases': {'C[0-1]-S[0]-P[0.01, 0.01]': [0.297, 0.495, 0.208], 'C[0-1]-S[1]-P[0.01-0.02]': [0.297, 0.490, 0.213], ...}}
    """

    def _generate_weights(weights: np.ndarray, combinations: np.ndarray, percentages: np.ndarray, increase: bool):
        """
        Generate weight scenarios based on the specified combinations of criteria and percentage thresholds.

        Parameters
        ----------
        weights : ndarray
            Vector of initial criteria weights.

        combinations: ndarray
            2D array of combinations of criteria indices for each scenario.

        percentages: ndarray
            2D array of percentage thresholds for each criteria weight modification.

        increase: bool
            Flag indicating whether to increase (True) or decrease (False) the weights.

        Returns
        -------
        dict
            Dictionary of weight vectors modified based on criteria combinations and percentage thresholds.

        Notes
        -----
        This function generates weight vectors for different scenarios where specific criteria weights are modified based on given combinations of criteria indices and corresponding percentage thresholds.

        The function ensures that the generated scenarios adhere to specific constraints (between 0 and 1 for each weight).

        Examples
        --------
        # Example 1: Generate weight scenarios with increasing weights
        >>> weights = np.array([0.3, 0.5, 0.2])
        >>> combinations = np.array([[0, 1], [1, 2]])
        >>> percentages = np.array([[[0.01, 0.01], [0.01, 0.02], [0.01, 0.03], ..., [1.00, 0.01], [1.00, 0.02], ...], [...]])
        >>> increase = True
        >>> scenarios_increase = _generate_weights(weights, combinations, percentages, increase)
        >>> print(scenarios_increase)
        >>> {'C[0-1]-S[0]-P[0.01, 0.01]': [0.303, 0.505, 0.192], 'C[0-1]-S[1]-P[0.01-0.02]': [0.303, 0.510, 0.187], ...}

        # Example 2: Generate weight scenarios with decreasing weights
        >>> weights = np.array([0.3, 0.5, 0.2])
        >>> combinations = np.array([[0, 2], [0, 1, 2]])
        >>> percentages = np.array([[[0.01, 0.01], [0.01, 0.02], [0.01, 0.03], ..., [1.00, 0.01], [1.00, 0.02], ...], [...]])
        >>> increase = False
        >>> scenarios_decrease = _generate_weights(weights, combinations, percentages, increase)
        >>> print(scenarios_decrease)
        >>> {'C[0-1]-S[0]-P[0.01, 0.01]': [0.297, 0.495, 0.208], 'C[0-1]-S[1]-P[0.01-0.02]': [0.297, 0.490, 0.213], ...}
        """

        weight_scenarios = {}

        for criteria, percentages_comb in zip(combinations, percentages):
            for scenario_idx, percentages in enumerate(percentages_comb):
                new_weights = weights.copy()
    
                diff = 0
                for idx_c, criterion in enumerate(criteria):
                    if increase:
                        new_weights[criterion] = weights[criterion] + weights[criterion] * percentages[idx_c]
                    else:
                        new_weights[criterion] = weights[criterion] - weights[criterion] * percentages[idx_c]
                    diff += weights[criterion] * percentages[idx_c]

                
                equal_diff = diff / (weights.shape[0] - len(criteria))

                for idx_w, w in enumerate(weights):
                    if idx_w not in list(criteria):
                        if increase: 
                            new_weights[idx_w] = w - equal_diff
                        else:
                            new_weights[idx_w] = w + equal_diff


                if any(new_weights < 0) or any(new_weights > 1):
                    continue
                                
                weight_scenarios[f'C[{"-".join(map(str, criteria))}]-S[{scenario_idx}]-P[{"-".join(map(str, percentages))}]'] = new_weights

        return weight_scenarios

    percentages_combs = []
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
        
    percentages_lists = np.array([np.arange(step, percentages[i]+step, step) / 100 if i in list(percentages.keys()) else np.array([]) for i in range(weights.shape[0])], dtype='object')
        

    for criteria in criteria_combs:
        temp_combs = []
        for c in criteria:
            temp_combs.append(percentages_lists[c])
        percentages_combs.append(np.array(list(itertools.product(*temp_combs))))

    scenarios = {}

    if direction == 'increases' or direction == 'both':
        scenarios['increases'] = _generate_weights(weights, criteria_combs, percentages_combs, True)
    if direction == 'decreases' or direction == 'both':
        scenarios['decreases'] = _generate_weights(weights, criteria_combs, percentages_combs, False)

    if direction == 'both':
        return scenarios

    return scenarios


if __name__ == '__main__':
    # weights = np.array([0.3, 0.3, 0.2, 0.2])
    # percentages = {
    #     0: 100,
    #     1: 50, 
    #     2: 70,
    #     3: 90
    # }
    weights = np.array([0.3, 0.5, 0.2])
    percentages = {
        0: 100,
        1: 100, 
        2: 100,
    }
    direction = 'both'
    
    # single
    # scenarios = single_percentage(weights, percentages, direction)
    # keys = list(scenarios['increases'].keys())
    # values = list(scenarios['increases'].values())
    # for i in range(0, 10):
    #     print(keys[i], values[i])

    # multiple
    multiple_scenarios = multiple_percentage(weights, percentages, direction=direction)
    keys = list(multiple_scenarios['increases'].keys())
    values = list(multiple_scenarios['increases'].values())
    for i in range(0, 10):
        print(keys[i], values[i])



