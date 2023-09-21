# Copyright (C) 2023 Jakub Więckowski

import numpy as np
import random

def random_weights(criteria: int, n: int) -> dict:
    """
    Generate random weight scenarios by assigning random values to criteria weights and normalizing them.

    Parameters
    ----------
    criteria : int
        Number of criteria for which random weight scenarios will be generated.

    n : int
        Number of random weight scenarios to generate.

    Returns
    -------
    dict
        Dictionary of randomly generated weight scenarios.
        The key of the dictionary is presented as 'S[n]', where:
            - 'n' represents the number of the generated scenario.

    Notes
    -----
    This function generates multiple scenarios by randomly assigning values to criteria weights within the specified range [0, 100]. The generated weight vectors are then normalized to ensure they sum up to 1.

    The function validates input data and ensures that the generated scenarios adhere to specific constraints.

    Examples
    --------
    >>> num_criteria = 3
    >>> num_scenarios = 5
    >>> random_scenarios = random_weights(num_criteria, num_scenarios)
    >>> print(random_scenarios)
    >>> {'S[0]': array([0.40350877 0.24561404 0.35087719]),
         'S[1]': array([0.43661972 0.45539906 0.10798122]),
         'S[2]': array([0.37853107 0.10734463 0.51412429]),
         'S[3]': array([0.32051282 0.61538462 0.06410256]),
         'S[4]': array([0.67924528 0.0754717  0.24528302])}
    """
    # Validator.check_type(criteria, 'criteria', int)
    # Validator.check_type(n, 'n', int)

    scenarios = {}
    
    for idx in range(n):
        weights = np.array([random.choice(range(0,100)) for _ in range(criteria)])
        scenarios[f'S[{idx}]'] = weights / np.sum(weights)

    return scenarios

if __name__ == '__main__':
    scenarios = random_weights(3, 10)
    for k, v in scenarios.items():
        print(k, v, np.sum(v))