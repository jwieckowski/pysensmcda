# Copyright (C) 2023 Jakub Więckowski

import numpy as np

def perturbed_weights(weights: np.ndarray, num_iterations: int = 5000) -> dict:
    """
    Perform sensitivity analysis using the Monte Carlo method to analyze the impact of varying criteria weights on the decision outcome.

    Parameters
    ----------
    weights : ndarray
        The initial criteria weights for each criterion.

    num_iterations : int, optional (default=5000)
        The number of Monte Carlo iterations to generate weights.

    Returns
    -------
    dict
        A dictionary of perturbed criteria weights scenarios.
        The key of the dictionary is presented as 'S[n]', where:
            - 'n' represents the number of the generated scenario.
    Notes
    -----
    This function performs sensitivity analysis by randomly perturbing the initial criteria weights and generating multiple scenarios.
    Each scenario represents a set of perturbed criteria weights, normalized to ensure they sum to 1.

    Examples
    --------
    # Example usage:
    >>> initial_weights = np.array([0.4, 0.3, 0.3])
    >>> num_iterations = 1000
    >>> scenarios = perturbed_weights(initial_weights, num_iterations)
    >>> print(scenarios)
        { 'S[0]': array([0.46645079, 0.19048653, 0.34306268]), 'S[1]': array([0.2646803, 0.4187305, 0.3165892]), 'S[2]': array([0.4220199, 0.24833998, 0.32964013]), ... }
    """

    scenarios = {}
    for scenario_idx in range(num_iterations):
        # Randomly perturb the criteria weights for sensitivity analysis
        # Adjust the standard deviation as needed
        perturbed_weights = weights + np.random.normal(0, 0.1, weights.shape[0])  

        # Normalize the perturbed weights to ensure they sum to 1
        perturbed_weights /= np.sum(perturbed_weights)

        scenarios[f'S[{scenario_idx}]'] = perturbed_weights

    return scenarios

if __name__ == "__main__":

    # Example initial criteria weights
    initial_weights = np.array([0.4, 0.3, 0.3])

    # Number of Monte Carlo iterations
    num_iterations = 1000

    # Perform sensitivity analysis
    scenarios = perturbed_weights(initial_weights, num_iterations)
    for k, v in scenarios.items():
        print(k, v)

    
