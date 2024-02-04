# Copyright (C) 2023 Jakub Więckowski

import numpy as np

def perturbed_matrix(matrix: np.ndarray, num_iterations: int = 5000, stddev: float = 0.1) -> dict:
    """
    Modify a decision matrix to perform Monte Carlo simulation for sensitivity analysis.

    Parameters
    ----------
    matrix : ndarray
        The 2d original decision matrix where rows represent alternatives and columns represent criteria.

    num_iterations : int, optional (default=5000)
        The number of Monte Carlo iterations to generate modified decision matrices.

    stddev : float, optional (default=0.1)
        The standard deviation for perturbing matrix elements.

    Returns
    -------
    dict
        A dictionary of modified decision matrices representing different scenarios.
        The key of the dictionary is presented as 'S[n]', where:
            - 'n' represents the number of the generated scenario.

    Notes
    -----
    This function generates multiple scenarios by perturbing the elements of the input decision matrix.
    The perturbations are based on a normal distribution with a specified standard deviation.

    Examples
    --------
    # Example usage:
    >>> matrix = np.array([[3, 5, 4], [4, 3, 5], [5, 4, 3]])
    >>> num_iterations = 1000
    >>> stddev = 0.1
    >>> scenarios = perturbed_matrix(matrix, num_iterations, stddev)
    >>> print(scenarios)
        { 'S[0]': array([[3.06183882, 5.11213018, 3.79713934], [4.24053485, 2.96722643, 5.03065071], [4.97993142, 4.07086121, 2.94555314]]), 'S[1]': array([[2.96975763, 5.16077357, 3.88214365], [3.99248436, 3.13021628, 4.81736085], [5.02818202, 4.027828, 2.85484162]]) ... }
    """

    scenarios = {}
    for scenario_idx in range(num_iterations):
        # Create a perturbed decision matrix by adding random noise
        perturbed_matrix = matrix + np.random.normal(0, stddev, matrix.shape)

        # Ensure values are non-negative
        perturbed_matrix[perturbed_matrix < 0] = 0

        scenarios[f'S[{scenario_idx}]'] = perturbed_matrix

    return scenarios

# Example usage
if __name__ == "__main__":
    matrix = np.array([[3, 5, 4], [4, 3, 5], [5, 4, 3]])
    num_iterations = 1000
    stddev = 0.1

    scenarios = perturbed_matrix(matrix, num_iterations, stddev)
    for k, v in scenarios.items():
        print(k, v)
