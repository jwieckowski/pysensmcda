# Copyright (C) 2024 Jakub Więckowski

import numpy as np

def perturbed_weights(weights: np.ndarray, simulations: int, precision: int = 6, perturbation_scale: float | np.ndarray = 0.1):
    """
    Generate perturbed weights based on the given initial criteria weights based on the given perturbation scale and uniform distribution.

    Parameters
    ----------
    weights : ndarray
        1D array representing the existing criteria weights

    simulations : int
        Number of perturbed weight simulations to generate

    precision : int, optional, default=6
        Precision for rounding the perturbed weights

    perturbation_scale : float | np.ndarray, optional, default=0.1
        Scale for random perturbation added to each weight.
        If float, then all criteria weights modeled with the same perturbation scale.
        If ndarray, then each criterion modeled with given perturbation scale.

    Returns
    -------
    ndarray
        A ndarray of perturbed weights based on the given criteria weights

    Examples
    --------
    ### Example 1 - run with default parameters
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> simulations = 1000
    >>> results = perturbed_weights(weights, simulations)
    >>> for r in results:
    ...     print(r)

    ### Example 2 - run with given precision and perturbation scale
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> simulations = 1000
    >>> precision = 3
    >>> perturbation_scale = 0.05
    >>> results = perturbed_weights(weights, simulations, precision, perturbation_scale)
    >>> for r in results:
    ...     print(r)

    ### Example 3 - run with perturbation scale defined for each criterion
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> simulations = 1000
    >>> precision = 3
    >>> perturbation_scale = np.array([0.05, 0.1, 0.04])
    >>> results = perturbed_weights(weights, simulations, precision, perturbation_scale)
    >>> for r in results:
    ...     print(r)

    """

    if not isinstance(weights, np.ndarray):
        raise TypeError("Weights should be given as a numpy array")

    if weights.ndim != 1:
        raise ValueError("Weights should be a 1D vector")

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Sum of weights should be equal to 1")

    if not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("Number of simulations should be a positive integer")

    if not isinstance(precision, int) or precision < 0:
        raise ValueError("Precision should be a non-negative integer")

    if isinstance(perturbation_scale, (float)):
        perturbation_scale = np.full(weights.shape[0], perturbation_scale)
    elif isinstance(perturbation_scale, np.ndarray):
        if perturbation_scale.shape[0] != weights.shape[0]:
            raise ValueError("Length of perturbation_scale should be equal to the number of criteria")

    modified_weights = []

    for _ in range(simulations):
        perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, weights.shape[0])
        modified_weights_candidate = weights + perturbation
        modified_weights_candidate = np.clip(modified_weights_candidate, 0, 1)
        normalized_weights = modified_weights_candidate / np.sum(modified_weights_candidate)

        modified_weights.append(list(np.round(normalized_weights, precision)))

    return np.array(modified_weights)
