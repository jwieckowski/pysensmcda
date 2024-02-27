# Copyright (C) 2024 Jakub Więckowski

import numpy as np
from ..validator import Validator

def perturbed_matrix(matrix: np.ndarray, simulations: int, precision: int = 6, perturbation_scale: float | np.ndarray = 0.1):
    """
    Generate perturbed decision matrices based on the given initial decision matrix using random perturbation based on uniform distribution.

    Parameters
    ----------
    matrix : ndarray
        2D array representing the initial decision matrix.

    simulations : int
        Number of perturbed decision matrix simulations to generate.

    precision : int, optional, default=6
        Precision for rounding the perturbed values from the decision matrix.

    perturbation_scale : float | np.ndarray, optional, default=0.1
        Scale for random perturbation added to each value from the decision matrix.
        If float, then all decision matrix is modeled with the same perturbation scale.
        If ndarray, then each criterion is modeled with a given perturbation scale.

    Returns
    -------
    ndarray
        A ndarray of simulations length with perturbed decision matrices based on the given initial decision matrix.

    Examples
    --------
    ### Example 1 - Run with default parameters
    >>> matrix = np.array([ [4, 3, 7], [1, 9, 6], [7, 5, 3] ])
    >>> simulations = 1000
    >>> results = perturbed_matrix(matrix, simulations)
    >>> for r in results:
    ...     print(r)

    ### Example 2 - Run with given precision and perturbation scale
    >>> matrix = np.array([ [4, 3, 7], [1, 9, 6], [7, 5, 3] ])
    >>> simulations = 500
    >>> precision = 3
    >>> perturbation_scale = 1
    >>> results = perturbed_matrix(matrix, simulations, precision, perturbation_scale)
    >>> for r in results:
    ...     print(r)

    ### Example 3 - Run with perturbation scale defined for each column
    >>> matrix = np.array([ [4, 3, 7], [1, 9, 6], [7, 5, 3] ])
    >>> simulations = 100
    >>> precision = 3
    >>> perturbation_scale = np.array([0.5, 1, 0.4])
    >>> results = perturbed_matrix(matrix, simulations, precision, perturbation_scale)
    >>> for r in results:
    ...     print(r)

    ### Example 4 - Run with 2D perturbation scale array
    >>> matrix = np.array([ [4, 3, 7], [1, 9, 6], [7, 5, 3] ])
    >>> simulations = 100
    >>> precision = 3
    >>> perturbation_scale = np.array([ [0.4, 0.5, 1], [0.7, 0.3, 1.2], [0.5, 0.1, 1.5] ])
    >>> results = perturbed_matrix(matrix, simulations, precision, perturbation_scale)
    >>> for r in results:
    ...     print(r)
    """

    Validator.is_type_valid(matrix, np.ndarray)
    # if not isinstance(matrix, np.ndarray):
    #     raise TypeError("Matrix should be given as a numpy array")

    Validator.is_dimension_valid(matrix, 2)
    # if matrix.ndim != 2:
    #     raise ValueError("Matrix should be a 2D array")

    Validator.is_type_valid(simulations, int)
    Validator.is_positive_value(simulations)
    # if not isinstance(simulations, int) or simulations <= 0:
    #     raise ValueError("Number of simulations should be a positive integer")

    Validator.is_type_valid(precision, int)
    Validator.is_positive_value(precision)
    # if not isinstance(precision, int) or precision < 0:
    #     raise ValueError("Precision should be a non-negative integer")

    Validator.is_type_valid(perturbation, (int, float, np.ndarray))

    if isinstance(perturbation_scale, (float, int)):
        perturbation_scale = np.full(matrix.shape[0], perturbation_scale)
    elif isinstance(perturbation_scale, np.ndarray):
        if perturbation_scale.ndim == 1:
            # if perturbation_scale.shape[0] != matrix.shape[0]:
            #     raise ValueError("Length of perturbation_scale should be equal to the number of criteria")
            Validator.is_shape_equal(matrix.shape[0], perturbation_scale.shape[0])
        elif perturbation_scale.ndim == 2:
            Validator.is_shape_equal(matrix.shape, perturbation_scale.shape)
            # if perturbation_scale.shape[0] != matrix.shape[0] and perturbation_scale.shape[1] != matrix.shape[1]:
            #     raise ValueError("Shape of perturbation_scale and matrix should be equal")

    modified_matrices = []

    for _ in range(simulations):
        perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, matrix.shape)
        modified_matrix = matrix + perturbation
        modified_matrices.append(list(np.round(modified_matrix, precision)))

    return np.array(modified_matrices)
