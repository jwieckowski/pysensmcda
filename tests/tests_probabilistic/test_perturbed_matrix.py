# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.probabilistic import perturbed_matrix

def test_perturbed_matrix_default_parameters():
    matrix = np.array([[4, 3, 7], [1, 9, 6], [7, 5, 3]])
    simulations = 1000
    results = perturbed_matrix(matrix, simulations)
    assert len(results) == simulations
    assert all(np.array_equal(r.shape, matrix.shape) for r in results)

def test_perturbed_matrix_custom_parameters():
    matrix = np.array([[4, 3, 7], [1, 9, 6], [7, 5, 3]])
    simulations = 500
    precision = 3
    perturbation_scale = 1
    results = perturbed_matrix(matrix, simulations, precision, perturbation_scale)
    assert len(results) == simulations
    assert all(np.array_equal(r.shape, matrix.shape) for r in results)

def test_perturbed_matrix_column_scale():
    matrix = np.array([[4, 3, 7], [1, 9, 6], [7, 5, 3]])
    simulations = 100
    precision = 3
    perturbation_scale = np.array([0.5, 1, 0.4])
    results = perturbed_matrix(matrix, simulations, precision, perturbation_scale)
    assert len(results) == simulations
    assert all(np.array_equal(r.shape, matrix.shape) for r in results)

def test_perturbed_matrix_2d_scale():
    matrix = np.array([[4, 3, 7], [1, 9, 6], [7, 5, 3]])
    simulations = 100
    precision = 3
    perturbation_scale = np.array([[0.4, 0.5, 1], [0.7, 0.3, 1.2], [0.5, 0.1, 1.5]])
    results = perturbed_matrix(matrix, simulations, precision, perturbation_scale)
    assert len(results) == simulations
    assert all(np.array_equal(r.shape, matrix.shape) for r in results)