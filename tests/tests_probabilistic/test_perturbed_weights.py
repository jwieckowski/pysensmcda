# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.probabilistic import perturbed_weights

def test_perturbed_weights_default_parameters():
    weights = np.array([0.3, 0.4, 0.3])
    simulations = 1000
    results = perturbed_weights(weights, simulations)
    assert len(results) == simulations
    assert all(np.array_equal(len(r), len(weights)) for r in results)

def test_perturbed_weights_custom_parameters():
    weights = np.array([0.3, 0.4, 0.3])
    simulations = 1000
    precision = 3
    perturbation_scale = 0.05
    results = perturbed_weights(weights, simulations, precision, perturbation_scale)
    assert len(results) == simulations
    assert all(np.array_equal(len(r), len(weights)) for r in results)

def test_perturbed_weights_criterion_scale():
    weights = np.array([0.3, 0.4, 0.3])
    simulations = 1000
    precision = 3
    perturbation_scale = np.array([0.05, 0.1, 0.04])
    results = perturbed_weights(weights, simulations, precision, perturbation_scale)
    assert len(results) == simulations
    assert all(np.array_equal(len(r), len(weights)) for r in results)

def test_perturbed_weights_error():
    weights = 1
    simulations = 1000
    precision = 3
    perturbation_scale = np.array([0.05, 0.1, 0.04])
    with raises(TypeError):
        perturbed_weights(weights, simulations, precision, perturbation_scale)
        