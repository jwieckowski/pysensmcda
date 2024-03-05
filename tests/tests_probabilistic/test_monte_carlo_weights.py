# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.probabilistic import monte_carlo_weights

def test_monte_carlo_weights():
    n = 3
    modified_weights = monte_carlo_weights(n, num_samples=1000, distribution='normal', params={'loc': 0.5, 'scale': 0.1})
    assert isinstance(modified_weights, np.ndarray)
    print(modified_weights[0])
    assert len(modified_weights[0]) == 3

def test_monte_carlo_weights_error():
    with raises(TypeError):
        monte_carlo_weights(5.5, num_samples=1000, distribution='normal', params={'loc': 0.5, 'scale': 0.1})

def test_monte_carlo_weights_distribution_error():
    with raises(ValueError):
        monte_carlo_weights(5, num_samples=1000, distribution='wrong', params={'loc': 0.5, 'scale': 0.1})
