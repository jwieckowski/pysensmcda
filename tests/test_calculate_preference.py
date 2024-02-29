# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.calculate_preference import calculate_preference
from pymcdm.methods import TOPSIS
from pysenscraft.criteria import percentage_modification
from pysenscraft.alternative import discrete_modification

def test_calculate_preference_alternative_sensitivity():
    topsis = TOPSIS()

    matrix = np.array([
        [4, 1, 6],
        [2, 6, 3],
        [9, 5, 7],
    ])
    discrete_values = np.array([
        [[5, 6], [2, 4], [5, 8]],
        [[3, 5.5], [4], [3.5, 4.5]],
        [[7, 8], [6], [8, 9]],
    ], dtype='object')
    indexes = np.array([[0, 2], 1], dtype='object')
    results = discrete_modification(matrix, discrete_values, indexes)
    kwargs = {
        'matrix': matrix,
        'weights': np.ones(matrix.shape[0]) / matrix.shape[0],
        'types': np.ones(matrix.shape[0])
    }

    results = calculate_preference(discrete_modification, results, topsis, kwargs)
    assert len(results) == 16
    assert len(results[0]) == 3

def test_calculate_preference_criteria_sensitivity_rankings():
    topsis = TOPSIS()

    weights = np.array([0.3, 0.3, 0.4])
    percentage = 5
    results = percentage_modification(weights, percentage)

    kwargs = {
        'matrix': np.random.random((10, 3)),
        'weights': weights,
        'types': np.ones(3)
    }

    results = calculate_preference(percentage_modification, results, topsis, kwargs, method_type=1)
    assert len(results) == 30
    assert len(results[0]) == 2
    assert len(results[0][1]) == kwargs['matrix'].shape[0]

def test_calculate_preference_criteria_sensitivity_aggregated_results():
    topsis = TOPSIS()

    weights = np.array([0.3, 0.3, 0.4])
    percentage = 5
    results = percentage_modification(weights, percentage)

    kwargs = {
        'matrix': np.random.random((10, 3)),
        'weights': weights,
        'types': np.ones(3)
    }

    results= calculate_preference(percentage_modification, results, topsis, kwargs, only_preference=False, method_type=1)
    print(results[0])
    assert len(results) == 30
    assert len(results[0]) == 5
    assert results[0][0] == 0
    assert results[0][1] == -0.01
    assert isinstance(results[0][2], np.ndarray)
    assert len(results[0][3]) == kwargs['matrix'].shape[0]
    assert len(results[0][4]) == kwargs['matrix'].shape[0]

def test_calculate_preference_error():
    topsis = TOPSIS()

    weights = np.array([0.3, 0.3, 0.4])
    percentage = 5
    results = percentage_modification(weights, percentage)

    kwargs = {
        'matrix': np.random.random((10, 3)),
        'weights': weights,
        'types': np.ones(3)
    }

    def error_fun():
        pass

    with raises(ValueError):
        calculate_preference(error_fun, results, topsis, kwargs, only_preference=False, method_type=1)