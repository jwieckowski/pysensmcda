# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.criteria import range_modification

def test_range_modification_single_change():
    weights = np.array([0.3, 0.3, 0.4])
    range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    results = range_modification(weights, range_values)
    assert results[0][0] == 0
    assert results[0][1] == 0.25
    assert np.isclose(np.sum(results[0][2]), 1.0)

def test_range_modification_specific_indexes_step():
    weights = np.array([0.3, 0.3, 0.4])
    indexes = np.array([[0, 1], 2], dtype='object')
    range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    results = range_modification(weights, range_values, indexes=indexes)
    assert results[0][0] == [0, 1]
    assert results[0][1] == (0.25, 0.3)
    assert np.isclose(np.sum(results[0][2]), 1.0)

def test_range_modification_individual_step():
    weights = np.array([0.3, 0.3, 0.4])
    range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    step = 0.02
    results = range_modification(weights, range_values, step=step)
    assert results[0][0] == 0
    assert results[0][1] == 0.25
    assert np.isclose(np.sum(results[0][2]), 1.0)

def test_range_modification_error():
    weights = 1
    range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    step = 0.02
    with raises(TypeError):
        range_modification(weights, range_values, step=step)
