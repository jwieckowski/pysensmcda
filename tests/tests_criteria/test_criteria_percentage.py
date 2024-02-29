# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.criteria import percentage_modification

def test_percentage_modification_single_change():
    weights = np.array([0.3, 0.3, 0.4])
    percentage = 5
    results = percentage_modification(weights, percentage)
    print(results[0])
    assert results[0][0] == 0
    assert results[0][1] == -0.01 
    assert isinstance(results[0][2], np.ndarray) 
    assert np.isclose(np.sum(results[0][2]), 1.0)

def test_percentage_modification_specific_indexes_step():
    weights = np.array([0.3, 0.3, 0.4])
    percentages = np.array([5, 5, 5])
    indexes = np.array([[0, 1], 2], dtype='object')
    results = percentage_modification(weights, percentages, indexes=indexes)
    assert results[0][0] == (0, 1)
    assert results[0][1] == (-0.01, -0.01) 
    assert isinstance(results[0][2], np.ndarray) 
    assert np.isclose(np.sum(results[0][2]), 1.0)


def test_percentage_modification_specific_direction():
    weights = np.array([0.3, 0.3, 0.4])
    percentages = np.array([6, 4, 5])
    direction = np.array([-1, 1, -1])
    results = percentage_modification(weights, percentages, direction=direction)
    assert results[0][0] == 0
    assert results[0][1] == -0.01  
    assert isinstance(results[0][2], np.ndarray) 
    assert np.isclose(np.sum(results[0][2]), 1.0)


def test_percentage_modification_specific_indexes_individual_step():
    weights = np.array([0.3, 0.3, 0.4])
    percentages = np.array([6, 4, 8])
    indexes = np.array([0, 2])
    step = 2
    results = percentage_modification(weights, percentages, indexes=indexes, step=step)
    assert results[0][0] == 0
    assert results[0][1] == -0.02
    assert isinstance(results[0][2], np.ndarray) 
    assert np.isclose(np.sum(results[0][2]), 1.0)

def test_percentage_modification_error():
    weights = 1
    percentages = np.array([6, 4, 8])
    indexes = np.array([0, 2])
    step = 2
    with raises(TypeError):
        percentage_modification(weights, percentages, indexes=indexes, step=step)

