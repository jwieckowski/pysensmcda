# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.criteria import remove_criteria

def test_remove_criteria_no_indexes():
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights)
    assert results[0][0] == 0
    assert isinstance(results[0][1], np.ndarray)
    assert results[0][1].shape[1] == matrix.shape[1] -1
    assert results[0][2].shape[0] == weights.shape[0] -1

def test_remove_criteria_int_index():
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights, 3)
    assert results[0][0] == 3
    assert isinstance(results[0][1], np.ndarray)
    assert results[0][1].shape[1] == matrix.shape[1] -1
    assert results[0][2].shape[0] == weights.shape[0] -1

def test_remove_criteria_array_indexes_one_dimensional():
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights, np.array([1, 2, 3]))
    assert results[0][0] == 1
    assert isinstance(results[0][1], np.ndarray)
    assert results[0][1].shape[1] == matrix.shape[1] -1
    assert results[0][2].shape[0] == weights.shape[0] -1

def test_remove_criteria_array_indexes_elements_list():
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = np.array([0.25, 0.25, 0.2, 0.2, 0.1])
    results = remove_criteria(matrix, weights, np.array([[0, 4], 2, 3], dtype='object'))
    assert results[0][0] == [0, 4]
    assert isinstance(results[0][1], np.ndarray)
    assert results[0][1].shape[1] == matrix.shape[1] -2
    assert results[0][2].shape[0] == weights.shape[0] -2

def test_range_modification_error():
    matrix = np.array([
        [1, 2, 3, 4, 4],
        [1, 2, 3, 4, 4],
        [4, 3, 2, 1, 4]
    ])
    weights = 1

    with raises(TypeError):
        remove_criteria(matrix, weights, np.array([[0, 4], 2, 3], dtype='object'))
        