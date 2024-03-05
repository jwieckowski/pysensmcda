# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.alternative import range_modification

def test_range_modification_2D_range_change():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    results = range_modification(matrix, range_values)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == 6.0 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix

def test_range_modification_3D_range_change_array():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    range_values = np.array([[[3, 5], [1, 3], [4, 6]], [[2, 5], [5, 7], [2, 4]], [[8, 11], [4, 7], [7, 9]]])
    results = range_modification(matrix, range_values)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == 3.0 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix

def test_range_modification_with_indexes():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    indexes = np.array([[0, 2], 1], dtype='object')
    results = range_modification(matrix, range_values, indexes)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == (0, 2) # index of modified criterion
    assert results[0][2] == (6.0, 4.0) # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix

def test_range_modification_with_step_size():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    step = 0.5
    results = range_modification(matrix, range_values, step=step)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == 6.0 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix
    
    assert results[1][0] == 0 # index of modified alternative
    assert results[1][1] == 0 # index of modified criterion
    assert results[1][2] == 6.5 # modified value
    assert isinstance(results[1][3], np.ndarray) # modified matrix

def test_range_modification_with_individual_step_sizes():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    range_values = np.array([[6, 8], [2, 4], [4, 6.5]])
    step = np.array([0.25, 0.4, 0.5])
    results = range_modification(matrix, range_values, step=step)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == 6.0 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix
    
    assert results[1][0] == 0 # index of modified alternative
    assert results[1][1] == 0 # index of modified criterion
    assert results[1][2] == 6.25 # modified value
    assert isinstance(results[1][3], np.ndarray) # modified matrix

def test_range_modification_error():
    matrix = 1
    range_values = np.array([[6, 8], [2, 4], [4, 6.5]])

    with raises(TypeError):
        range_modification(matrix, range_values)