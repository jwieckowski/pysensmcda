# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.alternative import percentage_modification

def test_percentage_modification_single_percentage():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    percentages = 5
    results = percentage_modification(matrix, percentages)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == -0.01 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix
    
def test_percentage_modification_percentage_list():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    percentages = np.array([3, 5, 8])
    results = percentage_modification(matrix, percentages)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == -0.01 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix
    
def test_percentage_modification_percentage_list_with_direction():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    percentages = np.array([2, 4, 6])
    direction = np.array([1, 1, -1])
    results = percentage_modification(matrix, percentages, direction)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == 0.01 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix

def test_percentage_modification_percentage_list_with_indexes():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    percentages = np.array([1, 4, 2])
    indexes = np.array([[0, 2], 1], dtype='object')
    results = percentage_modification(matrix, percentages, indexes=indexes)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == (0, 2) # index of modified criterion
    assert results[0][2] == (-0.01, -0.01) # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix

def test_percentage_modification_percentage_list_with_step():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    percentages = np.array([2, 4, 9])
    step = np.array([2, 2, 3])
    results = percentage_modification(matrix, percentages, step=step)
    assert results[0][0] == 0 # index of modified alternative
    assert results[0][1] == 0 # index of modified criterion
    assert results[0][2] == -0.02 # modified value
    assert isinstance(results[0][3], np.ndarray) # modified matrix
    
def test_percentage_modification_error():
    matrix = 1
    percentages = np.array([2, 4, 9])
    with raises(TypeError):
        percentage_modification(matrix, percentages)