# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.alternative import discrete_modification

def test_discrete_modification_example1():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    discrete_values = np.array([[2, 3, 4], [1, 5, 6], [3, 4]], dtype='object')
    results = discrete_modification(matrix, discrete_values)
    assert len(results) == 24  # Number of combinations
    assert results[0][0] == 0 # Number of modified alternative
    assert results[0][3][0][0] == 2 # value in modified alternative

def test_discrete_modification_example2():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    discrete_values = np.array([[2, 3, 4], [1, 5, 6], [3, 4]], dtype='object')
    indexes = np.array([[0, 2], 1], dtype='object')
    results = discrete_modification(matrix, discrete_values, indexes)
    assert len(results) == 27  # Number of combinations with specified indexes

def test_discrete_modification_example3():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    discrete_values = np.array([[[5, 6], [2, 4], [5, 8]], [[3, 5.5], [4], [3.5, 4.5]], [[7, 8], [6], [8, 9]]], dtype='object')
    results = discrete_modification(matrix, discrete_values)
    assert len(results) == 16  # Number of combinations with 3D discrete values

def test_discrete_modification_example4():
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    discrete_values = np.array([[[5, 6], [2, 4], [5, 8]], [[3, 5.5], [4], [3.5, 4.5]], [[7, 8], [6], [8, 9]]], dtype='object')
    indexes = np.array([[0, 2], 1], dtype='object')
    results = discrete_modification(matrix, discrete_values, indexes)
    assert len(results) == 16  # Number of combinations with 3D discrete values and specified indexes

def test_discrete_modification_invalid_matrix():
    matrix = "invalid_matrix"  # This will raise a TypeError in the function
    discrete_values = np.array([[2, 3, 4], [1, 5, 6], [3, 4]], dtype='object')
    
    with raises(TypeError):
        discrete_modification(matrix, discrete_values)
