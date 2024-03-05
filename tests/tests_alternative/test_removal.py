# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.alternative import remove_alternatives

def test_remove_alternatives_default_behavior():
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    results = remove_alternatives(matrix)
    assert isinstance(results[0], tuple) 
    assert results[0][0] == 0 
    assert (results[0][1], np.ndarray)
    assert len(results[0][1]) == matrix.shape[0] - 1

def test_remove_alternatives_remove_at_specific_index():
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    results = remove_alternatives(matrix, 3)
    assert isinstance(results[0], tuple) 
    assert results[0][0] == 3
    assert (results[0][1], np.ndarray)
    assert len(results[0][1]) == matrix.shape[0] - 1
    

def test_remove_alternatives_remove_with_specified_indexes_1D():
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    results = remove_alternatives(matrix, np.array([1, 2, 3]))
    assert isinstance(results[0], tuple) 
    assert results[0][0] == 1
    assert (results[0][1], np.ndarray)
    assert len(results[0][1]) == matrix.shape[0] - 1
    assert len(results) == 3

def test_remove_alternatives_remove_with_specified_indexes_mixed_type():
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    results = remove_alternatives(matrix, np.array([[0, 4], 2, 3], dtype='object'))
    assert isinstance(results[0], tuple) 
    assert results[0][0] == [0, 4] 
    assert (results[0][1], np.ndarray)
    assert len(results[0][1]) == matrix.shape[0] - 2
    
def test_remove_alternatives_error():
    matrix = 1

    with raises(TypeError):
        remove_alternatives(matrix)