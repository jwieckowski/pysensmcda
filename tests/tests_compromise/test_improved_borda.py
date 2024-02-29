# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.compromise import improved_borda
from pymcdm.methods import TOPSIS, VIKOR

def test_improved_borda():
    topsis = TOPSIS()
    vikor = VIKOR()
    matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    criteria_num = matrix.shape[1]
    weights = np.ones(criteria_num)/criteria_num
    types = np.ones(criteria_num)
    preferences = np.array([topsis(matrix, weights, types), vikor(matrix, weights, types)]).T
    
    compromise_ranking = improved_borda(preferences, [1, -1])

    assert np.array_equal(compromise_ranking, np.array([2, 3, 1]))

def test_improved_borda_invalid_rankings():
    # This will raise a TypeError in the function
    preferences = 'wrong type'
    
    with raises(TypeError):
        improved_borda(preferences, [1, -1])
        