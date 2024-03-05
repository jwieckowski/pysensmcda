# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.compromise import HQ_compromise

def test_HQ_compromise_default_parameters():
    rankings = np.array([[2, 2, 2],
                        [3, 4, 5],
                        [1, 1, 4],
                        [4, 3, 1],
                        [7, 5, 7],
                        [8, 8, 8],
                        [5, 6, 3],
                        [6, 7, 6]])
    ((consensus, trust), weights, ranking) = HQ_compromise(rankings)
    assert np.round(consensus, 2) == 0.85 # consensus index
    assert np.round(trust, 2) == 0.95 # trust index
    assert np.array_equal(np.round(weights, 3), np.array([0.5  , 0.495, 0.006])) # weights of rankings
    assert np.array_equal(np.round(ranking), np.array([2., 4., 1., 3., 6., 8., 5., 6.])) # rankings

def test_HQ_compromise_set_tolerance():
    rankings = np.array([[2, 2, 2],
                        [3, 4, 5],
                        [1, 1, 4],
                        [4, 3, 1],
                        [7, 5, 7],
                        [8, 8, 8],
                        [5, 6, 3],
                        [6, 7, 6]])
    ((consensus, trust), weights, ranking) = HQ_compromise(rankings, max_iters=100, tol=10e-6)
    assert np.round(consensus, 2) == 0.85 # consensus index
    assert np.round(trust, 2) == 0.95 # trust index
    assert np.array_equal(np.round(weights, 3), np.array([0.5  , 0.495, 0.006])) # weights of rankings
    assert np.array_equal(np.round(ranking), np.array([2., 4., 1., 3., 6., 8., 5., 6.])) # rankings
    
def test_HQ_compromise_invalid_rankings():
    # This will raise a TypeError in the function
    rankings = "invalid_ranking"  

    with raises(TypeError):
        HQ_compromise(rankings)
