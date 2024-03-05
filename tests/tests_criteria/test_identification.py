# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.criteria import relevance_identification
from pymcdm.methods import TOPSIS
from pymcdm.weights import equal_weights
import pymcdm.normalizations as norm
import pymcdm.correlations as correlations

def test_relevance_identification_example1():
    matrix = np.array([
        [4, 3, 5, 7],
        [7, 4, 2, 4],
        [9, 5, 7, 3],
        [3, 5, 6, 3]
    ])
    criteria_types = np.array([1, 1, -1, 1])
    weights = equal_weights(matrix)
    topsis = TOPSIS(normalization_function=norm.vector_normalization)
    call_kwargs = {
        'matrix': matrix,
        'weights': weights,
        'types': criteria_types
    }
    results = relevance_identification(topsis, call_kwargs, ranking_descending=True)
    assert isinstance(results[0][0], tuple)
    assert results[0][1] == 1
    assert isinstance(results[0][2], float)
    assert isinstance(results[0][3], np.ndarray)

def test_relevance_identification_example2():
    matrix = np.array([
        [106.78,  6.75,  2.  , 220.  ,  6.  ,  1.  , 52.  , 455.5 ,  8.9 , 36.8 ],
        [ 86.37,  7.12,  3.  , 400.  , 10.  ,  0.  , 20.  , 336.5 ,  7.2 , 29.8 ],
        [104.85,  6.95, 60.  , 220.  ,  7.  ,  1.  , 60.  , 416.  ,  8.7 , 36.2 ],
        [ 46.6 ,  6.04,  1.  , 220.  ,  3.  ,  0.  , 50.  , 277.  ,  3.9 , 16.  ],
        [ 69.18,  7.05, 33.16, 220.  ,  8.  ,  0.  , 35.49, 364.79,  5.39, 33.71],
        [ 66.48,  6.06, 26.32, 220.  ,  6.53,  0.  , 34.82, 304.02,  4.67, 27.07],
        [ 74.48,  6.61, 48.25, 400.  ,  4.76,  1.  , 44.19, 349.45,  4.93, 28.89],
        [ 73.67,  6.06, 19.54, 400.  ,  3.19,  0.  , 46.41, 354.65,  8.01, 21.09],
        [100.58,  6.37, 39.27, 220.  ,  8.43,  1.  , 22.07, 449.42,  7.89, 17.62],
        [ 94.81,  6.13, 50.58, 220.  ,  4.18,  1.  , 21.14, 450.88,  5.12, 17.3 ],
        [ 48.93,  7.12, 21.48, 220.  ,  5.47,  1.  , 55.72, 454.71,  8.39, 19.16],
        [ 74.75,  6.58,  7.08, 400.  ,  9.9 ,  1.  , 26.01, 455.17,  4.78, 18.44]
    ])
    criteria_types = np.array([1, 1, -1, 1, -1, -1, 1, -1, -1, 1])
    weights = equal_weights(matrix)
    topsis = TOPSIS(normalization_function=norm.vector_normalization)
    call_kwargs = {
        'matrix': matrix,
        'weights': weights,
        'types': criteria_types
    }
    results = relevance_identification(topsis, call_kwargs, corr_coef=[correlations.rw, correlations.ws, correlations.rs], ranking_descending=True, excluded_criteria=3)
    assert isinstance(results[0][0], tuple)
    assert results[0][1] == 1
    assert results[0][2] == 1
    assert results[0][3] == 1
    assert isinstance(results[0][4], float)
    assert isinstance(results[0][5], np.ndarray)

def test_relevance_identification_error():
    topsis = TOPSIS(normalization_function=norm.vector_normalization)
    call_kwargs = {}
    
    with raises(ValueError):
        relevance_identification(topsis, call_kwargs, ranking_descending=True)