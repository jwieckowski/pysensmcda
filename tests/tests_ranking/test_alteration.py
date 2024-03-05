# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.ranking import ranking_alteration
from pymcdm.methods import ARAS

def test_ranking_alteration_default_parameters():
    weights = np.array([0.4, 0.5, 0.1])
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    types = np.array([-1, 1, -1])
    aras = ARAS()
    pref = aras(matrix, weights, types)
    initial_ranking = aras.rank(pref)
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    results = ranking_alteration(weights, initial_ranking, aras, call_kwargs, ranking_descending)
    assert len(results) == len(initial_ranking)
    assert isinstance(results[0], tuple)
    assert results[0][0] == 0
    assert isinstance(results[0][1], np.ndarray)
    assert isinstance(results[0][2], np.ndarray)

def test_ranking_alteration_custom_step():
    weights = np.array([0.4, 0.5, 0.1])
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    types = np.array([-1, 1, -1])
    aras = ARAS()
    pref = aras(matrix, weights, types)
    initial_ranking = aras.rank(pref)
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    step = 0.05
    results = ranking_alteration(weights, initial_ranking, aras, call_kwargs, ranking_descending, step)
    assert len(results) == len(initial_ranking)
    assert isinstance(results[0], tuple)
    assert results[0][0] == 0
    assert isinstance(results[0][1], np.ndarray)
    assert isinstance(results[0][2], np.ndarray)

def test_ranking_alteration_error():
    weights = np.array([0.4, 0.5, 0.1])
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    types = np.array([-1, 1, -1])
    aras = ARAS()
    pref = aras(matrix, weights, types)
    initial_ranking = aras.rank(pref)
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    step = 0.05
    with raises(TypeError):
        ranking_alteration(1, initial_ranking, aras, call_kwargs, ranking_descending, step)
