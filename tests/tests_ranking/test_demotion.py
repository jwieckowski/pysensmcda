# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysensmcda.ranking import ranking_demotion
from pymcdm.methods import COPRAS

def test_ranking_demotion_default_parameters():
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    weights = np.array([0.4, 0.5, 0.1])
    types = np.array([-1, 1, -1])
    copras = COPRAS()
    pref = copras(matrix, weights, types)
    initial_ranking = copras.rank(pref)
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    direction = np.array([1, -1, 1])
    step = 0.5
    max_modification = 100
    results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, max_modification=max_modification)
    assert len(results) == 9
    assert len(results[0]) == 4
    assert isinstance(results[0][0], int)
    assert isinstance(results[0][1], int)
    assert isinstance(results[0][2], float)

def test_ranking_demotion_custom_bounds():
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    weights = np.array([0.4, 0.5, 0.1])
    types = np.array([-1, 1, -1])
    copras = COPRAS()
    initial_ranking = np.array([2, 3, 1])
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    direction = np.array([1, -1, 1])
    step = 0.5
    bounds = np.array([15, 0, 20])
    results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, bounds)
    assert len(results) == 9
    assert len(results[0]) == 4
    assert isinstance(results[0][0], int)
    assert isinstance(results[0][1], int)
    assert isinstance(results[0][2], float)

def test_ranking_demotion_custom_positions():
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    weights = np.array([0.4, 0.5, 0.1])
    types = np.array([-1, 1, -1])
    copras = COPRAS()
    initial_ranking = np.array([2, 3, 1])
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    direction = types * -1
    step = 0.5
    bounds = np.array([15, 0, 20])
    positions = np.array([3, 3, 1])
    results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, bounds, positions)
    assert len(results) == 9
    assert len(results[0]) == 4
    assert isinstance(results[0][0], int)
    assert isinstance(results[0][1], int)
    assert isinstance(results[0][2], float)

def test_ranking_demotion_without_zeros():
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    weights = np.array([0.4, 0.5, 0.1])
    types = np.array([-1, 1, -1])
    copras = COPRAS()
    initial_ranking = np.array([2, 3, 1])
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    direction = types * -1
    step = 0.5
    max_modification = 10
    return_zeros = False
    results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, max_modification=max_modification, return_zeros=return_zeros)
    assert len(results) == 5
    assert len(results[0]) == 4
    assert isinstance(results[0][0], int)
    assert isinstance(results[0][1], int)
    assert isinstance(results[0][2], float)

def test_ranking_demotion_error():
    matrix = np.array([
        [4, 2, 6],
        [7, 3, 2],
        [9, 6, 8]
    ])
    weights = np.array([0.4, 0.5, 0.1])
    types = np.array([-1, 1, -1])
    copras = COPRAS()
    initial_ranking = np.array([2, 3, 1])
    call_kwargs = {
        "matrix": matrix,
        "weights": weights,
        "types": types
    }
    ranking_descending = True
    direction = types * -1
    step = 0.5
    max_modification = 10
    return_zeros = False
    with raises(TypeError):
        ranking_demotion(1, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, max_modification=max_modification, return_zeros=return_zeros)