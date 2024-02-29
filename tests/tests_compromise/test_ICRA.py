# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.compromise.ICRA import iterative_compromise
from pymcdm.methods import TOPSIS, VIKOR, EDAS

def test_ICRA_default_parameters():
    decision_matrix = np.array([[8, 1, 6], [2, 4, 3], [9, 5, 7]])
    decision_problem_weights = np.ones(decision_matrix.shape[1])/decision_matrix.shape[1]
    decision_problem_types = np.ones(decision_matrix.shape[1])

    edas = EDAS()
    topsis = TOPSIS()
    vikor = VIKOR()

    edas_pref = edas(decision_matrix, decision_problem_weights, decision_problem_types)
    topsis_pref = topsis(decision_matrix, decision_problem_weights, decision_problem_types)
    vikor_pref = vikor(decision_matrix, decision_problem_weights, decision_problem_types)

    ## ICRA variables preparation
    methods = {
            edas: ['matrix', 'weights', 'types'],
            topsis: ['matrix', 'weights', 'types'],
            vikor: ['matrix', 'weights', 'types']
            }

    ICRA_matrix = np.array([edas_pref, topsis_pref, vikor_pref]).T
    method_types = np.array([1, 1, -1])

    result = iterative_compromise(methods, ICRA_matrix, method_types)
    assert len(result.all_rankings) == 2

def test_ICRA_invalid_rankings():
    # This will raise a TypeError in the function
    decision_matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    decision_problem_weights = np.ones(decision_matrix.shape[1])/decision_matrix.shape[1]
    decision_problem_types = np.ones(decision_matrix.shape[1])
    
    edas = EDAS()
    topsis = TOPSIS()
    vikor = VIKOR()
    
    edas_pref = edas(decision_matrix, decision_problem_weights, decision_problem_types)
    topsis_pref = topsis(decision_matrix, decision_problem_weights, decision_problem_types)
    vikor_pref = vikor(decision_matrix, decision_problem_weights, decision_problem_types)
    
    methods = 'wrong methods'
    
    ICRA_matrix = np.array([edas_pref, topsis_pref, vikor_pref]).T
    method_types = np.array([1, 1, -1])
    
    with raises(TypeError):
        iterative_compromise(methods, ICRA_matrix, method_types)
    