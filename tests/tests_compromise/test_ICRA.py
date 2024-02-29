# Copyright (c) 2024 Jakub Więckowski

import numpy as np
from pytest import raises
from pysenscraft.compromise.ICRA import iterative_compromise
from pymcdm.methods import COMET, TOPSIS, VIKOR
from pymcdm.methods.comet_tools import MethodExpert

def test_ICRA_default_parameters():
    decision_matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    decision_problem_weights = np.ones(decision_matrix.shape[1])/decision_matrix.shape[1]
    decision_problem_types = np.ones(decision_matrix.shape[1])

    comet = COMET(np.vstack((np.min(decision_matrix, axis=0), np.max(decision_matrix, axis=0))).T, MethodExpert(TOPSIS(), decision_problem_weights, decision_problem_types))
    topsis = TOPSIS()
    vikor = VIKOR()

    comet_pref = comet(decision_matrix)
    topsis_pref = topsis(decision_matrix, decision_problem_weights, decision_problem_types)
    vikor_pref = vikor(decision_matrix, decision_problem_weights, decision_problem_types)

    ## ICRA variables preparation
    methods = {
            COMET: [['np.vstack((np.min(matrix, axis=0), np.max(matrix, axis=0))).T', 'MethodExpert(TOPSIS(), weights, types)'], ['matrix']],
            topsis: ['matrix', 'weights', 'types'],
            vikor: ['matrix', 'weights', 'types']
            }

    ICRA_matrix = np.array([comet_pref, topsis_pref, vikor_pref]).T
    method_types = np.array([1, 1, -1])

    result = iterative_compromise(methods, ICRA_matrix, method_types)
    print(result.all_rankings)
    assert np.array_equal(result.all_rankings, np.array([1, 2, 3]))

def test_ICRA_invalid_rankings():
    # This will raise a TypeError in the function
    decision_matrix = np.array([[4, 1, 6], [2, 6, 3], [9, 5, 7]])
    decision_problem_weights = np.ones(decision_matrix.shape[1])/decision_matrix.shape[1]
    decision_problem_types = np.ones(decision_matrix.shape[1])
    
    comet = COMET(np.vstack((np.min(decision_matrix, axis=0), np.max(decision_matrix, axis=0))).T, MethodExpert(TOPSIS(), decision_problem_weights, decision_problem_types))
    topsis = TOPSIS()
    vikor = VIKOR()
    
    comet_pref = comet(decision_matrix)
    topsis_pref = topsis(decision_matrix, decision_problem_weights, decision_problem_types)
    vikor_pref = vikor(decision_matrix, decision_problem_weights, decision_problem_types)
    
    methods = 'wrong methods'
    
    ICRA_matrix = np.array([comet_pref, topsis_pref, vikor_pref]).T
    method_types = np.array([1, 1, -1])
    
    with raises(TypeError):
        iterative_compromise(methods, ICRA_matrix, method_types)
    