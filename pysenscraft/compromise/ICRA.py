# Copyright (C) 2024 Bartosz Paradowski

import inspect
import numpy as np
from scipy.stats import rankdata
from dataclasses import dataclass
from pymcdm.correlations import weighted_spearman

@dataclass
class ICRAResults:
    """Class for keeping ICRA results."""
    initial_preferences: np.ndarray
    initial_rankings: np.ndarray
    final_preferences: np.ndarray
    final_rankings: np.ndarray
    iters_number: int
    all_preferences: np.ndarray
    all_rankings: np.ndarray
    all_corrs: np.ndarray

def iterative_compromise(methods: dict, preferences: np.ndarray, rankings: np.ndarray, types, corr_coef: callable=weighted_spearman, max_iters: int=1000, compromise_weights: np.ndarray | None=None) -> ICRAResults:
    """ Iterative Compromise Ranking Analysis (ICRA).
        ---------------------------------------------

        The ICRA is an approach that provides decision maker a compromise ranking for specific set of considered rankings using specific mutli-criteria decision-making methods ICRA main advantage is using preference values and weights for each expert / decision-making method

        Parameters
        ----------
        methods: array of callable
            Array that should contain callable functions for all considered MCDMs. The function should return preference value. Support for pymcdm methods.
        preferences: ndarray
            Decision matrix consisting of preferences. Alternatives are in rows and Criteria (methods / experts) are in columns
        rankings: ndarray
        types: ndarray
            Array with definitions of method types types: 1 if method ranks in ascending order and -1 if the method ranks in descending order
        corr_coef: callable, optional, default=pymcdm.weighted_spearman
            Function which will be used to check similarity of rankings while achieving compromise.
        max_iters: int, optional, default=1000
            Maximum iterations number to seek compromise.
        compromise_weights: ndarray, optional, default=equal weights
            Weights of methods in compromise seeking. Sum of the weights should be 1. (e.g. sum(weights) == 1).

        Returns
        -------
        results: object
            ICRAResults object is returned, which consists of:
            initial_preferences: ndarray
            initial_rankings: ndarray
            final_preferences: ndarray
            final_rankings: ndarray
            iters_number: int
            all_preferences: ndarray
            all_rankings: ndarray
            all_corrs: ndarray

        Example
        --------
        >>> ## Initial decision problem evaluation - random problem
        >>> decision_matrix = np.random.random((7, 5))
        >>> 
        >>> decision_problem_weights = np.ones(decision_matrix.shape[1])/decision_matrix.shape[1]
        >>> decision_problem_types = np.ones(decision_matrix.shape[1])
        >>> 
        >>> comet = COMET(np.vstack((np.min(decision_matrix, axis=0), np.max(decision_matrix, axis=0))).T, MethodExpert(TOPSIS(), decision_problem_weights, decision_problem_types))
        >>> topsis = TOPSIS()
        >>> vikor = VIKOR()
        >>> 
        >>> comet_pref = comet(decision_matrix)
        >>> topsis_pref = topsis(decision_matrix, decision_problem_weights, decision_problem_types)
        >>> vikor_pref = vikor(decision_matrix, decision_problem_weights, decision_problem_types)
        >>> 
        >>> ## ICRA variables preparation
        >>> methods = {
        >>>     COMET: [['np.vstack((np.min(matrix, axis=0), np.max(matrix, axis=0))).T', 
        >>>                     'MethodExpert(TOPSIS(), weights, types)'], 
        >>>             ['matrix']],
        >>>     topsis: ['matrix', 'weights', 'types'],
        >>>     vikor: ['matrix', 'weights', 'types']
        >>>     }
        >>> 
        >>> ICRA_matrix = np.array([comet_pref, topsis_pref, vikor_pref]).T
        >>> method_types = np.array([1, 1, -1])
        >>> 
        >>> result = iterative_compromise(methods, ICRA_matrix, np.array([comet.rank(comet_pref), topsis.rank(topsis_pref), vikor.rank(vikor_pref)]), method_types)
        >>> print(result.all_rankings)
    """

    def assign_results(results: ICRAResults, new_preferences: list, new_rankings: list, all_preferences: list, all_rankings: list, all_corrs: list):
        results.final_preferences = np.array(new_preferences)
        results.final_rankings = np.array(new_rankings)
        results.all_preferences = np.array(all_preferences)
        results.all_rankings = np.array(all_rankings)
        results.all_corrs = np.array(all_corrs)

    results = ICRAResults(preferences, rankings, np.array([]), np.array([]), 0, np.array([]), np.array([]), np.array([]))
    all_corrs = []
    all_preferences = []
    all_rankings = []

    matrix = preferences

    if compromise_weights is None:
        weights = np.ones(matrix.shape[1])/matrix.shape[1]
    else:
        weights = compromise_weights

    prev_rankings = rankings
    corrs = np.zeros(matrix.shape[1])
    internal_corrs = np.zeros(matrix.shape[1]-1)
    all_preferences.append(matrix)
    all_rankings.append(rankings)

    local_scope = locals()

    while np.any(internal_corrs != 1):
        results.iters_number += 1

        new_preferences = []
        new_rankings = []
        
        for idx, key in enumerate(methods.keys()):
            if inspect.isclass(key):
                class_params = methods[key][0]
                method_params = methods[key][1]
                method = key(*[eval(param, globals(), local_scope) for param in class_params])
            else:
                method_params = methods[key]
                method = key
            pref = method(*[eval(param, globals(), local_scope) for param in method_params])
            new_preferences.append(pref)
            if types[idx] == 1:
                new_rankings.append(rankdata(pref * -1))
            else:
                new_rankings.append(rankdata(pref))

        for i in range(len(new_rankings)-1):
            internal_corrs[i] = corr_coef(new_rankings[i], new_rankings[i+1])

        for i, new_ranking in enumerate(new_rankings):
            corrs[i] = corr_coef(new_ranking, np.asarray(prev_rankings)[i])

        prev_rankings = new_rankings
        matrix = np.vstack(new_preferences).T
        local_scope = locals()
        
        all_corrs.append(corrs)
        all_preferences.append(matrix)
        all_rankings.append(new_rankings)

        if results.iters_number > max_iters:
            print(f"Compromise not obtained in {max_iters} iterations.")
            break

    assign_results(results, new_preferences, new_rankings, all_preferences, all_rankings, all_corrs)
    return results
