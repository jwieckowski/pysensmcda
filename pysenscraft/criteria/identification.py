# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import pymcdm
import pymcdm.correlations as corr
from pymcdm.weights import equal_weights

def relevance_identification(method: callable, call_kwargs: dict, ranking_descending: bool, excluded_criteria: int = 1, precision: int = 6):
    """
    The core idea behind this method is to iteratively exclude a specified number of criteria and evaluate the impact on the ranking of alternatives. 
    By systematically excluding different criteria and analyzing the resulting changes in the ranking, the function aims to identify which criteria significantly influence the final decision.

    Parameters
    ----------
    method : callable
        The evaluation function to be used for preference and ranking calculation.
        Should include `matrix`, `weights`, and `types` as one of the parameters to pass modified input data to the assessment process.

    call_kwargs : dict
        Dictionary with keyword arguments to be passed to the evaluation function.
        Should include `matrix`, `weights`, and `types` as one of the parameters to pass modified input data to the assessment process.

    ranking_descending: bool
        Flag determining the direction of alternatives ordering in ranking.
        By setting the flag to True, greater values will have better positions in the ranking.

    excluded_criteria: int, optional, default=1
        Number of criteria to be excluded in each iteration.

    precision: int, optional, default=6
        Precision for rounding the results.

    Returns
    -------
    list
        List of tuples containing information about the relevance identification process.
        Each tuple includes the excluded criteria indices, Weighted Spearman coefficient value, Weighted Similarity coefficient value, Pearson coefficient value, distance calculated as the sum of the Euclidean distance between preferences, and the modified matrix.

    Examples
    --------
    ### Example 1: Identify relevant criteria in a custom matrix using TOPSIS method
    >>> matrix = np.array([
    ...    [4, 3, 5, 7],
    ...    [7, 4, 2, 4],
    ...    [9, 5, 7, 3],
    ...    [3, 5, 6, 3]
    ... ])
    >>> criteria_types = np.array([1, 1, -1])
    >>> weights = equal_weights(matrix)
    >>> topsis = pm.TOPSIS(normalization_function=norm.vector_normalization)
    >>> call_kwargs = {
    ...     'matrix': matrix,
    ...     'weights': weights,
    ...     'types': criteria_types
    ... }
    >>> results = relevance_identification(topsis, call_kwargs, ranking_descending=True)
    >>> for r in results:
    ...     print(r)

    ### Example 2: Identify relevant criteria in using TOPSIS method and excluding 3 criteria in the problem
    >>> matrix = np.array([
    ...    [106.78,  6.75,  2.  , 220.  ,  6.  ,  1.  , 52.  , 455.5 ,  8.9 , 36.8 ],
    ...    [ 86.37,  7.12,  3.  , 400.  , 10.  ,  0.  , 20.  , 336.5 ,  7.2 , 29.8 ],
    ...    [104.85,  6.95, 60.  , 220.  ,  7.  ,  1.  , 60.  , 416.  ,  8.7 , 36.2 ],
    ...    [ 46.6 ,  6.04,  1.  , 220.  ,  3.  ,  0.  , 50.  , 277.  ,  3.9 , 16.  ],
    ...    [ 69.18,  7.05, 33.16, 220.  ,  8.  ,  0.  , 35.49, 364.79,  5.39, 33.71],
    ...    [ 66.48,  6.06, 26.32, 220.  ,  6.53,  0.  , 34.82, 304.02,  4.67, 27.07],
    ...    [ 74.48,  6.61, 48.25, 400.  ,  4.76,  1.  , 44.19, 349.45,  4.93, 28.89],
    ...    [ 73.67,  6.06, 19.54, 400.  ,  3.19,  0.  , 46.41, 354.65,  8.01, 21.09],
    ...    [100.58,  6.37, 39.27, 220.  ,  8.43,  1.  , 22.07, 449.42,  7.89, 17.62],
    ...    [ 94.81,  6.13, 50.58, 220.  ,  4.18,  1.  , 21.14, 450.88,  5.12, 17.3 ],
    ...    [ 48.93,  7.12, 21.48, 220.  ,  5.47,  1.  , 55.72, 454.71,  8.39, 19.16],
    ...    [ 74.75,  6.58,  7.08, 400.  ,  9.9 ,  1.  , 26.01, 455.17,  4.78, 18.44]
    ... ])
    >>> criteria_types = np.array([1, 1, -1, 1, -1, -1, 1, -1, -1, 1])
    >>> weights = equal_weights(matrix)
    >>> topsis = pm.TOPSIS(normalization_function=norm.vector_normalization)
    >>> call_kwargs = {
    ...     'matrix': matrix,
    ...     'weights': weights,
    ...     'types': criteria_types
    ... }
    >>> results = relevance_identification(topsis, call_kwargs, ranking_descending=True, excluded_criteria=3)
    >>> for r in results:
    ...     print(r)
    """
    
    if not callable(method):
        raise TypeError('Method should be callable')

    if not isinstance(excluded_criteria, int) or excluded_criteria <= 0:
        raise ValueError('`excluded_criteria` should be a positive integer')

    for key in ['matrix', 'weights', 'types']:
        if key not in list(call_kwargs.keys()):
            raise ValueError(f'Call kwargs dictionary should include `{key}` as one of the keys')

    try:
        initial_matrix = np.array(call_kwargs['matrix'].copy())
    except:
        raise TypeError('Matrix in `call_kwargs` should be given as numpy array')

    if initial_matrix.ndim != 2:
        raise ValueError('Matrix in `call_kwargs` should be a 2D array')

    try:
        types = np.array(call_kwargs['types'].copy())
    except:
        raise TypeError('Types in `call_kwargs` should be given as numpy array')

    if types.ndim != 1:
        raise ValueError('Types in `call_kwargs` should be a 2D array')

    call_kwargs['weights'] = equal_weights(initial_matrix)

    results = []

    excluded = []
    for _ in range(excluded_criteria):
        # remove already excluded criteria
        new_matrix = np.delete(initial_matrix, excluded, axis=1)
        new_weights = equal_weights(new_matrix)
        new_types = np.delete(types, excluded, axis=0)

        # update call parameters
        call_kwargs['matrix'] = new_matrix
        call_kwargs['weights'] = new_weights
        call_kwargs['types'] = new_types 

        # calculate the initial evaluation
        try:
            ref_preferences = method(**call_kwargs)
            ref_ranking = pymcdm.helpers.rankdata(ref_preferences, ranking_descending)
        except Exception as err:
            raise ValueError(err)

        # index of minimum change
        min_change_idx = None
        min_distance = 1 * new_matrix.shape[0]

        temp_results = []
        for i in range(initial_matrix.shape[1]):
            if i in excluded:
                continue
            index = i - len([e_idx for e_idx in excluded if e_idx < i]) if any([i > e_idx for e_idx in excluded]) else i

            # modify input data
            modified_matrix = np.delete(new_matrix, index, axis=1)
            modified_weights = equal_weights(modified_matrix)
            modified_types = np.delete(new_types, index, axis=0)

            # update call parameters
            call_kwargs['matrix'] = modified_matrix
            call_kwargs['weights'] = modified_weights
            call_kwargs['types'] = modified_types 

            # calculate results
            new_preferences = method(**call_kwargs)
            new_ranking = pymcdm.helpers.rankdata(new_preferences, ranking_descending)
            rw = np.round(corr.rw(ref_ranking, new_ranking), precision)
            ws = np.round(corr.ws(ref_ranking, new_ranking), precision)
            rs = np.round(corr.rs(ref_ranking, new_ranking), precision)
            distance = np.round(np.sum(np.sqrt((ref_preferences - new_preferences)**2)), precision)

            temp_results.append((tuple(excluded + [i]), rw, ws, rs, distance, modified_matrix))

            # update minimum change index
            if distance < min_distance:
                min_distance = distance
                min_change_idx = index

        excluded.append(min_change_idx)
        results.append(temp_results[min_change_idx])

    return results

