# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import pymcdm
from ..validator import Validator
from ..utils import memory_guard
from collections.abc import Generator

@memory_guard
def ranking_demotion(matrix: np.ndarray, 
                        initial_ranking: np.ndarray, 
                        method: callable, 
                        call_kwargs: dict, 
                        ranking_descending: bool, 
                        direction: np.ndarray, 
                        step: int | float, 
                        bounds: None | np.ndarray = None, 
                        positions: None | np.ndarray = None, 
                        return_zeros: bool = True, 
                        max_modification: None | int = None) -> list[tuple[int, int, float, int]]:
    """
    Demote alternatives in a decision matrix by adjusting specific criteria values, considering constraints on rankings. 
    With only required parameters given, the analysis is looking for changes that cause demotion for last position in ranking.

    Parameters:
    ------------
    matrix : ndarray
        2D array with a decision matrix containing alternatives in rows and criteria in columns.

    initial_ranking : ndarray
        1D vector representing the initial ranking of alternatives.

    method : callable
        The evaluation function to be used for preference and ranking calculation.
        Should include `matrix` as one of the parameters to pass decision matrix to assessment process.

    call_kwargs : dict
        Dictionary with keyword arguments to be passed to the evaluation function.
        Should include `matrix` as one of the parameters to pass modified decision matrix to assessment process.

    ranking_descending: bool
        Flag determining the direction of alternatives ordering in ranking.
        By setting the flag to True, greater values will have better positions in ranking.

    direction : ndarray
        1D vector specifying the direction of the modification for each column in decision matrix (1 for increase, -1 for decrease).

    step : int | float
        Step size for the modification.

    bounds : None | ndarray, optional, default=None
        Bounds representing the size of the modifications for columns in decision matrix. If None, then modifications are introduced in decision matrix until the last position in the ranking is achieved for a given alternative.

    positions : None | ndarray, optional, default=None
        Target positions for the alternatives in the ranking after modification. 
        If None, the positions are not constrained and the last position is targeted.

    return_zeros : bool, optional, default=True
        Flag determining whether results without noticed demotion in ranking will be returned.

    max_modification : None | int, optional, default=None
        Value determining maximum modification size represented as percent of initial value.
        Required if `bounds` parameter is not given.

    Returns:
    ---------
    List[Tuple[int, int, float, int]]
        A list of tuples containing information about alternative index, criterion index, size of change,
        and achieved new positions based on demotion analysis.

    Examples:
    ----------
    Example 1: Demotion analysis based on the COPRAS method with only required parameters
    
    >>> matrix = np.array([
    ...     [4, 2, 6],
    ...     [7, 3, 2],
    ...     [9, 6, 8]
    ... ])
    >>> weights = np.array([0.4, 0.5, 0.1])
    >>> types = np.array([-1, 1, -1])
    >>> copras = pymcdm.methods.COPRAS()
    >>> pref = copras(matrix, weights, types)
    >>> initial_ranking = copras.rank(pref)
    >>> call_kwargs = {
    ...     "matrix": matrix,
    ...     "weights": weights,
    ...     "types": types
    ... }
    >>> ranking_descending = True
    >>> direction = np.array([1, -1, 1])
    >>> step = 0.5
    >>> max_modification = 100
    >>> results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, max_modification=max_modification)
    >>> for r in results:
    ...     print(r)

    Example 2: Demotion analysis based on the COPRAS method with explicitly defined modification bounds
    
    >>> matrix = np.array([
    ...     [4, 2, 6],
    ...     [7, 3, 2],
    ...     [9, 6, 8]
    ... ])
    >>> weights = np.array([0.4, 0.5, 0.1])
    >>> types = np.array([-1, 1, -1])
    >>> copras = pymcdm.methods.COPRAS()
    >>> initial_ranking = np.array([2, 3, 1])
    >>> call_kwargs = {
    ...     "matrix": matrix,
    ...     "weights": weights,
    ...     "types": types
    ... }
    >>> ranking_descending = True
    >>> direction = np.array([1, -1, 1])
    >>> step = 0.5
    >>> bounds = np.array([15, 0, 20])
    >>> results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, bounds)
    >>> for r in results:
    ...     print(r)

    Example 3: Demotion analysis based on the COPRAS method with explicitly defined modification bounds and targeted positions
    
    >>> matrix = np.array([
    ...     [4, 2, 6],
    ...     [7, 3, 2],
    ...     [9, 6, 8]
    ... ])
    >>> weights = np.array([0.4, 0.5, 0.1])
    >>> types = np.array([-1, 1, -1])
    >>> copras = pymcdm.methods.COPRAS()
    >>> initial_ranking = np.array([2, 3, 1])
    >>> call_kwargs = {
    ...     "matrix": matrix,
    ...     "weights": weights,
    ...     "types": types
    ... }
    >>> ranking_descending = True
    >>> direction = types * -1
    >>> step = 0.5
    >>> bounds = np.array([15, 0, 20])
    >>> positions = np.array([3, 3, 1])
    >>> results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, bounds, positions)
    >>> for r in results:
    ...     print(r)

    Example 4: Demotion analysis based on the COPRAS method with not returned values without noticed demotion
    
    >>> matrix = np.array([
    ...     [4, 2, 6],
    ...     [7, 3, 2],
    ...     [9, 6, 8]
    ... ])
    >>> weights = np.array([0.4, 0.5, 0.1])
    >>> types = np.array([-1, 1, -1])
    >>> copras = pymcdm.methods.COPRAS()
    >>> initial_ranking = np.array([2, 3, 1])
    >>> call_kwargs = {
    ...     "matrix": matrix,
    ...     "weights": weights,
    ...     "types": types
    ... }
    >>> ranking_descending = True
    >>> direction = types * -1
    >>> step = 0.5
    >>> max_modification = 10
    >>> return_zeros = False
    >>> results = ranking_demotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, max_modification=max_modification, return_zeros=return_zeros)
    >>> for r in results:
    ...     print(r)
    """

    def generate_crit_changes(matrix: np.ndarray, alt_idx: int, crit_idx: int, direction: int, step: float, bounds: np.ndarray, max_modification: float) -> Generator[float]:
        # set modification bounds
        if bounds is None:
            limit = matrix[alt_idx, crit_idx] * max_modification * direction[crit_idx]
            crit_values = np.arange(matrix[alt_idx, crit_idx], limit, step * direction[crit_idx])
        else:
            crit_values = np.arange(matrix[alt_idx, crit_idx], bounds[crit_idx], step * direction[crit_idx])

        for change in crit_values:
            yield change

    Validator.is_callable(method, 'method')
    Validator.is_type_valid(matrix, np.ndarray, 'matrix')
    Validator.is_dimension_valid(matrix, 2, 'matrix')
    Validator.is_type_valid(initial_ranking, np.ndarray, 'initial_ranking')
    Validator.is_type_valid(direction, np.ndarray, 'direction')
    Validator.is_in_list(direction, [-1, 1], 'direction')
    Validator.is_shape_equal(matrix.shape[0], initial_ranking.shape[0], custom_message="Number of rows in 'matrix' and length of 'initial_ranking' are different")
    Validator.is_shape_equal(matrix.shape[1], direction.shape[0], custom_message="Number of rows in 'matrix' and length of 'direction' are different")
    if bounds is not None:
        Validator.is_type_valid(bounds, np.ndarray, 'bounds')    
    Validator.is_type_valid(call_kwargs, dict, 'call_kwargs')
    Validator.is_type_valid(ranking_descending, bool, 'ranking_descending')
    Validator.is_type_valid(step, (int, np.integer, float, np.floating), 'step')
    Validator.is_type_valid(return_zeros, bool, 'return_zeros')
    if max_modification is not None:
        Validator.is_type_valid(max_modification, (int, np.integer), 'max_modification')
    if bounds is None and max_modification is None:
        raise TypeError("'max_modification' parameter must be given when 'bounds' is None")
    if positions is not None:
        Validator.is_type_valid(positions, np.ndarray, 'positions')
        Validator.is_shape_equal(matrix.shape[0], positions.shape[0], custom_message="Number of rows in 'matrix' and length of 'positions' are different")
        Validator.is_in_range(positions, 1, positions.shape[0], 'positions')
    Validator.is_key_in_dict(['matrix'], call_kwargs, 'call_kwargs')
    
    # store demoted positions and changes that caused the demotions
    new_positions = np.full((matrix.shape), 0, dtype=int)
    changes = np.full((matrix.shape), 0, dtype=float)

    results = []

    for alt_idx in range(matrix.shape[0]):
        for crit_idx in range(matrix.shape[1]):
            # set desired position to demote given alternative
            if positions is None:
                new_positions[alt_idx, crit_idx] = initial_ranking[alt_idx]
            else:
                new_positions[alt_idx, crit_idx] = positions[alt_idx]

            # put new changed value in decision matrix and assess alternatives
            for change in generate_crit_changes(matrix, alt_idx, crit_idx, direction, step, bounds, max_modification):
                
                new_matrix = matrix.copy()
                new_matrix[alt_idx, crit_idx] = change

                # swap matrix with new changed matrix
                call_kwargs['matrix'] = new_matrix
                try:
                    new_preferences = method(**call_kwargs)
                    new_ranking = pymcdm.helpers.rankdata(new_preferences, ranking_descending)
                except Exception as err:
                    raise ValueError(err)

                # check if position changed and adjust values which cause demotion
                if new_ranking[alt_idx] > new_positions[alt_idx, crit_idx]:
                    new_positions[alt_idx, crit_idx] = new_ranking[alt_idx]
                    changes[alt_idx, crit_idx] = change

                if positions is None:
                    # if last in new ranking then end analysis for given alternative and criterion
                    if new_ranking[alt_idx] == matrix.shape[0]:
                        break
                else:
                    # check if desired position achieved
                    if new_ranking[alt_idx] == positions[alt_idx]:
                        # update values that cause changes
                        if initial_ranking[alt_idx] != matrix.shape[0]:
                            new_positions[alt_idx, crit_idx] = new_ranking[alt_idx]
                            changes[alt_idx, crit_idx] = change
                        break

            if return_zeros:
                results.append([alt_idx, crit_idx, changes[alt_idx, crit_idx], new_positions[alt_idx, crit_idx]])
            else:
                if new_positions[alt_idx, crit_idx] != initial_ranking[alt_idx]:        
                    results.append([alt_idx, crit_idx, changes[alt_idx, crit_idx], new_positions[alt_idx, crit_idx]])
            
    return results
