# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import pymcdm
import inspect

def ranking_promotion(matrix: np.ndarray, initial_ranking: np.ndarray, method: object, call_kwargs: dict, direction: np.ndarray, step: int | float, bounds: None | np.ndarray = None, positions: None | np.ndarray = None):
    """
    Promote alternatives in a decision matrix by adjusting specific criteria values, considering constraints on rankings. 
    With only required parameters given, the analysis is looking for changes that cause promotion for 1st position in ranking.

    Parameters
    ----------
    matrix : ndarray
        2D array with a decision matrix containing alternatives in rows and criteria in columns.

    initial_ranking : ndarray
        1D vector representing the initial ranking of alternatives.

    method : object
        An instance of the method from the pymcdm package to be used for preference and ranking calculation.

    call_kwargs : dict
        Dictionary with keyword arguments to be passed to the pymcdm method object.

    direction : ndarray
        1D vector specifying the direction of the modification for each column in decision matrix (1 for increase, -1 for decrease).

    step : int | float
        Step size for the modification.

    bounds : None | ndarray, optional, default=None
        Bounds representing the size of the modifications for columns in decision matrix. If None, then modifications are introduced in decision matrix until the 1st position in the ranking is achieved for a given alternative.

    positions : None | ndarray, optional, default=None
        Target positions for the alternatives in the ranking after modification. 
        If None, the positions are not constrained and the 1st position is targeted.

    Returns
    -------
    List[Tuple[int, int, float, int]]
        A list of tuples containing information about alternative index, criterion index, size of change,
        and achieved new positions based on promotion analysis.

    ## Examples
    --------
    ### Example 1: Promotion analysis based on the COPRAS method with only required parameters
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
    >>> direction = np.array([-1, 1, -1])
    >>> step = 0.5
    >>> results = ranking_promotion(matrix, initial_ranking, copras, call_kwargs, direction, step)
    >>> for r in results:
    ...     print(r)

    ### Example 2: Promotion analysis based on the COPRAS method with explicitly defined modification bounds
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
    >>> direction = np.array([-1, 1, -1])
    >>> step = 0.5
    >>> bounds = np.array([1, 15, 0])
    >>> results = ranking_promotion(matrix, initial_ranking, copras, call_kwargs, direction, step, bounds)
    >>> for r in results:
    ...     print(r)

    ### Example 3: Promotion analysis based on the COPRAS method with explicitly defined modification bounds and targeted positions
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
    >>> step = 0.5
    >>> bounds = np.array([1, 15, 0])
    >>> positions = np.array([1, 2, 1])
    >>> results = ranking_promotion(matrix, initial_ranking, copras, call_kwargs, direction, step, bounds, positions)
    >>> for r in results:
    ...     print(r)
    """

    methods_types = [obj for _, obj in inspect.getmembers(pymcdm.methods) if inspect.isclass(obj)]
    if type(method) not in methods_types:
        raise TypeError('Method object should be one of the pymcdm package methods')

    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix should be a numpy array type')
        
    if not isinstance(initial_ranking, np.ndarray):
        raise TypeError('Initial ranking should be a numpy array type')
    
    if not isinstance(direction, np.ndarray):
        raise TypeError('Direction should be a numpy array type')

    if any([d not in [-1, 1] for d in direction]):
        raise ValueError('Direction vector should contain only values 1 or -1')

    if matrix.ndim != 2:
        raise ValueError('Matrix should be a 2D array')

    if matrix.shape[1] != initial_ranking.shape[0]:
        raise ValueError("Number of alternatives in matrix and positions in initial ranking should be the same")
    
    if matrix.shape[1] != direction.shape[0]:
        raise ValueError("Number of alternatives in matrix and length of direction should be the same")

    if bounds is not None and not isinstance(bounds, np.ndarray):
        raise TypeError('Bounds should be a numpy array type')

    if positions is not None:
        if not isinstance(positions, np.ndarray):
            raise TypeError('Positions should be a numpy array type')
    
        if matrix.shape[1] != positions.shape[0]:
            raise ValueError("Number of alternatives in matrix and length of positions should be the same")

        if any([p <= 0 or p > positions.shape[0] for p in positions]):
            raise ValueError('Values in positions should not exceed possible ranking placements')

    # store promoted positions and changes that caused the promotions
    new_positions = np.full((matrix.shape), 0, dtype=int)
    changes = np.full((matrix.shape), 0, dtype=float)

    results = []

    for alt_idx in range(matrix.shape[0]):
        for crit_idx in range(matrix.shape[1]):
            # set desired position to promote given alternative
            if positions is None:
                new_positions[alt_idx, crit_idx] = initial_ranking[alt_idx]
            else:
                new_positions[alt_idx, crit_idx] = positions[alt_idx]

            # set modification bounds
            if bounds is None:
                # to tysiąc ustawione tak na sztywno żeby  była jakaś wartośc graniczna mimo wszystko dla których modyfikacje sie odbywają, może masz jakis inny pomysł jak to rozwiązać
                crit_changes = np.arange(matrix[alt_idx, crit_idx], matrix[alt_idx, crit_idx] * 1000 * direction[crit_idx], step * direction[crit_idx])
            else:
                crit_changes = np.arange(matrix[alt_idx, crit_idx], bounds[crit_idx], step * direction[crit_idx])

            # put new changed value in decision matrix and assess alternatives
            for change in crit_changes:
                
                new_matrix = matrix.copy()
                new_matrix[alt_idx, crit_idx] = change

                # swap matrix with new changed matrix
                call_kwargs['matrix'] = new_matrix
                try:
                    new_preferences = method(**call_kwargs)
                    new_ranking = method.rank(new_preferences)
                except Exception as err:
                    raise ValueError(err)

                # check if position changed and adjust values which cause promotion
                if new_ranking[alt_idx] < new_positions[alt_idx, crit_idx]:
                    new_positions[alt_idx, crit_idx] = new_ranking[alt_idx]
                    changes[alt_idx, crit_idx] = change

                if positions is None:
                    # if first in new ranking then end analysis for given alternative and criterion
                    if new_ranking[alt_idx] == 1:
                        break
                else:
                    # check if desired position achieved
                    if new_ranking[alt_idx] == positions[alt_idx]:
                        # update values that cause changes
                        if initial_ranking[alt_idx] != 1:
                            new_positions[alt_idx, crit_idx] = new_ranking[alt_idx]
                            changes[alt_idx, crit_idx] = change
                        break

            results.append([alt_idx, crit_idx, changes[alt_idx, crit_idx], new_positions[alt_idx, crit_idx]])
                    
    return results
