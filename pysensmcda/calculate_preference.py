# Copyright (C) 2024 Bartosz Paradowski

from . import alternative 
from . import criteria 
from . import probabilistic 
import numpy as np
from scipy.stats import rankdata
from .validator import Validator
from .utils import memory_guard

@memory_guard
def calculate_preference(func: callable, 
                            results: list | np.ndarray | tuple, 
                            method: callable, 
                            call_kwargs: dict, 
                            only_preference: bool = True, 
                            method_type: int | None = None) -> np.ndarray | tuple:
    """
    Wrapper for calculating preference depending on the sensitivity analysis function.

    Parameters:
    ------------
        func: callable
            Function for pysensmcda library that was used to acquire results.
        results: depending on func
            Results of the function which should be given as `func`.
        method: callable
            Method that should be used to calculate preferences.
        call_kwargs: dict
            Parameters that should be passed to `method` in order to calculate preferences.
            Used internally:
                `matrix` for decision matrix
                `weights` for criteria weights
        only_preference: bool, optional, default=True
            If `True` only preferences are returned in `ndarray`. If `False` list of tuples that resembles results is returned, where preferences are in the last column.
        method_type: int or None, optional, default=None
            If set, rankings are returned. Supported values: -1 for ascending ranking; 1 for descending ranking

    Returns:
    ---------
        ndarray or list[tuple]
            If only_preference=True, array of preferences calculated for different matrices / weights depending on the type of sensitivity analysis is returned. Else the preferences are appended to results as last column. If `method_type` is set, the rankings are appended to column after preferences.
    
    Examples:
    ----------
    Example 1: Alternative sensitivity analysis - return only preferences
    
        >>> from pymcdm.methods import TOPSIS
        >>> 
        >>> topsis = TOPSIS()
        >>> 
        >>> matrix = np.array([
        >>> [4, 1, 6],
        >>> [2, 6, 3],
        >>> [9, 5, 7],
        >>> ])
        >>> discrete_values = np.array([
        >>>     [[5, 6], [2, 4], [5, 8]],
        >>>     [[3, 5.5], [4], [3.5, 4.5]],
        >>>     [[7, 8], [6], [8, 9]],
        >>> ], dtype='object')
        >>> indexes = np.array([[0, 2], 1], dtype='object')
        >>> results = discrete_modification(matrix, discrete_values, indexes)
        >>> kwargs = {
        >>>     'matrix': matrix,,
        >>>     'weights': np.ones(matrix.shape[0])/matrix.shape[0],
        >>>     'types': np.ones(matrix.shape[0])
        >>> }
        >>> 
        >>> calculate_preference(discrete_modification, results, topsis, kwargs)

    Example 2: Criteria sensitivity analysis - return preferences and rankings
    
        >>> from pysensmcda.criteria import percentage_modification
        >>> 
        >>> weights = np.array([0.3, 0.3, 0.4])
        >>> percentage = 5
        >>> results = percentage_modification(weights, percentage)
        >>> 
        >>> kwargs = {
        >>>     'matrix': np.random.random((10, 3)),
        >>>     'weights': weights,
        >>>     'types': np.ones(3)
        >>> }
        >>> 
        >>> calculate_preference(percentage_modification, results, topsis, kwargs, method_type=1)

    Example 3: Criteria sensitivity analysis - return rankings and aggregated results
    
        >>> from pysensmcda.criteria import percentage_modification
        >>> 
        >>> weights = np.array([0.3, 0.3, 0.4])
        >>> percentage = 5
        >>> results = percentage_modification(weights, percentage)
        >>> 
        >>> kwargs = {
        >>>     'matrix': np.random.random((10, 3)),
        >>>     'weights': weights,
        >>>     'types': np.ones(3)
        >>> }
        >>> 
        >>> calculate_preference(percentage_modification, results, topsis, kwargs, only_preference=False, method_type=1)

    """

    Validator.is_callable(func, 'func')
    Validator.is_type_valid(results, (list, np.ndarray, tuple), 'results')
    Validator.is_callable(method, 'method')
    Validator.is_type_valid(call_kwargs, dict, 'call_kwargs')
    Validator.is_type_valid(only_preference, bool, 'only_preference')
    if method_type is not None:
        Validator.is_type_valid(method_type, (int, np.integer), 'method_type')
        Validator.is_in_list(method_type, [-1, 1], 'method_type')

    def preference_aggregator(results: tuple, 
                                val_list: np.ndarray, 
                                method: callable, 
                                call_kwargs: dict, 
                                param_name: str | list[str], 
                                only_preference: bool = True, 
                                method_type: int | None = None) -> tuple | np.ndarray:
        preferences = []
        for val in val_list:
            if isinstance(param_name, list):
                for p, v in zip(param_name, val):
                    call_kwargs[p] = v
            else:
                call_kwargs[param_name] = val
            pref = method(**call_kwargs)

            if method_type is None:
                preferences.append(pref)
            elif method_type == 1:
                preferences.append([pref, rankdata(-pref)])
            elif method_type == -1:
                preferences.append([pref, rankdata(pref)])

        if only_preference:
            return np.asarray(preferences)
        else:
            for idx, pref in enumerate(preferences):
                results[idx] = tuple([*results[idx], *pref])
            return results

    if func in [alternative.discrete_modification, alternative.percentage_modification, alternative.range_modification]:
        Validator.is_key_in_dict(['weights', 'types'], call_kwargs, 'call_kwargs')
        val_list = np.asarray(results, dtype='object')[:, 3]
        params = 'matrix'
    elif func in [alternative.remove_alternatives]:
        Validator.is_key_in_dict(['weights', 'types'], call_kwargs, 'call_kwargs')
        val_list = np.asarray(results, dtype='object')[:, 1]
        params = 'matrix'
    elif func in [criteria.percentage_modification, criteria.range_modification]:
        Validator.is_key_in_dict(['matrix', 'types'], call_kwargs, 'call_kwargs')
        val_list = np.array(results, dtype='object')[:, 2]
        params = 'weights'
    elif func in [probabilistic.monte_carlo_weights, probabilistic.perturbed_weights]:
        Validator.is_key_in_dict(['matrix', 'types'], call_kwargs, 'call_kwargs')
        val_list = np.array(results)
        params = 'weights'
    elif func in [probabilistic.perturbed_matrix]:
        Validator.is_key_in_dict(['weights', 'types'], call_kwargs, 'call_kwargs')
        val_list = np.array(results)
        params = 'matrix'
    else:
        raise ValueError(f'Function {func.__name__} is not supported for preference calculation.')
    return preference_aggregator(results, val_list, method, call_kwargs, params, only_preference, method_type)

