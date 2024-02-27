# Copyright (C) 2024 Bartosz Paradowski
from .alternative import discrete_modification, percentage_modification, range_modification, remove_alternatives
import numpy as np

def calculate_preference(func: callable, results, method: callable, kwargs: dict):
    """
    Wrapper for calculating preference depening on the sensitivity analysis function.

    Parameters
    ----------
        func: callable
            Function for pysenscraft library that was used to acquire results.
        results: depending on func
            Results of the function which should be given as `func`.
        method: callable
            Method that should be used to calculate preferences.
        kwargs: dict
            Parameters that should be passed to `method` in order to calculate preferences.

    Examples
    --------
    ### Example 1: Alternative sensitivity analysis
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
        >>>     'weights': np.ones(matrix.shape[0])/matrix.shape[0],
        >>>     'types': np.ones(matrix.shape[0])
        >>> }
        >>> 
        >>> calculate_preference(discrete_modification, results, topsis, kwargs)

    Returns
    -------
        ndarray
            Array of preferences calculated for different matrices / weights depending on the type of sensitivity analysis.
    
    """
    def preference_aggregator(val_list, method: callable, kwargs: dict, param_name: str):
        preferences = []
        for val in val_list:
            kwargs[param_name] = val
            preferences.append(method(**kwargs))
        return np.asarray(preferences)

    if func in [discrete_modification, percentage_modification, range_modification]:
        matrices = np.asarray(results, dtype='object')[:, 3]
        return preference_aggregator(matrices, method, kwargs, 'matrix')
    elif func in [remove_alternatives]:
        matrices = np.asarray(results, dtype='object')[:, 1]
        return preference_aggregator(matrices, method, kwargs, 'matrix')

