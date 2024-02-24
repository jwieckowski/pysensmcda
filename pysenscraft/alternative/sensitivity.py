# Copyright (C) 2024 Jakub Więckowski

# from ..abc import SensitivityAnalysis
from abc import ABC, abstractmethod
import numpy as np

# functions
# from .discrete import discrete_modification
# from .percentage import percentage_modification
# from .range import range_modification
# from .removal import remove_alternatives

class SensitivityAnalysis(ABC):
    """
    Abstract base class for Sensitivity Analysis.

    Methods:
    --------
    - __init__(self)
        Constructor for the SensitivityAnalysis class.

    - change_data(self):
        Change data for sensitivity analysis

    - print_results(self)
        Print formatted results from sensitivity analysis
    """

    def __init__(self):
        pass

    @abstractmethod
    def change_data(self):
        pass

    @abstractmethod
    def perform_analysis(self):
        pass

    @abstractmethod
    def print_results(self):
        pass

def remove_alternatives(matrix: np.ndarray, indexes: None | int | np.ndarray = None):
    """
    Remove one or more alternatives from a decision matrix.

    Parameters
    ----------
    matrix : ndarray
        2D array with decision matrix containing multiple criteria and alternatives.

    indexes : None | int | ndarray, optional, default=None
        Index or array of indexes specifying which alternative to remove. 
        If None, one alternative will be subsequently removed by default

    Returns
    -------
    List[Tuple[int, ndarray]]
        A list of tuples containing information about new decision matrix.

    ## Examples
    --------
    ### Example 1: 
    TODO
    ### Example 2: 
    TODO
    ### Example 3: 
    TODO
    ### Example 4: 
    TODO
    """
    
    matrix = np.array(matrix)

    # # matrix dimension - can be done not only for crisp matrix
    # if matrix.ndim != 2:
    #     raise ValueError('Matrix should be given as at two dimensional array')

    alt_indexes = None

    if indexes is None:
        # generate vector of subsequent alternative indexes to remove
        alt_indexes = np.arange(0, matrix.shape[1])
    
    if isinstance(indexes, int):
        if indexes >= matrix.shape[0] or indexes < 0:
            raise IndexError(f'Given index ({indexes}) out of range')
        alt_indexes = np.array([indexes])

    if isinstance(indexes, np.ndarray):
        for c_idx in indexes:
            if isinstance(c_idx, int):
                if c_idx < 0 or c_idx >= matrix.shape[0]:
                    raise IndexError(f'Given index ({indexes}) out of range')
            elif isinstance(c_idx, list):
                if any([idx < 0 or idx >= matrix.shape[0] for idx in c_idx]):
                    raise IndexError(f'Given indexes ({c_idx}) out of range')

        alt_indexes = indexes

    data = []
    # remove row in decision matrix
    for i, a_idx in enumerate(alt_indexes):
        try:
            new_matrix = np.delete(matrix, a_idx, axis=0)

            data.append((a_idx, new_matrix))
        except:
            raise ValueError(f'Calculation error. Check elements in {i} index')

    return data
    


class ModificationStrategy(ABC):
    """
    Abstract base class for modification strategies.
    """

    @abstractmethod
    def modify(self, *args, **kwargs):
        pass

# class DiscreteModificationStrategy(ModificationStrategy):
#     def __init__(self, matrix: np.ndarray, discrete_values: np.ndarray, indexes: None | np.ndarray = None):
#         self.matrix = matrix
#         self.discrete_values = discrete_values
#         self.indexes = indexes

#     def modify(self):
#         return discrete_modification(self.matrix, self.discrete_values, self.indexes)

# class PercentageModificationStrategy(ModificationStrategy):
#     def __init__(self, matrix: np.ndarray, percentages: int | np.ndarray, direction: None | np.ndarray = None, indexes: None | np.ndarray = None, step: int | np.ndarray = 1):
#         self.matrix = matrix
#         self.percentages = percentages
#         self.direction = direction
#         self.indexes = indexes
#         self.step = step

#     def modify(self):
#         return percentage_modification(self.matrix, self.percentages, self.direction, self.indexes, self.step)

# class RangeModificationStrategy(ModificationStrategy):
#     def __init__(self, matrix: np.ndarray, range_values: np.ndarray, indexes: None | np.ndarray = None, step: int | float | np.ndarray = 1):
#         self.matrix = matrix
#         self.range_values = range_values
#         self.indexes = indexes
#         self.step = step

#     def modify(self):
#         return range_modification(self.matrix, self.range_values, self.indexes, self.step)

class RemovalModificationStrategy(ModificationStrategy):
    def __init__(self, matrix: np.ndarray, indexes: None | int | np.ndarray = None):
        self.matrix = matrix
        self.indexes = indexes

    def modify(self):
        return remove_alternatives(self.matrix, self.indexes)

class AlternativeSensitivity(SensitivityAnalysis):
    """
    Abstract base class for Sensitivity Analysis.

    Methods:
    --------
    - __init__(self)
        Constructor for the SensitivityAnalysis class.

    - print_results(self)
        Print formatted results from sensitivity analysis
    """

    def __init__(self, strategy: ModificationStrategy):
        self.strategy = strategy

    def change_data(self):
        return super().change_data()

    def perform_analysis(self):
        # Use the specified modification strategy
        modified_data = self.strategy.modify()

        return modified_data

    def print_results(self):
        # return super().print_results()
        return


# Example
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
strategy = RemovalModificationStrategy(matrix)
sensitivity_analysis = AlternativeSensitivity(strategy)

data = sensitivity_analysis.perform_analysis()
for d in data:
    print(d)
