# Copyright (C) 2023 Jakub Więckowski

import numpy as np

class Validator:

    @staticmethod
    def weights_sum(weights: np.array):
        Validator.check_type(weights, 'weights', np.ndarray)
        if np.round(np.sum(weights), 4) != 1:
            raise ValueError(f'Weights should sum up to 1, not {np.round(np.sum(weights), 4)}')

    @staticmethod
    def check_type(param, param_name: str, param_type):
        if not isinstance(param, param_type):
            if param_name == 'dict key':
                raise TypeError(f"Dictionary key should be type of {param_type}, not {type(param)}")
            else:
                raise TypeError(f"Parameter '{param_name}' should be type of {param_type}, not {type(param)}")

    @staticmethod
    def dict_keys(dict: dict, max: int, key_type: str = int):
        # types
        for key in list(dict.keys()):
            Validator.check_type(key, 'dict key', key_type)

        # keys range
        if any([key > max for key in list(dict.keys())]):
            raise ValueError(f'Keys in dict should not be greater than maximum index of the modified data. Check the index {np.where([key > max for key in list(dict.keys())])[0][0]}')

    @staticmethod
    def percentage_direction(type: str):
        types = ['increase', 'decrease', 'both']
        if type not in types:
            raise ValueError(f'Type {type} is not allowed. Should be one of ({", ".join(types)})')

    @staticmethod
    def percentage_range(percentage: (np.ndarray | int | float)):
        if isinstance(percentage, np.ndarray):
            
                raise ValueError('Percentage value should be greater than 0.')
        elif isinstance(percentage, int) or isinstance(percentage, float):
            if percentage <= 0:
                raise ValueError('Percentage value should be greater than 0.')

    @staticmethod
    def percentage_bounds(percentages: np.ndarray, bounds: np.ndarray):
        if any([p < bounds[0] or p > bounds[1] for p in percentages]):
            raise ValueError(f'Percentages should be placed between range of {bounds[0]} and {bounds[1]}. Check the index {np.where(np.array([p < bounds[0] or p > bounds[1] for p in percentages]) == True)[0][0]}')

    @staticmethod
    def step_bound(step, step_type, bounds: np.ndarray):
        Validator.check_type(step, step_type)

        if step < bounds[0] or step > bounds[1]:
            raise ValueError(f'Given step ({step}) cannot be used properly')

    @staticmethod
    def check_dimension(param, param_name: str, dimension=2):
        Validator.check_type(param, param_name, np.ndarray)
        if param.ndim != dimension:
            raise ValueError(f"The dimension of parameter '{param_name}' should be {dimension}, not {param.ndim}")

    @staticmethod
    def check_length(param, param_name: str, length: int):
        if param.shape[0] != length:
            raise ValueError(f'Parameter "{param_name} should have length of {length}, not {param.shape[0]}"')

    @staticmethod
    def weights_extension(param: np.ndarray):
        Validator.check_type(param, 'weights', np.ndarray)
        # crisp
        if param.ndim == 1:
            if param.dtype.kind != 'f':
                raise TypeError(f"Crisp weight values are wrongly formatted") 
        else: 
            raise TypeError(f"Parameter 'weights' is wrongly formatted")

    @staticmethod
    def matrix_extension(param: np.ndarray):
        Validator.check_type(param, 'matrix', np.ndarray)
        # crisp
        if param.ndim == 2:
            if param.dtype.kind != 'f' or param.dtype.kind == 'i':
                raise TypeError(f"Crisp matrix is wrongly formatted") 
        else: 
            raise TypeError(f"Parameter 'matrix' is wrongly formatted")

    @staticmethod
    def range_weights_size(ranges, weights):
        # check types
        Validator.check_type(weights, 'weights', np.ndarray)
        Validator.check_type(ranges, 'ranges', np.ndarray)

        # check dimension
        Validator.check_dimension(ranges, 'ranges')

        if isinstance(ranges, np.ndarray) and weights.shape[0] < ranges.shape[0]:
            raise ValueError(f"Number ranges should not exceed the weights amount, ranges: {ranges.shape[0]}, weights: {weights.shape[0]}")


    @staticmethod
    def ranges_bounds(ranges: np.ndarray):
        """
        Parameters
        ----------
            ranges: ndarray
                Vectors for criteria to generate weights scenarios with specific value from given ranges

        """

        
        if any([0 < len(r) < 2 for r in ranges]):
            raise ValueError(f'All ranges should have lower and upper bound of weight modification. Check the index {np.where(np.array([0 < len(r) < 2 for r in ranges]) == True)[0][0]}')

        if any([len(r) > 2 for r in ranges]):
            raise ValueError(f'All ranges should have only lower and upper bound of weight modification. Check the index {np.where(np.array([0 < len(r) > 2 for r in ranges]) == True)[0][0]}')

        if any([r[0] - r[1] > 0 if len(r) > 0 else False for r in ranges]):
            raise ValueError(f'All left bounds should be lower than right bounds or equals them. Check the index {np.where(np.array([r[0] - r[1] > 0 if len(r) > 0 else False for r in ranges]) == True)[0][0]}')

    @staticmethod
    def check_combinations(combinations: np.ndarray, weights: np.ndarray):
        # check types
        Validator.check_type(combinations, 'combinations', np.ndarray)
        Validator.check_dimension(combinations, 'combinations', 2)
        
        if any([max(c) >= weights.shape[0] for c in combinations]):
            raise ValueError(f'Weights index cannot not greater than the amount of criteria weights. Check the index {np.where(np.array([max(c) >= weights.shape[0] for c in combinations]) == True)[0][0]}')

    @staticmethod
    def check_minimum_length(param: np.ndarray, param_name: str, length: int = 0):
        if param.shape[0] == length:
            raise ValueError(f'Parameter "{param_name} should have length greater than {length}, currently {param.shape[0]}"')

    @staticmethod
    def check_maximum_length(param: np.ndarray, param_name: str, length: int):
        if param.shape[0] > length:
            raise ValueError(f'Parameter "{param_name} should have length lower than {length}, currently {param.shape[0]}"')

    @staticmethod
    def check_minimum_value(param: np.ndarray, param_name: str, value: (int | float)):
        if np.min(param) < value:
            raise ValueError(f'Minimum value of {param_name} should not be lower than {value}. Check the index {np.argmin(param)}')
    
    @staticmethod
    def check_maximum_value(param: np.ndarray, param_name: str, value: (int | float)):
        if np.min(param) < value:
            raise ValueError(f'Maximum value of {param_name} should not be greater than {value}. Check the index {np.argmax(param)}')

    @staticmethod
    def check_unique(param: np.ndarray, param_name: str):
        if np.unique(param).shape[0] != param.shape[0]:
            raise ValueError(f'{param_name} array should not contain duplicates')

if __name__ == '__main__':

    a = np.array([
        [0.3, 0.3],
        [0.3, 0.3],
        [0.3, 0.4]
    ])
    weights = np.array([0.3, 0.4, 0.3])
    # Validator.weights_sum(weights)
    Validator.range_weights_size(ranges=a, weights=weights)
    Validator.ranges_bounds(ranges=a)
