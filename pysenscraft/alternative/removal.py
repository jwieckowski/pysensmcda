# Copyright (C) 2024 Jakub Więckowski

import numpy as np

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
    

if __name__ == '__main__':
    # Example 1, no indexes given
    # by default one alternative removed from data subsequently 
    print('Example 1 ---------------------')
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    results = remove_alternatives(matrix)
    for result in results:
        print(result)

    # Example 2 int index given
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    print('Example 2 ---------------------')
    results = remove_alternatives(matrix, 3)
    for result in results:
        print(result)

    # Example 3 np array indexes given, one-dimensional
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    print('Example 3 ---------------------')
    results = remove_alternatives(matrix, np.array([1, 2, 3]))
    for result in results:
        print(result)

    # Example 4 np array indexes given, elements of array as list
    matrix = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [3, 5, 3, 2],
        [4, 2, 5, 5],
    ])
    print('Example 4 ---------------------')
    results = remove_alternatives(matrix, np.array([[0, 4], 2, 3], dtype='object'))
    for result in results:
        print(result)

