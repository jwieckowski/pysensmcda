# Copyright (C) 2023 - 2024 Jakub Więckowski, Bartosz Paradowski

import numpy as np
import multiprocessing
from npy_append_array import NpyAppendArray
from collections import deque
import os
from joblib import Parallel, delayed
import tempfile
import tqdm
import pickle

def generate_weights_scenarios(crit_num: int, step: float, precision: int = 4, cores_num: int | None = None, file_name: str | None = None, return_array: bool = False, sequential: bool = False, save_zeros: bool = True) -> list | None:
    """
    Generate scenarios for examining criteria weights based on given criteria number and step of weights space exploration

    Parameters
    ----------
    crit_num : int
        The number of criteria.

    step : float
        The step size used for generating criteria weights.

    precision : int, optional, default=4
        The number of decimal places to round the generated criteria weights.

    cores_num : int or None, optional, default=None
        If provided, the generated scenarios will be generated with given number of processes. 
        If None, all available CPU cores will be used.

    filename : str or None, optional, default=None or 'out'
        Using parallel version files are always created. Temporary for subsequent processes and main file that contains all results. Temporary files are deleted after completion.
        If provided, the generated scenarios will be saved to the specified file. 
        If None, scenarios will be returned as a list.

    return_array : bool, optional, default=False
        Returns results in a format of nd.array (numpy)

    sequential: bool, optional, default=False
        If True code will be run sequentially. Progressbar will be shown and non temporary files created.

    save_zeros: bool, optional, default=True
        If True saves weights vectors where zeros are present.

    Returns
    -------
    list or None
        Depending on return_array parameter, None or nd.array will be returned.

    Examples
    --------
    ### Example 1: parallel without array return
    >>> generate_weights_scenarios(4, 0.1, 3)
    >>> # results will be saved to 'out.npy'

    ### Example 2: parallel with array return
    >>> scenarios = generate_weights_scenarios(4, 0.1, 3, return_array=True)
    >>> print(scenarios)
    >>> [(0.9, 0.1, 0.0, 0.0), (0.8, 0.2, 0.0, 0.0), ...]
    >>> # results will be saved to 'out.npy'

    ### Example 3: parallel with custom file name
    >>> generate_weights_scenarios(4, 0.1, 3, file_name='4crit_0.1')
    >>> # results will be saved to '4crit_0.1.npy'

    ### Example 3: sequential
    >>> generate_weights_scenarios(4, 0.1, 3, sequential=True)
    >>> print(scenarios)
    >>> [(0.9, 0.1, 0.0, 0.0), (0.8, 0.2, 0.0, 0.0), ...]
    >>> # results will not be saved

    ### Example 4: sequential with saving to file
    >>> generate_weights_scenarios(4, 0.1, 3, sequential=True, file_name='4crit_0.1')
    >>> print(scenarios)
    >>> [(0.9, 0.1, 0.0, 0.0), (0.8, 0.2, 0.0, 0.0), ...]
    >>> # results will be saved to '4crit_0.1.npy'
"""
    def weight_gen_worker(stack: deque, worker_list: np.ndarray, no_crit: int, worker_id: int, temp_dir: str) -> None:
        """
        Internal worker function for weights generation on multiple processes with partial results saving during runtime to temporary files
        """
        local_results = []
        with open(f'{temp_dir}\w_gen__temp_{worker_id}.pkl', 'wb') as f:
            while stack:
                n, max_points, current = stack.pop()
                if n == 2:
                    for i in range(max_points + 1):
                        result = [i, (max_points - i)] + current
                        local_results.append(tuple(result))
                        if len(local_results) == 10000:
                            pickle.dump(local_results, f)
                            local_results = []
                elif n == no_crit:
                    for i in worker_list:
                        stack.append((n - 1, i, [max_points - i] + current))
                else:
                    for i in range(max_points + 1):
                        stack.append((n - 1, i, [max_points - i] + current))
            pickle.dump(local_results, f)

    def weight_gen(stack: deque, iters_num: int) -> np.ndarray:
        """
        Internal sequential weights generation function.
        """
        with tqdm.tqdm(total=iters_num) as pbar:
            results = []
            while stack:
                n, max_points, current = stack.pop()

                if n == 2:
                    for i in range(max_points + 1):
                        result = [i, (max_points - i)] + current
                        results.append(tuple(result))
                        pbar.update(1)
                else:
                    for i in range(max_points + 1):
                        stack.append((n - 1, i, [max_points - i] + current))
        return np.round(np.array(results) * step, precision)

    def delete_temp_files(cores_num: int, temp_dir: str) -> None:
        """
        Internal function for temporary files deletion.
        """
        for worker_id in range(cores_num):
            try:
                os.remove(f'{temp_dir}\w_gen__temp_{worker_id}.pkl')
            except:
                pass

    def calc_iterations(max_points: int) -> int:
        """
        Internal function to calculate necessary iterations for each process. Can be used to calculate combinations number.
        """
        def sum_of_seq(n):
            return (n*(n+1))/2
        
        s = []
        for i in range(max_points+2):
            s.append(sum_of_seq(i))
        s = np.array(s)
        for _ in range(crit_num - 4):
            s_copy = s.copy()
            for idx, _ in enumerate(s):
                s_copy[idx] += np.sum(s[0:idx])
            s = s_copy.copy()
        return s

    def run_parallel(cores_num: int, temp_dir: str, file_name: str, save_zeros: bool, return_array: bool) -> None | np.ndarray:
        """
        Internal function for parallel initialization.
        """
        if file_name is None:
            file_name = 'out'

        max_points = int(1 / step)

        stack = deque()
        stack.append((crit_num, max_points, []))

        workers_idx = np.tile([*np.arange(1, cores_num+1), *np.arange(cores_num, 0, -1)], int(np.ceil((max_points+1)/(cores_num*2))))[0:max_points+1]

        Parallel(n_jobs=cores_num)(delayed(weight_gen_worker)(stack.copy(), np.where(workers_idx == i+1)[0], crit_num, i, temp_dir) for i in range(cores_num))

        with NpyAppendArray(f'{file_name}.npy', delete_if_exists=True) as npaa:
            for worker_id in range(cores_num):
                with open(f'{temp_dir}\w_gen__temp_{worker_id}.pkl', "rb") as f:
                    while True:
                        try:
                            temp_results = np.round(np.array(pickle.load(f)) * step, precision)
                            if not save_zeros:
                                npaa.append(temp_results[np.all(temp_results != 0, axis=1)])
                            else:
                                npaa.append(temp_results)
                        except EOFError:
                            break

        if return_array:
            return np.load(f'{file_name}.npy')

        
    def run_sequential(file_name: str, save_zeros: bool, return_array: bool) -> None | np.ndarray:
        """
        Internal function for initialization of sequential run.
        """
        max_points = int(1 / step)
        iters_num = np.sum(calc_iterations(max_points))

        stack = deque()
        stack.append((crit_num, max_points, []))
        results = weight_gen(stack, iters_num)

        if not save_zeros:
            results = results[np.all(results != 0, axis=1)]

        if file_name is not None:
            np.save(f'{file_name}.npy', results)

        if return_array:
            return results

    
    if cores_num is None:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = cores_num
    temp_dir = tempfile.gettempdir()
    if sequential:
        return run_sequential(file_name, save_zeros, return_array)
    else:
        try:
            return run_parallel(num_cores, temp_dir, file_name, save_zeros, return_array)
        finally:
            delete_temp_files(num_cores, temp_dir)