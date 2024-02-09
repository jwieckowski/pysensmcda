# Copyright (C) 2023 - 2024 Jakub Więckowski

from tqdm import tqdm
import threading
import os

def generate_weights_scenarios(n: int, step: float, precision: int = 3, threads_num: None | int = None, filename: str = None):
    """
    Generate scenarios for examining criteria weights based on given criteria number and step of weights space exploration

    Parameters
    ----------
    n : int
        The number of criteria.

    step : float
        The step size used for generating criteria weights.

    precision : int, optional, default=3
        The number of decimal places to round the generated criteria weights.

    threads_num : int or None, optional, default=None
        If provided, the generated scenarios will be generated with given number of threads. 
        If None, all threads from CPU will be used

    filename : str or None, optional, default=None
        If provided, the generated scenarios will be saved to the specified file. 
        If None, scenarios will be returned as a list.

    Returns
    -------
    list or None
        If filename is None, the function returns a list of scenarios, where each scenario is represented as a list of adjusted criteria weights. If filename is provided, no return value.

    Notes
    -----
    This function generates scenarios by iteratively adjusting criteria weights for a given number of criteria and step size. It starts with the maximum value for one criterion and decreases it by the step size while increasing the values of other criteria to maintain the sum. The scenarios can be saved to a file or returned as a list.

    ## Examples
    --------
    ### Example 1: Generate and save scenarios to a file
    >>> generate_weights_scenarios(3, 0.2, 2, "scenarios.txt")

    ### Example 2: Generate scenarios and return as a list
    >>> scenarios = generate_weights_scenarios(4, 0.1, 3)
    >>> print(scenarios)
    >>> [(0.9, 0.1, 0.0, 0.0), (0.8, 0.2, 0.0, 0.0), ...]

    """

    def worker(stack, results, step, precision):
        """
        Worker function for adjusting criteria weights and generating scenarios in parallel.

        Parameters
        ----------
        stack : list
            A stack containing criteria adjustment tasks to be performed.

        results : list
            A list to store the generated scenarios.

        step : float
            The step size to adjust criteria weights.

        precision : int
            The number of decimal places to round the generated criteria weights.

        """

        while True:
            try:
                n, max_points, current = stack.pop()
            except IndexError:
                break

            if n == 2:
                for i in range(max_points + 1):
                    result = [round(i * step, precision), round((max_points - i) * step, precision)] + [round(x * step, precision) for x in current]

                    results.append(tuple(result))
            else:
                for i in range(max_points + 1):
                    stack.append((n - 1, i, [max_points - i] + current))


    max_points = int(1 / step)

    results = []
    stack = [(n, max_points, [])]
    
    if threads_num is None:
        num_cores = os.cpu_count()
        num_threads = min(num_cores * 2, 8)
    else:
        num_threads = threads_num

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(stack, results, step, precision))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if filename:
        with open(f'{filename}', 'a+') as outfile:
            for weights in tqdm(results):
                outfile.write(weights)
                outfile.write('\n')
    else:
        return results
