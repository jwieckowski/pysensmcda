# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def interval_plot(results, figsize: tuple =(12, 8), xlabel: str ='Criteria Weights',
                                label_fontsize: int=12, title: str ='Weights ranges', line_color: str ='blue',
                                line_width: int=2, grid_alpha: float =0.7, max_elements_in_row: int = 4, shareX: bool = True, ax: mpl.axes.Axes|None=None):
    """
    Visualize the modified weights with range changes as intervals in a plot.

    Parameters
    ----------
    results : List[Tuple[int, Union[float, Tuple[float, ...]], ndarray]]
        List of tuples containing information about modified criteria index, 
        range change, and the resulting criteria weights.

    figsize : tuple, optional, default=(12, 8)
        Size of the figure.

    xlabel : str, optional, default='Criteria Weights'
        Label for the x-axis.

    label_fontsize : int, optional, default=12
        Fontsize for axis labels.

    title : str, optional, default='Modified Weights with Range Changes'
        Title of the plot.

    line_color : str, optional, default='blue'
        Color of the lines representing intervals.

    line_width : int, optional, default=2
        Width of the lines representing intervals.

    grid_alpha : float, optional, default=0.7
        Alpha (transparency) value for the grid.

    max_elements_in_row : int, optional, default=4
        Maximum number of elements in each row for the 'grid' layout.
    
    shareX : bool, optional, default=True
        Whether to share X-axis among subplots.

    ax: Axis | None, optional, default=None
        Matplotlib Axis to draw on. If None, current axis is used.

    # Example Usage
    --------------
    ### Example 1: Interval plot based on weights range modification
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> range_values = np.array([[0.25, 0.3], [0.3, 0.35], [0.37, 0.43]])
    >>> results = range_modification(weights, range_values, indexes=np.array([0, 1, 2, [0, 2]], dtype='object'))
    >>> interval_plot(results, max_elements_in_row=2)

    ### Example 2: Interval plot based on weights percentage modification
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentages = np.array([5, 5, 5])
    >>> indexes = np.array([[0, 1], 2], dtype='object')
    >>> results = percentage_modification(weights, percentages, indexes=indexes)
    >>> interval_plot(results, max_elements_in_row=2)

    """


    results_cases = []
    for r in results:
        if r[0] not in results_cases:
            results_cases.append(r[0])
    cases_num = len(results_cases)

    num_rows = int(np.ceil(cases_num / max_elements_in_row))
    num_cols = min(cases_num, max_elements_in_row)
    if ax is None:
        fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=shareX)

    for idx, case in enumerate(results_cases):
        row_idx = idx // max_elements_in_row
        col_idx = idx % max_elements_in_row

        data = []
        for result in results:
            if result[0] == case:  
                data.append(list(result[2]))

        axes_idx = (row_idx, col_idx) if num_rows > 1 else (col_idx)

        intervals = np.vstack([np.min(data, axis=0), np.max(data, axis=0)]).T
        for idx, interval in enumerate(intervals):
            ax[axes_idx].plot(interval, [idx, idx], color=line_color, linewidth=line_width)

        if isinstance(case, int):
            criteria_labels = f'$C_{{{case + 1}}}$'
        else:
            criteria_labels = ''
            for i in case:
                criteria_labels += f'$C_{{{i + 1}}}$ '
        
        ax[axes_idx].set_title(f'Change in {criteria_labels}', fontsize=label_fontsize)
        ax[axes_idx].set_yticks(np.arange(cases_num))
        ax[axes_idx].set_yticklabels([f'$C_{{{i + 1}}}$' for i in range(cases_num)], fontsize=label_fontsize)
        
        if row_idx == num_rows - 1:
            ax[axes_idx].set_xlabel(xlabel, fontsize=label_fontsize)

        ax[axes_idx].grid(axis='both', alpha=grid_alpha)

    fig.suptitle(title, fontsize=label_fontsize + 2)
    plt.tight_layout()
    return ax