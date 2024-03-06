# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ..validator import Validator

def weights_barplot(weights: np.ndarray, 
                    title: str, 
                    ax: None | plt.Axes = None, 
                    width: float = 0.8, 
                    color: str = 'dodgerblue', 
                    alpha: int = 1, 
                    grid_on: bool = False, 
                    annotate_bars: bool = False) -> plt.Axes:
    """
    Generate a bar plot to visualize criteria weights.

    Parameters:
    ------------
    weights : np.ndarray
        1D array representing criteria weights.

    title : str
        Title of the bar plot.

    ax : plt.Axes or None, optional, default=None
        Matplotlib Axes on which to draw the bar plot. If None, the current Axes is used.

    width : float, optional, default=0.8
        Width of the bars in the bar plot.

    color : str, optional, default='dodgerblue'
        Color of the bars in the bar plot.

    alpha : float, optional, default=1
        Opacity of the bars in the bar plot. Should be in range [0, 1].

    grid_on : bool, optional, default=False
        If True, display grid lines on the plot.

    annotate_bars : bool, optional, default=False
        If True, annotate each bar with its corresponding weight value.

    Returns:
    ---------
    mpl.axes.Axes, mpl.container.BarContainer
        Matplotlib Axes and BarContainer objects for the generated bar plot.

    Examples:
    ----------
    Example 1: Plot visualization with required parameters
    
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> title = 'Criteria Weights'
    >>> ax, bars = weights_barplot(weights, title)
    >>> plt.show()

    Example 2: Plot visualization with additional parameters
    
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> title = 'Criteria Weights'
    >>> ax, bars = weights_barplot(weights, title, color='green', width=0.5, alpha=0.7, grid_on=True, annotate_bars=True)
    >>> plt.show()
    """
    Validator.is_type_valid(weights, np.ndarray, 'weights')
    Validator.is_type_valid(title, str, 'title')
    if ax is not None:
        Validator.is_type_valid(ax, plt.Axes, 'ax')
    Validator.is_type_valid(width, (float, np.floating), 'width')
    Validator.is_type_valid(color, str, 'color')
    Validator.is_type_valid(alpha, (int, np.integer, float, np.floating), 'alpha')
    Validator.is_in_range(alpha, 0, 1, 'alpha')
    Validator.is_type_valid(grid_on, bool, 'grid_on')
    Validator.is_type_valid(annotate_bars, bool, 'annotate_bars')

    if ax is None:
        ax = plt.gca()

    bars = ax.bar(np.arange(len(weights)), weights, align='center', color=color, width=width, alpha=alpha)

    if annotate_bars:
        for idx, w in enumerate(weights):
            ax.text(idx, w, np.round(w, 4), ha='center', va='bottom', fontsize=10, color='black')

    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(weights)))
    ax.set_xticklabels([f'$C_{{{i+1}}}$' for i in range(len(weights))])
    if grid_on:
        ax.grid(which='both', alpha=0.7)
        ax.set_axisbelow(True)
    
    ax.set_title(title)

    return ax, bars


def slider_weights_barplot(initial_weights: np.ndarray, 
                                results: list[tuple[int | tuple, tuple, np.ndarray]], 
                                ax: None | plt.Axes = None, 
                                width: float = 0.8, 
                                color: str = 'dodgerblue', 
                                sort_values: bool = True, 
                                grid_on: bool = False, 
                                percentage_change: bool = False, 
                                annotate_bars: bool = False) -> tuple[plt.Axes, Slider, Slider]:
    """
    Create an interactive slider-based bar plot to visualize changes in criteria weights.

    Parameters:
    ------------
    initial_weights : np.ndarray
        1D array representing the initial criteria weights.

    results : List[Tuple[int | tuple, tuple, np.ndarray]]
        A list of tuples containing information about the modified criteria index, percentage change,
        and the resulting criteria weights.

    ax : plt.Axes or None, optional, default=None
        Matplotlib Axes on which to draw the bar plot. If None, a new Axes is created.

    width : float, optional, default=0.8
        Width of the bars in the bar plot.

    color : str, optional, default='dodgerblue'
        Color of the bars in the bar plot.

    sort_values : bool, optional, default=True
        If True, sort the values when plotting.

    grid_on : bool, optional, default=False
        If True, display grid lines on the plot.

    percentage_change : bool, optional, default=False
        If True, interpret changes as percentages and add '%' in labels.

    annotate_bars : bool, optional, default=False
        If True, annotate each bar with its corresponding weight value.

    Returns:
    ---------
    Tuple[plt.Axes, mpl.widgets.Slider, mpl.widgets.Slider]
        Matplotlib Axes and two Slider objects for interactive use.

    Examples:
    ----------
    Example 1: Plot visualization with required parameters and percentage_modification
    
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentages = np.array([5, 5, 5])
    >>> indexes = np.array([[0, 1], 2], dtype='object')
    >>> results = percentage_modification(weights, percentages, indexes=indexes)
    >>> # In the case of using sliders, the reference should be kept, so Python wouldn't GC
    >>> ax, criteria_slider, change_slider = slider_weights_barplot(weights, results, percentage_change=True, annotate_bars=True)
    >>> plt.show()

    Example 2: Plot visualization with additional parameters and percentage_modification
    
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> percentages = np.array([5, 5, 5])
    >>> indexes = np.array([[0, 1], 2], dtype='object')
    >>> results = percentage_modification(weights, percentages, indexes=indexes)
    >>> # In the case of using sliders, the reference should be kept, so Python wouldn't GC
    >>> ax, criteria_slider, change_slider = slider_weights_barplot(weights, results, percentage_change=True, annotate_bars=True, width=0.7, grid_on=True, color='red)
    >>> plt.show()

    Example 3: Plot visualization with range modification
    
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> range_values = np.array([[0.28, 0.32], [0.30, 0.33], [0.37, 0.44]])
    >>> indexes = np.array([[0, 1], 2], dtype='object')
    >>> results = range_modification(weights, range_values, indexes=indexes)
    >>> # In the case of using sliders, the reference should be kept, so Python wouldn't GC
    >>> ax, criteria_slider, change_slider = slider_weights_barplot(weights, results, annotate_bars=True, grid_on=True)
    >>> plt.show()
    """
    Validator.is_type_valid(initial_weights, np.ndarray, initial_weights)
    Validator.is_type_valid(results, list, 'results')
    Validator.is_type_valid(width, (float, np.floating), 'width')
    Validator.is_type_valid(color, str, 'color')
    Validator.is_type_valid(width, (float, np.floating), 'width')
    Validator.is_type_valid(sort_values, bool, 'sort_values')
    Validator.is_type_valid(grid_on, bool, 'grid_on')
    Validator.is_type_valid(percentage_change, bool, 'percentage_change')
    Validator.is_type_valid(annotate_bars, bool, 'annotate_bars')
    if ax is not None:
        Validator.is_type_valid(ax, plt.Axes, 'ax')

    if ax is None:
        fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.25)

    ax, _ = weights_barplot(initial_weights, 'Initial weights', ax=ax, width=width, color=color, grid_on=grid_on, annotate_bars=annotate_bars)

    # get data for interactive plotting
    crit_indexes = [None]
    for r in results:
        if r[0] not in crit_indexes:
            crit_indexes.append(r[0])

    change_values = []
    change_values_sizes = []
    for cidx in crit_indexes:
        temp = []
        for r in results:
            if r[0] == cidx:
                temp.append(r[1])

        if sort_values:
            change_values.append(sorted(temp))
        else:
            change_values.append(temp)
        change_values_sizes.append(len(temp))

    # Create a sliders on the left side
    crit_slider_ax = plt.axes([0.02, 0.3, 0.05, 0.6])
    criteria_slider = Slider(crit_slider_ax, 'Criteria\nindexes', 0, len(crit_indexes)-1, valinit=0, valstep=np.arange(0, len(crit_indexes)+1), orientation='vertical')
    
    change_slider_ax = plt.axes([0.12, 0.3, 0.05, 0.6])
    change_slider = Slider(change_slider_ax, 'Change', 0, 1, valinit=0, valstep=1, orientation='vertical')

    def update_criteria(val: float | int) -> None:
        ax.clear()
        
        # get criteria idx
        criteria_idx = criteria_slider.val
        change_slider.set_val(0)

        # adjust change slider values
        change_slider.valmax = change_values_sizes[criteria_idx]-1
        change_slider.ax.set_ylim(0, change_slider.valmax)

    def update_change(val: float | int) -> None:
        ax.clear()
        
        # get criteria idx
        criteria_idx = criteria_slider.val
        change = change_slider.val

        # if criteria_idx == 0 or change == 0:
        if criteria_idx == 0:
            modified_weights = initial_weights
        else:
            modified_weights = [r[2] for r in results if crit_indexes[criteria_idx] == r[0] and change_values[criteria_idx][change] == r[1]][0]

        # Plot modified state
        _, bars = weights_barplot(modified_weights, '', ax=ax, width=width, color=color, grid_on=grid_on, annotate_bars=annotate_bars, alpha=0.5)

        title = ''
        change_label = '% change' if percentage_change else ' change'
        if criteria_idx == 0:
            title = 'Initial weights'
            for bar in bars:
                bar.set_alpha(1)
        if crit_indexes[criteria_idx] is not None:    
            if isinstance(crit_indexes[criteria_idx], (tuple)):
                for i, crit_idx in enumerate(crit_indexes[criteria_idx]):
                    bars[crit_idx].set_alpha(1)
                    change_value = change_values[criteria_idx][change][i] * 100 if percentage_change else change_values[criteria_idx][change][i]
                    title += f'$C_{{{crit_idx+1}}}$ {change_value} {change_label} '   
            else:
                bars[criteria_idx].set_alpha(1)
                change_value = change_values[criteria_idx][change] * 100 if percentage_change else change_values[criteria_idx][change]
                title = f'$C_{{{criteria_idx+1}}}$ {change_value} {change_label}'
        
        ax.set_title(title)

        plt.draw()

    # Attach the update function to the slider
    criteria_slider.on_changed(update_criteria)
    change_slider.on_changed(update_change)

    return ax, criteria_slider, change_slider
