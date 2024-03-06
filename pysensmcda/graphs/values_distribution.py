# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..validator import Validator

def hist_dist(data: np.ndarray, 
            ax: plt.Axes | None = None, 
            fig: plt.Figure = None, 
            xlabel: str = 'Value', 
            kind: str = 'hist+kde', 
            show_slider: bool = True, 
            title: str = '', 
            slider_label: str = 'Number\nof bins', 
            slider_pad: float | None = None, 
            bins_count: str | int = 'auto', 
            slider_size: str | float = '5%', 
            min_bins: int = 1, 
            max_bins: int = 20) -> tuple[plt.Axes, Slider] | plt.Axes:
    """
    Visualization of distribution of values with histograms

    Parameters:
    ------------
        data: ndarray
            Values of criteria, where columns designate separate criteria.
        ax: Axis | None, optional, default=None
            Matplotlib Axis to draw on. If None, current axis is used.
        fig: matplotlib.Figure|None
            Matplotlib Figure to draw on. If show_slider=True and ax is passed, the fig needs to be passed as well.
        xlabel: str, optional, default='Value'
            Label of x axis.
        kind: 'hist+kde'|'hist'|'kde', optional, default='hist+kde'
            Kind of distribution.
        show_slider: bool, optional, default=True
            If True slider to change number of bins is shown.
        title: str, optional, default=''
            Title of the axes.
        slider_label: str
            Label for the slider which is responsible for the number of bins.
        slider_pad: float|None, optional, default=None
            Padding that should be applied between axes and slider.
        bins_count: int|'auto', optional, default='auto'
            Number of initial bins.
        slider_size: float|str, optional, default='5%'
            The value of how much of space the slider should take.
        min_bins: int, optional, default=1
            Minimum amount of bins available to select with slider.
        max_bins: int, optional, default=20
            Maximum amount of bins available to select with slider.

    Returns:
    ---------
        tuple(ax, slider) if show_slider=True else ax
        ax: matplotlib.Axes
            Axes object or list of Axes objects on which plots were drawn.
        slider: matplotlib.widgets.Slider
            Slider object used in plot
    
    Examples:
    ----------

    Example 1: Hist with slider
    
    >>> fig, ax = plt.subplots()
    >>> results = np.array([0.294, 0.306, 0.288, 0.312, 0.282, 0.318, 0.304, 0.296, 0.308, 0.292, 0.312, 0.288, 0.316, 0.284])
    >>> # In the case of using sliders, the reference should be kept, so Python wouldn't GC
    >>> _, bins_slider = hist_dist(results, ax, fig=fig, slider_label='Number of bins', kind='hist', xlabel='Value', title='Criterion value distribution')
    >>> plt.show()

    Example 2: Hist+kde without slider
    
    >>> fig, ax = plt.subplots()
    >>> results = np.array([0.294, 0.306, 0.288, 0.312, 0.282, 0.318, 0.304, 0.296, 0.308, 0.292, 0.312, 0.288, 0.316, 0.284])
    >>> hist_dist(results, ax, fig=fig, slider_label='Number of bins', show_slider=False, xlabel='Value', title='Criterion value distribution')
    >>> plt.show()
    
    """
    
    Validator.is_type_valid(data, np.ndarray, 'data')
    if ax is not None:
        Validator.is_type_valid(ax, plt.Axes, 'ax')
    if fig is not None:
        Validator.is_type_valid(fig, plt.Figure, 'fig')
    Validator.is_type_valid(xlabel, str, 'xlabel')
    Validator.is_type_valid(kind, str, 'kind')
    Validator.is_in_list(kind, ['hist+kde', 'hist', 'kde'], 'kind')
    Validator.is_type_valid(show_slider, bool, 'show_slider')
    Validator.is_type_valid(title, str, 'title')
    Validator.is_type_valid(slider_label, str, 'slider_label')
    if slider_pad is not None:
        Validator.is_type_valid(slider_pad, (float, np.floating), 'slider_pad')
    Validator.is_type_valid(bins_count, (str, int, np.integer), 'bins_count')
    Validator.is_type_valid(slider_size, (str, float, np.floating), 'slider_size')
    Validator.is_type_valid(min_bins, (int, np.integer), 'min_bins')
    Validator.is_positive_value(min_bins, var_name='min_bins')
    Validator.is_type_valid(max_bins, (int, np.integer), 'max_bins')
    Validator.is_positive_value(max_bins, var_name='max_bins')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        if fig is None and show_slider:
            raise ValueError("Parameter 'fig' needs to be passed when 'ax' is passed.")

    if show_slider:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size=slider_size, pad=slider_pad)

    def create_slider(init_bins: int) -> Slider:
        def update(val: float | int) -> None:
            ax.clear()
            sns.histplot(data, ax=ax, bins=val)
            if kind == 'hist+kde':
                sns.kdeplot(data, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            fig.canvas.draw_idle()

        bins_slider = Slider(
            ax=cax,
            label=slider_label,
            valmin=min_bins,
            valstep=1,
            valmax=max_bins,
            valinit=init_bins,
        )
        bins_slider.on_changed(update)
        return bins_slider

    initial_bin_number = len(np.histogram(data)[0])
    if kind == 'hist+kde':
        if show_slider:
            bins_slider = create_slider(initial_bin_number)
        sns.histplot(data, ax=ax, bins=bins_count)
        sns.kdeplot(data, ax=ax)
    elif kind == 'kde':
        sns.kdeplot(data, ax=ax)
    elif kind == 'hist':
        sns.histplot(data, ax=ax, bins=bins_count)
        if show_slider:
            bins_slider = create_slider(initial_bin_number)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if show_slider:
        return (ax, bins_slider)
    else:
        return ax
    
def mutli_hist_dist(data: np.ndarray, 
                    nrows: int, 
                    ncols:int, 
                    figsize: tuple[int], 
                    ax_title: bool = True, 
                    slider_label: bool = True, 
                    slider_pad: float = 0.5, 
                    slider_size: str | float = '5%', 
                    title: str = 'Distribution of criteria values', 
                    kind: str = 'hist+kde', 
                    title_pos: float = 0.5, 
                    w_pad: float = 1.5, 
                    min_bins: int = 1, 
                    max_bins: int = 20, 
                    show_slider: bool = True, 
                    bins_count: str | int = 'auto', 
                    main_slider_label: str = 'Number of bins', 
                    xlabel: str = 'Value') -> tuple[plt.Axes, plt.Figure, Slider, list[Slider]] | tuple[plt.Axes, plt.Figure]:
    """
    Visualization of distribution of multiple values with histograms


    Parameters:
    ------------
        data: ndarray
            Values of criteria, where columns designate separate criteria.
        nrows: int
            Number of rows in subplots.
        ncols: int
            Number of columns in subplots.
        figsize: tuple[int]
            Size of figure in tuple (width, height).
        ax_title: bool, optional, default=True
            If True for each axes title is set to f'Crit {idx+1}'.
        slider_label: bool, optional, default=True
            If True for each axes slider label is set to f'$C_{{{idx+1}}}$ bins'.
        slider_pad: float, optional, default=0.5
            Padding that should be applied between axes and slider.
        slider_size: float|str, optional, default='5%'
            The value of how much of space the slider should take.
        title: str, optional, default='Distribution of criteria values'
            Title of the figure. Set with suptitle().
        kind: 'hist+kde'|'hist'|'kde', optional, default='hist+kde'
            Kind of distribution.
        title_pos: float, optional, default=0.5
            Position of suptitle if set.
        w_pad: float, optional, default=1.5
            Padding between axes when multiple subplots present.
        min_bins: int, optional, default=1
            Minimum amount of bins available to select with slider.
        max_bins: int, optional, default=20
            Maximum amount of bins available to select with slider.
        show_slider: bool, optional, default=True
            If True slider to change number of bins is shown.
        bins_count: int|'auto', optional, default='auto'
            Number of initial bins.
        main_slider_label: str, optional, default='Number of bins'
            Label of main slider that controls all sliders at once.
        xlabel: str, optional, default='Value'
            Label of x axis.

    Example
    ---------
        >>> results = np.array([[0, -0.02, np.array([0.294, 0.303, 0.403])],
        >>>     [0, 0.02, np.array([0.306, 0.297, 0.397])],
        >>>     [0, -0.04, np.array([0.288, 0.306, 0.406])],
        >>>     [0, 0.04, np.array([0.312, 0.294, 0.394])],
        >>>     [0, -0.06, np.array([0.282, 0.309, 0.409])],
        >>>     [0, 0.06, np.array([0.318, 0.291, 0.391])],
        >>>     [2, -0.02, np.array([0.304, 0.304, 0.392])],
        >>>     [2, 0.02, np.array([0.296, 0.296, 0.408])],
        >>>     [2, -0.04, np.array([0.308, 0.308, 0.384])],
        >>>     [2, 0.04, np.array([0.292, 0.292, 0.416])],
        >>>     [2, -0.06, np.array([0.312, 0.312, 0.376])],
        >>>     [2, 0.06, np.array([0.288, 0.288, 0.424])],
        >>>     [2, -0.08, np.array([0.316, 0.316, 0.368])],
        >>>     [2, 0.08, np.array([0.284, 0.284, 0.432])]], dtype=object)
        >>> critera_values = np.array([*results[:, 2]], dtype=float)
        >>> # In the case of using sliders, the reference should be kept, so Python wouldn't GC
        >>> _, _, sliders, main_slider = mutli_hist_dist(critera_values, title_pos=0.5, nrows=1, ncols=3, figsize=(8, 4))
        >>> plt.show()

    Returns:
    ---------
        tuple(ax, fig, main_slider, sliders) if show_slider=True else tuple(ax, fig)
        ax: matplotlib.Axes
            Axes object or list of Axes objects on which plots were drawn.
        fig: matplotlib.Figure
            Figure object on which axes were drawn.
        main_slider: matplotlib.widgets.Slider
            Slider object that controlls bins count for all subplots
        sliders: list[matplotlib.widgets.Slider]
            Slider objects that controlls bins count for each subplot individually
    """
    
    Validator.is_type_valid(data, np.ndarray, 'data')
    Validator.is_type_valid(nrows, (int, np.integer), 'nrows')
    Validator.is_positive_value(nrows, var_name='nrows')
    Validator.is_type_valid(ncols, (int, np.integer), 'ncols')
    Validator.is_positive_value(ncols, var_name='ncols')
    Validator.is_type_valid(figsize, tuple, 'figsize')
    Validator.is_type_valid(ax_title, bool, 'ax_title')
    Validator.is_type_valid(slider_label, bool, 'slider_label')
    Validator.is_type_valid(slider_pad, (float, np.floating), 'slider_pad')
    Validator.is_type_valid(slider_size, (str, float, np.floating), 'slider_size')
    Validator.is_type_valid(title, str, 'title')
    Validator.is_type_valid(kind, str, 'kind')
    Validator.is_in_list(kind, ['hist+kde', 'hist', 'kde'], 'kind')
    Validator.is_type_valid(title_pos, (float, np.floating), 'title_pos')
    Validator.is_type_valid(w_pad, (float, np.floating), 'w_pad')
    Validator.is_type_valid(min_bins, (int, np.integer), 'min_bins')
    Validator.is_positive_value(min_bins, var_name='min_bins')
    Validator.is_type_valid(max_bins, (int, np.integer), 'max_bins')
    Validator.is_positive_value(max_bins, var_name='max_bins')
    Validator.is_type_valid(show_slider, bool, 'show_slider')
    Validator.is_type_valid(bins_count, (str, int, np.integer), 'bins_count')
    Validator.is_type_valid(main_slider_label, str, 'main_slider_label')
    Validator.is_type_valid(xlabel, str, 'xlabel')

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    ax = ax.flatten()
    sliders = []
    for idx in range(len(data[2])):
        axes_title = f'Crit {idx+1}' if ax_title else ''
        s_label = f'$C_{{{idx+1}}}$ bins' if slider_label else ''
        if show_slider:
            _, ax_slider = hist_dist(data[:, idx], ax[idx], fig=fig, title=axes_title, slider_label=s_label, slider_pad=slider_pad, slider_size=slider_size, show_slider=show_slider, bins_count=bins_count, kind=kind, xlabel=xlabel, min_bins=min_bins, max_bins=max_bins)
            sliders.append(ax_slider)
        else:
            hist_dist(data[:, idx], ax[idx], fig=fig, title=axes_title, slider_label=s_label, slider_pad=slider_pad, slider_size=slider_size, show_slider=show_slider, bins_count=bins_count, kind=kind, xlabel=xlabel, min_bins=min_bins, max_bins=max_bins)
    plt.suptitle(title, x=title_pos)
    plt.tight_layout(w_pad=w_pad)
    
    if show_slider:
        fig.subplots_adjust(left=0.25)
        axfreq = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        main_slider = Slider(axfreq, main_slider_label, min_bins, max_bins, valstep=1, orientation='vertical')
        def update_all_sliders(val):
            for slider in sliders:
                slider.set_val(val)
        main_slider.on_changed(update_all_sliders)
    if show_slider:
        return (fig, ax, sliders, main_slider)
    else:
        return (fig, ax)