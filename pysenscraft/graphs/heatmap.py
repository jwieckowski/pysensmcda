# Copyright (C) 2024 Jakub Więckowski

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ..validator import Validator

def heatmap(matrix: np.ndarray, 
                title: str = "Fuzzy Ranking Matrix",
                xlabel: str = "Alternatives",
                ylabel: str = "Positions",
                cmap: str | list = "Blues",
                annotate: bool = True,
                fmt: str = ".2f",
                linewidths: float = .5,
                cbar_kwargs: dict = {'label': 'Membership Degree'},
                figsize: tuple = (8, 6),
                label_fontsize: int = 10,
                title_fontsize: int = 12,
                ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize the fuzzy ranking matrix using a heatmap.

    Parameters
    ----------
    matrix : np.ndarray
        2D matrix, for example, obtained from the 'fuzzy_ranking' function.

    title : str, optional
        Title for the visualization.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    cmap : str or Colormap, optional
        Colormap for the heatmap.

    annotate : bool, optional
        If True, write the data values in each cell.

    fmt : str, optional
        String formatting code to use when adding annotations.

    linewidths : float, optional
        Width of the lines that will divide each cell.

    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar.

    figsize : tuple, optional
        Figure size (width, height) in inches.

    label_fontsize : int, optional
        Font size for axis labels.

    title_fontsize : int, optional
        Font size for the title.

    ax : plt.Axes, optional
        The axes on which to draw the heatmap. If not provided, a new figure will be created.

    Returns
    -------
    plt.Axes

    Examples
    --------
    >>> rankings = np.array([
    ...     [1, 2, 3, 4, 5],
    ...     [2, 1, 5, 3, 4],
    ...     [4, 3, 2, 5, 1],
    ...     [3, 2, 1, 4, 5],
    ... ])
    >>> fuzzy_rank = fuzzy_ranking(rankings, normalization_axis=0)
    >>> heatmap(fuzzy_rank, title="Fuzzy Ranking Matrix", figsize=(10, 8))

    """
    
    Validator.is_type_valid(matrix, np.ndarray)
    Validator.is_type_valid(title, str)
    Validator.is_type_valid(xlabel, str)
    Validator.is_type_valid(ylabel, str)
    Validator.is_type_valid(cmap, str)
    Validator.is_type_valid(annotate, bool)
    Validator.is_type_valid(fmt, str)
    Validator.is_type_valid(linewidths, float)
    Validator.is_type_valid(cbar_kwargs, dict)
    Validator.is_type_valid(figsize, tuple)
    Validator.is_type_valid(label_fontsize, int)
    Validator.is_type_valid(title_fontsize, int)
    if ax is not None:
        Validator.is_type_valid(ax, plt.Axes)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = ax

    sns.heatmap(matrix, cmap=cmap, annot=annotate, fmt=fmt, linewidths=linewidths, cbar_kws=cbar_kwargs, ax=ax)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    return ax
