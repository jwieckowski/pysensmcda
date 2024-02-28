# Copyright (C) 2024 Jakub Więckowski

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def heatmap(fuzzy_ranking_matrix: np.ndarray, 
                             title: str = "Fuzzy Ranking Matrix",
                             xlabel: str = "Alternatives",
                             ylabel: str = "Positions",
                             cmap: str | list = "Blues",
                             annot: bool = True,
                             fmt: str = ".2f",
                             linewidths: float = .5,
                             cbar_kws: dict = {'label': 'Membership Degree'},
                             figsize: tuple = (8, 6),
                             label_fontsize: int = 10,
                             title_fontsize: int = 12,
                             ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize the fuzzy ranking matrix using a heatmap.

    Parameters
    ----------
    fuzzy_ranking_matrix : np.ndarray
        Fuzzy ranking matrix obtained from the `fuzzy_ranking` function.

    title : str, optional
        Title for the visualization.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    cmap : str or Colormap, optional
        Colormap for the heatmap.

    annot : bool, optional
        If True, write the data values in each cell.

    fmt : str, optional
        String formatting code to use when adding annotations.

    linewidths : float, optional
        Width of the lines that will divide each cell.

    cbar_kws : dict, optional
        Additional keyword arguments for the colorbar.

    figsize : tuple, optional
        Figure size (width, height) in inches.

    label_fontsize : int, optional
        Font size for axis labels.

    title_fontsize : int, optional
        Font size for the title.

    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the heatmap. If not provided, a new figure will be created.

    Returns
    -------
    matplotlib.axes.Axes

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
    
    if ax is None:
        plt.figure(figsize=figsize)
        axs = plt.gca()
    else:
        axs = ax

    sns.heatmap(fuzzy_ranking_matrix, cmap=cmap, annot=annot, fmt=fmt, linewidths=linewidths, cbar_kws=cbar_kws, ax=axs)
    axs.set_title(title, fontsize=title_fontsize)
    axs.set_xlabel(xlabel, fontsize=label_fontsize)
    axs.set_ylabel(ylabel, fontsize=label_fontsize)

    return axs
