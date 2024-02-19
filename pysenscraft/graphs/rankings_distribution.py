# Copyright (C) 2024 Bartosz Paradowski

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def rankings_distribution(rankings, ax=None, title='', methods=None, legend_loc='upper', plot_type='box',
                           plot_kwargs=dict(), xlabel='Alternative', ylabel='Position', show_legend=True):
    """
    Parameters
    ----------
    rankings: nd.array
        3d or 2d array of rankings to plot distribution for.
    ax: Axis | None, optional, default=None
        Matplotlib Axis to draw on. If None, current axis is used.
    title: str, optional, default=''
        Plot title.
    methods: list[str] | None, optional, default=None
        Name of methods for which distribution will be plotted. If not provided in case of multiple methods, the legend is shown as `Method 'ordinal number'`
    legend_loc: str, optional, default='upper'
        Legend location, all options provide legend outside axis. Supported options: 'upper', 'lower', 'right'
    plot_type: str, optional, default='box'
        Type of distribution plot, based on seaborn package. Supported options: 'box', 'boxen', 'violin'
    plot_kwargs: dict, optional, default=dict()
        Keyword arguments to pass into plot function.
    xlabel: str, optional, default='Alternative'
        Label for x axis.
    ylabel: str, optional, default='Position'
        Label for y axis.
    show_legend: bool, optional, default='True'
        Boolean responsible for whether the legend is visible.

    Examples
    --------
    ### Example 1 - one method
    >>> rankings = np.array([[1, 2, 3, 4 ,5],
    >>>                      [2, 3, 5, 4, 1],
    >>>                      [5, 3, 2, 1, 4]])
    >>> rankings_distribution(rankings, title='TOPSIS ranking distribution')
    >>> plt.show()
    
    ### Example 2 - multiple methods
    >>> rankings = np.array([[[1, 2, 3, 4 ,5],
    >>>              [2, 3, 5, 4, 1],
    >>>              [5, 3, 2, 1, 4]],
    >>>              [[1, 2, 3, 4 ,5],
    >>>              [3, 2, 5, 4, 1],
    >>>              [5, 2, 3, 1, 4]]])
    >>> rankings_distribution(rankings, title='Ranking distribution')
    >>> plt.show()

    ### Example 3 - multiple methods with names
    >>> rankings = np.array([[[1, 2, 3, 4 ,5],
    >>>              [2, 3, 5, 4, 1],
    >>>              [5, 3, 2, 1, 4]],
    >>>              [[1, 2, 3, 4 ,5],
    >>>              [3, 2, 5, 4, 1],
    >>>              [5, 2, 3, 1, 4]]])
    >>> fig, ax = plt.subplots(1, 1)
    >>> methods = ['TOPSIS', 'VIKOR']
    >>> rankings_distribution(rankings, methods=methods, title='Ranking distribution', ax=ax)
    >>> plt.show()

    ### Example 4 - single method, no legend, custom labels
    >>> rankings = np.array([[1, 2, 3, 4 ,5],
    >>>                      [2, 3, 5, 4, 1],
    >>>                      [5, 3, 2, 1, 4]])
    >>> rankings_distribution(rankings, title='TOPSIS ranking distribution', show_legend=False, xlabel='Alt', ylabel='Ranking position')
    >>> plt.show()

    Returns
    -------
    ax: Axis
        Axis on which plot was drawn.
    """
    def create_df(rankings, method=None):
        """
        Internal function for dataframe creation for the purpose of plotting rankings distribution

        Parameters
        ----------
        rankings: nd.array
            2d array of rankings for specific method
        method: str | None, optional, default=None
            Method name

        Example
        -------
        >>> rankings = np.array([[1, 2, 3, 4 ,5],
        >>>              [2, 3, 5, 4, 1],
        >>>              [5, 3, 2, 1, 4]])
        >>> create_df(rankings)
        
        Returns
        -------
        df: pd.DataFrame
        """
        df = []
        for ranking in rankings:
            for alt, pos in enumerate(ranking):
                df.append([f'$A_{alt+1}$', pos])
        df = pd.DataFrame(df, columns=[xlabel, ylabel])
        if method is not None:
            df['Method'] = method
        return df
    
    if ax is None:
        ax = plt.gca()

    if plot_type == 'box':
        plot_f = sns.boxplot
    elif plot_type == 'boxen':
        plot_f = sns.boxenplot
    elif plot_type == 'violin':
        plot_f = sns.violinplot

    if rankings.ndim == 2:
        df = create_df(rankings)
        plot_f(data=df, x=xlabel, y=ylabel, ax=ax, **plot_kwargs)
        ax.set_title(title)
    elif rankings.ndim == 3:
        if methods is None:
            df = pd.concat([create_df(rankings[idx], f'Method {idx+1}') for idx in range(len(rankings))])
        elif len(rankings) == len(methods):
            df = pd.concat([create_df(method) for method in methods])
        else:
            raise ValueError('Number of method names inconsistent with number of rankings.')
        plot_f(data=df, x=xlabel, y=ylabel, hue='Method', ax=ax, **plot_kwargs)
    if show_legend:
        if legend_loc == 'upper':
            ax.set_title(title, y=1.12)
            sns.move_legend(ax, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=3, title=None)
        elif legend_loc == 'lower':
            ax.set_title(title)
            sns.move_legend(ax, bbox_to_anchor=(0, -.25, 1, 0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=3, title=None)
        elif legend_loc == 'right':
            ax.set_title(title)
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.04, 1), borderaxespad=0)
    else:
        ax.set_title(title)

    return ax