# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ..compromise.ICRA import ICRAResults
from ..validator import Validator

def preference_distribution(g: sns.FacetGrid, xlabel: str) -> None:
    """
    Function for kde distribution based on `sns.FacetGrid`
    Code from https://seaborn.pydata.org/examples/kde_ridgeplot.html Adapted for ICRA approach, however can be used separately
    
    Parameters:
    ----------
    g: sns.FacetGrid
        `See provided link`
    xlabel: str
        Label of x axis

    Returns:
    -------
        None

    Example:
    -------
    >>> methods = ["TOPSIS", "VIKOR", "COMET"]
    >>> preference = np.random.random((8, 3))
    >>> df = pd.DataFrame(preference, columns=methods)
    >>> df = df.stack().reset_index()
    >>> df.rename(columns={df.columns[1]: 'Method', df.columns[2]: 'Preference'}, inplace=True)
    >>> with sns.axes_style('white', rc={"axes.facecolor": (0, 0, 0, 0)}):
    >>>     g = sns.FacetGrid(df, row='Method', hue='Method', aspect=10, height=.75)
    >>>     preference_distribution(g, 'Preference')
    >>>     plt.suptitle('Preference distribution')
    >>>     plt.show()
    """

    Validator.is_type_valid(g, sns.FacetGrid, 'g')
    Validator.is_type_valid(xlabel, str, 'xlabel')

    g.map(sns.kdeplot, xlabel, bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, xlabel, clip_on=False, color="w", lw=2, bw_adjust=.5)

    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax.transAxes)

    g.map(label, xlabel)

    g.figure.subplots_adjust(hspace=-.25)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

def ICRA_pref_distribution(results: ICRAResults, 
                        methods: list[str], 
                        palettes: list | None = None, 
                        by: str = 'methods', 
                        file_name: str = 'ICRA_pref', 
                        save: bool = False, 
                        format: str = 'png', 
                        show: bool = True, 
                        indexes: list | None = None, 
                        FacetGrid_kwargs: dict = dict()) -> None:
    """
    Function for plotting distibution of preferences for ICRA approach across different methods or iterations.

    Parameters:
    ------------
    results: ICRAResults
        Results obtained form ICRA procedure.
    methods: list[str]
        List of method names from ICRA procedure.
    palettes: list, optional, default=None
        List of lists of RGB tuples or `matplotlib.colors.ListedColormap`. Should contain palette for each method or iteration `see examples`.
    by: str, optional, default='methods'
        The distribution is shown for a given method over the subsequent iterations if by=`methods`
        The distribution is shown for a given iteration over methods if by=`iters`
    file_name: str, optional, default='ICRA_pref'
        If save=`True`, the plots are saved as f'{file_name}_{value}.{format}', where value is either method name or iteration number depending on `by` parameter
    save: bool, optional, default=False
        If save=`True`, the plots are saved
    format: str, optional, default='png'
        If save=`True`, the plots are saved as f'{file_name}_{value}.{format}'
    show: bool, optional, default=True
        If show=`True`, plots are shown
    indexes: list|None, optional, default=None
        Indexes of iterations or methods for which distribution should be plotted
    FacetGrid_kwargs: dict, optional, default=dict()
        Keyword arguments to pass into `sns.FacetGrid()`.

    Returns:
    ---------
        None
        
    Example:
    ----------
    
    Example of obtaining ICRA results
    
    >>> ## Initial decision problem evaluation - random problem
    >>> decision_matrix = np.random.random((7, 5))
    >>> 
    >>> decision_problem_weights = np.ones(decision_matrix.shape[1])/decision_matrix.shape[1]
    >>> decision_problem_types = np.ones(decision_matrix.shape[1])
    >>> 
    >>> comet = COMET(np.vstack((np.min(decision_matrix, axis=0), np.max(decision_matrix, axis=0))).T, MethodExpert(TOPSIS(), decision_problem_weights, decision_problem_types))
    >>> topsis = TOPSIS()
    >>> vikor = VIKOR()
    >>> 
    >>> comet_pref = comet(decision_matrix)
    >>> topsis_pref = topsis(decision_matrix, decision_problem_weights, decision_problem_types)
    >>> vikor_pref = vikor(decision_matrix, decision_problem_weights, decision_problem_types)
    >>> 
    >>> ## ICRA variables preparation
    >>> methods = {
    >>>     COMET: [['np.vstack((np.min(matrix, axis=0), np.max(matrix, axis=0))).T', 
    >>>                     'MethodExpert(TOPSIS(), weights, types)'], 
    >>>             ['matrix']],
    >>>     topsis: ['matrix', 'weights', 'types'],
    >>>     vikor: ['matrix', 'weights', 'types']
    >>>     }
    >>> 
    >>> ICRA_matrix = np.array([comet_pref, topsis_pref, vikor_pref]).T
    >>> method_types = np.array([1, 1, -1])
    >>> 
    >>> result = iterative_compromise(methods, ICRA_matrix, method_types)

    Example 1: Show and save plots for TOPSIS and COMET distribution across all iterations
    
    >>> methods = ["TOPSIS", "VIKOR", "COMET"]
    >>> ICRA_pref_distribution(result, methods, save=True, by='methods', indexes=[0, 2])
    
    Example 2: Save plots for first and third iterations across all methods under custom file name with pdf format
    
    >>> methods = ["TOPSIS", "VIKOR", "COMET"]
    >>> ICRA_pref_distribution(result, methods, file_name='custom_file_name', format='pdf', save=True, show=False, by='iters', indexes=[0, 2])
    
    Example 3: Using custom palettes
    
    >>> methods = ["TOPSIS", "VIKOR", "COMET"]
    >>> palettes = []
    >>> for idx in range(2):
    >>>     palettes.append(sns.cubehelix_palette(5, rot=-.25, start=0.3*idx, hue=1))
    >>> ICRA_pref_distribution(result, methods, by='methods', indexes=[0, 2], palettes=palettes)
    """

    Validator.is_type_valid(results, ICRAResults, 'results')
    Validator.is_type_valid(methods, list, 'methods')
    if palettes is not None:
        Validator.is_type_valid(palettes, list, 'palettes')
    Validator.is_type_valid(by, str, 'by')
    Validator.is_in_list(by, ['methods', 'iters'], 'by')
    Validator.is_type_valid(file_name, str, 'file_name')
    Validator.is_type_valid(save, bool, 'save')
    Validator.is_type_valid(format, str, 'format')
    Validator.is_type_valid(show, bool, 'show')
    if indexes is not None:
        Validator.is_type_valid(indexes, list, 'indexes')
    Validator.is_type_valid(FacetGrid_kwargs, dict, 'FacetGrid_kwargs')

    df = pd.concat([pd.DataFrame(arr, columns=methods) 
                        for arr in results.all_preferences], 
                        keys=[f'Iter. {i+1}' for i in range(len(results.all_preferences))])
    df = df.stack().reset_index()
    df.drop(df.columns[1], axis=1, inplace = True)
    df.rename(columns={df.columns[0]: 'Iteration', df.columns[1]: 'Method', df.columns[2]: 'Preference'}, inplace=True)

    iters_number = results.all_preferences[:, 0].shape[0]

    if by == 'iters':
        dist_by = [f'Iter. {i+1}' for i in range(iters_number)]
    elif by == 'methods':
        dist_by = methods
    
    if indexes is None:
        indexes = np.arange(len(dist_by))
    
    if not FacetGrid_kwargs:
        FacetGrid_kwargs.update({'aspect': 10, 'height': .75})

    for idx, value in enumerate(dist_by):
        if idx not in indexes:
            continue

        if palettes is None:
            if by == 'iters':
                pal = None
            elif by == 'methods':
                if iters_number <= 2:
                    pal = sns.light_palette(sns.color_palette('tab10')[idx], n_colors=iters_number+1)[1:]
                else:
                    pal = [*sns.light_palette(sns.color_palette('tab10')[idx], n_colors=iters_number)[1:-1], 
                        *sns.dark_palette(sns.color_palette('tab10')[idx], reverse=True, n_colors=iters_number)[:-1]]
        else:
            pal = palettes[indexes.index(idx)]
        with sns.axes_style('white', rc={"axes.facecolor": (0, 0, 0, 0)}):
            if by == 'iters':
                g = sns.FacetGrid(df[df.Iteration == value], row='Method', hue='Method', palette=pal, **FacetGrid_kwargs)
                title = f'Iteration {idx+1}'
            elif by == 'methods':
                g = sns.FacetGrid(df[df.Method == value], row='Iteration', hue='Iteration', palette=pal, **FacetGrid_kwargs)
                title = value
            preference_distribution(g, 'Preference')
            plt.suptitle(title, x=(g.figure.subplotpars.right + g.figure.subplotpars.left)/2)
            if save:
                plt.savefig(f'{file_name}_{value}.{format}')
            if show:
                plt.show()