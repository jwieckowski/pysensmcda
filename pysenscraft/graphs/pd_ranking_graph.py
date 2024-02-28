# Copyright (C) 2024 Bartosz Paradowski, Jakub Więckowski

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ..validator import Validator

def percentage_graph(percentage_changes: list, new_positions: list, ax: mpl.axes.Axes|None=None, xticks: list|None=None, percentage_kwargs: dict=dict(), kind: str='bar', palette: dict=dict()):
    """
    Graph for showing percentage changes in criteria values and changes in alternative rank.

    Parameters
    ----------
    percentage_changes: list
        Changes of values of criteria in percentage values.
    new_positions: list
        List of positions acquired in promotion or demotion process.
    ax: Axes, optional, default=None
        Axes object on which the graphs will be drawn.
    xticks: list, optional, default=None
    percentage_kwargs: dict, optional, default=dict()
        Dictionary for styling plots. Available keys: 'show_percenatage_value', 'show_ranks', 'base_linestyle', 'base_linewidth', 'linestyle', 'linewidth', 'ylabel', 'title', 'positive_marker', 'positive_markersize', 'neutral_marker', 'neutral_markersize', 'negative_marker', 'negative_markersize'.
    kind: str, optional, default='bar'
        Changes style of the plot. Available options: 'bar', 'line'.
    palette: dict, optional, default=dict()
        Sets colors for specific part of the plot. Available keys: 'positive', 'neutral', 'negative'.
        
    Example
    -------
    >>> palette = {
    >>>     'positive': '#4c72b0',
    >>>     'neutral': 'black',
    >>>     'negative': '#c44e52',
    >>>     }
    >>> percentage_kwargs = {
    >>>     'title':'Promotion $A_1$', 
    >>>     'ylabel':'Percentage change'
    >>>     }
    >>> percentage_changes = [-3895.4559558861965, 276.3139391152694, -4453.417374310514, 1776.7036818539723, 0] 
    >>> new_positions = [1.0, 2.0, 1.0, 1.0, 5.0]
    >>> percentage_graph(percentage_changes, new_positions, palette=palette, percentage_kwargs=percentage_kwargs)
    >>> plt.show()

    Returns
    -------
    ax : Axes
        Axes object on which graph was drawn.
    """
    Validator.is_type_valid(percentage_changes, list)
    Validator.is_type_valid(new_positions, list)
    Validator.is_type_valid(ax, (None, mpl.axes.Axes))
    Validator.is_type_valid(xticks, (None, list))
    Validator.is_type_valid(percentage_kwargs, dict)
    Validator.is_type_valid(kind, str)
    Validator.is_in_list(kind, ['bar', 'line'])
    Validator.is_type_valid(palette, dict)

    # if kind not in ['bar', 'line']:
    #     raise ValueError('Accepted kind parameter values are: `bar`, `kind`.')
    
    # if xticks is not None and len(xticks) != len(percentage_changes):
    #     raise ValueError('`xticks` should contain the same number of values as `percentage_changes`')
    if xticks is not None:
        Validator.is_shape_equal(len(xticks), len(percentage_changes))

    if ax is None:
        ax = plt.gca()

    crit_num = len(percentage_changes)
    min_change = np.round(np.min(percentage_changes), -2)
    max_change = np.round(np.max(percentage_changes), -2)
    step = int(np.max(np.abs([min_change/5, max_change/5])))

    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)

    ax.plot([-1, len(percentage_changes)], [0, 0], percentage_kwargs.get('base_linestyle', '--'), color=palette.get('neutral', 'black'), linewidth=percentage_kwargs.get('base_linewidth', 1))

    if kind == 'bar':
        colors = [palette.get('positive', 'blue') if change > 0  else palette.get('negative', 'red') for change in percentage_changes]
        p = ax.bar(np.arange(0, crit_num), percentage_changes, color=colors)
        if percentage_kwargs.get('show_percenatage_value', True):
            ax.bar_label(p, label_type='center', fmt=lambda x: '' if x == 0 else f'{x:.0f} %')
        if percentage_kwargs.get('show_ranks', True):
            ax.bar_label(p, labels=[f'Rank {rank}' for rank in new_positions])
        for idx, change in enumerate(percentage_changes):
            if change == 0:
                ax.plot([idx-0.4, idx+0.4], [0, 0], color=palette.get('neutral', 'black'))
    elif kind == 'line':
        for idx, change in enumerate(percentage_changes):
            if change > 0:
                color = palette.get('positive', 'blue')
                marker = percentage_kwargs.get('positive_marker', '^')
                marker_size = percentage_kwargs.get('marker_size', percentage_kwargs.get('positive_markersize', 8))
            elif change < 0:
                color = palette.get('negative', 'red')
                marker = percentage_kwargs.get('negative_marker', 'v')
                marker_size = percentage_kwargs.get('marker_size', percentage_kwargs.get('negative_markersize', 8))
            else:
                color = palette.get('neutral', 'black')
                marker = percentage_kwargs.get('neutral_marker', 'o')
                marker_size = percentage_kwargs.get('marker_size', percentage_kwargs.get('neutral_markersize', 8))
            ax.plot([idx], [change], color=color, marker=marker, markersize=marker_size)
            ax.plot([idx, idx], [0, change], percentage_kwargs.get('linestyle', '-'), linewidth=percentage_kwargs.get('linewidth', 3), color=color)
        if percentage_kwargs.get('show_ranks', True):
            for idx, change in enumerate(percentage_changes):
                dist = np.sign(change)*step/2 if np.sign(change) else step/2
                ax.text(x=idx , y=change+dist, s=f'Rank {new_positions[idx]}', ha='center')
    
    ax.set_ylim(np.min(percentage_changes) - step, np.max(percentage_changes) + step)
    ax.set_xlim(-0.5, crit_num-0.5)
    
    if xticks is None:
        ax.set_xticks(np.arange(0, crit_num), [f'$C_{{{i+1}}}$' for i in range(crit_num)])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel(percentage_kwargs.get('ylabel', ''))
    ax.set_title(percentage_kwargs.get('title', ''))

    return ax

def rank_graph(initial_rank: int|float, new_positions: list, ax: mpl.axes.Axes|None=None, palette: dict=dict(), rank_kwargs: dict=dict()):
    """
    Graph for showing promotion / demotion of position of specific alternative.

    Parameters
    ----------
    initial_rank: int|int
        Initial position of the alternative.
    new_positions: list
        List of positions acquired in promotion or demotion process.
    ax: Axes, optional, default=None
        Axes object on which the graphs will be drawn.
    palette: dict, optional, default=dict()
        Sets colors for specific part of the plot. Available keys: 'positive', 'neutral', 'negative'.
    rank_kwargs: dict, optional, default=dict()
        Dictionary for styling plots. Available keys: 'base_linestyle', 'base_linewidth', 'linestyle', 'linewidth', 'ylabel', 'title', 'marker', 'markersize'.

    Example
    -------
    >>> palette = {
    >>>     'positive': '#4c72b0',
    >>>     'neutral': 'black',
    >>>     'negative': '#c44e52',
    >>>     }
    >>> rank_kwargs = {
    >>>     'title':'Promotion $A_1$', 
    >>>     'ylabel':'Rank change'
    >>>     }
    >>> initial_rank = 5
    >>> new_positions = [1.0, 2.0, 1.0, 1.0, 5.0]
    >>> rank_graph(initial_rank, new_positions, palette=palette, rank_kwargs=rank_kwargs)
    >>> plt.xlim(-1, 5)
    >>> plt.show()

    Returns
    -------
    ax : Axes
        Axes object on which graph was drawn.
    """
    Validator.is_type_valid(initial_rank, (int, float))
    Validator.is_type_valid(new_positions, list)
    Validator.is_type_valid(ax, (None, mpl.axes.Axes))
    Validator.is_type_valid(palette, dict)
    Validator.is_type_valid(rank_kwargs, dict)

    if ax is None:
        ax = plt.gca()

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    ax.plot([-1, len(new_positions)], [initial_rank, initial_rank], rank_kwargs.get('base_linestyle', '--'), color=palette.get('neutral', 'black'), linewidth=rank_kwargs.get('base_linewidth', 1))

    for idx, rank in enumerate(new_positions):
        if rank < initial_rank:
            color = palette.get('positive', 'blue')
        elif rank > initial_rank:
            color = palette.get('negative', 'red')
        else:
            color = palette.get('neutral', 'black')
        ax.plot([idx, idx], [initial_rank, rank], rank_kwargs.get('linestyle', '--'), color=color)
        ax.plot(idx, rank, rank_kwargs.get('marker', '*'), color=color, markersize=rank_kwargs.get('markersize', 10))
    
    ax.set_yticks(np.arange(np.min([initial_rank, *new_positions]), np.max([initial_rank, *new_positions])+1))
    ax.set_ylim(np.min([initial_rank, *new_positions])-0.5, np.max([initial_rank, *new_positions])+0.5)
    ax.invert_yaxis()
    ax.set_ylabel(rank_kwargs.get('ylabel', ''))
    ax.set_title(rank_kwargs.get('title', ''))

    return ax

def pd_rankings_graph(initial_rank: int|float, new_positions: list, percentage_changes: list, xticks: None | list =None, kind: str='bar', title: str='', ax: mpl.axes.Axes|None=None, draw_ranking_change: bool=True, height_ratio: list=[1, 3], percentage_kwargs:dict=dict(), rank_kwargs:dict=dict(), palette:dict=dict()):
    """
    Graph for plotting results of promotion / demotion ranking procedure

    Parameters
    ----------
    initial_rank: int|int
        Initial position of the alternative.
    new_positions: list
        List of positions acquired in promotion or demotion process.
    percentage_changes: list
        Changes of values of criteria in percentage values.
    kind: str, optional, default='bar'
        Changes style of the plot. Available options: 'bar', 'line'.
    title: str, optional, default=''
        Title that will be displayed as suptitle.
    ax: Axes, optional, default=None
        Axes object on which the graphs will be drawn.
    draw_ranking_change: bool, optional, default=True
        If True changes in ranking will be additionally drawn on second axis.
    height_ratio: list[int], optional, default=[1, 3]
        Sets ratio of rank_graph to percentage_graph.
    percentage_kwargs: dict, optional, default=dict()

    rank_kwargs: dict, optional, default=dict()
        Dictionary for styling rank_graph plots. Available keys: 'base_linestyle', 'base_linewidth', 'linestyle', 'linewidth', 'ylabel', 'title', 'marker', 'markersize'.
    palette: dict, optional, default=dict()
        Sets colors for specific part of the plot. Available keys: 'positive', 'neutral', 'negative'.

    Example
    -------
    >>> results = ranking_promotion(matrix, initial_ranking, copras, call_kwargs, ranking_descending, direction, step, max_modification=max_modification, return_zeros=return_zeros)
    >>> results = np.array(results)
    >>> for alt in range(matrix.shape[1]):
    >>>     alt_results = results[results[:, 0] == alt]
    >>>     percentage_changes = []
    >>>     new_positions = []
    >>>     if len(alt_results):
    >>>         for crit in range(matrix.shape[0]):
    >>>             r = alt_results[alt_results[:, 1] == crit]
    >>>             if len(r):
    >>>                 _ , crit, change, new_pos = r[0]
    >>>                 crit = int(crit)
    >>>                 percentage_changes.append((change - matrix[alt, crit])/matrix[alt, crit]*100)
    >>>                 new_positions.append(new_pos)
    >>>             else:
    >>>                 percentage_changes.append(0)
    >>>                 new_positions.append(initial_ranking[alt])
    >>>         
    >>>         promotion_graph(initial_ranking[alt], new_positions, np.array(percentage_changes), kind='bar', title=f'Rank promotion - $A_{{{alt+1}}}$')

    Returns
    -------
    ax : Axes
        Axes object on which graphs were drawn.

    """
    Validator.is_type_valid(initial_rank, (int, float))
    Validator.is_type_valid(new_positions, list)
    Validator.is_type_valid(percentage_changes, list)
    Validator.is_type_valid(xticks, (None, list))
    Validator.is_type_valid(kind, str)
    Validator.is_in_list(kind, ['bar', 'line'])
    Validator.is_type_valid(title, str)
    Validator.is_type_valid(ax, (None, mpl.axes.Axes))
    Validator.is_type_valid(draw_ranking_change, bool)
    Validator.is_type_valid(height_ratio, list)
    Validator.is_type_valid(percentage_kwargs, dict)
    Validator.is_type_valid(rank_kwargs, dict)
    Validator.is_type_valid(palette, dict)


    # if kind not in ['bar', 'line']:
    #     raise ValueError('Accepted kind parameter values are: `bar`, `kind`.')
    
    # if xticks is not None and len(xticks) != len(percentage_changes):
    #     raise ValueError('`xticks` should contain the same number of values as `percentage_changes`')
    if xticks is not None:
        Validator.is_shape_equal(len(xticks), len(percentage_changes))


    if not palette:
        palette = {
            'positive': '#4c72b0',
            'neutral': 'black',
            'negative': '#c44e52',
        }
    if not rank_kwargs:
        rank_kwargs = {
            'ylabel': 'Rank change',
        }

    if not percentage_kwargs:
        percentage_kwargs = {
            'ylabel': 'Criterion value change'
        }
                
    if ax is None:
        if draw_ranking_change:
            fig, (cax, main_ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratio})
            cax.tick_params(bottom=False, labelbottom=False)
        else:
            fig, main_ax = plt.subplots()
    else:
        if draw_ranking_change:
            try:
                (cax, main_ax) = ax
            except TypeError:
                raise TypeError("If 'draw_ranking_change'='True', the ax parameter needs to consist of two axes")
        else:
            main_ax = ax

    percentage_graph(percentage_changes, new_positions, xticks=xticks, percentage_kwargs=percentage_kwargs, palette=palette, ax=main_ax, kind=kind)

    if draw_ranking_change:
        rank_graph(initial_rank, new_positions, palette=palette, rank_kwargs=rank_kwargs, ax=cax)
        cax.set_xlim(main_ax.get_xlim())

    fig.align_ylabels()
    plt.suptitle(title)
    
    return ax