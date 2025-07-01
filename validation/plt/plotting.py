
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

from typing import Tuple, Union, Dict, Literal, List

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from matplotlib.ticker import ScalarFormatter

from validation.utils.val_utils import eval_wrapper_sample_wise


def plot_param_lineplot(
        res_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        xlog10: bool = False,  # Trafo for the x-Axis
        xlog10plusone: bool = False,  # Trafo for the x-Axis
        custom_x_ticks: Union[List[Union[int, float]], Literal['log10_scale'], None] = None,
        x_label: Union[str, None] = None,
        y_label: Union[str, None] = None,
        markersize: float = 6.0,
        x_axis_grid: bool = False,
        abline_param_values: bool = False,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if xlog10 and xlog10plusone:
        raise ValueError("xlog10 and xlog10plusone cannot both be True")

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    x = res_df[x_col].to_numpy()
    x_og = x.copy()
    y = res_df[y_col].to_numpy()

    if xlog10:
        x = np.log10(x)
    if xlog10plusone:
        x = np.log10(1 + x)

    ax.plot(x, y, marker='o', markersize=markersize)

    if custom_x_ticks is not None:
        if custom_x_ticks == 'log10_scale':
            x_low = x_og.min()
            x_high = x_og.max()
            xt = [x_low, ]
            current_value = x_low
            while True:
                # Determine the current step size based on magnitude
                step = 10 ** (len(str(current_value)) - 1)

                # Find the next value based on the step
                next_value = ((current_value // step) + 1) * step

                # Stop if the next value exceeds x_high
                if next_value >= x_high:
                    break

                xt.append(next_value)
                current_value = next_value

            xt.append(x_high)

            def is_power_of_ten(n):
                while n % 10 == 0:
                    n //= 10
                return n == 1

            xt_labels = [tick if is_power_of_ten(n=tick) else '' for tick in xt]
            xt_labels[0] = x_low
            # xt_labels[-1] = x_high

        else:
            xt = custom_x_ticks
            xt_labels = custom_x_ticks

        if xlog10:
            xt = np.log10(xt)
        elif xlog10plusone:
            xt = np.log10(1 + xt)

        ax.set_xticks(xt, labels=xt_labels)

    ax.xaxis.grid(x_axis_grid)
    ax.yaxis.grid(False)

    if abline_param_values:
        offset = 0.05

        for i, value in enumerate(x):
            ax.axvline(x=value, color='grey', linestyle='-', linewidth=0.8, zorder=1)

            y_position = y.max() * 0.9 + (i % 2) * offset
            ax.text(
                value, y_position, f"{x_og[i]:.2f}",
                ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontsize=10, color='grey'
            )

    if x_label is not None:
        ax.set_xlabel(x_label)
    else:
        if xlog10:
            ax.set_xlabel(f'log10({x_col})')
        elif xlog10plusone:
            ax.set_xlabel(f'log10(1 + {x_col})')
        else:
            ax.set_xlabel(x_col)

    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(y_col)

    return ax


def plot_param_stripplot(
        res_df: pd.DataFrame,
        id_var: str,
        val_var: str,
        val_name: Union[str, None] = None,
        jitter: Union[bool, float] = True,
        xlabel: str = '',
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:
    val_name = val_name if val_name is not None else 'Mean Test Score'

    try:
        res_df[id_var] = res_df[id_var].astype(str)
    except TypeError:
        print("'id_var' values could not be turned into str")
        return

    # val_name = 'Score'
    df_melted = res_df.reset_index().melt(
        id_vars=['index', id_var], value_vars=val_var, value_name=val_name)

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    ax = sns.stripplot(
        data=df_melted, x=id_var, y=val_name, jitter=jitter,
        size=11, color='orange', edgecolor='auto', linewidth=1.0, ax=ax)
    ax.set_xlabel(xlabel)

    categories = df_melted[id_var].unique()
    collections = ax.collections

    if len(categories) != len(collections):
        raise ValueError("Mismatch between categories and collections. Check plot setup.")

    for category, collection in zip(categories, collections):

        # Filter DataFrame for this category
        category_data = df_melted[df_melted[id_var] == category]

        # Retrieve offsets for this collection
        offsets = collection.get_offsets()
        x_coords = offsets[:, 0]
        y_coords = offsets[:, 1]

        # Annotate points
        for (x, y), (_, row) in zip(zip(x_coords, y_coords), category_data.iterrows()):
            ax.text(
                x=x,
                y=y,
                s=str(row['index']),
                # Annotate with sample index
                fontsize=8,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )

    return ax


def plot_cv_results_mean_vs_std_scatter(
        res_df: pd.DataFrame,
        score_name: Union[str, None] = None,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    x_coords = res_df['mean_test_score'].to_numpy()
    y_coords = res_df['std_test_score'].to_numpy()

    ax.scatter(x=x_coords, y=y_coords)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.text(
            x=x,
            y=y,
            s=str(i),
            fontsize=8,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )

    if score_name is None:
        score_name = 'score'

    ax.set_xlabel(f'Mean {score_name}')
    ax.set_ylabel(f'Std {score_name}')

    ax.set_title(
        f'Min {score_name}: {np.round(x_coords.min(), 4)}, '
        f'max {score_name}: {np.round(x_coords.max(), 4)}, '
        f'range: {np.round(x_coords.max() - x_coords.min(), 4)}'
    )

    return ax


def plot_param_heatmap(
        res_df: pd.DataFrame,
        param_row: str,
        param_col: str,
        performance_score: str = 'mean_test_score',
        other_params: Union[dict, None] = None,
        title_fontsize: float = 12,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    # Problem, param pairs are not unique, two options:
    # 1.) Aggregate to remove duplicates
    #     agg_res_df = res_df.groupby([param_row, param_col])[performance_score].mean().reset_index()
    # 2.) Subset dataframe ~= fix all other parameters to specified value

    res_df = res_df.copy()

    if other_params is None:
        other_params = {}

    title_str = ''
    for key, val in other_params.items():
        res_df = res_df[res_df[key] == str(val)]
        title_str += f'{key.removeprefix('param_')}: {val} '

    # Pivot the data
    heatmap_data = res_df.pivot(index=param_row, columns=param_col, values=performance_score)

    # Plot the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".5f", ax=ax)
    ax.set_xlabel(param_col.removeprefix('param_'))
    ax.set_ylabel(param_row.removeprefix('param_'))
    ax.set_title(title_str, fontsize=title_fontsize)

    return ax


########################################################################################################################

def plot_performance_score_box_plot(
        sample_wise_res_dfs: List[List[pd.DataFrame]],
        method_names: Union[List[str], None] = None,
        dataset_names: Union[List[str], None] = None,
        score_mode: Literal['macro', 'micro', 'weighted'] = 'macro',
        y_label: Union[str, None] = None,
        title: Union[str, None] = None,
        palette: Union[str, List[str], Dict[str, str], Dict[str, Tuple[float, ...]], None] = None,  # {method: color}
        sns_boxplot_kwargs: Union[Dict, None] = None,
        plot_points: bool = False,
        point_kwargs: Union[Dict, None] = None,
        boxplot_alpha: Union[float, None] = None,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    num_datasets = len(sample_wise_res_dfs)
    num_methods = len(sample_wise_res_dfs[0])

    if dataset_names is None:
        dataset_names = [f"DS_{i + 1}" for i in range(num_datasets)]

    if method_names is None:
        method_names = [f"M_{i + 1}" for i in range(num_methods)]

    # Build long-form dataframe for seaborn
    long_data = []

    for dataset_idx, dataset_dfs in enumerate(sample_wise_res_dfs):  # Iterate over the datasets
        for method_idx in range(num_methods):
            method_name = method_names[method_idx]

            # Check if df is provided for this dataset-method combo
            try:
                df = dataset_dfs[method_idx]
                if df is not None and not df.empty and score_mode in df.columns:
                    scores = df[score_mode].dropna().tolist()
                else:
                    scores = []

            except IndexError:
                scores = []

            # Even if scores are empty, add NaNs for consistency
            if scores:
                for score in scores:
                    long_data.append({
                        'Dataset': dataset_names[dataset_idx],
                        'Method': method_name,
                        'Score': score
                    })
            else:
                # Add a single NaN row to preserve grouping/hue
                long_data.append({
                    'Dataset': dataset_names[dataset_idx],
                    'Method': method_name,
                    'Score': float('nan')
                })

    long_df = pd.DataFrame(long_data)


    if palette is None:
        palette = 'Set2'

    if sns_boxplot_kwargs is None:
        sns_boxplot_kwargs = dict()

    if point_kwargs is None:
        point_kwargs = dict()

    if plot_points:
        # Disable outliers
        sns_boxplot_kwargs.setdefault('showfliers', False)

    ax = sns.boxplot(
        data=long_df,
        x='Dataset',
        y='Score',
        hue='Method',
        palette=palette,
        zorder=2,
        ax=ax,
        **sns_boxplot_kwargs
    )

    if boxplot_alpha is not None:
        for patch in ax.patches:  # box patches
            patch.set_alpha(boxplot_alpha)
        # Lines: whiskers, caps, medians (in order of plotting)
        for line in ax.lines:
            line.set_alpha(boxplot_alpha)

        # Fliers (outlier dots)
        for col in ax.collections:
            col.set_alpha(boxplot_alpha)


    # Overlay individual scores
    if plot_points:
        point_kwargs.setdefault('alpha', 0.4)
        point_kwargs.setdefault('dodge', True)
        point_kwargs.setdefault('linewidth', 0.5)
        point_kwargs.setdefault('size', 3.0)
        point_kwargs.setdefault('jitter', True)

        ax = sns.stripplot(
            data=long_df,
            x='Dataset',
            y='Score',
            hue='Method',
            palette=palette,
            zorder=1,
            ax=ax,
            **point_kwargs
        )

        # Avoid duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        n = len(method_names)
        ax.legend(handles[:n], labels[:n], title='Method')

    ax.set_xlabel('Dataset')
    if y_label is None:
        y_label = 'Score'
    ax.set_ylabel(f'{score_mode.capitalize()} {y_label}')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    if title is not None:
        ax.set_title(title)

    return ax


def plot_cell_pop_size_pred_vs_gt(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
        percentage: bool = False,
        palette: Union[str, List[str], Dict[str, str], Dict[str, Tuple[float, ...]], None] = None,  # {method: color}
        title: Union[str, None] = None,
        point_size: Union[float, None] = None,
        show_r2: bool = False,
        show_pearson: bool = False,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    # Get the data
    plot_df = _get_plot_df(y_trues=y_trues, y_preds=y_preds)

    x_col = 'counts_yt'
    y_col = 'counts_yp'

    # Compute percentages
    if percentage:
        plot_df['percent_yt'] = plot_df['counts_yt'] / plot_df['totals'] * 100
        plot_df['percent_yp'] = plot_df['counts_yp'] / plot_df['totals'] * 100

        x_col = 'percent_yt'
        y_col = 'percent_yp'

    # Convert label column to string or categorical for better Seaborn color handling
    plot_df['labels'] = plot_df['labels'].astype(str)

    if palette is None:
        palette = 'Set2'

    # Plot
    scatter_kwargs = dict(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue='labels',
        palette=palette,
        alpha=0.7,
        edgecolor='k',
        ax=ax,
    )

    # Add size parameter
    if point_size is not None:
        scatter_kwargs['s'] = point_size

    ax = sns.scatterplot(**scatter_kwargs)

    # Diagonal reference line (ideal match)
    max_val_x = plot_df[x_col].max()
    max_val_y = plot_df[y_col].max()
    ax.plot([0, max_val_x], [0, max_val_y], linestyle='--', color='grey', linewidth=1.0, zorder=0)

    # Labels & formatting
    x_label = 'Population Size'
    y_label = 'Predicted Population Size'
    if percentage:
        x_label += ' (%)'
        y_label += ' (%)'
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title is not None:
        # 'Predicted vs True Cell Type Proportions'
        ax.set_title(title)

    # ax.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(title='Cell Type', loc='best')
    ax.set_axisbelow(True)
    ax.grid(True)

    if show_r2 or show_pearson:

        # Compute and annotate stats
        x_vals = plot_df[x_col].to_numpy()
        y_vals = plot_df[y_col].to_numpy()

        stats_text = []

        if show_r2:
            r2 = r2_score(x_vals, y_vals)
            stats_text.append(f"$R^2 = {r2:.4f}$")

        if show_pearson:
            r, p = pearsonr(x_vals, y_vals)
            stats_text.append("Pearson's " + f"$r = {r:.4f}$")

        if stats_text:
            ax.text(
                0.05, 0.95, "\n".join(stats_text),
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

    return ax


def _get_plot_df(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
) -> pd.DataFrame:

    unique_labels = np.unique(np.concatenate(y_preds + y_trues))

    sample_ids = []
    labels = []
    counts_yt = []
    counts_yp = []
    totals = []

    for i, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):

        total = y_true.shape[0]

        yt_vals, yt_counts = np.unique(y_true, return_counts=True)
        yp_vals, yp_counts = np.unique(y_pred, return_counts=True)

        yt_count_label_count_mapping = dict(zip(yt_vals, yt_counts))
        yp_count_label_count_mapping = dict(zip(yp_vals, yp_counts))

        for label in unique_labels:

            sample_ids.append(i)
            labels.append(label)
            counts_yt.append(yt_count_label_count_mapping.get(label, 0))
            counts_yp.append(yp_count_label_count_mapping.get(label, 0))
            totals.append(total)

    df = pd.DataFrame()
    df['sample_ids'] = sample_ids
    df['labels'] = labels
    df['counts_yt'] = counts_yt
    df['counts_yp'] = counts_yp
    df['totals'] = totals

    return df


def plot_performance_score_box_plot_cw(
        sample_wise_res_dfs: List[pd.DataFrame],
        method_names: Union[List[str], None] = None,
        y_label: Union[str, None] = None,
        title: Union[str, None] = None,
        palette: Union[str, List[str], Dict[str, str], Dict[str, Tuple[float, ...]], None] = None,  # {method: color}
        sns_boxplot_kwargs: Union[Dict, None] = None,
        plot_points: bool = False,
        point_kwargs: Union[Dict, None] = None,
        boxplot_alpha: Union[float, None] = None,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    # Get the number of methods
    num_methods = len(sample_wise_res_dfs)

    # Get the classes
    classes = []
    for res_df in sample_wise_res_dfs:
        for c in res_df.columns.tolist():
            if c != -1 and c != 'NA':
                classes.append(c)
    classes = list(set(classes))

    # print(classes)
    # num_classes = len(classes)

    if method_names is None:
        method_names = [f"M_{i + 1}" for i in range(num_methods)]

    # Build long-form dataframe
    long_data = []

    for method_idx, method_df in enumerate(sample_wise_res_dfs):
        method_name = method_names[method_idx]
        for class_label in method_df.columns:
            if class_label == -1 or class_label == str(-1) or class_label == 'NA':
                continue
            scores = method_df[class_label].tolist()
            for score in scores:
                long_data.append({
                    'Method': method_name,
                    'Cell Type Label': class_label,  # str(int(class_label)),
                    'Score': score
                })

    long_df = pd.DataFrame(long_data)

    if palette is None:
        palette = 'Set2'

    if sns_boxplot_kwargs is None:
        sns_boxplot_kwargs = dict()

    if point_kwargs is None:
        point_kwargs = dict()

    if plot_points:
        # Disable outliers
        sns_boxplot_kwargs.setdefault('showfliers', False)

    # Change order such that others is always plotted at the end
    label_order = sorted(set(long_df['Cell Type Label']))
    if 'O' in label_order:
        label_order.remove('O')
        label_order.append('O')

    ax = sns.boxplot(
        data=long_df,
        x='Cell Type Label',
        y='Score',
        hue='Method',
        order=label_order,
        palette=palette,
        zorder=2,
        ax=ax,
        **sns_boxplot_kwargs
    )

    if boxplot_alpha is not None:
        for patch in ax.patches:  # box patches
            patch.set_alpha(boxplot_alpha)
        # Lines: whiskers, caps, medians (in order of plotting)
        for line in ax.lines:
            line.set_alpha(boxplot_alpha)

        # Fliers (outlier dots)
        for col in ax.collections:
            col.set_alpha(boxplot_alpha)


    # Overlay individual scores
    if plot_points:
        point_kwargs.setdefault('alpha', 0.4)
        point_kwargs.setdefault('dodge', True)
        point_kwargs.setdefault('linewidth', 0.5)
        point_kwargs.setdefault('size', 3.0)
        point_kwargs.setdefault('jitter', True)

        ax = sns.stripplot(
            data=long_df,
            x='Cell Type Label',
            y='Score',
            hue='Method',
            order=label_order,
            palette=palette,
            legend=False,
            zorder=1,
            ax=ax,
            **point_kwargs
        )

        # Avoid duplicate legends
        # handles, labels = ax.get_legend_handles_labels()
        # n = len(method_names)
        # ax.legend(handles[:n], labels[:n], title='Method')

    ax.set_xlabel('Cell Type')
    if y_label is None:
        y_label = 'Score'
    ax.set_ylabel(f'{y_label}')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    if title is not None:
        ax.set_title(title)

    return ax


def plot_sample_sizes(
        ys: List[np.ndarray],
        title: Union[str, None] = None,
        abline_mean: bool = False,
        abline_std: bool = False,
        print_total: bool = False,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    # Enforce scientific notation for y axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((3, 3))
    ax.yaxis.set_major_formatter(formatter)

    sample_sizes = sorted([a.shape[0] for a in ys])
    sample_sizes = np.array(sample_sizes)

    labels = list(range(1, len(ys) + 1))

    ax.bar(
        x=labels,
        height=sample_sizes,
        color='lightblue',
        edgecolor='darkgray',
        linewidth=1.0
    )

    if print_total:
        total = sample_sizes.sum()
        ax.text(
            0.98, 0.98, f'Total Events: {total}',
            transform=ax.transAxes,  # Axes coordinates (0–1)
            ha='right', va='top',  # Align top-right
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Transparent box
        )

    if abline_mean:
        m = sample_sizes.mean()
        ax.axhline(y=m, color='darkred', linestyle='-', linewidth=1.5, label=f'Mean: {int(m)}')
        ax.legend()

    if abline_std:
        m = sample_sizes.mean()
        std = sample_sizes.std()

        ax.axhline(y=m - std, color='gold', linestyle='--', linewidth=1.5, label=f'Std: {np.round(std, 3)}')
        ax.axhline(y=m + std, color='gold', linestyle='--', linewidth=1.5)
        ax.legend()

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(f'Sample ID (1--{len(ys)})')

    ax.set_ylabel('Number of Events')


    return ax


def plot_class_balance(
        ys: List[np.ndarray],
        title: Union[str, None] = None,
        palette: Union[str, List[str], Dict[str, str], Dict[str, Tuple[float, ...]], None] = None,  # {method: color}
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    if palette is None:
        palette = 'Accent'

    # Enforce scientific notation for y axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((3, 3))
    ax.yaxis.set_major_formatter(formatter)

    # Concatenate label vectors
    all_labels = np.concatenate(ys)
    class_counts = pd.Series(all_labels).value_counts().sort_index()
    total = class_counts.sum()
    percentages = (class_counts / total * 100).round(2)

    # Prepare DataFrame for plotting
    plot_data = pd.DataFrame({
        'Cell Type Label': class_counts.index,
        'Count': class_counts.to_numpy(),
        'Percentage': percentages.to_numpy(),
    })

    # Change order such that others is always plotted at the end
    label_order = sorted(set(plot_data['Cell Type Label']))
    if 'Out' in label_order:
        label_order.remove('Out')
        label_order.append('Out')
    if 'O' in label_order:
        label_order.remove('O')
        label_order.append('O')

    # Barplot
    sns.barplot(
        data=plot_data,
        x='Cell Type Label',
        y='Count',
        hue='Cell Type Label',
        order=label_order,
        palette=palette,
        legend=False,
        ax=ax
    )

    # Add vertical labels
    max_count = plot_data['Count'].max()
    for idx, label in enumerate(label_order):
        row = plot_data[plot_data['Cell Type Label'] == label].iloc[0]
        is_max = row['Count'] == max_count
        y_pos = (row['Count'] * 0.85) if is_max else (row['Count'] + (0.01 * total))

        ax.text(
            x=idx,
            y=y_pos,
            s=f"{int(row['Count'])}\n{row['Percentage']}%",
            ha='center',
            va='bottom',
            fontsize=8
        )

    if title is not None:
        ax.set_title(title)


    ax.set_ylabel('Number of Events')


    return ax



















def plot_prec_rec_vs_thresh(
        y_trues: List[np.ndarray],
        y_probs: List[np.ndarray],
        thresholds: Union[List[float], None] = None,
        pos_label: int = 1,
        neg_label: int = 0,
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    if any([np.unique(y_true).shape[0] >= 3 for y_true in y_trues]):
        raise ValueError("Function assumes binary labels. 'y_trues' contains more than two classes.")

    if ax is None:
        fig, ax = plt.subplots()

    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    precs = []
    recs = []

    for threshold in thresholds:
        # Get prediction for current threshold
        y_preds = [(y_prob >= threshold).astype(int) for y_prob in y_probs]

        # Relabel if necessary
        if pos_label != 1:
            for y_pred in y_preds:
                y_pred[y_pred == 1] = pos_label
        if neg_label != 0:
            for y_pred in y_preds:
                y_pred[y_pred == 0] = neg_label

        out = eval_wrapper_sample_wise(
            y_trues=y_trues,
            y_preds=y_preds,
            abstention_label=None,
            others_label=None,
            pos_label=pos_label,
            verbosity=1,
        )

        res_df_avg_prec = out[0]
        res_df_avg_rec = out[1]

        precs.append(res_df_avg_prec.loc['mean', 'binary'])
        recs.append(res_df_avg_rec.loc['mean', 'binary'])

    ax.plot(thresholds, precs, label='Precision', marker='o', c='b')
    ax.plot(thresholds, recs, label='Recall', marker='o', c='g')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score (prec, rec), binary')

    plt.legend(loc='best')

    return ax


def plot_n_samples_n_events(
        res_df: pd.DataFrame,
        cmap_name: str = 'magma',
        ax: Union[plt.Axes, None] = None,
) -> plt.Axes:

    # Instantiate axis
    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    # Get cmap
    cmap = plt.get_cmap(cmap_name)

    for i, (idx, row) in enumerate(res_df.iterrows()):
        # Get row values that are not NaNs
        valid_values = row.dropna()
        # Skip rows with all NaNs
        if valid_values.empty:
            continue

        ax.plot(
            valid_values.index.astype(int),
            valid_values.values,
            marker='o',
            c=cmap(i / (res_df.shape[0] - 1)),
            label=idx
        )

    # Optional: add labels, title, legend, etc.
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Macro F1')
    ax.legend(title='Number of Events')

    return ax




def annotate_mosaic(fig: plt.Figure, axd: Dict[str, plt.Axes], fontsize: Union[float, None] = None):
    # Annotate subplot mosaic tiles with labels
    for label, ax in axd.items():
        # ax = fig.add_subplot(axd[label])
        # ax.annotate(label, xy=(0.1, 1.1), xycoords='axes fraction', ha='center', fontsize=16)
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            0.95,
            label,
            transform=ax.transAxes + trans,
            fontsize=fontsize,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )














def plot_support_hists(
        som_c,  # Todo
        n_bins: int = 100,
        verbosity: int = 0,
        dpi: int = 100,
        plot_class_wise: bool = False,
        n_bins_class_wise: int = 20,
        plot_act_freq: bool = False,
        save_p: Union[str, None] = None,
        ax: Union[plt.Axes, None] = None,
):
    support = som_c.class_counts_per_unit_.sum(axis=2)
    if verbosity >= 1:
        print(f'# ### Total support (n events): {support.sum()}')
        print(f'# ### Max support: {support.max()}')
        print(f'# ### Min support: {support.min()}')
        print(f'# ### Class of unit with smallest support in training set: '
              f'{som_c.som_unit_labels_[np.unravel_index(np.argmin(support), support.shape)]}')

    if ax is None:
        _, ax = plt.subplots(dpi=dpi)

    ax.hist(support.flatten(), bins=n_bins, color='lightblue', edgecolor='grey')

    if save_p is not None:
        plt.savefig(os.path.join(save_p, 'support_hist.png'))
        plt.close('all')

    if plot_class_wise:
        # Plot the for each class the size of the support (events for which the unit is bmu)
        # of the units that predict this class
        # => Are there classes with BMUs that predict them that are bmu only for very few cells
        colormap = plt.get_cmap('tab10', len(som_c.og_classes_))
        for i, c in enumerate(som_c.classes_):
            c_bool = (som_c.som_unit_labels_ == c)
            c_supp_vals = support[c_bool]
            fig, ax_dummy = plt.subplots(dpi=dpi)
            ax_dummy.hist(
                c_supp_vals, color=colormap(i), alpha=0.6, edgecolor='grey', label=f'{int(c)}', bins=n_bins_class_wise)
            ax_dummy.set_title(
                f'Class: {som_c.new_to_og_classes_dict_[c]}, '
                f'Total units: {c_bool.sum()}, '
                f'total support: {int(c_supp_vals.sum())}')
            if save_p is not None:
                plt.savefig(os.path.join(save_p, f'support_hist_c{som_c.new_to_og_classes_dict_[c]}.png'))
                plt.close('all')

    if plot_act_freq:
        freq = support / support.sum()
        freq_sci = np.vectorize(lambda x: f"{x:.2e}")(freq)

        fig, ax_dummy = plt.subplots(dpi=dpi)
        sns.heatmap(freq, cmap='Blues', annot=freq_sci, annot_kws={'size': 2}, fmt='', ax=ax_dummy)
        if save_p is not None:
            plt.savefig(os.path.join(save_p, 'act_freq_heatmap.png'))
            plt.close('all')

        fig, ax_dummy = plt.subplots(dpi=dpi)
        ax_dummy.hist(freq.flatten(), color='green', alpha=0.6, edgecolor='grey', bins=n_bins)
        if save_p is not None:
            plt.savefig(os.path.join(save_p, 'act_freq_hist.png'))
            plt.close('all')

        if verbosity >= 1:
            print(f'Activation frequencies sorted: {np.sort(freq.flatten()).tolist()}')
            print(f'Activation frequency is zero {(freq == 0).sum()} times')




















# Todo: Assess if this is needed at some point

def plot_support_hist_w_class_perc(
        som_c,  # Todo
        class_label: Union[int, None] = None,
        n_bins: int = 20,
        plot_percentages: bool = False,
        fontsize: Union[float, None] = None,
        plot_title: bool = False,
        ax: Union[plt.Axes, None] = None,
        dpi: int = 100,
):
    # ### Get class count per unit and number of classes from SOM classifier
    cc_per_unit = som_c.class_counts_per_unit_.copy()  # somdim0 x somdim1 x n_classes
    n_classes = som_c.og_classes_.shape[0]

    # ### Calculate support of each unit
    support = cc_per_unit.sum(axis=2).flatten()  # shape: n_units

    # Flatten the class count per unit array
    cc_per_unit_flat = cc_per_unit.reshape(-1, n_classes)  # shape: n_units x n_classes
    # Note: flatten, reshape works row by row
    # => entries in 0-dimension correspond to same unit in 'support' and 'cc_per_unit_flat'

    # ### If passed, subset 'support' and 'cc_per_unit_flat' to units associated with 'class_label'
    if class_label is not None:
        new_class_label = None
        for key, value in som_c.new_to_og_classes_dict_.items():
            if value == class_label:
                new_class_label = key
        if new_class_label is not None:
            c_bool = (som_c.som_unit_labels_ == new_class_label).flatten()
            support = support[c_bool]
            cc_per_unit_flat = cc_per_unit_flat[c_bool, :]
        else:
            warnings.warn(
                "'class_label' is not a class that the SomClassifier can predict, "
                "proceeding without subsetting SOM units", UserWarning)

    # Define histogram (bin values and bins edges)
    bin_vals, bins = np.histogram(support, bins=n_bins)

    # ### For each bin calculate the label fraction across all events that are in the support of units of that bin
    # Initialize counts for each label in the bins, shape: n_bins x n_classes
    bin_wise_label_count = np.zeros((n_bins, n_classes))

    # Calculate counts for each bin and label
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        # Create bool for which units the support value is in the respective bin
        if i == len(bins) - 2:  # Last bin, included upper bound
            in_bin = (low <= support) & (support <= high)
        else:
            in_bin = (low <= support) & (support < high)
        # in_bin = (support >= low) & (support < high)
        if in_bin.any():
            # Sum label counts across units that are in the bin
            bin_wise_label_count[i, :] = cc_per_unit_flat[in_bin].sum(axis=0)

    # Calculate the label fractions in each bin, shape: n_bins x n_classes
    relative_bin_wise_label_count = np.zeros_like(bin_wise_label_count)
    row_sums = bin_wise_label_count.sum(axis=1)
    non_zero_rows = row_sums != 0
    relative_bin_wise_label_count[non_zero_rows] = (
            bin_wise_label_count[non_zero_rows] / row_sums[non_zero_rows, None]
    )

    # ### Plotting
    # Define centers of bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    # Define bottom and top of the respective bars (one barplot per class),
    bar_edges = np.zeros((n_bins, n_classes + 1))
    for i in range(n_classes):
        bar_edges[:, i + 1] = bar_edges[:, i].copy() + relative_bin_wise_label_count[:, i] * bin_vals

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    # Plot the stacked bar histogram
    for i in range(n_classes):
        ax.bar(
            bin_centers,
            bar_edges[:, i + 1] - bar_edges[:, i],  # Bar height starts from bottom
            width=np.diff(bins),
            bottom=bar_edges[:, i],
            color=colors[i],
            edgecolor='black',
            align='center',
            label=f'{int(som_c.new_to_og_classes_dict_[i])}'
        )

    if plot_percentages:
        for i in range(n_bins):
            percentages_str = '%: '
            for j in range(n_classes):
                frac = relative_bin_wise_label_count[i, j]
                if frac > 0:
                    percentages_str += f'{int(som_c.new_to_og_classes_dict_[j])}: {np.round(frac * 100, 2)}'
                    percentages_str += ', ' if j <= n_classes - 2 else ''
            ax.text(
                x=bin_centers[i], y=ax.get_ylim()[1] * 0.05, s=percentages_str, rotation=90, ha='center',
                fontsize=fontsize
            )

    # Add legend and labels
    # ax.set_title('Distribution of support sizes across SOM units')
    ax.set_xlabel('Support, all units' if class_label is None else f'Support, Class {class_label} units')
    ax.set_ylabel('# SOM units')
    ax.legend(title='Labels')

    if plot_title:
        ax.set_title(
            f'Class: {class_label}, '
            f'Total units: {support.shape[0]}, '
            f'Total support: {int(support.sum())}')

































































# Todo: Add some of the som plotting functoins to the flagx package




def plot_som(
        labels_unit_wise: np.ndarray,
        label_mapping: Union[dict, None] = None,
        mask: Union[np.ndarray, None] = None,
        mask_color: str = 'oldlace',
        cmap: Union[str, mcolors.ListedColormap, None] = None,
        prefer_seaborn_cmap: bool = True,
        plot_legend: bool = True,
        title: Union[str, None] = None,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
        **heatmap_kwargs,
) -> plt.Axes:
    # Get Axes object
    if ax == None:
        fig, ax = plt.subplots(dpi=dpi)

    # Get cmap
    if cmap is None:
        cmap_instance = plt.get_cmap('tab20')
    elif isinstance(cmap, str):
        cmap_instance = _get_cmap(cmap, use_seaborn_cmap=prefer_seaborn_cmap)
    else:
        cmap_instance = cmap

    # Get original labels
    if label_mapping is not None:
        og_labels_unit_wise = np.vectorize(lambda v: label_mapping.get(v, v))(labels_unit_wise)
    else:
        og_labels_unit_wise = labels_unit_wise.copy()

    # Apply mask
    if mask is not None:
        og_labels_unit_wise = np.where(mask, np.nan, og_labels_unit_wise)

    # Create a BoundaryNorm to ensure each unique label gets a distinct color
    # unique_labels = np.unique(og_labels_unit_wise)
    # norm = mcolors.BoundaryNorm(unique_labels - 0.5, len(unique_labels))
    # sns.heatmap(og_labels_unit_wise, cmap=cmap_instance, cbar=False, annot=True, ax=ax, norm=norm, **heatmap_kwargs)
    sns.heatmap(
        labels_unit_wise, cmap=cmap_instance, cbar=False, annot=og_labels_unit_wise,  mask=mask,ax=ax, **heatmap_kwargs
    )

    outline_thickness = heatmap_kwargs.get('linewidths', 2.0)
    outline_color = heatmap_kwargs.get('linecolor', 'white')
    for _, spine in ax.spines.items():
        spine.set_color(outline_color)
        spine.set_linewidth(outline_thickness)

    fs = heatmap_kwargs.get('annot_kws', {}).get('fontsize', matplotlib.rcParams['font.size'])

    if mask is not None:
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i), 1, 1,
                            color=mask_color, ec=outline_color, lw=outline_thickness, zorder=-1
                        )
                    )
                    ax.text(j + 0.5, i + 0.5, 'x', ha='center', va='center', fontsize=fs)

    if plot_legend:
        unique_labels = np.unique(labels_unit_wise)

        colors = [cmap_instance(i) for i in range(len(unique_labels))]

        # Create patches for legend
        patches = [mpatches.Patch(color=colors[i], label=label_mapping.get(lbl, lbl) if label_mapping else lbl)
                   for i, lbl in enumerate(unique_labels)]

        # Add custom legend
        ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10)

    if title is not None:
        ax.set_title(title)

    return ax


def plot_som_pies(
        class_counts_per_unit: np.ndarray,
        som_dimensions: Tuple[int, int],
        label_mapping: Union[dict, None] = None,
        radius: float = 1.0,
        cmap: Union[str, mcolors.ListedColormap, None] = None,
        prefer_seaborn_cmap: bool = True,
        title: Union[str, None] = None,
        add_grid_labels: bool = True,
        plot_legend: bool = True,
        apply_tightlayout: bool = True,
        figsize: Tuple[int, int] = (9, 9),
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
        cache_dir: Union[str, None] = None,
):

    # Create a new figure and axes if not provided
    fig, axs = plt.subplots(
        som_dimensions[0], som_dimensions[1],
        figsize=figsize, subplot_kw={'aspect': 'equal'}, dpi=dpi,
    )

    fig.subplots_adjust(wspace=0, hspace=0)  # Remove space around subplots

    # Calculate fractions and iterate over axes and plot the corresponding pie chart
    som_grid_fractions = np.zeros_like(class_counts_per_unit)
    n_events_per_unit = class_counts_per_unit.sum(axis=2, keepdims=True)
    nonzero_bool = (n_events_per_unit != 0).squeeze(axis=2)
    som_grid_fractions[nonzero_bool, :] = class_counts_per_unit[nonzero_bool, :] / n_events_per_unit[nonzero_bool, :]
    # som_grid_fractions = class_counts_per_unit / class_counts_per_unit.sum(axis=2, keepdims=True)

    # Define colormap if provided
    if cmap is not None:

        if isinstance(cmap, str):
            cmap_instance = _get_cmap(cmap, use_seaborn_cmap=prefer_seaborn_cmap)
        else:
            cmap_instance = cmap

        # colors = [cmap_instance(i) for i in range(som_grid_fractions.shape[2])]
        colors = [cmap_instance(i / max(som_grid_fractions.shape[2] - 1, 1)) for i in
                  range(som_grid_fractions.shape[2])]

    else:
        colors = None

    for i in range(som_dimensions[0]):
        for j in range(som_dimensions[1]):
            class_fracs = som_grid_fractions[i, j, :]

            ax_pie = axs[i, j]

            if np.all(class_fracs == 0):
                ax_pie.set_facecolor('black')
                ax_pie.spines['top'].set_visible(False)
                ax_pie.spines['right'].set_visible(False)
                ax_pie.spines['left'].set_visible(False)
                ax_pie.spines['bottom'].set_visible(False)
                ax_pie.set_xticks([])
                ax_pie.set_yticks([])

            else:
                # ax_pie.text(0.5, 0.5, f'{i, j}')  # just for checking

                # Plot the pie chart on the current axis
                patches, _ = ax_pie.pie(class_fracs, colors=colors, radius=radius)
                ax_pie.set_xticks([])
                ax_pie.set_yticks([])

    if add_grid_labels:
        # Add labels to the columns
        for col, ax_pie in enumerate(axs[0]):
            fig.text(
                ax_pie.get_position().x0 + ax_pie.get_position().width / 2,
                0.92,
                f'{col}',
                ha='center',
                va='bottom'
            )

        # Add labels to the rows
        for row, ax_pie in enumerate(axs[:, 0]):
            fig.text(
                0.05,
                ax_pie.get_position().y0 + ax_pie.get_position().height / 2,
                f'{row}',
                ha='right',
                va='center'
            )

    # Create a legend for the figure using patches from one of the plots
    if plot_legend:
        if label_mapping is not None:
            labels = [label_mapping[key] for key in range(som_grid_fractions.shape[2])]
        else:
            labels = np.array(list(range(som_grid_fractions.shape[2])))

        # fig.legend(handles=patches, labels=labels, loc='center right', ncol=1)

        fig.legend(
            handles=patches,
            labels=labels,
            loc='center right',
            fontsize=20,
            frameon=True,
            borderpad=1.2,
            ncol=1,
        )

    if title is not None:
        if ax is None:
            fig.suptitle(title)
        else:
            ax.set_title(title)

    if apply_tightlayout:
        fig.tight_layout()

    if ax is not None:
        if cache_dir is None:
            cache_dir = os.getcwd()

        fig.savefig(f'{cache_dir}/som_pies.png', bbox_inches='tight', pad_inches=0.05, dpi=dpi)
        plt.close(fig)
        img = mpimg.imread(f'{cache_dir}/som_pies.png')
        ax.imshow(img)
        ax.axis('off')
        os.remove(f'{cache_dir}/som_pies.png')

    return fig, axs


def plot_som_disks(
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        scatter_kwargs: dict,
        labels: Union[np.ndarray, None] = None,
        cmap: Union[str, mcolors.ListedColormap, None] = None,
        prefer_seaborn_cmap: bool = True,
        plot_legend: bool = True,
        circle_x_y_r: Union[np.ndarray, None] = None,
        title: Union[str, None] = None,
        dpi: int = 100,
        aspect_ratio: Union[Literal['auto', 'equal'], float, None] = 'equal',
        ax: Union[plt.Axes, None] = None,

) -> plt.Axes:

    # Create fig and ax
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    # Set aspect ratio
    if aspect_ratio is not None:
        ax.set_aspect(aspect_ratio)

    if labels is not None:
        # Get unique labels and assign them sequential indices
        unique_labels, label_indices = np.unique(labels, return_inverse=True)

        # Define colormap
        if cmap is None:
            cmap_instance = plt.get_cmap('tab20')
        elif isinstance(cmap, str):
            cmap_instance = _get_cmap(cmap, use_seaborn_cmap=prefer_seaborn_cmap)
        else:
            cmap_instance = cmap

        if plot_legend:
            # Get correct colors from the colormap
            colors = [cmap_instance(i / max(len(unique_labels) - 1, 1)) for i in range(len(unique_labels))]

            # Create patches for legend
            patches = [mpatches.Patch(color=colors[i], label=str(lbl)) for i, lbl in enumerate(unique_labels)]

            # Add custom legend
            ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10)

    else:
        label_indices = None
        cmap_instance = None

    ax.scatter(x_vals, y_vals, c=label_indices, cmap=cmap_instance, edgecolors='none', **scatter_kwargs)

    if circle_x_y_r is not None:
        for i in range(circle_x_y_r.shape[0]):
            x = circle_x_y_r[i, 0]
            y = circle_x_y_r[i, 1]
            r = circle_x_y_r[i, 2]

            circle = plt.Circle((x, y), r, alpha=0.6, edgecolor='red', facecolor='none', linewidth=0.5)
            ax.add_patch(circle)

        ax.set_xticks(np.unique(circle_x_y_r[:, 0]))
        ax.set_yticks(np.unique(circle_x_y_r[:, 1]))

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    return ax


def plot_support(
        support: np.ndarray,
        plot_activation_frequency: bool = False,
        cmap: Union[str, mcolors.Colormap, None] = None,
        prefer_seaborn_cmap: bool = True,
        plot_cbar: bool = True,
        annotate: bool = True,
        title: Union[str, None] = None,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
        **heatmap_kwargs,
) -> plt.Axes:

    # Compute activation frequencies
    if plot_activation_frequency:
        support = support / support.sum()

    # Get Axes object
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    # Get cmap
    if cmap is None:
        cmap_instance = plt.get_cmap('viridis')
    elif isinstance(cmap, str):
        cmap_instance = _get_cmap(cmap, use_seaborn_cmap=prefer_seaborn_cmap)
    else:
        cmap_instance = cmap

    sns.heatmap(
        support,
        cmap=cmap_instance,
        cbar=plot_cbar,
        annot=annotate,
        ax=ax,
        **heatmap_kwargs
    )

    outline_thickness = heatmap_kwargs.get('linewidths', 2.0)
    outline_color = heatmap_kwargs.get('linecolor', 'white')
    for _, spine in ax.spines.items():
        spine.set_color(outline_color)
        spine.set_linewidth(outline_thickness)

    if title is not None:
        ax.set_title(title)

    return ax


def plot_support_entropy_scatter(
        class_counts_per_unit: np.ndarray,
        label_mapping: Union[dict, None] = None,
        cmap: Union[str, mcolors.ListedColormap, None] = None,
        prefer_seaborn_cmap: bool = True,
        plot_legend: bool = True,
        title: Union[str, None] = None,
        ax: Union[plt.Axes, None] = None,
        **scatter_kwargs
):

    support = class_counts_per_unit.sum(axis=2)

    # Get class frequencies for each unit
    frequencies = np.divide(
        class_counts_per_unit,
        class_counts_per_unit.sum(axis=2, keepdims=True),
        where=class_counts_per_unit.sum(axis=2, keepdims=True) != 0
    )

    # Calculate entropy only on nonzero entries, if support was 0 then entropy is 0
    mask = frequencies > 0
    log_freq = np.full_like(frequencies, 0)
    log_freq[mask] = np.log2(frequencies[mask])
    entropies = - np.sum(frequencies * log_freq, axis=2)

    # Set entropy to Nan for units with support 0
    entropies[support == 0] = np.nan

    # Get unit labels
    labels_unit_wise = class_counts_per_unit.argmax(axis=2)
    if label_mapping is not None:
        og_labels_unit_wise = np.vectorize(lambda v: label_mapping.get(v, v))(labels_unit_wise)
    else:
        og_labels_unit_wise = labels_unit_wise.copy()

    if ax is None:
        fig, ax = plt.subplots()

    if cmap is None:
        cmap_instance = plt.get_cmap('tab20')
    elif isinstance(cmap, str):
        cmap_instance = _get_cmap(cmap, use_seaborn_cmap=prefer_seaborn_cmap)
    else:
        cmap_instance = cmap

    # Get unique labels and assign them sequential indices
    unique_labels, label_indices = np.unique(og_labels_unit_wise, return_inverse=True)

    ax.scatter(support.flatten(), entropies.flatten(), c=label_indices, cmap=cmap_instance, **scatter_kwargs)

    if plot_legend:
        # unique_labels = np.unique(og_labels_unit_wise)
        # colors = [cmap_instance(i / len(unique_labels)) for i in range(len(unique_labels))]

        # Get correct colors from the colormap
        colors = [cmap_instance(i / max(len(unique_labels) - 1, 1)) for i in range(len(unique_labels))]

        # Create patches for legend
        # patches = [mpatches.Patch(color=colors[i], label=label_mapping.get(lbl, lbl) if label_mapping else lbl)
        #            for i, lbl in enumerate(unique_labels)]
        patches = [mpatches.Patch(color=colors[i], label=str(lbl))
                   for i, lbl in enumerate(unique_labels)]

        # Add custom legend
        ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10)

    ax.set_xlabel('Support')
    ax.set_ylabel('Entropy')

    if title is not None:
        ax.set_title(title)

    return ax


def plot_umatrix(
        umatrix: np.ndarray,
        cmap: Union[str, mcolors.Colormap, None] = None,
        prefer_seaborn_cmap: bool = True,
        plot_cbar: bool = True,
        title: Union[str, None] = None,
        ax: Union[plt.Axes, None] = None,
        **imshow_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    if cmap is None:
        cmap_instance = plt.get_cmap('viridis')
    elif isinstance(cmap, str):
        cmap_instance = _get_cmap(cmap, use_seaborn_cmap=prefer_seaborn_cmap)
    else:
        cmap_instance = cmap

    out = ax.imshow(
        umatrix,
        cmap=cmap_instance,
        **imshow_kwargs
    )

    if plot_cbar:
        cbar = plt.colorbar(out, ax=ax, orientation='horizontal', shrink=0.5)
        cbar.set_label("U-matrix Values")

    ax.axis('off')

    if title is not None:
        ax.set_title(title)

    return ax


def _get_cmap(cmap: str, use_seaborn_cmap: bool = False) -> mcolors.Colormap:
    try:
        if not use_seaborn_cmap:
            cmap_instance = cm.get_cmap(cmap)
        else:
            # sns_palette = sns.color_palette(cmap)
            # cmap_instance = mcolors.ListedColormap(sns_palette)
            cmap_instance = sns.color_palette(cmap, as_cmap=True)
    except ValueError:
        warnings.warn(f"'{cmap}' is not a valid {'matplotlib' if not use_seaborn_cmap else 'seaborn'} colormap, "
                      f"trying {'matplotlib' if use_seaborn_cmap else 'seaborn'}.")

        try:
            if use_seaborn_cmap:
                cmap_instance = cm.get_cmap(cmap)
            else:
                # sns_palette = sns.color_palette(cmap)
                # cmap_instance = mcolors.ListedColormap(sns_palette)
                cmap_instance = sns.color_palette(cmap, as_cmap=True)

        except ValueError:
            # Fallback to a default Matplotlib colormap
            cmap_instance = cm.get_cmap('tab20')
            warnings.warn(f"'{cmap}' is not a recognized colormap, using matplotlib's 'tab20' instead.")

    return cmap_instance

