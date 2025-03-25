
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


def annotate_mosaic(
        fig: plt.Figure,
        axd: Dict[str, plt.Axes],
        fontsize: int = 14
) -> None:
    for key, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 0.95, key, transform=ax.transAxes + trans,
                fontsize=fontsize, va='bottom', fontfamily='sans-serif', fontweight='bold')


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

