
import os
import warnings

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

from typing import Tuple, Union, Dict, Literal


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

