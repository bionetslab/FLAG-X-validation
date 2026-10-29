"""
Dataset overview figures: sample sizes and class distribution.

Part 1 collects the plotting data from the curated datasets (data/curated, see curate_data.py) into CSVs in
results/dataset_overview:
- sample_sizes.csv: dataset, sample, n_events + all columns of meta_samples.csv
- class_balance.csv: dataset, label, name, count, percentage (pooled over all samples of a dataset)
- class_balance_samples.csv: dataset, sample, label, name, count, percentage (per sample) + all columns of
  meta_samples.csv

Part 2 plots one figure per main function from these CSVs into figures/dataset_overview. Bar panel style as in the
legacy figures (legacy/main_plotting.py: fig8s_dataset_size, fig9s_class_balance).

Hue: the plotting functions take a list of meta_samples.csv columns (e.g. ['sample_type', 'subtype']). Each dataset uses
the first listed column it has; datasets without any of them are plotted without hue.

Env: fx_plot (envs/plot.yml)
"""

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from config import DATA_CURATED, RESULTS, FIGURES, DATASETS


OVERWRITE = False  # If False, existing CSVs of part 1 are reused

RES_DIR = os.path.join(RESULTS, 'dataset_overview')
FIG_DIR = os.path.join(FIGURES, 'dataset_overview')

DATASET_NAMES = {
    'imstat': 'Imstat', 'lt1': 'Lt1', 'lt2': 'Lt2', 'lt1b': 'Lt1b', 'lt2b': 'Lt2b', 'flowcyt': 'Flowcyt',
    'blast110_p1': 'BLAST110 P1', 'blast110_p2': 'BLAST110 P2', 'blast110_p3': 'BLAST110 P3',
    'blast110_p4': 'BLAST110 P4',
}

# Order of the class labels (short names from meta_labels.csv) on the x-axis of the class balance panels
LABEL_ORDER = ['HSPC', 'M', 'M16', 'Ma', 'T', 'Th', 'NK', 'G', 'B', 'dyB', 'Blast', 'X', 'O']

# Class balance figures: wide panels for the multi-class datasets, narrow ones for the binary datasets
CLASS_BALANCE_DATASETS = [
    'imstat', 'flowcyt', 'lt1', 'lt2', 'lt1b', 'lt2b', 'blast110_p1', 'blast110_p2', 'blast110_p3', 'blast110_p4',
]
CLASS_BALANCE_MOSAIC = """
    AAABBB
    CCCDDD
    EFGHIJ
    """

# Marker styles of the hue groups in the strip plots (colors: Set2)
HUE_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


# ### Part 1: plotting data ############################################################################################

def collect_plotting_data():
    sample_sizes = []
    class_balance = []
    class_balance_samples = []
    for dataset in DATASETS:  # Iterate over datasets
        # Load meta data
        data_dir = os.path.join(DATA_CURATED, dataset)
        meta_samples = pd.read_csv(os.path.join(data_dir, 'meta_samples.csv'))
        meta_labels = pd.read_csv(os.path.join(data_dir, 'meta_labels.csv'))

        counts = pd.Series(0, index=meta_labels['label'])
        for _, row in meta_samples.iterrows():  # Iterate over samples 
            # Add entry to sample_sizes
            y = np.load(os.path.join(data_dir, row['sample'] + '_y.npy'))
            sample_sizes.append({'dataset': dataset, 'sample': row['sample'], 'n_events': y.shape[0], **row.to_dict()})

            # Add entries to class_balance_samples
            sample_counts = pd.Series(y).value_counts()
            counts = counts.add(sample_counts, fill_value=0)
            for label, name in zip(meta_labels['label'], meta_labels['name']):
                count = int(sample_counts.get(label, 0))
                class_balance_samples.append({
                    'dataset': dataset,
                    'sample': row['sample'],
                    'label': label,
                    'name': name,
                    'count': count,
                    'percentage': count / y.shape[0] * 100,
                    **row.to_dict(),
                })

        for label, name in zip(meta_labels['label'], meta_labels['name']):
            class_balance.append({
                'dataset': dataset,
                'label': label,
                'name': name,
                'count': int(counts[label]),
                'percentage': round(counts[label] / counts.sum() * 100, 2),
            })

    os.makedirs(RES_DIR, exist_ok=True)
    pd.DataFrame(class_balance).to_csv(os.path.join(RES_DIR, 'class_balance.csv'), index=False)
    pd.DataFrame(class_balance_samples).to_csv(os.path.join(RES_DIR, 'class_balance_samples.csv'), index=False)
    pd.DataFrame(sample_sizes).to_csv(os.path.join(RES_DIR, 'sample_sizes.csv'), index=False)


# ### Part 2: figures ##################################################################################################

def plot_sample_sizes(hue: list[str] | None = None):
    df = pd.read_csv(os.path.join(RES_DIR, 'sample_sizes.csv'))

    fig = plt.figure(figsize=(8, 13), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        """
    )

    for dataset, key in zip(DATASETS, 'ABCDEFGHIJ'):
        df_dataset = df[df['dataset'] == dataset]
        hue_col = get_hue_column(df_dataset, hue)
        sample_sizes_panel(
            sample_sizes=df_dataset['n_events'].to_numpy(),
            title=DATASET_NAMES[dataset],
            ax=axd[key],
            groups=df_dataset[hue_col].to_numpy() if hue_col is not None else None,
        )

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, 'sample_sizes.pdf'), dpi=fig.dpi)
    plt.close(fig)


def plot_sample_sizes_box(hue: list[str] | None = None, log_scale: bool = True):
    df = pd.read_csv(os.path.join(RES_DIR, 'sample_sizes.csv'))
    df['dataset_name'] = df['dataset'].map(DATASET_NAMES)
    df['group'] = add_hue_groups(df, hue)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True, dpi=300)

    styles = get_hue_styles(df['group'])
    box_strip_panel(
        data=df, x='dataset_name', y='n_events', order=[DATASET_NAMES[d] for d in DATASETS], styles=styles,
        log_scale=log_scale, ax=ax,
    )
    if styles:
        ax.legend(handles=hue_legend_handles(styles), loc='upper left', fontsize=8)

    ax.set_xlabel(None)
    ax.set_ylabel('Number of events per sample')

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, 'sample_sizes_box.pdf'), dpi=fig.dpi)
    plt.close(fig)


def plot_class_balance():
    df = pd.read_csv(os.path.join(RES_DIR, 'class_balance.csv'))

    palette = get_label_palette(df['name'])

    fig = plt.figure(figsize=(12, 11), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(CLASS_BALANCE_MOSAIC)

    for dataset, key in zip(CLASS_BALANCE_DATASETS, 'ABCDEFGHIJ'):
        class_balance_panel(
            class_balance=df[df['dataset'] == dataset],
            title=DATASET_NAMES[dataset],
            palette=palette,
            ax=axd[key],
        )
        if key not in 'AE':
            axd[key].set_ylabel(None)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, 'class_balance.pdf'), dpi=fig.dpi)
    plt.close(fig)


def plot_class_balance_box(hue: list[str] | None = None, log_scale: bool = True):
    df = pd.read_csv(os.path.join(RES_DIR, 'class_balance_samples.csv'))
    df['group'] = add_hue_groups(df, hue)

    palette = get_label_palette(df['name'])
    styles = get_hue_styles(df['group'])

    fig = plt.figure(figsize=(12, 11), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(CLASS_BALANCE_MOSAIC)

    for dataset, key in zip(CLASS_BALANCE_DATASETS, 'ABCDEFGHIJ'):
        df_dataset = df[df['dataset'] == dataset]
        box_strip_panel(
            data=df_dataset, x='name',  y='percentage',
            order=[name for name in LABEL_ORDER if name in set(df_dataset['name'])],
            styles=styles, log_scale=log_scale, ax=axd[key], box_palette=palette,
        )
        axd[key].set_title(DATASET_NAMES[dataset])
        axd[key].set_xlabel('Cell type label')
        axd[key].set_ylabel('Events per sample (%)' if key in 'AE' else None)

    if styles:
        fig.legend(handles=hue_legend_handles(styles), loc='outside lower center', ncol=len(styles), fontsize=9)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, 'class_balance_box.pdf'), dpi=fig.dpi)
    plt.close(fig)


# ### Helpers ##########################################################################################################

# Adapted from legacy/validation/plt/plotting.py, plot_sample_sizes (mean, std and total always shown; optional hue)
def sample_sizes_panel(
        sample_sizes: np.ndarray, title: str, ax: plt.Axes, groups: np.ndarray | None = None
) -> None:

    # Enforce scientific notation for y axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((3, 3))
    ax.yaxis.set_major_formatter(formatter)

    order = np.argsort(sample_sizes, kind='stable')
    sample_sizes = sample_sizes[order]
    n = len(sample_sizes)
    x = np.arange(1, n + 1)

    if groups is None:
        ax.bar(x=x, height=sample_sizes, color='lightblue', edgecolor='darkgray', linewidth=1.0)
    else:
        # One color per group, bars still sorted by size; no bar edges, they would hide the colors of the thin bars
        groups = groups[order]
        for group, color in zip(sorted(set(groups)), sns.color_palette('Set2')):
            mask = groups == group
            ax.bar(x=x[mask], height=sample_sizes[mask], color=color, edgecolor='none', label=group)

    ax.text(
        0.02, 0.95, f'Total events: {sample_sizes.sum()}',
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
    )

    m = sample_sizes.mean()
    std = sample_sizes.std()
    ax.axhline(y=m, color='darkred', linestyle='-', linewidth=1.5, label=f'Mean: {int(m)}')
    if m - std > sample_sizes.min():
        ax.axhline(y=m - std, color='gold', linestyle='--', linewidth=1.5, label=f'Std: {np.round(std, 1)}')
    if m + std < sample_sizes.max():
        ax.axhline(y=m + std, color='gold', linestyle='--', linewidth=1.5,
                   label=None if m - std > sample_sizes.min() else f'Std: {np.round(std, 1)}')
    if groups is None:
        ax.legend(loc='lower right', fontsize=8)
    else:
        # Bars are sorted ascending, so the upper left is free (below the total events text)
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=8)

    # x-ticks: always 1 and n, plus matplotlib's ticks in between; an in-between tick closer than half a tick spacing to
    # 1 or n is dropped so that the labels do not overlap (e.g. n=107: 1, 20, ..., 80, 107 instead of ..., 80, 100, 107)
    inner = [round(t) for t in ax.get_xticks() if 1 < t < n]
    spacing = inner[1] - inner[0] if len(inner) >= 2 else n
    inner = [t for t in inner if t - 1 >= spacing / 2 and n - t >= spacing / 2]
    ax.set_xticks([1] + inner + [n])
    ax.set_xticklabels([str(t) for t in [1] + inner + [n]])

    ax.set_title(title)
    ax.set_xlabel(f'Sample ID (1–{n})')
    ax.set_ylabel('Number of events')


# Adapted from legacy/validation/plt/plotting.py, plot_class_balance (takes precomputed counts)
def class_balance_panel(class_balance: pd.DataFrame, title: str, palette: dict, ax: plt.Axes) -> None:

    # Enforce scientific notation for y axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((3, 3))
    ax.yaxis.set_major_formatter(formatter)

    order = [name for name in LABEL_ORDER if name in set(class_balance['name'])]

    sns.barplot(
        data=class_balance, x='name', y='count', hue='name', order=order, palette=palette, legend=False, ax=ax
    )

    # Count and percentage above each bar (inside the bar for the largest class)
    total = class_balance['count'].sum()
    max_count = class_balance['count'].max()
    for idx, name in enumerate(order):
        row = class_balance[class_balance['name'] == name].iloc[0]
        y_pos = row['count'] * 0.85 if row['count'] == max_count else row['count'] + 0.01 * total
        ax.text(x=idx, y=y_pos, s=f"{int(row['count'])}\n{row['percentage']}%", ha='center', va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Cell type label')
    ax.set_ylabel('Number of events')


def box_strip_panel(
        data: pd.DataFrame,
        x: str,
        y: str,
        order: list[str],
        styles: dict,
        log_scale: bool,
        ax: plt.Axes,
        box_palette: dict | None = None,
) -> None:
    # One box per x category (outliers not drawn, all samples are shown as strip points). Strip points are colored and
    # marked by their hue group (column 'group', see add_hue_groups), points without group are gray circles.

    # Zero values cannot be shown on a log axis: they are left out and counted in the x tick labels
    n_zero = data[data[y] <= 0].groupby(x).size() if log_scale else pd.Series(dtype=int)
    if log_scale:
        data = data[data[y] > 0]

    if box_palette is None:
        sns.boxplot(data=data, x=x, y=y, order=order, color='lightblue', showfliers=False, log_scale=log_scale, ax=ax)
    else:
        sns.boxplot(
            data=data, x=x, y=y, order=order, hue=x, palette=box_palette, legend=False, showfliers=False,
            log_scale=log_scale, ax=ax,
        )

    rng_state = np.random.get_state()
    np.random.seed(0)  # Reproducible jitter
    strip_kwargs = dict(x=x, y=y, order=order, jitter=0.25, size=2.5, alpha=0.8, log_scale=log_scale, ax=ax)
    no_group = data[data['group'].isna()]
    if len(no_group) > 0:
        sns.stripplot(data=no_group, color='dimgray', marker='o', **strip_kwargs)
    for group, (color, marker) in styles.items():
        data_group = data[data['group'] == group]
        if len(data_group) > 0:
            sns.stripplot(data=data_group, color=color, marker=marker, **strip_kwargs)
    np.random.set_state(rng_state)

    if len(n_zero) > 0:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([
            f'{c}\n(0 in {n_zero[c]})' if c in n_zero.index else c for c in order
        ])


def get_hue_column(df: pd.DataFrame, hue: list[str] | None) -> str | None:
    # First column of hue that the dataset has (non-empty in the CSV, which holds the meta columns of all datasets)
    for col in hue or []:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def add_hue_groups(df: pd.DataFrame, hue: list[str] | None) -> pd.Series:
    # Per dataset: the values of its hue column (see get_hue_column), NaN for datasets without hue
    groups = pd.Series(np.nan, index=df.index, dtype=object)
    for dataset, df_dataset in df.groupby('dataset'):
        hue_col = get_hue_column(df_dataset, hue)
        if hue_col is not None:
            groups[df_dataset.index] = df_dataset[hue_col].astype(str)
    return groups


def get_hue_styles(groups: pd.Series) -> dict:
    # Hue group -> (color, marker)
    group_values = sorted(groups.dropna().unique())
    return {g: (c, m) for g, c, m in zip(group_values, sns.color_palette('Set2'), HUE_MARKERS)}


def hue_legend_handles(styles: dict) -> list[Line2D]:
    return [
        Line2D([], [], color=color, marker=marker, linestyle='none', markersize=6, label=group)
        for group, (color, marker) in styles.items()
    ]


def get_label_palette(names: pd.Series) -> dict:
    # One color per class label name across all datasets (as in legacy: shuffled with a fixed seed)
    names = sorted(set(names))
    random.Random(42).shuffle(names)
    return dict(zip(names, sns.color_palette('Set3') + sns.color_palette('Pastel1')))


# Copied from legacy/validation/plt/plotting.py, annotate_mosaic (unused arguments removed)
def annotate_mosaic(fig: plt.Figure, axd: dict, fontsize: float) -> None:
    # Annotate subplot mosaic tiles with their labels, shifted to the left and up
    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0, 0.95, label,
            transform=ax.transAxes + trans, fontsize=fontsize, va='bottom', fontfamily='sans-serif',
            fontweight='bold',
        )


if __name__ == '__main__':

    # Part 1
    csv_files = ('sample_sizes.csv', 'class_balance.csv', 'class_balance_samples.csv')
    csvs_exist = all(os.path.exists(os.path.join(RES_DIR, f)) for f in csv_files)
    if csvs_exist and not OVERWRITE:
        print('Plotting data already collected, skipped')
    else:
        print('Collecting plotting data ...')
        collect_plotting_data()

    # Part 2
    plot_sample_sizes(hue=['sample_type'])
    plot_sample_sizes_box(hue=['sample_type'])
    plot_class_balance()
    plot_class_balance_box(hue=['sample_type'])

    print('done')
