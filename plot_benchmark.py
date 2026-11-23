"""
Benchmark figures (PLAN.md step 7 / 10).

Main figure: box plots of the sample-wise scores, one panel per dataset, one box per method, the individual test
samples jittered on top, the outer test fold shown by the marker.

    scores_box.pdf   one point = one test sample: its score averaged over the model seeds (stochastic methods: 3
                     seeds, deterministic: 1); box = distribution over all test samples (all outer folds pooled):
                     median and quartiles, whiskers 1.5 IQR, no fliers (all points are drawn). Panel letters A, B, ...
                     in PLOT_DATASETS order (the same letters as in pairwise_f1.pdf).

Score of a sample (as in the tuning, validation.core.scores.f1_from_counts, scoring rule D8): macro F1 = NaN-mean of
the per-class f1_<l> columns; datasets with 2 labels (lt1b / lt2b: B cells, BLAST110: blasts) F1 of the positive
class, label 1 (f1_1). The y-axis label of each panel: 'F1 macro' or 'F1 binary'.
Unpredicted samples (failed / killed / missing atoms) are left out; a seed mean uses the seeds that predicted the
sample. A box is annotated with 'k/n' only if some predictions of the method failed: k = predicted, n = expected
predictions (test samples of the dataset over all outer splits x model seeds), so a failed seed is counted even if
the other seeds predicted the sample; a method without any predicted sample is marked 'n/a'.

Supplementary figures: predicted vs. true population sizes (as legacy fig2s_population_sizes), one figure per dataset.

    population_<dataset>.pdf   one panel per method (3 x 4 grid); x = true share of a class in a test sample (%),
                     y = predicted share (%, events predicted as the class / all events of the sample, averaged over
                     the seeds that predicted the sample); one point per test sample x class, colored by class;
                     dashed diagonal = perfect agreement. Abstained events belong to no class, so the predicted
                     shares of a sample can sum to < 100 %. Statistics pooled over the points of a panel (SHOW_R2,
                     SHOW_PEARSON): R^2 = 1 - sum (pred - true)^2 / sum (true - mean true)^2 (agreement with the
                     diagonal, not a fitted line; as legacy r2_score(true, pred)) and Pearson r.

    bland_altman_<dataset>.pdf   (multi-class datasets) Bland-Altman plots of the same points (seed means): one row
                     per method, one column per class; x = (predicted + true) / 2 (%), y = predicted - true
                     (percentage points); only points where the class is in the sample or predicted. Lines: bias
                     (solid) and 95 % limits of agreement (bias -/+ 1.96 SD, dashed) of the panel's points
                     (validation.eval_helpers.ba_stats, = benchmark_eval.py's population_agreement.csv); text: bias
                     [lower, upper limit] in pp (SHOW_BA_TEXT). x and y axes shared within a column, y symmetric.
    bland_altman_binary.pdf   the same for the binary datasets, positive class (label 1: B, Blast) only: one row per
                     method, one column per dataset.

Supplementary figures: class-wise F1 (multi-class datasets; the binary datasets' main-figure score is already the F1
of the positive class, and the F1 of the other class adds nothing).

    class_scores_box_{imstat,flowcyt,lt}.pdf   (CLASS_FIGURES; Lt1 and Lt2 stacked in one figure) one row per class,
                     one box per method, style of scores_box.pdf; one point per test sample: per-class F1 (f1_<l>,
                     D8) averaged over the seeds; samples where the class is neither present nor predicted (F1
                     undefined) are left out, so no 'k/n' annotation. Row label: class (median share in the test
                     samples).

Pairwise comparison (benchmark_eval.py pairwise_deltas.csv, no statistical tests):

    pairwise_f1.pdf  one heatmap per dataset: cell (row A, column B) = mean paired difference F1(A) - F1(B) over the
                     test samples predicted by both methods (seed means over the predicting seeds; macro / binary as
                     in scores_box.pdf); a method with failed predictions has '(k/n)' (as in scores_box.pdf) in front
                     of its name; monochrome blue scale 0 .. DELTA_LIM (darker = row method better; saturates
                     beyond); a negative delta (row method worse on the common samples despite the higher mean score;
                     only possible if one of the two methods has failed predictions) is grey (DELTA_NEG_COLOR), the
                     colorbar extends to -DELTA_NEG for it; a dot marks equivalent pairs (benchmark_eval.py column
                     'equivalent': 95 % bootstrap CI of the mean difference within +-config.DELTA); light grey =
                     not available (fewer than 2 test samples predicted by both methods). The legend lists the grey
                     cell types only if they occur.
                     Upper triangle only (the matrix is antisymmetric); methods sorted per dataset by mean score (as
                     in scores_box.pdf), best at the top; the method names stand in the empty lower triangle, ending
                     in the diagonal cell (name i labels row i and column i). Colorbar and legend in the empty panel
                     slots of the last row (under the panels if there are none).

Order, names and colors of the datasets, methods and classes: config.py, section Plotting (PLOT_DATASETS,
DATASET_NAMES, PLOT_METHODS, METHOD_NAMES, METHOD_COLORS, CLASS_ORDER, CLASS_COLORS).

Env: fx_plot (envs/plot.yml)

Usage:
    python plot_benchmark.py [--in DIR] [--out DIR]
        --in: output of benchmark_eval.py (default results/benchmark_eval; pilot: results/benchmark_pilot_eval)
        --out: figure directory (default figures/benchmark; pilot: e.g. figures/benchmark_pilot)

Inputs: <in>/scores/<dataset>.csv, <in>/populations/<dataset>.csv
Outputs: <out>/scores_box.pdf, <out>/class_scores_box_*.pdf, <out>/pairwise_f1.pdf, <out>/population_<dataset>.pdf,
    <out>/bland_altman_<dataset>.pdf (multi-class),
    <out>/bland_altman_binary.pdf
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr

from config import (DATA_CURATED, RESULTS, FIGURES, PLOT_DATASETS, DATASET_NAMES, PLOT_METHODS, METHOD_NAMES,
                    METHOD_COLORS, CLASS_ORDER, CLASS_COLORS, PAGE_WIDTH, PAGE_HEIGHT, PLOT_RC, FONT_TICK,
                    FONT_LABEL, FONT_TITLE, FONT_SUPTITLE, FONT_PANEL, FONT_ANNOT, FONT_MIN, DELTA)
from validation.eval_helpers import ba_stats, population_seed_mean

# Final printed size: every figure is PAGE_WIDTH wide and at most PAGE_HEIGHT tall, font sizes FONT_* (config.py)
plt.rcParams.update(PLOT_RC)


OVERWRITE = True  # If False, existing figures are skipped

IN_DIR = os.path.join(RESULTS, 'benchmark_eval')
OUT_DIR = os.path.join(FIGURES, 'benchmark')

FOLD_MARKERS = ['o', 's', '^', 'D', 'v']  # Marker of outer split 0, 1, ...
N_COLS = 2  # Panels per row
N_LEGEND_COLS = 6  # Methods per legend row
JITTER = 0.15  # Half-width of the horizontal point jitter (deterministic, seed 0)

# Population figures
POP_N_COLS = 4  # Method panels per row
SHOW_R2 = True  # R^2 of predicted vs. true % (agreement with the diagonal) in each panel
SHOW_PEARSON = False  # Pearson r in each panel
# Class-wise figures (multi-class datasets only; binary datasets: the main figure already shows the positive class):
# output file -> datasets stacked in it (one row per class)
CLASS_FIGURES = [
    ('class_scores_box_imstat.pdf', ['imstat']),
    ('class_scores_box_flowcyt.pdf', ['flowcyt']),
    ('class_scores_box_lt.pdf', ['lt1', 'lt2']),
]
SHOW_BA_TEXT = True  # Bland-Altman panels: 'bias [lower, upper limit of agreement]' in pp
# Pairwise heatmaps
DELTA_LIM = 0.2  # Color scale of the mean paired delta: 0 .. DELTA_LIM (larger values saturate)
DELTA_NEG = 0.02  # The colorbar extends to -DELTA_NEG: negative deltas, all in DELTA_NEG_COLOR
DELTA_NEG_COLOR = '0.6'  # Darker than the grey of unavailable cells (0.9)
PAIR_N_COLS = 3  # Dataset panels per row


def load_scores(in_dir, dataset):
    """
    Sample-wise scores of one dataset, averaged over the model seeds.

    Args:
        in_dir (str): Output directory of benchmark_eval.py.
        dataset (str): Dataset name.

    Returns:
        (pd.DataFrame, str): One row per method x outer split x test sample (all expected ones, also unpredicted) with
        columns method, split, sample, score (seed mean; NaN if no seed predicted the sample), n_seeds (seeds that
        predicted it), n_seeds_expected (seeds of the method); and the y-axis label naming the score.
    """
    t = pd.read_csv(os.path.join(in_dir, 'scores', dataset + '.csv'))
    f1 = [c for c in t.columns if c.startswith('f1_')]
    if len(f1) == 2:  # Binary: F1 of the positive class (label 1)
        t['score'] = t['f1_1']
        y_label = 'F1 binary'
    else:  # Macro F1 (classes neither present nor predicted are NaN)
        t['score'] = t[f1].mean(axis=1, skipna=True)
        y_label = 'F1 macro'
    t.loc[~t['predicted'], 'score'] = np.nan
    g = t.groupby(['method', 'split', 'sample'])
    scores = pd.concat([g['score'].mean(), g['predicted'].sum().rename('n_seeds'),
                        g.size().rename('n_seeds_expected')], axis=1).reset_index()
    return scores, y_label


def prediction_counts(s):
    """(k, n) of one method: predicted / expected test sample x seed predictions (load_scores rows of the method)."""
    return int(s['n_seeds'].sum()), int(s['n_seeds_expected'].sum())


def plot_panel(ax, scores, methods, rng, show_counts=True):
    """
    Boxes per method (colored), the samples as points (marker = outer split), 'k/n' / 'n/a' annotations.

    Args:
        ax (plt.Axes): Panel of one dataset.
        scores (pd.DataFrame): load_scores output of the dataset.
        methods (list of str): Methods on the x-axis, in order (position i = methods[i]).
        rng (np.random.Generator): For the jitter.
        show_counts (bool): Annotate boxes of methods with failed predictions with 'k/n' (predicted / expected test
            sample x seed predictions).
    """
    for i, m in enumerate(methods):
        s = scores[scores['method'] == m]
        valid = s.dropna(subset=['score'])
        if valid.empty:
            ax.text(i, 0.02, 'n/a', ha='center', va='bottom', fontsize=FONT_ANNOT, color='0.5')
            continue
        color = METHOD_COLORS[m]
        ax.boxplot([valid['score'].values], positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops={'facecolor': color, 'alpha': 0.9, 'edgecolor': 'black', 'linewidth': 0.6},
                   medianprops={'color': 'black', 'linewidth': 0.9},
                   whiskerprops={'linewidth': 0.6}, capprops={'linewidth': 0.6}, zorder=2)
        for split, g in valid.groupby('split'):
            ax.scatter(i + rng.uniform(-JITTER, JITTER, len(g)), g['score'], s=4,
                       marker=FOLD_MARKERS[int(split) % len(FOLD_MARKERS)], color=color, edgecolors='black',
                       linewidths=0.25, alpha=0.6, zorder=3)
        if not show_counts:
            continue
        k, n = prediction_counts(s)
        if k < n:
            ax.text(i, 1.01, '{}/{}'.format(k, n), ha='center', va='bottom', fontsize=FONT_MIN, color='0.3')


def plot_scores_box(in_dir, path):
    datasets = [d for d in PLOT_DATASETS if os.path.exists(os.path.join(in_dir, 'scores', d + '.csv'))]
    if not datasets:
        raise SystemExit('no score tables in {} (--in must be an output directory of benchmark_eval.py, e.g. '
                         'results/benchmark_pilot_eval)'.format(os.path.join(in_dir, 'scores')))
    loaded = dict((d, load_scores(in_dir, d)) for d in datasets)
    scores = dict((d, loaded[d][0]) for d in datasets)
    present = set().union(*(set(s['method']) for s in scores.values()))
    methods = [m for m in PLOT_METHODS if m in present]
    splits = sorted(set().union(*(set(s.dropna(subset=['score'])['split']) for s in scores.values())))

    n_rows = int(np.ceil(len(datasets) / float(N_COLS)))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(PAGE_WIDTH, min(1.5 * n_rows + 1.3, PAGE_HEIGHT)),
                             sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    rng = np.random.default_rng(0)
    for k, ax in enumerate(axes.flat):
        if k >= len(datasets):
            ax.set_visible(False)
            continue
        plot_panel(ax, scores[datasets[k]], methods, rng)
        ax.set_title(DATASET_NAMES[datasets[k]])
        panel_letter(ax, k)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_ylabel(loaded[datasets[k]][1])
        ax.tick_params(axis='y', labelleft=True)  # Every panel: y label and ticks (score type differs per panel)
    for ax in axes[-1]:
        ax.set_xticks(range(len(methods)), [METHOD_NAMES[m] for m in methods], rotation=45, ha='right',
                      rotation_mode='anchor')
        ax.tick_params(axis='x', labelbottom=True)
    ax_last = axes.flat[len(datasets) - 1]  # Bottom panel of an odd grid gets the tick labels too
    ax_last.tick_params(axis='x', labelbottom=True)

    method_fold_legend(fig, methods, splits)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def panel_letter(ax, k):
    """Bold panel letter (A, B, ... for k = 0, 1, ...) at the top left of the panel, level with its title."""
    ax.set_title('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[k], loc='left', fontweight='bold', fontsize=FONT_PANEL)


def method_fold_legend(fig, methods, splits):
    """Legend under the figure: methods (colors) in rows of N_LEGEND_COLS, then the test folds (markers) in one row."""
    method_handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor='black', linewidth=0.6, label=METHOD_NAMES[m])
                      for m in methods]
    fold_handles = [Line2D([], [], linestyle='', marker=FOLD_MARKERS[int(s) % len(FOLD_MARKERS)], color='0.5',
                           markersize=4, markeredgecolor='black', markeredgewidth=0.3,
                           label='Test fold {}'.format(int(s)))
                    for s in splits]
    blank = Line2D([], [], linestyle='', label='')

    def rows(h):  # Pad to full rows of N_LEGEND_COLS
        return [h[i:i + N_LEGEND_COLS] + [blank] * (N_LEGEND_COLS - len(h[i:i + N_LEGEND_COLS]))
                for i in range(0, len(h), N_LEGEND_COLS)]
    grid = rows(method_handles) + rows(fold_handles)
    handles = [r[c] for c in range(N_LEGEND_COLS) for r in grid]  # matplotlib fills the legend column by column
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc='outside lower center',
               ncol=N_LEGEND_COLS, frameon=False, columnspacing=1.2, handlelength=1.5)


def load_populations(in_dir, dataset):
    """
    Population sizes of one dataset averaged over the model seeds (validation.eval_helpers.population_seed_mean on
    benchmark_eval.py's populations/<dataset>.csv): one row per method x outer split x predicted test sample x class
    with name, true_pct, pred_pct, d_pp, mean_pct, present. Methods without any predicted sample have no rows.
    """
    return population_seed_mean(pd.read_csv(os.path.join(in_dir, 'populations', dataset + '.csv')))


def agreement_text(true, pred):
    """Statistics text of one panel (SHOW_R2, SHOW_PEARSON); R^2 against the diagonal (true as reference)."""
    lines = []
    if SHOW_R2:
        r2 = 1.0 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
        lines.append('$R^2$ = {:.3f}'.format(r2))
    if SHOW_PEARSON:
        lines.append('$r$ = {:.3f}'.format(pearsonr(true, pred)[0]))
    return '\n'.join(lines)


def method_grid(in_dir, dataset, pop):
    """
    Figure with one panel per method of the dataset (POP_N_COLS per row) for the population figures.

    Returns:
        (fig, list of (method, ax, pop rows of the method), list of str): the classes of the dataset in CLASS_ORDER.
    """
    present = set(pd.read_csv(os.path.join(in_dir, 'scores', dataset + '.csv'), usecols=['method'])['method'])
    methods = [m for m in PLOT_METHODS if m in present]
    classes = [c for c in CLASS_ORDER if c in set(pop['name'])] + sorted(set(pop['name']) - set(CLASS_ORDER))
    n_rows = int(np.ceil(len(methods) / float(POP_N_COLS)))
    fig, axes = plt.subplots(n_rows, POP_N_COLS, figsize=(PAGE_WIDTH, min(1.65 * n_rows + 1.0, PAGE_HEIGHT)),
                             sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    panels = []
    for k, ax in enumerate(axes.flat):
        if k >= len(methods):
            ax.set_visible(False)
            continue
        ax.set_title(METHOD_NAMES[methods[k]])
        ax.grid(linestyle='--', linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.first_col = k % POP_N_COLS == 0
        ax.bottom = k >= len(methods) - POP_N_COLS  # Bottom panel of its column
        if ax.bottom:
            ax.tick_params(axis='x', labelbottom=True)
        panels.append((methods[k], ax, pop[pop['method'] == methods[k]]))
    return fig, panels, classes


def finish_figure(fig, classes, title, path):
    """Suptitle, class legend under the figure, save."""
    fig.suptitle(title)
    handles = [Line2D([], [], linestyle='', marker='o', markersize=4, color=CLASS_COLORS.get(c, '0.5'),
                      markeredgecolor='black', markeredgewidth=0.3, label=c) for c in classes]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False, title='Class',
               columnspacing=1.0, handletextpad=0.2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def scatter_classes(ax, p, classes, x, y):
    for c in classes:  # Classes in legend order; later ones on top
        g = p[p['name'] == c]
        ax.scatter(g[x], g[y], s=5, color=CLASS_COLORS.get(c, '0.5'), edgecolors='black', linewidths=0.2, alpha=0.8,
                   zorder=2)


def plot_populations(in_dir, dataset, path):
    """
    Predicted vs. true population sizes of one dataset: one panel per method (columns: module docstring).

    Args:
        in_dir (str): Output directory of benchmark_eval.py.
        dataset (str): Dataset name.
        path (str): Output PDF.
    """
    pop = load_populations(in_dir, dataset)
    fig, panels, classes = method_grid(in_dir, dataset, pop)
    for method, ax, p in panels:
        ax.plot([0, 100], [0, 100], linestyle='--', color='0.5', linewidth=0.6, zorder=1)
        if p.empty:
            ax.text(0.75, 0.25, 'n/a', transform=ax.transAxes, ha='center', va='center', fontsize=FONT_ANNOT,
                    color='0.5')
        else:
            scatter_classes(ax, p, classes, 'true_pct', 'pred_pct')
            text = agreement_text(p['true_pct'].to_numpy(), p['pred_pct'].to_numpy())
            if text:
                ax.text(0.04, 0.96, text, transform=ax.transAxes, ha='left', va='top', fontsize=FONT_ANNOT,
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 1.5})
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.set_aspect('equal')
        if ax.first_col:
            ax.set_ylabel('Predicted (%)')
        if ax.bottom:
            ax.set_xlabel('True (%)')
    finish_figure(fig, classes, DATASET_NAMES[dataset] + ': predicted vs. true population size per test sample', path)


def plot_ba_grid(columns, title, path):
    """
    Bland-Altman figure: one row per method (PLOT_METHODS), one column per (title, points, color).

    Args:
        columns (list of (str, pd.DataFrame, str)): Column title, its points (seed means, present only, columns
            method, mean_pct, d_pp), point color.
        title (str): Figure title.
        path (str): Output PDF.
    """
    present = set().union(*(set(c[1]['method']) for c in columns))
    methods = [m for m in PLOT_METHODS if m in present]
    fig, axes = plt.subplots(len(methods), len(columns),
                             figsize=(PAGE_WIDTH, min(0.7 * len(methods) + 1.0, PAGE_HEIGHT)),
                             sharex='col', sharey='col', squeeze=False, layout='constrained')
    for j, (col_title, pts, color) in enumerate(columns):
        lim = max(1.0, float(pts['d_pp'].abs().max()) * 1.15) if not pts.empty else 1.0  # Symmetric per column
        for i, m in enumerate(methods):
            ax = axes[i, j]
            p = pts[pts['method'] == m]
            ax.axhline(0, color='0.6', linewidth=0.5, zorder=1)
            if p.empty:
                ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes, ha='center', va='center', fontsize=FONT_ANNOT,
                        color='0.5')
            else:
                ax.scatter(p['mean_pct'], p['d_pp'], s=3, color=color, edgecolors='black', linewidths=0.15, alpha=0.8,
                           zorder=2)
                st = ba_stats(p['d_pp'])
                ax.axhline(st['bias_pp'], color='black', linewidth=0.7, zorder=3)
                for v in (st['loa_low_pp'], st['loa_high_pp']):
                    if np.isfinite(v):
                        ax.axhline(v, color='black', linewidth=0.6, linestyle='--', zorder=3)
                if SHOW_BA_TEXT:
                    # Two lines (bias, [limits]): one line is wider than a panel at PAGE_WIDTH with 8 classes
                    ax.text(0.03, 0.97, '{:.2f}\n[{:.2f}, {:.2f}]'.format(st['bias_pp'], st['loa_low_pp'],
                                                                        st['loa_high_pp']),
                            transform=ax.transAxes, ha='left', va='top', fontsize=FONT_MIN,
                            bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 0.8})
            ax.set_ylim(-lim, lim)
            ax.tick_params(labelsize=FONT_MIN, length=2, pad=1)  # Dense grid: smallest tick labels
            ax.grid(linestyle='--', linewidth=0.4, alpha=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(col_title)
            if j == 0:
                ax.set_ylabel(METHOD_NAMES[m], fontsize=FONT_ANNOT)  # Row label of a dense grid
    fig.supxlabel('(Predicted + true) / 2 (%)')
    fig.supylabel('Predicted − true (pp)')
    fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def ba_points(in_dir, dataset):
    """Seed-mean population points of a dataset, present only (class in the sample or predicted)."""
    pop = load_populations(in_dir, dataset)
    return pop[pop['present']]


def plot_bland_altman_multiclass(in_dir, dataset, path):
    """Bland-Altman figure of one multi-class dataset: rows = methods, columns = classes (CLASS_ORDER)."""
    pts = ba_points(in_dir, dataset)
    names = [c for c in CLASS_ORDER if c in set(pts['name'])] + sorted(set(pts['name']) - set(CLASS_ORDER))
    columns = [(c, pts[pts['name'] == c], CLASS_COLORS.get(c, '0.5')) for c in names]
    plot_ba_grid(columns, '{}: Bland–Altman of the population sizes per test sample\nbias [95 % limits of agreement] '
                          '(pp)'.format(DATASET_NAMES[dataset]), path)


def plot_bland_altman_binary(in_dir, datasets, path):
    """Bland-Altman figure of the binary datasets, positive class (label 1) only: rows = methods, columns = datasets."""
    columns = []
    for d in datasets:
        pts = ba_points(in_dir, d)
        pos = pts[pts['label'] == 1]
        name = pos['name'].iloc[0] if not pos.empty else ''
        columns.append(('{}\n({})'.format(DATASET_NAMES[d], name), pos, CLASS_COLORS.get(name, '0.5')))
    plot_ba_grid(columns, 'Binary datasets: Bland–Altman of the positive-class population size per test sample\n'
                          'bias [95 % limits of agreement] (pp)', path)


def load_class_scores(in_dir, dataset):
    """
    Per-class F1 per test sample, averaged over the model seeds.

    Returns:
        list of (str, float, pd.DataFrame): Per class (CLASS_ORDER): short name, median share of the class in the test
        samples (%), and one row per method x outer split x test sample with column score (seed mean of f1_<l>; NaN
        if no seed predicted the sample or the class is neither in the sample nor predicted).
    """
    t = pd.read_csv(os.path.join(in_dir, 'scores', dataset + '.csv'))
    labels = pd.read_csv(os.path.join(DATA_CURATED, dataset, 'meta_labels.csv'))
    name_of = dict(zip(labels['name'], labels['label']))
    out = []
    for name in [c for c in CLASS_ORDER if c in name_of] + sorted(set(name_of) - set(CLASS_ORDER)):
        l = name_of[name]
        t['score'] = t['f1_{}'.format(l)].where(t['predicted'])
        sc = t.groupby(['method', 'split', 'sample'])['score'].mean().reset_index()
        share = (100.0 * t['n_{}'.format(l)] / t['n_cells']).groupby(t['sample']).first().median()
        out.append((name, share, sc))
    return out


def plot_class_scores(in_dir, datasets, path):
    """
    Class-wise box plots: one row per dataset x class (stacked), one box per method; style of scores_box.pdf.

    Args:
        in_dir (str): Output directory of benchmark_eval.py.
        datasets (list of str): Datasets stacked in the figure.
        path (str): Output PDF.
    """
    rows = [(d, name, share, sc) for d in datasets for name, share, sc in load_class_scores(in_dir, d)]
    present = set().union(*(set(r[3]['method']) for r in rows))
    methods = [m for m in PLOT_METHODS if m in present]
    splits = sorted(set().union(*(set(r[3].dropna(subset=['score'])['split']) for r in rows)))
    fig, axes = plt.subplots(len(rows), 1, figsize=(PAGE_WIDTH, min(0.95 * len(rows) + 1.4, PAGE_HEIGHT)),
                             sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    rng = np.random.default_rng(0)
    for ax, (d, name, share, sc) in zip(axes[:, 0], rows):
        plot_panel(ax, sc, methods, rng, show_counts=False)
        label = '{} ({:#.3g} %)'.format(name, share)  # 3 significant digits, e.g. 59.4 %, 0.183 %
        ax.set_ylabel(label if len(datasets) == 1 else '{}: {}'.format(DATASET_NAMES[d], label))
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xticks(range(len(methods)), [METHOD_NAMES[m] for m in methods], rotation=45, ha='right',
                           rotation_mode='anchor')
    title = ' and '.join(DATASET_NAMES[d] for d in datasets)
    fig.suptitle('{}: class-wise F1 per test sample (class: median share in the test samples)'.format(title))
    fig.supylabel('F1')
    method_fold_legend(fig, methods, splits)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_pairwise(in_dir, path, metric='f1'):
    """
    Heatmaps of the mean paired differences between methods, one panel per dataset (columns: module docstring).

    Args:
        in_dir (str): Output directory of benchmark_eval.py (pairwise_deltas.csv).
        path (str): Output PDF.
        metric (str): 'f1', 'prec' or 'rec'.
    """
    pairs = pd.read_csv(os.path.join(in_dir, 'pairwise_deltas.csv'))
    pairs = pairs[pairs['metric'] == metric]
    datasets = [d for d in PLOT_DATASETS if d in set(pairs['dataset'])]
    n_rows = int(np.ceil(len(datasets) / float(PAIR_N_COLS)))
    n_empty = n_rows * PAIR_N_COLS - len(datasets)  # Empty slots of the last row: colorbar and legend go there
    fig, axes = plt.subplots(n_rows, PAIR_N_COLS,
                             figsize=(PAGE_WIDTH, min(2.0 * n_rows + (0.2 if n_empty else 0.7), PAGE_HEIGHT)),
                             squeeze=False, layout='constrained')
    # Monochrome: rows are sorted by mean score, so the upper-triangle deltas are >= 0 unless a method has failed
    # predictions (the pairs use the common samples). Cells: Blues 0 .. DELTA_LIM; negative cells are grey patches
    # (not mapped through the colormap, so values just above 0 cannot fall into the grey segment). The colorbar shows
    # the grey segment -DELTA_NEG .. 0 in front of the Blues scale.
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('none')  # Lower triangle, diagonal, negative and missing cells: patches below
    n_neg = int(round(256 * DELTA_NEG / DELTA_LIM))
    cbar_cmap = ListedColormap([DELTA_NEG_COLOR] * n_neg + list(cmap(np.linspace(0, 1, 256))))
    cbar_cmap.set_under(DELTA_NEG_COLOR)
    cbar_cmap.set_over(cmap(1.0))
    cbar = ScalarMappable(norm=Normalize(-DELTA_NEG, DELTA_LIM), cmap=cbar_cmap)
    any_missing = any_negative = False  # Legend entries only for cell types that occur
    for n, ax in enumerate(axes.flat):
        if n >= len(datasets):
            ax.set_visible(False)
            continue
        p = pairs[pairs['dataset'] == datasets[n]].set_index(['method_a', 'method_b'])
        # Methods sorted by mean score (mean over the test samples of the seed-mean score, as in scores_box.pdf;
        # unpredicted samples left out), best first; methods without any score last
        scores, _ = load_scores(in_dir, datasets[n])
        mean = scores.groupby('method')['score'].mean()
        methods = [m for m in PLOT_METHODS if m in set(p.index.get_level_values('method_a'))]
        methods = sorted(methods, key=lambda m: -mean.get(m, np.nan) if np.isfinite(mean.get(m, np.nan)) else np.inf)
        k = len(methods)
        # The matrix is antisymmetric (delta(B, A) = -delta(A, B), mirrored CI): only the upper triangle is drawn
        mat = np.full((k, k), np.nan)
        equiv = np.zeros((k, k), dtype=bool)  # Equivalent within +-DELTA (benchmark_eval.py column 'equivalent')
        missing = []  # Upper cells without a value (grey)
        for i, a in enumerate(methods):
            for j, b in enumerate(methods):
                if j <= i:
                    continue
                if (a, b) in p.index and np.isfinite(p.loc[(a, b), 'mean_delta']):
                    r = p.loc[(a, b)]
                    mat[i, j] = r['mean_delta']
                    equiv[i, j] = str(r['equivalent']) == 'True'  # Bool or NaN in the CSV
                else:
                    missing.append((i, j))
        # pcolormesh (vector cells) instead of imshow (embedded bitmap, smoothed by PDF viewers); cell edges at
        # +-0.5 and inverted y axis: cell (i, j) centred at x=j, y=i, row 0 (best method) at the top
        edges = np.arange(k + 1) - 0.5
        ax.pcolormesh(edges, edges, np.ma.masked_invalid(np.where(mat < 0, np.nan, mat)), cmap=cmap, vmin=0,
                      vmax=DELTA_LIM)
        any_missing |= bool(missing)
        any_negative |= bool((mat < 0).any())
        for i, j in missing:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='0.9', linewidth=0))
        for i, j in zip(*np.nonzero(mat < 0)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=DELTA_NEG_COLOR, linewidth=0))
        ax.set_xlim(-0.5, k - 0.5)
        ax.set_ylim(k - 0.5, -0.5)
        ax.set_aspect('equal')
        yy, xx = np.nonzero(equiv)
        ax.scatter(xx, yy, s=1.5, color='black', linewidths=0)
        # Method names in the empty lower triangle, ending in the diagonal cell: name i labels row i (to its right)
        # and column i (above it); '(k/n) ' in front if some predictions of the method failed (as in scores_box.pdf)
        for i, m in enumerate(methods):
            k, n_exp = prediction_counts(scores[scores['method'] == m])
            name = METHOD_NAMES[m] if k == n_exp else '({}/{}) {}'.format(k, n_exp, METHOD_NAMES[m])
            ax.text(i + 0.4, i, name, ha='right', va='center', fontsize=FONT_MIN)
        panel_letter(ax, n)
        ax.set_title('{} ({})'.format(DATASET_NAMES[datasets[n]], 'binary' if
                                      (p['average'] == 'binary').any() else 'macro'))
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    label = 'Mean paired Δ{} per test sample\n(row − column)'.format(
        {'f1': 'F1', 'prec': ' precision', 'rec': ' recall'}[metric])
    legend = [Line2D([], [], linestyle='', marker='o', markersize=2, color='black',
                     label='Equivalent (95 % CI of Δ within ±{:g})'.format(DELTA))]
    if any_negative:
        legend.append(Patch(facecolor=DELTA_NEG_COLOR, edgecolor='none',
                            label='Negative Δ (row method worse on the common samples)'))
    if any_missing:
        legend.append(Patch(facecolor='0.9', edgecolor='none', label='Not available (no common predicted samples)'))
    if n_empty:  # Colorbar and legend in the empty slots of the last row (one axes spanning them)
        gs = axes[-1, 0].get_gridspec()
        for ax in axes[-1, PAIR_N_COLS - n_empty:]:
            ax.remove()
        host = fig.add_subplot(gs[n_rows - 1, PAIR_N_COLS - n_empty:])
        host.axis('off')
        cax = host.inset_axes([0.1, 0.62, 0.8, 0.06])
        cb = fig.colorbar(cbar, cax=cax, orientation='horizontal', extend='both')
        host.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.3), frameon=False)
    else:
        cb = fig.colorbar(cbar, ax=axes, location='bottom', shrink=0.5, extend='both', pad=0.02, aspect=40)
        fig.legend(handles=legend, loc='outside lower center', ncol=2, frameon=False)
    cb.set_label(label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(in_dir, out_dir):
    path = os.path.join(out_dir, 'scores_box.pdf')
    if os.path.exists(path) and not OVERWRITE:
        print('exists, skipped:', path)
    else:
        plot_scores_box(in_dir, path)
        print('written:', path)

    for dataset in PLOT_DATASETS:
        if not os.path.exists(os.path.join(in_dir, 'scores', dataset + '.csv')):
            continue
        path = os.path.join(out_dir, 'population_{}.pdf'.format(dataset))
        if os.path.exists(path) and not OVERWRITE:
            print('exists, skipped:', path)
            continue
        plot_populations(in_dir, dataset, path)
        print('written:', path)

    for name, datasets in CLASS_FIGURES:
        if not all(os.path.exists(os.path.join(in_dir, 'scores', d + '.csv')) for d in datasets):
            continue
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and not OVERWRITE:
            print('exists, skipped:', path)
            continue
        plot_class_scores(in_dir, datasets, path)
        print('written:', path)

    path = os.path.join(out_dir, 'pairwise_f1.pdf')
    if not os.path.exists(os.path.join(in_dir, 'pairwise_deltas.csv')):
        pass
    elif os.path.exists(path) and not OVERWRITE:
        print('exists, skipped:', path)
    else:
        plot_pairwise(in_dir, path)
        print('written:', path)

    # Bland-Altman: one figure per multi-class dataset, one for all binary datasets
    datasets = [d for d in PLOT_DATASETS if os.path.exists(os.path.join(in_dir, 'populations', d + '.csv'))]
    binary = [d for d in datasets if pd.read_csv(os.path.join(in_dir, 'populations', d + '.csv'),
                                                 usecols=['label'])['label'].nunique() == 2]
    jobs = [('bland_altman_{}.pdf'.format(d), plot_bland_altman_multiclass, d) for d in datasets if d not in binary]
    if binary:
        jobs.append(('bland_altman_binary.pdf', plot_bland_altman_binary, binary))
    for name, fn, arg in jobs:
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and not OVERWRITE:
            print('exists, skipped:', path)
            continue
        fn(in_dir, arg, path)
        print('written:', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in', dest='in_dir', default=IN_DIR, help='benchmark_eval.py output (default: %(default)s)')
    parser.add_argument('--out', default=OUT_DIR, help='figure directory (default: %(default)s)')
    args = parser.parse_args()
    main(os.path.abspath(args.in_dir), os.path.abspath(args.out))
