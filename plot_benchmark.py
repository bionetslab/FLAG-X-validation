"""
Benchmark figures (PLAN.md step 7 / 10).

Main figure: box plots of the sample-wise scores, one panel per dataset, one box per method, the individual test
samples jittered on top, the outer test fold shown by the marker.

    scores_box.pdf   one point = one test sample: its score averaged over the model seeds (stochastic methods: 3
                     seeds, deterministic: 1); box = distribution over all test samples (all outer folds pooled):
                     median and quartiles, whiskers 1.5 IQR, no fliers (all points are drawn).

Score of a sample (as in the tuning, validation.core.scores.f1_from_counts, scoring rule D8): macro F1 = NaN-mean of
the per-class f1_<l> columns; datasets with 2 labels (lt1b / lt2b: B cells, BLAST110: blasts) F1 of the positive
class, label 1 (f1_1). The y-axis label of each panel: 'F1 macro' or 'F1 binary'.
Unpredicted samples (failed / killed / missing atoms) are left out; a seed mean uses the seeds that predicted the
sample. A box with fewer samples than expected is annotated with 'k/n' (n = test samples of the dataset over all
outer splits); a method without any predicted sample is marked 'n/a'.

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

Order, names and colors of the datasets, methods and classes: config.py, section Plotting (PLOT_DATASETS,
DATASET_NAMES, PLOT_METHODS, METHOD_NAMES, METHOD_COLORS, CLASS_ORDER, CLASS_COLORS).

Env: fx_plot (envs/plot.yml)

Usage:
    python plot_benchmark.py [--in DIR] [--out DIR]
        --in: output of benchmark_eval.py (default results/benchmark_eval; pilot: results/benchmark_pilot_eval)
        --out: figure directory (default figures/benchmark; pilot: e.g. figures/benchmark_pilot)

Inputs: <in>/scores/<dataset>.csv, <in>/populations/<dataset>.csv
Outputs: <out>/scores_box.pdf, <out>/population_<dataset>.pdf, <out>/bland_altman_<dataset>.pdf (multi-class),
    <out>/bland_altman_binary.pdf
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr

from config import (RESULTS, FIGURES, PLOT_DATASETS, DATASET_NAMES, PLOT_METHODS, METHOD_NAMES,
                    METHOD_COLORS, CLASS_ORDER, CLASS_COLORS)
from validation.eval_helpers import ba_stats, population_seed_mean


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
SHOW_BA_TEXT = True  # Bland-Altman panels: 'bias [lower, upper limit of agreement]' in pp


def load_scores(in_dir, dataset):
    """
    Sample-wise scores of one dataset, averaged over the model seeds.

    Args:
        in_dir (str): Output directory of benchmark_eval.py.
        dataset (str): Dataset name.

    Returns:
        (pd.DataFrame, str): One row per method x outer split x test sample (all expected ones, also unpredicted) with
        columns method, split, sample, score (seed mean; NaN if no seed predicted the sample), n_seeds (seeds that
        predicted it); and the y-axis label naming the score.
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
    scores = (t.groupby(['method', 'split', 'sample'])['score'].agg(['mean', 'count']).reset_index()
              .rename(columns={'mean': 'score', 'count': 'n_seeds'}))
    return scores, y_label


def plot_panel(ax, scores, methods, rng):
    """
    Boxes per method (colored), the samples as points (marker = outer split), 'k/n' / 'n/a' annotations.

    Args:
        ax (plt.Axes): Panel of one dataset.
        scores (pd.DataFrame): load_scores output of the dataset.
        methods (list of str): Methods on the x-axis, in order (position i = methods[i]).
        rng (np.random.Generator): For the jitter.
    """
    for i, m in enumerate(methods):
        s = scores[scores['method'] == m]
        valid = s.dropna(subset=['score'])
        if valid.empty:
            ax.text(i, 0.02, 'n/a', ha='center', va='bottom', fontsize=7, color='0.5')
            continue
        color = METHOD_COLORS[m]
        ax.boxplot([valid['score'].values], positions=[i], widths=0.6, showfliers=False, patch_artist=True,
                   boxprops={'facecolor': color, 'alpha': 0.9, 'edgecolor': 'black', 'linewidth': 0.8},
                   medianprops={'color': 'black', 'linewidth': 1.2},
                   whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8}, zorder=2)
        for split, g in valid.groupby('split'):
            ax.scatter(i + rng.uniform(-JITTER, JITTER, len(g)), g['score'], s=10,
                       marker=FOLD_MARKERS[int(split) % len(FOLD_MARKERS)], color=color, edgecolors='black',
                       linewidths=0.4, alpha=0.6, zorder=3)
        if len(valid) < len(s):
            ax.text(i, 1.01, '{}/{}'.format(len(valid), len(s)), ha='center', va='bottom', fontsize=6, color='0.3')


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
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(11, 2.4 * n_rows + 1.2), sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    rng = np.random.default_rng(0)
    for k, ax in enumerate(axes.flat):
        if k >= len(datasets):
            ax.set_visible(False)
            continue
        plot_panel(ax, scores[datasets[k]], methods, rng)
        ax.set_title(DATASET_NAMES[datasets[k]], fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_ylabel(loaded[datasets[k]][1], fontsize=8)
        ax.tick_params(axis='y', labelleft=True)  # Every panel: y label and ticks (score type differs per panel)
    for ax in axes[-1]:
        ax.set_xticks(range(len(methods)), [METHOD_NAMES[m] for m in methods], rotation=45, ha='right',
                      rotation_mode='anchor', fontsize=8)
        ax.tick_params(axis='x', labelbottom=True)
    ax_last = axes.flat[len(datasets) - 1]  # Bottom panel of an odd grid gets the tick labels too
    ax_last.tick_params(axis='x', labelbottom=True)

    # Legend under the figure: methods (colors) in rows of N_LEGEND_COLS, then the test folds (markers) in one row
    method_handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor='black', linewidth=0.8, label=METHOD_NAMES[m])
                      for m in methods]
    fold_handles = [Line2D([], [], linestyle='', marker=FOLD_MARKERS[int(s) % len(FOLD_MARKERS)], color='0.5',
                           markeredgecolor='black', markeredgewidth=0.4, label='Test fold {}'.format(int(s)))
                    for s in splits]
    blank = Line2D([], [], linestyle='', label='')

    def rows(h):  # Pad to full rows of N_LEGEND_COLS
        return [h[i:i + N_LEGEND_COLS] + [blank] * (N_LEGEND_COLS - len(h[i:i + N_LEGEND_COLS]))
                for i in range(0, len(h), N_LEGEND_COLS)]
    grid = rows(method_handles) + rows(fold_handles)
    handles = [r[c] for c in range(N_LEGEND_COLS) for r in grid]  # matplotlib fills the legend column by column
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc='outside lower center',
               ncol=N_LEGEND_COLS, frameon=False, fontsize=8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


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
    fig, axes = plt.subplots(n_rows, POP_N_COLS, figsize=(9, 2.3 * n_rows + 1.0), sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    panels = []
    for k, ax in enumerate(axes.flat):
        if k >= len(methods):
            ax.set_visible(False)
            continue
        ax.set_title(METHOD_NAMES[methods[k]], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(linestyle='--', linewidth=0.5, alpha=0.7)
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
    fig.suptitle(title, fontsize=10)
    handles = [Line2D([], [], linestyle='', marker='o', color=CLASS_COLORS.get(c, '0.5'), markeredgecolor='black',
                      markeredgewidth=0.3, label=c) for c in classes]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False, fontsize=8,
               title='Class', title_fontsize=8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def scatter_classes(ax, p, classes, x, y):
    for c in classes:  # Classes in legend order; later ones on top
        g = p[p['name'] == c]
        ax.scatter(g[x], g[y], s=12, color=CLASS_COLORS.get(c, '0.5'), edgecolors='black', linewidths=0.3, alpha=0.8,
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
        ax.plot([0, 100], [0, 100], linestyle='--', color='0.5', linewidth=0.8, zorder=1)
        if p.empty:
            ax.text(0.75, 0.25, 'n/a', transform=ax.transAxes, ha='center', va='center', fontsize=8, color='0.5')
        else:
            scatter_classes(ax, p, classes, 'true_pct', 'pred_pct')
            text = agreement_text(p['true_pct'].to_numpy(), p['pred_pct'].to_numpy())
            if text:
                ax.text(0.04, 0.96, text, transform=ax.transAxes, ha='left', va='top', fontsize=7,
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 1.5})
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.set_aspect('equal')
        if ax.first_col:
            ax.set_ylabel('Predicted (%)', fontsize=8)
        if ax.bottom:
            ax.set_xlabel('True (%)', fontsize=8)
    finish_figure(fig, classes, DATASET_NAMES[dataset] + ': predicted vs. true population size per test sample', path)


def plot_ba_grid(columns, title, path, col_width=1.05):
    """
    Bland-Altman figure: one row per method (PLOT_METHODS), one column per (title, points, color).

    Args:
        columns (list of (str, pd.DataFrame, str)): Column title, its points (seed means, present only, columns
            method, mean_pct, d_pp), point color.
        title (str): Figure title.
        path (str): Output PDF.
        col_width (float): Width of one column (inch).
    """
    present = set().union(*(set(c[1]['method']) for c in columns))
    methods = [m for m in PLOT_METHODS if m in present]
    fig, axes = plt.subplots(len(methods), len(columns), figsize=(col_width * len(columns) + 1.0, 0.85 * len(methods) + 1.2),
                             sharex='col', sharey='col', squeeze=False, layout='constrained')
    for j, (col_title, pts, color) in enumerate(columns):
        lim = max(1.0, float(pts['d_pp'].abs().max()) * 1.15) if not pts.empty else 1.0  # Symmetric per column
        for i, m in enumerate(methods):
            ax = axes[i, j]
            p = pts[pts['method'] == m]
            ax.axhline(0, color='0.6', linewidth=0.5, zorder=1)
            if p.empty:
                ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes, ha='center', va='center', fontsize=6, color='0.5')
            else:
                ax.scatter(p['mean_pct'], p['d_pp'], s=6, color=color, edgecolors='black', linewidths=0.2, alpha=0.8,
                           zorder=2)
                st = ba_stats(p['d_pp'])
                ax.axhline(st['bias_pp'], color='black', linewidth=0.7, zorder=3)
                for v in (st['loa_low_pp'], st['loa_high_pp']):
                    if np.isfinite(v):
                        ax.axhline(v, color='black', linewidth=0.6, linestyle='--', zorder=3)
                if SHOW_BA_TEXT:
                    ax.text(0.03, 0.97, '{:.2f} [{:.2f}, {:.2f}]'.format(st['bias_pp'], st['loa_low_pp'],
                                                                      st['loa_high_pp']),
                            transform=ax.transAxes, ha='left', va='top', fontsize=5,
                            bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 0.8})
            ax.set_ylim(-lim, lim)
            ax.tick_params(labelsize=5, length=2, pad=1)
            ax.grid(linestyle='--', linewidth=0.4, alpha=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(col_title, fontsize=8)
            if j == 0:
                ax.set_ylabel(METHOD_NAMES[m], fontsize=7)
    fig.supxlabel('(Predicted + true) / 2 (%)', fontsize=8)
    fig.supylabel('Predicted − true (pp)', fontsize=8)
    fig.suptitle(title, fontsize=9)
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
                          'bias [95 % limits of agreement] (pp)', path, col_width=1.35)


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
