"""
Training data requirement figures (PLAN.md step 8, working_notes/training_data_requirements_eval_plan.md), from the
outputs of training_data_requirements_eval.py. Scores: macro F1, binary datasets (2 labels) the F1 of label 1
('F1 macro' / 'F1 binary' in the axis labels). Only the scope 'pooled' (all outer folds, trial means); only eligible
cells (all 5 folds x 3 trials in budget and ok); other cells are not drawn (a line has a gap there).

    scores_all_events.pdf   main figure (as legacy fig5_num_samples): one panel per dataset; x = training samples
                     (cells with all events per sample), y = mean score over the test samples (trial means, all folds
                     pooled) with its 95 % bootstrap CI (band; patients resampled, donors for Imstat); one line per
                     method (method colors); dashed horizontal line = the method's full-data score (step 7, seed
                     means, mean over all test samples). One y-range for all panels (R4.5: no truncated, differing
                     axes).
    delta_all_events.pdf   the main figure as paired differences: same layout and cells, y = mean paired difference d
                     = score(cell) - score(full-data model) over the test samples with its 95 % CI (band); solid line
                     at 0 (= full-data model), dotted line at -delta (PRIMARY_DELTA); top left corner: 'All data:
                     x | y | ...' = each method's full-data score (3 decimals, method colors). One y-range for all
                     panels (at least down to -2 delta).
    scores_per_events.pdf   supplementary (as legacy fig5s_num_samples_num_events): one row per dataset, one column
                     per method; x = training samples, y = mean score; one line per events cap (events per training
                     sample; color = cap); dashed line + grey band = full-data score and its 95 % CI. One y-range.
    delta_per_events.pdf   one row per dataset, one column per method; x = training samples, y = mean paired
                     difference d = score(cell) - score(full-data model) over the test samples, with its 95 % CI
                     (band); one line per events cap; line at 0 annotated with the full-data score (3 decimals);
                     dotted line at -delta (PRIMARY_DELTA); red ring = minimal cell (paired non-inferiority
                     at delta: lower CI bound > -delta; greedy: fewest samples, then fewest events; STABLE). One
                     common y-range D_YLIM (lower values are cut off).
    delta_sensitivity.pdf   minimal training configuration over the margin delta (the delta grid of the eval,
                     training_data_requirements_eval.DELTAS; log axis): two panels per dataset, left = training
                     samples of the greedy minimal cell, right = total training events of the joint minimal cell (mean
                     over the 15 lanes; log axis); one step line per method; vertical dotted line at PRIMARY_DELTA. No
                     line where no cell passes.

    minority_sensitivity.pdf   as delta_sensitivity.pdf, greedy rule only: left = training samples of the greedy
                     minimal cell, right = its minority count = the smallest class count of the whole training set
                     (per-sample class counts after downsampling summed over the training samples, minimum over all
                     classes of the dataset, no class excluded; 0 if a class is missing from the training set). One
                     value per lane (outer fold x trial, 15 per cell): line = median over the lanes, band = min-max.
                     symlog y-axis (linear below 1), so 0 is shown. No class label: the minority class is not the same
                     in every training set (mostly one class per dataset, other classes at 1-2 training samples).
                     Lanes where the cell is the step-7 model (('all', 'all') alias) have no class counts and are left
                     out.

    minority_scatter.pdf   (as legacy fig6s_minority_count, with Δ) one row per dataset, one column per method; one
                     point per completed atom (status ok and in budget; cell x outer fold x trial, i.e. one trained
                     model): x = minority count of its training set (as in minority_sensitivity.pdf; symlog, so 0 is
                     shown), y = Δ = mean over the fold's test samples of (atom score - full-data score); color =
                     training samples. Points below SCATTER_YMIN are drawn at it as open markers. r_S = Spearman
                     correlation of the panel's points, descriptive only (no p-value: the points share test samples
                     and nested training data, so they are not independent). Lines at 0 and -delta.

x-axis of the curve figures: the training sample grid of the default dataset grid (training_data_requirements.N_SAMPLES)
at equal spacing (the grid is roughly log-spaced); 'all' = all training samples of the fold (75-101, FlowCyt 24).

Order, names and colors of the datasets and methods: config.py, section Plotting.

Env: fx_plot (envs/plot.yml)

Usage:
    python plot_training_data_requirements.py [--in DIR] [--out DIR]
        --in: output of training_data_requirements_eval.py (default results/training_data_requirements_eval; dummy:
              results/training_data_requirements_dummy_eval)
        --out: figure directory (default figures/training_data_requirements)

Inputs: <in>/curves.csv, <in>/minimal.csv, <in>/sample_scores.csv, <in>/atoms.csv,
    <in>/train_class_counts/<dataset>.csv
Outputs: <out>/scores_all_events.pdf, <out>/delta_all_events.pdf, <out>/scores_per_events.pdf,
    <out>/delta_per_events.pdf, <out>/delta_sensitivity.pdf, <out>/minority_sensitivity.pdf,
    <out>/minority_scatter.pdf
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.stats import spearmanr

from config import (DATA_CURATED, RESULTS, FIGURES, PLOT_DATASETS, DATASET_NAMES, PLOT_METHODS, METHOD_NAMES,
                    METHOD_COLORS, PAGE_WIDTH, PAGE_HEIGHT, PLOT_RC, FONT_ANNOT, FONT_MIN, FONT_PANEL, DELTA)
from training_data_requirements import N_EVENTS, N_SAMPLES
from training_data_requirements_eval import DELTAS

# Final printed size: every figure is PAGE_WIDTH wide and at most PAGE_HEIGHT tall, font sizes FONT_* (config.py)
plt.rcParams.update(PLOT_RC)

OVERWRITE = True  # If False, existing figures are skipped

IN_DIR = os.path.join(RESULTS, 'training_data_requirements_eval')
OUT_DIR = os.path.join(FIGURES, 'training_data_requirements')

PRIMARY_DELTA = DELTA  # Margin of the minimal cell in delta_per_events.pdf
STABLE = False  # Minimal cell with the stability rule (primary still open, see the eval plan)
X_GRID = N_SAMPLES['default']  # x positions of the curve figures (equally spaced)
X_LABELS = ['1', '2', '', '4', '', '8', '', '16', '', '32', '', '', 'all']  # Tick labels (others unlabelled)
EVENT_LABELS = {100: '100', 300: '300', 1000: '1k', 3000: '3k', 10000: '10k', 30000: '30k', 100000: '100k',
                300000: '300k', 'all': 'all'}
EVENT_CMAP = plt.get_cmap('magma')  # Color of the events cap, from EVENT_CMAP_RANGE[0] (100) to [1] ('all')
EVENT_CMAP_RANGE = (0.0, 0.8)
D_YLIM = (-0.2, 0.05)  # y-range of delta_per_events.pdf
SCORE_YMAX = 1.0  # Upper end of the score axes; the lower end is the smallest plotted value (rounded down to 0.05)
SHOW_CI = True  # CI bands in the *_per_events.pdf figures (the *_all_events.pdf figures always have them)
MINIMAL_COLOR = '#CC3311'  # Ring of the minimal cell in delta_per_events.pdf
FULL_LABEL_COLOR = '0.3'  # 'All data:' and the separators of the full-data scores in the Δ figures
LW = 0.8
MS = 1.8  # Marker size
GRID_COLOR = '0.9'
SAMPLES_CMAP = plt.get_cmap('viridis')  # minority_scatter.pdf: color = training samples (1 ... all)
SCATTER_YMIN = -0.3  # minority_scatter.pdf: lower end of the y-axis; lower points are drawn at it as open markers
SCATTER_MS = 2.0  # Marker size (points^2 for scatter: SCATTER_MS ** 2)


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def grid_value(v):
    return 'all' if str(v) == 'all' else int(float(v))


def read_table(path):
    t = pd.read_csv(path, dtype={'n_samples': str, 'n_events': str}, keep_default_na=False, na_values=[''])
    for c in ('n_samples', 'n_events'):
        t[c] = [grid_value(v) if isinstance(v, str) else np.nan for v in t[c]]
    return t


def x_pos(ns):
    return X_GRID.index(ns)


def event_color(ne):
    lo, hi = EVENT_CMAP_RANGE
    return EVENT_CMAP(lo + (hi - lo) * N_EVENTS.index(ne) / (len(N_EVENTS) - 1.0))


def score_label(dataset):
    k = len(pd.read_csv(os.path.join(DATA_CURATED, dataset, 'meta_labels.csv')))
    return 'F1 binary' if k == 2 else 'F1 macro'


def datasets_methods(curves):
    datasets = [d for d in PLOT_DATASETS if d in set(curves['dataset'])]
    methods = [m for m in PLOT_METHODS if m in set(curves['method'])]
    return datasets, methods


def pooled(curves):
    """Eligible cells of the scope 'pooled', sorted by n_samples."""
    c = curves[(curves['scope'] == 'pooled') & curves['eligible'].astype(bool)].copy()
    c['x'] = c['n_samples'].map(x_pos)
    return c.sort_values('x')


def full_data_scores(samples):
    """Full-data (step 7) score per method x dataset: mean of the seed means over the test samples, and its n."""
    b = samples.drop_duplicates(['method', 'dataset', 'sample'])
    return b.groupby(['method', 'dataset'])['baseline'].agg(['mean', 'count'])


def setup_x(ax, labels=True):
    ax.set_xticks(range(len(X_GRID)))
    ax.set_xticklabels(X_LABELS if labels else [])
    ax.set_xlim(-0.4, len(X_GRID) - 0.6)
    ax.grid(axis='y', lw=0.3, color=GRID_COLOR)
    ax.set_axisbelow(True)


def common_score_ylim(values):
    lo = np.nanmin(values)
    return (np.floor(lo * 20) / 20.0, SCORE_YMAX)


def annotate_full(ax, items, corner=False):
    """
    Full-data scores in a Δ panel (3 decimals): items = [(score, color)], one per method. Default: the numbers only,
    above the y = 0 line at the left end; corner=True: 'All data: x | y | ...' in the top left corner, each number in
    its color.
    """
    stroke = [pe.withStroke(linewidth=1.5, foreground='white')]
    if not corner:  # One line of text at the 0 line (one color: the first item's)
        ax.annotate(' | '.join('{:.3f}'.format(v) for v, _ in items), xy=(0.02, 0),
                    xycoords=('axes fraction', 'data'), xytext=(0, 1), textcoords='offset points', ha='left',
                    va='bottom', fontsize=FONT_MIN, color=items[0][1], zorder=5, path_effects=stroke)
        return
    parts = [TextArea('All data:', textprops=dict(fontsize=FONT_MIN, color=FULL_LABEL_COLOR, path_effects=stroke))]
    for k, (v, color) in enumerate(items):
        if k:
            parts.append(TextArea('|', textprops=dict(fontsize=FONT_MIN, color=FULL_LABEL_COLOR)))
        parts.append(TextArea('{:.3f}'.format(v), textprops=dict(fontsize=FONT_MIN, color=color, path_effects=stroke)))
    box = AnchoredOffsetbox(loc='upper left', child=HPacker(children=parts, pad=0, sep=2), pad=0.1, borderpad=0.4,
                            frameon=False)
    box.set_zorder(5)
    ax.add_artist(box)


def event_lines(ax, c, col, lo=None, hi=None):
    """One line per events cap of a method x dataset (c: pooled rows); optional CI band from columns lo / hi."""
    for ne in N_EVENTS:
        g = c[c['n_events'] == ne]
        if g.empty:
            continue
        color = event_color(ne)
        ax.plot(g['x'], g[col], color=color, lw=LW, marker='o', ms=MS)
        if SHOW_CI and lo is not None:
            ax.fill_between(g['x'], g[lo], g[hi], color=color, alpha=0.15, lw=0)


def event_legend(fig, extra):
    handles = [Line2D([], [], color=event_color(ne), lw=LW, marker='o', ms=MS, label=EVENT_LABELS[ne])
               for ne in N_EVENTS] + extra
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False,
               title='Events per training sample', columnspacing=1.0, handlelength=1.5)


def row_col_labels(axes, datasets, methods, ylabel):
    for j, m in enumerate(methods):
        axes[0, j].set_title(METHOD_NAMES[m])
    for i, d in enumerate(datasets):
        axes[i, 0].set_ylabel('{}\n{}'.format(DATASET_NAMES[d], ylabel(d)))
    for j in range(len(methods)):
        axes[-1, j].set_xlabel('Training samples')


# ---------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------

def plot_all_events(curves, samples, path, delta=False):
    """
    Main figure (all events per sample), one panel per dataset, one line per method: the mean score with the full-data
    score as a dashed line (delta=False), or the mean paired difference d to the full-data model with 0 and -delta
    (delta=True).
    """
    c = pooled(curves)
    c = c[c['n_events'] == 'all']
    full = full_data_scores(samples)
    datasets, methods = datasets_methods(curves)
    col, lo, hi = ('mean_d', 'ci_low', 'ci_high') if delta else ('mean_score', 'score_ci_low', 'score_ci_high')
    n_cols = 3
    n_rows = int(np.ceil((len(datasets) + 1) / float(n_cols)))  # + 1 slot for the legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PAGE_WIDTH, 0.18 * PAGE_HEIGHT * n_rows), sharex=True,
                             sharey=True, constrained_layout=True)
    axes = axes.ravel()
    if delta:  # One range for all panels, at least down to -2 delta
        ylim = (min(np.floor(np.nanmin(c[lo]) * 50) / 50.0, -2 * PRIMARY_DELTA),
                max(np.ceil(np.nanmax(c[hi]) * 50) / 50.0, PRIMARY_DELTA))
    else:
        ylim = common_score_ylim(np.concatenate([c[lo].to_numpy(), full['mean'].to_numpy()]))
    for k, d in enumerate(datasets):
        ax = axes[k]
        if delta:
            ax.axhline(0, color='0.3', lw=0.5)
            ax.axhline(-PRIMARY_DELTA, color='0.3', lw=LW, ls=':')
        for m in methods:
            g = c[(c['dataset'] == d) & (c['method'] == m)]
            color = METHOD_COLORS[m]
            ax.plot(g['x'], g[col], color=color, lw=LW, marker='o', ms=MS)
            ax.fill_between(g['x'], g[lo], g[hi], color=color, alpha=0.2, lw=0)
            if not delta and (m, d) in full.index:
                ax.axhline(full.loc[(m, d), 'mean'], color=color, lw=LW, ls='--')
        if delta:
            annotate_full(ax, [(full.loc[(m, d), 'mean'], METHOD_COLORS[m]) for m in methods if (m, d) in full.index],
                          corner=True)
        setup_x(ax)
        ax.set_ylim(*ylim)
        ax.set_title(DATASET_NAMES[d])
        ax.set_ylabel(('Δ ' if delta else '') + score_label(d))
        ax.text(-0.02, 1.02, chr(ord('A') + k), transform=ax.transAxes, fontsize=FONT_PANEL, fontweight='bold',
                ha='right', va='bottom')
        if k >= len(datasets) - n_cols:
            ax.set_xlabel('Training samples')
            ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[len(datasets):]:
        ax.axis('off')
    handles = [Line2D([], [], color=METHOD_COLORS[m], lw=LW, marker='o', ms=MS, label=METHOD_NAMES[m])
               for m in methods]
    if delta:
        handles += [Line2D([], [], color='0.3', lw=LW, ls=':', label='−δ = −{}'.format(PRIMARY_DELTA))]
        title = 'All events per training sample\nΔ = score − full-data score\n(band: 95 % CI)'
    else:
        handles += [Line2D([], [], color='0.3', lw=LW, ls='--', label='All training data')]
        title = 'All events per training sample\n(band: 95 % CI)'
    axes[len(datasets)].legend(handles=handles, loc='center', frameon=False, title=title)
    fig.savefig(path)
    plt.close(fig)


def plot_scores_per_events(curves, samples, path):
    """Supplementary: mean score per events cap, one row per dataset, one column per method."""
    c = pooled(curves)
    # CI of the full-data score: base_ci of a cell with every test sample (all samples have a defined d)
    full_t = curves[(curves['scope'] == 'pooled') & (curves['n_test'] == curves['n_test_total'])].drop_duplicates(
        ['method', 'dataset'])
    full = full_data_scores(samples)
    datasets, methods = datasets_methods(curves)
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(PAGE_WIDTH, PAGE_HEIGHT), sharex=True,
                             sharey=True, constrained_layout=True, squeeze=False)
    ylim = common_score_ylim(np.concatenate([c['mean_score'].to_numpy(), full['mean'].to_numpy()]))
    for i, d in enumerate(datasets):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            g = c[(c['dataset'] == d) & (c['method'] == m)]
            event_lines(ax, g, 'mean_score', 'score_ci_low', 'score_ci_high')
            if (m, d) in full.index:
                ax.axhline(full.loc[(m, d), 'mean'], color='0.3', lw=LW, ls='--')
                b = full_t[(full_t['dataset'] == d) & (full_t['method'] == m)]
                if len(b):
                    ax.axhspan(b['base_ci_low'].iloc[0], b['base_ci_high'].iloc[0], color='0.5', alpha=0.15, lw=0)
            setup_x(ax, labels=i == len(datasets) - 1)
            ax.set_ylim(*ylim)
    row_col_labels(axes, datasets, methods, score_label)
    event_legend(fig, [Line2D([], [], color='0.3', lw=LW, ls='--', label='all training data')])
    fig.savefig(path)
    plt.close(fig)


def plot_delta_per_events(curves, minimal, samples, path):
    """Mean paired difference to the full-data model per events cap, with the minimal cell at PRIMARY_DELTA."""
    c = pooled(curves)
    full = full_data_scores(samples)
    datasets, methods = datasets_methods(curves)
    mn = minimal[(minimal['scope'] == 'pooled') & (minimal['criterion'] == 'noninf')
                 & np.isclose(minimal['delta'], PRIMARY_DELTA) & (minimal['order'] == 'greedy')
                 & (minimal['stable'].astype(bool) == STABLE) & minimal['found'].astype(bool)]
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(PAGE_WIDTH, PAGE_HEIGHT), sharex=True,
                             sharey=True, constrained_layout=True, squeeze=False)
    for i, d in enumerate(datasets):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            ax.axhline(0, color='0.3', lw=0.5)
            ax.axhline(-PRIMARY_DELTA, color='0.3', lw=LW, ls=':')
            g = c[(c['dataset'] == d) & (c['method'] == m)]
            event_lines(ax, g, 'mean_d', 'ci_low', 'ci_high')
            r = mn[(mn['dataset'] == d) & (mn['method'] == m)]
            if len(r):
                r = r.iloc[0]
                ax.plot(x_pos(r['n_samples']), r['mean_d'], 'o', ms=5.5, mfc='none', mec=MINIMAL_COLOR, mew=1.2,
                        zorder=4)
            if (m, d) in full.index:
                annotate_full(ax, [(full.loc[(m, d), 'mean'], '0.2')])
            setup_x(ax, labels=i == len(datasets) - 1)
            ax.set_ylim(*D_YLIM)
    row_col_labels(axes, datasets, methods, lambda d: 'Δ ' + score_label(d))
    event_legend(fig, [Line2D([], [], color='0.3', lw=LW, ls=':', label='−δ = −{}'.format(PRIMARY_DELTA)),
                       Line2D([], [], ls='', marker='o', ms=5.5, mfc='none', mec=MINIMAL_COLOR, mew=1.2,
                              label='minimal')])
    fig.savefig(path)
    plt.close(fig)


def count_label(v, _pos=None):
    """Tick label of a count axis: 0, 1, 10, 100, 1k, 10k, 100k, 1M, ..."""
    for div, suffix in ((1e6, 'M'), (1e3, 'k')):
        if v >= div:
            return '{:g}{}'.format(v / div, suffix)
    return '{:g}'.format(v)


def minority_counts(in_dir, dataset):
    """
    Minority count of every training set of the dataset: {(split, trial, n_samples, n_events): count}, count = the
    smallest class count of the whole training set (per-sample class counts summed over the training samples, minimum
    over all classes of the dataset; 0 if a class is missing). From <in>/train_class_counts/<dataset>.csv.
    """
    t = read_table(os.path.join(in_dir, 'train_class_counts', dataset + '.csv'))
    cols = [c for c in t.columns if c.startswith('n_') and c[2:].isdigit()]
    g = t.groupby(['split', 'trial', 'n_samples', 'n_events'], sort=False)[cols].sum()
    return dict(zip(g.index, g.to_numpy().min(axis=1)))


def cell_minority(mc, ns, ne):
    """
    Minority counts of a cell over its lanes (split x trial), (median, min, max). A cell missing in a lane is resolved
    as in the eval (training_data_requirements_eval.lane_table): (n_samples, 'all'), then ('all', n_events); a lane
    resolved to the step-7 model ('all', 'all') has no class counts and is left out.
    """
    lanes = sorted(set(k[:2] for k in mc))
    v = []
    for lane in lanes:
        for key in [(ns, ne), (ns, 'all'), ('all', ne)]:
            if lane + key in mc:
                v.append(mc[lane + key])
                break
    return (np.median(v), np.min(v), np.max(v)) if v else (np.nan, np.nan, np.nan)


def plot_sensitivity(curves, minimal, path, right='events', in_dir=None):
    """
    Minimal training configuration over delta, two panels per dataset: left the training samples of the greedy minimal
    cell; right (right='events') the total training events of the joint minimal cell, or (right='minority') the
    minority count of the greedy minimal cell (median over the lanes, band min-max; symlog axis, so 0 is shown).
    """
    datasets, methods = datasets_methods(curves)
    mn = minimal[(minimal['scope'] == 'pooled') & (minimal['criterion'] == 'noninf')
                 & (minimal['stable'].astype(bool) == STABLE)]
    n_cols = 4  # 2 datasets per row
    n_rows = int(np.ceil(len(datasets) / 2.0))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(PAGE_WIDTH, 0.16 * PAGE_HEIGHT * n_rows), sharex=True,
                             constrained_layout=True, squeeze=False)
    for k, d in enumerate(datasets):
        ax_s, ax_r = axes[k // 2, 2 * (k % 2)], axes[k // 2, 2 * (k % 2) + 1]
        mc = minority_counts(in_dir, d) if right == 'minority' else None
        for m in methods:
            color = METHOD_COLORS[m]
            g = mn[(mn['dataset'] == d) & (mn['method'] == m) & (mn['order'] == 'greedy')].sort_values('delta')
            found = g['found'].astype(bool).to_numpy()
            y = [x_pos(v) if f else np.nan for v, f in zip(g['n_samples'], found)]
            ax_s.step(g['delta'], y, where='post', color=color, lw=LW)
            if right == 'minority':
                q = np.array([cell_minority(mc, ns, ne) if f else (np.nan,) * 3
                              for ns, ne, f in zip(g['n_samples'], g['n_events'], found)])
                ax_r.step(g['delta'], q[:, 0], where='post', color=color, lw=LW)
                ax_r.fill_between(g['delta'], q[:, 1], q[:, 2], step='post', color=color, alpha=0.15, lw=0)
            else:
                g = mn[(mn['dataset'] == d) & (mn['method'] == m) & (mn['order'] == 'joint')].sort_values('delta')
                ax_r.step(g['delta'], g['n_train_events_mean'].where(g['found'].astype(bool)), where='post',
                          color=color, lw=LW)
        ax_s.set_yticks(range(len(X_GRID)))
        ax_s.set_yticklabels(X_LABELS)
        ax_s.set_ylim(-0.5, len(X_GRID) - 0.5)
        if right == 'minority':
            ax_r.set_yscale('symlog', linthresh=1)
            ax_r.set_ylim(bottom=0)
            ax_r.yaxis.set_major_formatter(FuncFormatter(count_label))
        else:
            ax_r.set_yscale('log')
        if k % 2 == 0:  # Left dataset of a row: y labels (the right dataset has the same axes)
            ax_s.set_ylabel('Training samples\n(greedy)')
            ax_r.set_ylabel('Minority count\n(greedy)' if right == 'minority' else 'Total training events\n(joint)')
        ax_s.set_title(DATASET_NAMES[d], loc='left')
        for ax in (ax_s, ax_r):
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([0.001, 0.01, 0.1])
            ax.set_xticklabels(['0.001', '0.01', '0.1'])
            ax.axvline(PRIMARY_DELTA, color='0.3', lw=LW, ls=':')
            ax.grid(axis='y', lw=0.3, color=GRID_COLOR)
            ax.set_axisbelow(True)
        if k // 2 == n_rows - 1:
            ax_s.set_xlabel('Margin δ')
            ax_r.set_xlabel('Margin δ')
    for k in range(len(datasets), 2 * n_rows):
        axes[k // 2, 2 * (k % 2)].axis('off')
        axes[k // 2, 2 * (k % 2) + 1].axis('off')
    handles = [Line2D([], [], color=METHOD_COLORS[m], lw=LW, label=METHOD_NAMES[m]) for m in methods]
    handles += [Line2D([], [], color='0.3', lw=LW, ls=':', label='δ = {}'.format(PRIMARY_DELTA))]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False)
    fig.savefig(path)
    plt.close(fig)


def atom_points(in_dir, samples):
    """
    One row per completed atom (status ok and in budget): method, dataset, split, trial, n_samples, n_events, minority
    (smallest class count of its training set, minority_counts) and d (mean over the fold's test samples of the
    atom's score - the full-data score; samples where either is undefined are left out).
    """
    a = read_table(os.path.join(in_dir, 'atoms.csv'))
    a = a[(a['status'] == 'ok') & a['in_budget'].astype(bool)]
    cell = ['method', 'dataset', 'n_samples', 'n_events', 'split']
    rows = []
    for d, g in a.groupby('dataset', sort=False):
        mc = minority_counts(in_dir, d)
        s = samples[samples['dataset'] == d]
        for t in sorted(g['trial'].unique()):
            diff = (s['score_t{}'.format(t)] - s['baseline']).rename('d')
            mean_d = pd.concat([s[cell], diff], axis=1).groupby(cell, sort=False)['d'].mean()
            gt = g[g['trial'] == t]
            keys = list(zip(gt['method'], gt['dataset'], gt['n_samples'], gt['n_events'], gt['split']))
            rows.append(pd.DataFrame({
                'method': gt['method'].to_numpy(), 'dataset': d, 'split': gt['split'].to_numpy(), 'trial': t,
                'n_samples': gt['n_samples'].to_numpy(), 'n_events': gt['n_events'].to_numpy(),
                'minority': [mc.get((sp, t, ns, ne), np.nan) for sp, ns, ne in
                             zip(gt['split'], gt['n_samples'], gt['n_events'])],
                'd': [mean_d.get(k, np.nan) for k in keys]}))
    return pd.concat(rows, ignore_index=True)


def plot_minority_scatter(curves, samples, in_dir, path):
    """Δ against the minority count of the training set, one point per completed atom, dataset x method grid."""
    pts = atom_points(in_dir, samples)
    pts = pts[np.isfinite(pts['d']) & np.isfinite(pts['minority'])]
    datasets, methods = datasets_methods(curves)
    ymax = max(PRIMARY_DELTA, np.ceil(pts['d'].max() * 50) / 50.0)
    xmax = pts['minority'].max()
    norm = len(X_GRID) - 1.0
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(PAGE_WIDTH, PAGE_HEIGHT), sharey=True,
                             constrained_layout=True, squeeze=False)
    for i, d in enumerate(datasets):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            ax.axhline(0, color='0.3', lw=0.5)
            ax.axhline(-PRIMARY_DELTA, color='0.3', lw=LW, ls=':')
            g = pts[(pts['dataset'] == d) & (pts['method'] == m)].copy()
            g['c'] = [x_pos(v) / norm for v in g['n_samples']]
            g = g.sort_values('c')  # More training samples drawn on top
            low = g['d'] < SCATTER_YMIN
            ax.scatter(g.loc[~low, 'minority'], g.loc[~low, 'd'], s=SCATTER_MS ** 2, c=SAMPLES_CMAP(g.loc[~low, 'c']),
                       lw=0, alpha=0.6, rasterized=True)
            ax.scatter(g.loc[low, 'minority'], np.full(low.sum(), SCATTER_YMIN), s=SCATTER_MS ** 2,
                       facecolors='none', edgecolors=SAMPLES_CMAP(g.loc[low, 'c']), lw=0.4, alpha=0.6, clip_on=False,
                       rasterized=True)
            if len(g) >= 3:
                r = spearmanr(g['minority'], g['d'])[0]
                ax.text(0.03, 0.95, 'r$_S$ = {:.2f}'.format(r), transform=ax.transAxes, ha='left', va='top',
                        fontsize=FONT_MIN, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(-0.3, xmax * 2)
            ax.xaxis.set_major_locator(FixedLocator([0] + [10 ** e for e in range(0, 9, 2) if 10 ** e <= xmax * 2]))
            ax.xaxis.set_major_formatter(FuncFormatter(count_label))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(SCATTER_YMIN, ymax)
            ax.grid(axis='y', lw=0.3, color=GRID_COLOR)
            ax.set_axisbelow(True)
            if i < len(datasets) - 1:
                ax.set_xticklabels([])
    row_col_labels(axes, datasets, methods, lambda d: 'Δ ' + score_label(d))
    for j in range(len(methods)):
        axes[-1, j].set_xlabel('Minority count')
    shown = ['1', '2', '4', '8', '16', '32', '64', 'all']
    handles = [Line2D([], [], ls='', marker='o', ms=SCATTER_MS + 1, mec='none',
                      mfc=SAMPLES_CMAP(X_GRID.index('all' if v == 'all' else int(v)) / norm), label=v) for v in shown]
    handles += [Line2D([], [], color='0.3', lw=LW, ls=':', label='−δ = −{}'.format(PRIMARY_DELTA)),
                Line2D([], [], ls='', marker='o', ms=SCATTER_MS + 1, mfc='none', mec='0.3', mew=0.4,
                       label='below {}'.format(SCATTER_YMIN))]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False,
               title='Training samples', columnspacing=1.0, handlelength=1.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    curves = read_table(os.path.join(in_dir, 'curves.csv'))
    minimal = read_table(os.path.join(in_dir, 'minimal.csv'))
    samples = read_table(os.path.join(in_dir, 'sample_scores.csv'))
    figures = [
        ('scores_all_events.pdf', lambda p: plot_all_events(curves, samples, p)),
        ('delta_all_events.pdf', lambda p: plot_all_events(curves, samples, p, delta=True)),
        ('scores_per_events.pdf', lambda p: plot_scores_per_events(curves, samples, p)),
        ('delta_per_events.pdf', lambda p: plot_delta_per_events(curves, minimal, samples, p)),
        ('delta_sensitivity.pdf', lambda p: plot_sensitivity(curves, minimal, p)),
        ('minority_sensitivity.pdf', lambda p: plot_sensitivity(curves, minimal, p, right='minority', in_dir=in_dir)),
        ('minority_scatter.pdf', lambda p: plot_minority_scatter(curves, samples, in_dir, p)),
    ]
    for name, plot in figures:
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and not OVERWRITE:
            continue
        plot(path)
        print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in', dest='in_dir', default=IN_DIR, help='eval output (default: %(default)s)')
    parser.add_argument('--out', default=OUT_DIR, help='figure directory (default: %(default)s)')
    args = parser.parse_args()
    main(os.path.abspath(args.in_dir), os.path.abspath(args.out))
