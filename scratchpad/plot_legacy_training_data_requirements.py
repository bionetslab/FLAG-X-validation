"""Env: fx_plot. Usage: python scratchpad/plot_legacy_training_data_requirements.py

The planned step-8 figures (plot_training_data_requirements.py) drawn on the legacy training data requirement results,
from the reformatted tables of scratchpad/legacy_training_data_requirements.py
(scratchpad/results/legacy_training_data_requirements/). Uses the plot functions of plot_training_data_requirements.py
unchanged; only its grid constants are set to the legacy grid, the score label to 'F1 macro' (legacy deltas are macro
F1 for all datasets), and the minority scatter reads points.csv (one point per cell: one split, one order) and has a
legend for the legacy grid. See the docstring of the table script for the differences to the step-8 eval.

-> scratchpad/figures/legacy_training_data_requirements/*.pdf
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import plot_training_data_requirements as P  # noqa: E402
import legacy_training_data_requirements as L  # noqa: E402

IN = L.OUT
OUT = os.path.join(HERE, 'figures', 'legacy_training_data_requirements')

P.X_GRID = L.X_GRID
P.X_LABELS = ['1', '2', '', '4', '', '10', '', '20', '', '40', '', 'all']
P.N_EVENTS = L.N_EVENTS
P.EVENT_LABELS = {100: '100', 1000: '1k', 5000: '5k', 10000: '10k', 20000: '20k', 50000: '50k', 'all': 'all'}
P.score_label = lambda dataset: 'F1 macro'
SCATTER_SHOWN = ['1', '2', '4', '10', '20', '40', 'all']  # legend of the minority scatter


def plot_minority_scatter(curves, path):
    """As P.plot_minority_scatter, points from points.csv (one per cell), legend for the legacy grid."""
    pts = pd.read_csv(os.path.join(IN, 'points.csv'), dtype={'n_samples': str, 'n_events': str})
    pts['n_samples'] = pts['n_samples'].map(P.grid_value)
    datasets, methods = P.datasets_methods(curves)
    ymax = max(P.PRIMARY_DELTA, np.ceil(pts['d'].max() * 50) / 50.0)
    xmax = pts['minority'].max()
    norm = len(P.X_GRID) - 1.0
    fig, axes = plt.subplots(len(datasets), len(methods), figsize=(P.PAGE_WIDTH, P.PAGE_HEIGHT), sharey=True,
                             constrained_layout=True, squeeze=False)
    for i, d in enumerate(datasets):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            ax.axhline(0, color='0.3', lw=0.5)
            ax.axhline(-P.PRIMARY_DELTA, color='0.3', lw=P.LW, ls=':')
            g = pts[(pts['dataset'] == d) & (pts['method'] == m)].copy()
            g['c'] = [P.x_pos(v) / norm for v in g['n_samples']]
            g = g.sort_values('c')
            low = g['d'] < P.SCATTER_YMIN
            ax.scatter(g.loc[~low, 'minority'], g.loc[~low, 'd'], s=P.SCATTER_MS ** 2,
                       c=P.SAMPLES_CMAP(g.loc[~low, 'c']), lw=0, alpha=0.6, rasterized=True)
            ax.scatter(g.loc[low, 'minority'], np.full(low.sum(), P.SCATTER_YMIN), s=P.SCATTER_MS ** 2,
                       facecolors='none', edgecolors=P.SAMPLES_CMAP(g.loc[low, 'c']), lw=0.4, alpha=0.6,
                       clip_on=False, rasterized=True)
            if len(g) >= 3:
                r = spearmanr(g['minority'], g['d'])[0]
                ax.text(0.03, 0.95, 'r$_S$ = {:.2f}'.format(r), transform=ax.transAxes, ha='left', va='top',
                        fontsize=P.FONT_MIN, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(-0.3, xmax * 2)
            ax.xaxis.set_major_locator(FixedLocator([0] + [10 ** e for e in range(0, 9, 2) if 10 ** e <= xmax * 2]))
            ax.xaxis.set_major_formatter(FuncFormatter(P.count_label))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(P.SCATTER_YMIN, ymax)
            ax.grid(axis='y', lw=0.3, color=P.GRID_COLOR)
            ax.set_axisbelow(True)
            if i < len(datasets) - 1:
                ax.set_xticklabels([])
    P.row_col_labels(axes, datasets, methods, lambda d: 'Δ ' + P.score_label(d))
    for j in range(len(methods)):
        axes[-1, j].set_xlabel('Minority count')
    handles = [Line2D([], [], ls='', marker='o', ms=P.SCATTER_MS + 1, mec='none',
                      mfc=P.SAMPLES_CMAP(P.X_GRID.index(P.grid_value(v)) / norm), label=v) for v in SCATTER_SHOWN]
    handles += [Line2D([], [], color='0.3', lw=P.LW, ls=':', label='−δ = −{}'.format(P.PRIMARY_DELTA)),
                Line2D([], [], ls='', marker='o', ms=P.SCATTER_MS + 1, mfc='none', mec='0.3', mew=0.4,
                       label='below {}'.format(P.SCATTER_YMIN))]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False,
               title='Training samples', columnspacing=1.0, handlelength=1.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    curves = P.read_table(os.path.join(IN, 'curves.csv'))
    minimal = P.read_table(os.path.join(IN, 'minimal.csv'))
    samples = P.read_table(os.path.join(IN, 'sample_scores.csv'))
    figures = [
        ('scores_all_events.pdf', lambda p: P.plot_all_events(curves, samples, p)),
        ('delta_all_events.pdf', lambda p: P.plot_all_events(curves, samples, p, delta=True)),
        ('scores_per_events.pdf', lambda p: P.plot_scores_per_events(curves, samples, p)),
        ('delta_per_events.pdf', lambda p: P.plot_delta_per_events(curves, minimal, samples, p)),
        ('delta_sensitivity.pdf', lambda p: P.plot_sensitivity(curves, minimal, p)),
        ('minority_sensitivity.pdf', lambda p: P.plot_sensitivity(curves, minimal, p, right='minority', in_dir=IN)),
        ('minority_scatter.pdf', lambda p: plot_minority_scatter(curves, p)),
    ]
    for name, plot in figures:
        path = os.path.join(OUT, name)
        plot(path)
        print(path)


if __name__ == '__main__':
    main()
