"""Env: fx_plot. Usage: python scratchpad/plot_legacy_criterion.py (reads scratchpad/results/legacy_criterion_check/).

Illustrates the non-inferiority criterion on the legacy training data requirement results: per method x dataset the
paired difference to the full-data baseline (mean over the test samples, 95 % paired bootstrap CI as band) over the
number of training samples, one line per events-per-sample cap. Reference lines: 0, -delta (new criterion; a cell
passes if its CI band lies above it), and the legacy threshold (lower bound of the baseline CI, in delta units).
Markers: minimal cell of the new criterion (circle) and of the legacy criterion (cross), full legacy grid.
-> scratchpad/figures/legacy_criterion_check/legacy_criterion.png (300 dpi).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
IN = os.path.join(HERE, 'results', 'legacy_criterion_check')
OUT = os.path.join(HERE, 'figures', 'legacy_criterion_check')
import config  # noqa: E402

DELTA = 0.02
EVENTS = ['5000', '20000', '50000', 'all']
YLIM = (-0.15, 0.04)
PANELS = [('Flowcyt', 'MLP'), ('LT1', 'MLP'), ('LT1b', 'MLP'), ('Imstat', 'MLP'), ('LT2', 'MLP'), ('LT2b', 'MLP'),
          ('Flowcyt', 'SOM'), ('LT1', 'SOM'), ('LT1b', 'SOM'), ('Imstat', 'SOM'), ('LT2', 'SOM'), ('LT2b', 'SOM')]

plt.rcParams.update(config.PLOT_RC)
t = pd.read_csv(os.path.join(IN, 'legacy_deltas.csv'), dtype={'n_events': str})
m = pd.read_csv(os.path.join(IN, 'legacy_minimal.csv'))
m = m[m['grid'] == 'full']
colors = dict(zip(EVENTS, plt.cm.viridis(np.linspace(0.0, 0.85, len(EVENTS)))))

fig, axes = plt.subplots(4, 3, figsize=(config.PAGE_WIDTH, 0.8 * config.PAGE_HEIGHT), sharey=True,
                         constrained_layout=True)
for ax, (ds, method) in zip(axes.ravel(), PANELS):
    sub = t[(t['dataset'] == ds) & (t['method'] == method) & (t['n_samples'] >= 2)]
    for ne in EVENTS:
        s = sub[sub['n_events'] == ne].sort_values('n_samples')
        ax.fill_between(s['n_samples'], s['delta_ci_low'], s['delta_ci_high'], color=colors[ne], alpha=0.18, lw=0)
        ax.plot(s['n_samples'], s['delta'], color=colors[ne], lw=0.8, label=ne)
    base = sub.iloc[0]
    ax.axhline(0, color='0.3', lw=0.6)
    ax.axhline(-DELTA, color='tab:red', lw=0.8, ls='--')
    ax.axhline(base['base_ci_low'] - base['baseline'], color='0.4', lw=0.8, ls=':')
    row = m[(m['dataset'] == ds) & (m['method'] == method)].iloc[0]
    for crit, marker, col in [('noninf_0.02', 'o', 'tab:red'), ('legacy_ci', 'x', '0.2')]:
        ns, ne = row[crit].split('x')
        c = sub[(sub['n_samples'] == int(ns)) & (sub['n_events'] == ne)].iloc[0]
        if marker == 'o':  # the criterion: lower CI bound of the selected cell above -delta
            ax.errorbar(c['n_samples'], c['delta'], yerr=[[c['delta'] - c['delta_ci_low']],
                        [c['delta_ci_high'] - c['delta']]], color=col, lw=0.9, capsize=2, zorder=5)
        ax.plot(c['n_samples'], np.clip(c['delta'], *YLIM), marker=marker, ms=5, mfc='none' if marker == 'o' else col,
                mec=col, mew=1.0, ls='none', zorder=5)
        ax.annotate(row[crit].replace('000x', 'k x').replace('x', ' x '), (c['n_samples'], np.clip(c['delta'], *YLIM)),
                    xytext=(4, -9 if marker == 'o' else 5), textcoords='offset points', fontsize=config.FONT_MIN,
                    color=col)
    ax.set_xscale('log')
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(['2', '5', '10', '20', '50'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(*YLIM)
    ax.set_title('{} {} (baseline {:.3f})'.format(ds, method, base['baseline']))
for ax in axes[-1]:
    ax.set_xlabel('Training samples')
for ax in axes[:, 0]:
    ax.set_ylabel('Δ macro F1 vs. full data')

handles = [plt.Line2D([], [], color=colors[e], lw=1.2, label='{} events'.format(e)) for e in EVENTS]
handles += [plt.Line2D([], [], color='tab:red', ls='--', lw=0.8, label='−δ = −{}'.format(DELTA)),
            plt.Line2D([], [], color='0.4', ls=':', lw=0.8, label='legacy threshold (baseline CI low)'),
            plt.Line2D([], [], marker='o', mfc='none', mec='tab:red', ls='none', label='minimal: non-inferiority'),
            plt.Line2D([], [], marker='x', mec='0.2', ls='none', label='minimal: legacy CI')]
fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
fig.suptitle('Paired difference to the full-data model (band: 95 % paired bootstrap CI); legacy results, one order')
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, 'legacy_criterion.png'), dpi=300)
print(os.path.join(OUT, 'legacy_criterion.png'))
