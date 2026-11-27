"""Env: fx_plot. Usage: python scratchpad/plot_delta_sensitivity.py (reads scratchpad/results/legacy_criterion_check/).

Sensitivity of the minimal training configuration to the non-inferiority margin delta, on the legacy training data
requirement results (one split, one sample order). For each delta on a fine grid, the minimal cell among the passing
cells (n_samples >= 2; lower bound of the 95 % paired bootstrap CI of the mean difference to the full-data model
> -delta; no stability rule) by two rules:
    greedy: fewest samples, then fewest events per sample (legacy rule, as legacy_criterion_check.py)
    joint:  fewest total training events (sum over the first n_samples of min(events of the sample, cap); counts from
            legacy_train_event_counts.py), ties -> fewer samples
Rows: minimal training samples, events per sample and total training events of the minimal cell (the accepted mean
difference to the full-data model is in the CSV, column delta). Columns: MLP / SOM x greedy / joint.
One line per dataset; vertical line at the default delta.
-> scratchpad/figures/legacy_criterion_check/delta_sensitivity.png (300 dpi) and
   scratchpad/results/legacy_criterion_check/delta_sensitivity.csv.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import config  # noqa: E402

IN = os.path.join(HERE, 'results', 'legacy_criterion_check')
OUT_FIG = os.path.join(HERE, 'figures', 'legacy_criterion_check')
DEFAULT_DELTA = 0.02
DELTAS = np.round(np.arange(0.001, 0.1001, 0.001), 3)
EVENTS = ['100', '1000', '5000', '10000', '20000', '50000', 'all']
DATASETS = ['Imstat', 'Flowcyt', 'LT1', 'LT2', 'LT1b', 'LT2b']  # order of config.PLOT_DATASETS
COLORS = dict(zip(DATASETS, ['#332288', '#88CCEE', '#117733', '#DDCC77', '#CC6677', '#AA4499']))
METHODS = ['MLP', 'SOM']


RULES = ['greedy', 'joint']


def total_events(counts, n_samples, n_events):
    """Training events of a cell: sum over the first n_samples samples of min(events, cap)."""
    n = counts[:n_samples]
    return int(n.sum() if n_events == 'all' else np.minimum(n, int(n_events)).sum())


def minimal_cell(sub, delta, rule):
    """Row of the minimal passing cell under the rule, or None."""
    ok = sub[(sub['n_samples'] >= 2) & (sub['delta_ci_low'] > -delta)]
    if ok.empty:
        return None
    keys = ['n_samples', 'ev'] if rule == 'greedy' else ['total_events', 'n_samples']
    return ok.sort_values(keys).iloc[0]


t = pd.read_csv(os.path.join(IN, 'legacy_deltas.csv'), dtype={'n_events': str})
ev = pd.read_csv(os.path.join(IN, 'legacy_train_event_counts.csv'))
counts = {ds: g.sort_values('sample_index')['n_events'].to_numpy() for ds, g in ev.groupby('dataset')}
t['ev'] = t['n_events'].map(EVENTS.index)
t['total_events'] = [total_events(counts[r.dataset], r.n_samples, r.n_events) for r in t.itertuples()]
rows = []
for (ds, method), sub in t.groupby(['dataset', 'method']):
    for rule in RULES:
        for d in DELTAS:
            c = minimal_cell(sub, d, rule)
            rows.append(dict(dataset=ds, method=method, rule=rule, delta_margin=d,
                             n_samples=c['n_samples'] if c is not None else np.nan,
                             n_events=c['n_events'] if c is not None else '',
                             total_events=c['total_events'] if c is not None else np.nan,
                             delta=c['delta'] if c is not None else np.nan))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(IN, 'delta_sensitivity.csv'), index=False)

plt.rcParams.update(config.PLOT_RC)
COLS = [(m, r) for m in METHODS for r in RULES]
fig, axes = plt.subplots(3, 4, figsize=(config.PAGE_WIDTH, 0.66 * config.PAGE_HEIGHT), sharex=True, sharey='row',
                         constrained_layout=True)
offsets = dict(zip(DATASETS, np.linspace(-0.25, 0.25, len(DATASETS))))
for j, (method, rule) in enumerate(COLS):
    for ds in DATASETS:
        s = res[(res['dataset'] == ds) & (res['method'] == method) & (res['rule'] == rule)].sort_values('delta_margin')
        kw = dict(color=COLORS[ds], lw=0.9, where='post', label=ds)
        axes[0, j].step(s['delta_margin'], s['n_samples'], **kw)
        axes[1, j].step(s['delta_margin'], s['n_events'].map(EVENTS.index) + offsets[ds], **kw)
        axes[2, j].step(s['delta_margin'], s['total_events'], **kw)
    axes[0, j].set_title('{} – {}'.format(method, rule))
    axes[2, j].set_xlabel('Margin δ')
    axes[2, j].set_xticks([0, 0.05, 0.1])
    for ax in axes[:, j]:
        ax.axvline(DEFAULT_DELTA, color='tab:red', lw=0.8, ls='--')
        ax.grid(axis='y', lw=0.3, color='0.85')
axes[0, 0].set_yscale('log')
axes[0, 0].set_yticks([2, 3, 5, 10, 20, 30, 58])
axes[0, 0].set_yticklabels(['2', '3', '5', '10', '20', '30', '58'])
axes[0, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0, 0].set_ylabel('Training samples')
axes[1, 0].set_yticks(range(len(EVENTS)))
axes[1, 0].set_yticklabels(['100', '1k', '5k', '10k', '20k', '50k', 'all'])
axes[1, 0].set_ylabel('Events per sample')
axes[2, 0].set_yscale('log')
axes[2, 0].set_ylabel('Total training events')
handles = [plt.Line2D([], [], color=COLORS[d], lw=1.2, label=d) for d in DATASETS]
handles += [plt.Line2D([], [], color='tab:red', ls='--', lw=0.8, label='default δ = {}'.format(DEFAULT_DELTA))]
fig.legend(handles=handles, loc='outside lower center', ncol=7, frameon=False)
fig.suptitle('Minimal training configuration over the margin δ: greedy (samples first) vs. joint (fewest total events)')
os.makedirs(OUT_FIG, exist_ok=True)
fig.savefig(os.path.join(OUT_FIG, 'delta_sensitivity.png'), dpi=300)
print(os.path.join(OUT_FIG, 'delta_sensitivity.png'))
