"""Env: fx_plot. Usage: python scratchpad/legacy_criterion_check.py [OUT_DIR]
(default: scratchpad/results/legacy_criterion_check).
Back-check of the step-8 criteria on the legacy num_samples x num_events results (one split, one order).
Per test sample macro F1 (res_df_sw_avg_f1.csv) vs. the full-data baseline (gating_performance), paired by row."""
import os
import sys
import numpy as np
import pandas as pd

LEG = '/data/bionets/ac07izid/FLAG-X-validation/legacy/results'
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                                                         'legacy_criterion_check')
os.makedirs(OUT, exist_ok=True)
DATASETS = {'Imstat': ('imstat', 75), 'LT1': ('lymphoma_tube1', 58), 'LT2': ('lymphoma_tube2', 58),
            'LT1b': ('lymphoma_tube1_binary', 58), 'LT2b': ('lymphoma_tube2_binary', 58),
            'Flowcyt': ('flowcyt', 18)}
METHODS = {'MLP': 'fcnn', 'SOM': 'som'}
EVENTS = [100, 1000, 5000, 10000, 20000, 50000, 'all']
LEGACY_EVENTS = [5000, 10000, 20000, 50000, 'all']   # grid of the legacy figure / Table 1
DELTAS = [0.01, 0.02, 0.05]
N_BOOT = 10000


def trafo(d):
    return 'log10_cutoff100' if d == 'flowcyt' else 'log10_w_custom_cutoffs'


def read(path):
    return pd.read_csv(path, index_col=0).drop(index=['mean', 'std'], errors='ignore')['macro'].to_numpy(float)


rows = []
for ds, (d, n_full) in DATASETS.items():
    for mname, m in METHODS.items():
        base = read(os.path.join(LEG, 'gating_performance', m, d, trafo(d), 'res_df_sw_avg_f1.csv'))
        rng = np.random.RandomState(42)
        idx = rng.randint(0, len(base), size=(N_BOOT, len(base)))   # same resamples for all cells (paired)
        b_lo, b_hi = np.percentile(base[idx].mean(1), [2.5, 97.5])
        for ns in range(1, n_full + 1):
            for ne in EVENTS:
                if ns == n_full and ne == 'all':
                    s = base
                else:
                    p = os.path.join(LEG, 'num_samples_num_events', m, d, trafo(d), 'random', 'detailed_res',
                                     'nevents_{}_nsamples_{}'.format(ne, ns), 'res_df_sw_avg_f1.csv')
                    if not os.path.exists(p):
                        continue
                    s = read(p)
                dd = s - base
                bm = dd[idx].mean(1)
                lo, hi = np.percentile(bm, [2.5, 97.5])
                rows.append(dict(dataset=ds, method=mname, n_samples=ns, n_events=ne, n_test=len(s),
                                 score=s.mean(), baseline=base.mean(), base_ci_low=b_lo, base_ci_high=b_hi,
                                 delta=dd.mean(), delta_sd=dd.std(ddof=1), delta_ci_low=lo, delta_ci_high=hi))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, 'legacy_deltas.csv'), index=False)


def first(sub, ok, legacy_grid):
    """Greedy (samples first, then events), n_samples >= 2 as in legacy."""
    sub = sub[sub['n_samples'] >= 2]
    if legacy_grid:
        sub = sub[(sub['n_samples'] % 5 == 0) | (sub['n_samples'] == sub['n_samples'].max())]
        sub = sub[sub['n_events'].isin(LEGACY_EVENTS)]
    for ns in sorted(sub['n_samples'].unique()):
        for ne in EVENTS:
            r = sub[(sub['n_samples'] == ns) & (sub['n_events'] == ne)]
            if len(r) and ok(r.iloc[0]):
                return '{}x{}'.format(ns, ne)
    return '-'


crit = [('legacy_ci', lambda r: r['score'] >= r['base_ci_low'])]
crit += [('point_{}'.format(x), (lambda x: lambda r: r['delta'] >= -x)(x)) for x in DELTAS]
crit += [('noninf_{}'.format(x), (lambda x: lambda r: r['delta_ci_low'] > -x)(x)) for x in DELTAS]
out = []
for grid in (True, False):
    for (ds, mname), sub in t.groupby(['dataset', 'method'], sort=False):
        r = dict(grid='legacy' if grid else 'full', dataset=ds, method=mname,
                 baseline=round(sub['baseline'].iloc[0], 3),
                 ci_halfwidth=round((sub['base_ci_high'].iloc[0] - sub['base_ci_low'].iloc[0]) / 2, 3))
        for name, ok in crit:
            r[name] = first(sub, ok, grid)
        out.append(r)
m = pd.DataFrame(out)
m.to_csv(os.path.join(OUT, 'legacy_minimal.csv'), index=False)
pd.set_option('display.width', 250)
print(m.to_string(index=False))
