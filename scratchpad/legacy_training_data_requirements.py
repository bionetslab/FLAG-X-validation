"""Env: fx_plot. Usage: python scratchpad/legacy_training_data_requirements.py

Reformats the legacy training data requirement results (one split, one 'random' sample order = sample_00 ...,
SOM and MLP) into the input tables of plot_training_data_requirements.py, so the planned step-8 figures can be drawn
on them (scratchpad/plot_legacy_training_data_requirements.py). The legacy results are only read, never changed.

Sources:
    scratchpad/results/legacy_criterion_check/legacy_deltas.csv   per cell (n_samples x n_events): mean score, paired
        mean difference d to the full-data model and its 95 % bootstrap CI (test samples resampled, 10k), baseline CI.
        Computed by legacy_criterion_check.py from the per-test-sample legacy results (detailed_res, only on the
        cluster); the per-sample scores of the cells are not on disk here.
    legacy/results/gating_performance/<m>/<d>/<trafo>/res_df_sw_avg_f1.csv   per-test-sample full-data macro F1.
    data/np_files/<d>/<trafo>/sample_wise_train/y_sample_<NN>_train.npy   legacy training labels (class counts).

Differences to the step-8 eval tables:
    - one lane (split 0, trial 0): no trial means; the CI is over test samples (legacy), not patients / donors.
    - score = macro F1 for every dataset (also LT1b / LT2b; legacy_deltas.csv has no binary F1).
    - no score CI per cell (per-sample cell scores missing): score_ci_low = score_ci_high = mean_score.
    - cells restricted to the plot grid X_GRID (legacy n_samples; the full legacy n_samples = 'all'), so the minimal
      cell is selected on that grid (step 8 also selects on its grid).
    - class counts after downsampling to n_events: the deterministic part of the legacy stratified downsampling
      (val_utils.get_downsampling_bool: per class min(round(cap * count / N), count)); the legacy random correction
      of the rounding remainder (a few events) is assigned to the largest classes. Minority counts can be off by a
      few events where the remainder hit a small class.

Outputs (scratchpad/results/legacy_training_data_requirements/):
    curves.csv, minimal.csv (training_data_requirements_eval.minimal_table, all deltas / orders / stable),
    sample_scores.csv (full-data cell only: per test sample baseline), train_class_counts/<dataset>.csv,
    points.csv (minority scatter: one row per cell except ('all', 'all'): minority count, d).
"""
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import config  # noqa: E402
from training_data_requirements_eval import minimal_table  # noqa: E402

OVERWRITE = True
LEG = os.path.join(config.ROOT, 'legacy', 'results')
NP_FILES = os.path.join(config.ROOT, 'data', 'np_files')
DELTAS_CSV = os.path.join(HERE, 'results', 'legacy_criterion_check', 'legacy_deltas.csv')
OUT = os.path.join(HERE, 'results', 'legacy_training_data_requirements')

# legacy name -> (new dataset, legacy dir, full n_samples)
DATASETS = OrderedDict([('Imstat', ('imstat', 'imstat', 75)), ('LT1', ('lt1', 'lymphoma_tube1', 58)),
                        ('LT2', ('lt2', 'lymphoma_tube2', 58)), ('LT1b', ('lt1b', 'lymphoma_tube1_binary', 58)),
                        ('LT2b', ('lt2b', 'lymphoma_tube2_binary', 58)), ('Flowcyt', ('flowcyt', 'flowcyt', 18))])
METHODS = OrderedDict([('MLP', ('mlp', 'fcnn')), ('SOM', ('som', 'som'))])
N_EVENTS = [100, 1000, 5000, 10000, 20000, 50000, 'all']  # legacy events grid
X_GRID = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 'all']  # plot grid (legacy n_samples, all datasets but Flowcyt)


def trafo(d):
    return 'log10_cutoff100' if d == 'flowcyt' else 'log10_w_custom_cutoffs'


def grid_value(v):
    return 'all' if str(v) == 'all' else int(v)


def downsampled_counts(counts, cap):
    """Class counts of one sample after the legacy stratified downsampling to cap events (deterministic part)."""
    n = counts.sum()
    if cap == 'all' or cap >= n:
        return counts.copy()
    keep = np.minimum(np.round(cap * counts / float(n)).astype(int), counts)
    order = np.argsort(-counts)  # rounding remainder: largest classes
    k = 0
    while keep.sum() != cap:
        i = order[k % len(order)]
        if keep.sum() > cap and keep[i] > 0:
            keep[i] -= 1
        elif keep.sum() < cap and keep[i] < counts[i]:
            keep[i] += 1
        k += 1
    return keep


def sample_class_counts(legacy_dir):
    """(labels, array n_train_samples x n_labels) of the legacy training samples in file order."""
    p = os.path.join(NP_FILES, legacy_dir, trafo(legacy_dir), 'sample_wise_train')
    ys = []
    while os.path.exists(os.path.join(p, 'y_sample_{:02d}_train.npy'.format(len(ys)))):
        ys.append(np.asarray(np.load(os.path.join(p, 'y_sample_{:02d}_train.npy'.format(len(ys))))))
    labels = np.unique(np.concatenate(ys))
    return labels, np.array([[np.sum(y == l) for l in labels] for y in ys])


def class_count_table(dataset, labels, counts, n_full):
    rows = []
    for ns in X_GRID:
        n = n_full if ns == 'all' else ns
        if n > n_full or (n == n_full and ns != 'all'):  # not on the dataset's legacy grid
            continue
        for ne in N_EVENTS:
            for s in range(n):
                row = OrderedDict([('dataset', dataset), ('split', 0), ('trial', 0), ('n_samples', ns),
                                   ('n_events', ne), ('sample', 'sample_{:02d}_train'.format(s))])
                for l, c in zip(labels, downsampled_counts(counts[s], ne)):
                    row['n_{}'.format(l)] = int(c)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    if os.path.exists(os.path.join(OUT, 'minimal.csv')) and not OVERWRITE:
        return
    os.makedirs(os.path.join(OUT, 'train_class_counts'), exist_ok=True)
    t = pd.read_csv(DELTAS_CSV, dtype={'n_events': str})
    curves, samples, points = [], [], []
    for ds, (dataset, ldir, n_full) in DATASETS.items():
        labels, counts = sample_class_counts(ldir)
        assert len(counts) == n_full, (ds, len(counts))
        cc = class_count_table(dataset, labels, counts, n_full)
        cc.to_csv(os.path.join(OUT, 'train_class_counts', dataset + '.csv'), index=False)
        ccols = [c for c in cc.columns if c.startswith('n_') and c[2:].isdigit()]
        g = cc.groupby(['n_samples', 'n_events'], sort=False)[ccols].sum()
        n_train = dict(zip(g.index, g.to_numpy().sum(axis=1)))
        minority = dict(zip(g.index, g.to_numpy().min(axis=1)))
        for mname, (method, mdir) in METHODS.items():
            base = pd.read_csv(os.path.join(LEG, 'gating_performance', mdir, ldir, trafo(ldir), 'res_df_sw_avg_f1.csv'),
                               index_col=0).drop(index=['mean', 'std'], errors='ignore')['macro'].to_numpy(float)
            samples.append(pd.DataFrame({'method': method, 'dataset': dataset, 'n_samples': 'all', 'n_events': 'all',
                                         'split': 0, 'sample': ['sample_{:02d}_test'.format(i)
                                                                for i in range(len(base))],
                                         'score_t0': base, 'n_trials_pred': 1, 'trial_mean': base,
                                         'baseline': base, 'd': 0.0}))
            sub = t[(t['dataset'] == ds) & (t['method'] == mname)]
            assert np.isclose(sub['baseline'].iloc[0], base.mean()), (ds, mname)
            for _, r in sub.iterrows():
                ns = 'all' if r['n_samples'] == n_full else int(r['n_samples'])
                ne = grid_value(r['n_events'])
                if ns not in X_GRID:
                    continue
                curves.append(OrderedDict([
                    ('method', method), ('dataset', dataset), ('n_samples', ns), ('n_events', ne),
                    ('scope', 'pooled'), ('n_lanes', 1), ('n_alias', 0), ('n_baseline_alias', 0), ('n_ok', 1),
                    ('eligible', True), ('n_train_events_mean', n_train[(ns, ne)]),
                    ('n_train_events_min', n_train[(ns, ne)]), ('n_train_events_max', n_train[(ns, ne)]),
                    ('n_test', r['n_test']), ('n_test_total', r['n_test']), ('mean_score', r['score']),
                    ('mean_baseline', r['baseline']), ('mean_d', r['delta']), ('sd_d', r['delta_sd']),
                    ('ci_low', r['delta_ci_low']), ('ci_high', r['delta_ci_high']),
                    ('base_ci_low', r['base_ci_low']), ('base_ci_high', r['base_ci_high']),
                    ('score_ci_low', r['score']), ('score_ci_high', r['score'])]))
                if (ns, ne) != ('all', 'all'):
                    points.append(OrderedDict([('method', method), ('dataset', dataset), ('split', 0), ('trial', 0),
                                               ('n_samples', ns), ('n_events', ne),
                                               ('minority', minority[(ns, ne)]), ('d', r['delta'])]))
    curves = pd.DataFrame(curves)
    curves.to_csv(os.path.join(OUT, 'curves.csv'), index=False)
    minimal_table(curves).to_csv(os.path.join(OUT, 'minimal.csv'), index=False)
    pd.concat(samples, ignore_index=True).to_csv(os.path.join(OUT, 'sample_scores.csv'), index=False)
    pd.DataFrame(points).to_csv(os.path.join(OUT, 'points.csv'), index=False)
    print(OUT)


if __name__ == '__main__':
    main()
