"""
Evaluation tables of the main benchmark (PLAN.md step 7, working_notes/benchmark_plan.md): the atoms of benchmark.py
(JSON + counts CSV per method x dataset x outer split x seed) as one fit table and one wide score table per dataset.
How to aggregate (macro / weighted / binary F1, over samples, folds, seeds, datasets) is decided later from these.

Env: fx_plot (pandas).

Usage:
    python benchmark_eval.py [--in DIR] [--out DIR]
        --in: benchmark output root (default results/benchmark; e.g. results/benchmark_pilot for the pilot)
        --out: output directory (default results/benchmark_eval)

Inputs: <in>/{manifests,atoms,counts,logs}, data/splits/<dataset>.csv (test samples), the curated meta_labels.csv
(label order) and sample_<NNN>_y.npy headers (n_cells).

Failure type of an atom (validation.eval_helpers.failure_type), columns failure / failure_phase / error:
    ''              ok
    missing         no JSON (never started)
    python_error    status 'failed' with a traceback (error = its last line), e.g. DGCyTOF's IndexError
    oom             status 'failed' with a MemoryError / CUDA out of memory, or a killed job whose SLURM log reports
                    an OOM kill
    timeout         killed job, SLURM log: '... DUE TO TIME LIMIT'
    node_failure    killed job, SLURM log: node failure
    killed_unknown  killed job (status 'started'), no reason found in its log
    The log of a killed atom is found by the line 'start <atom_id>' that benchmark.py prints; if the atom was started
    in several logs, the latest one (highest job id) counts, from that line on. failure_phase = the JSON phase
    ('fit': killed / failed before the fit was recorded; 'predict': during the prediction of the test samples).

Outputs (<out> = results/benchmark_eval by default):
    atoms.csv   fit table, one row per manifest atom: atom_id, method, dataset, split, seed, config_id, param_source,
                train_downsampling, atom_status, phase, failure, failure_phase, error, log, n_test_done, n_test,
                n_train_samples, n_train_events, fit_s, fit_peak_mb, fit_peak_increase_mb, fit_gpu_peak_mb, host,
                gpu, n_threads, cap_hit, fit_info_<key> (scalar fit_info entries)
                (fit_s = wall time of preprocessing fit + build + fit; *_peak_mb = peak RSS of the process tree,
                *_peak_increase_mb = peak - RSS before the block, both in MB; NaN where not recorded)
    scores/<dataset>.csv   one row per manifest atom x test sample of the dataset (test samples of the outer split):
                method, dataset, split, seed, atom_id, config_id, train_downsampling, sample, atom_status, failure,
                failure_phase, predicted, n_cells, n_abstained, abstention_rate, pred_s, pred_peak_mb,
                pred_peak_increase_mb, pred_gpu_peak_mb, then per label l (order of meta_labels.csv):
                n_<l> (true events of class l), npred_<l> (events predicted as l = tp + fp), prec_<l>, rec_<l>,
                f1_<l>. True / predicted class fractions: n_<l> / n_cells and npred_<l> / n_cells (the predicted
                fractions plus abstention_rate sum to 1).
                predicted = the sample was predicted (also the finished samples of a killed atom; check atom_status);
                otherwise all scores / prediction resources are NaN. n_cells is always filled.
                Class scores (D8: an abstained event is an FN of its true class and an FP of no class):
                prec = tp / (tp + fp), NaN if tp + fp = 0; rec = tp / (tp + fn), NaN if n_true = 0;
                f1 = 2 tp / (2 tp + fp + fn), NaN if the class is neither present nor predicted.
                So the NaN-skipping row mean of f1_* is the macro F1 of validation.core.scores.f1_from_counts (as in
                the tuning); weighted F1 uses n_<l> as weights; binary datasets use f1_1.
    failures.csv   number of atoms per method x dataset x failure x failure_phase x error
    populations/<dataset>.csv   population sizes, one row per predicted atom x test sample x class (all classes,
                also the negative class O of the binary datasets; seeds separate): method, dataset, split, seed, atom_id,
                sample, label, name (class short name, meta_labels.csv), n_cells, true_pct = 100 n_<l> / n_cells,
                pred_pct = 100 npred_<l> / n_cells, d_pp = pred_pct - true_pct (error in percentage points),
                mean_pct = (pred_pct + true_pct) / 2, present = n_<l> + npred_<l> > 0. Abstained events belong to no
                class, so per sample the d_pp of all classes sum to -100 abstention_rate.
    population_agreement.csv   Bland-Altman statistics per method x dataset x class (validation.eval_helpers.ba_stats)
                on the seed means (population_seed_mean: one point per test sample x class, as in plot_benchmark.py),
                only present points (class in the sample or predicted): n, bias_pp, sd_pp, mae_pp, loa_low_pp,
                loa_high_pp (bias -/+ 1.96 SD).
    fold_seed_scores.csv   one row per manifest atom (method x dataset x outer split = fold x seed) x metric: method,
                dataset, split, seed, metric (f1 / prec / rec), average ('macro' for multi-class datasets: NaN-mean of
                the per-class <metric>_<l> of a sample; 'binary' for datasets with 2 labels: <metric>_1, the positive
                class), m (mean of the sample scores over the fold's test samples), n_samples (test samples),
                n_valid (samples with a score). Missingness propagates: m is NaN if any test sample of the fold has
                no score (not predicted: failed / killed / missing atom, or the score is undefined).
    fold_seed_variability.csv   one row per method x dataset x metric: method, dataset, metric, average, mean,
                fold_sd, seed_sd (formulas: fold_seed_stats; seed_sd empty for deterministic methods with 1 seed),
                rerun_range (max - min over the seeds of the seed's mean score over all test samples of the dataset,
                i.e. the n_samples-weighted mean of its fold means: how much the overall score changes when the
                method is only retrained with another seed; empty for 1 seed), n_folds, n_seeds. NaN propagates from
                any NaN m.
    pairwise_deltas.csv   paired differences between all method pairs per dataset x metric (f1 / prec / rec, macro
                or binary as above), at the test-sample level: per sample the score averaged over the seeds that
                predicted it (NaN if none did; as in plot_benchmark.py), delta = score(method_a) - score(method_b)
                over the dataset's test samples (all outer folds) that both methods predicted (incomplete results:
                the available pairs are used). Columns: dataset, metric, average, method_a, method_b, n_samples
                (paired samples), n_total (test samples of the dataset), mean_delta, sd_delta (SD of the per-sample
                deltas, ddof 1), median_delta, q1_delta / q3_delta (quartiles of the per-sample deltas: IQR =
                q3 - q1), ci_low / ci_high (95 % percentile bootstrap CI of the mean delta: N_BOOT resamples of
                the paired samples with replacement, seed derived from dataset, metric and the pair). Both orders of
                each pair are listed (a - b and b - a). All NaN if fewer than 2 paired samples.
                Tests from the 95 % CI (each one-sided at alpha = 0.025), margin = config.DELTA (as in step 8):
                superior (a better than b: ci_low > 0), noninferior (a not worse than b by delta or more:
                ci_low > -delta; non-superiority of a over b = noninferior of the reversed row), equivalent (TOST:
                -delta < ci_low and ci_high < delta; symmetric in a, b). Empty if the CI is NaN.
                Each decision is followed by its one-sided bootstrap p-value (boot_p, from the same N_BOOT bootstrap
                means; no multiplicity correction): p_superior (H0: mean delta <= 0), p_noninferior (H0: mean
                delta <= -delta), p_equivalent (TOST: the larger of the p-values of H0: <= -delta and H0: >= delta).
                The decisions correspond to p < 0.025 (up to the percentile interpolation); the two-sided p-value of
                'a differs from b' is 2 min(p_superior of a - b, p_superior of b - a).
Each part is skipped if its outputs exist (OVERWRITE = False); the parts after it then read them.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

import config
from benchmark import atom_paths
from validation.core.runs import read_manifest
from validation.core.seeding import derive_seed
from validation.core.scores import class_scores
from validation.eval_helpers import ba_stats, cap_hit, failure_type, population_seed_mean, write_csv

IN_DIR = os.path.join(config.RESULTS, 'benchmark')
OUT_DIR = os.path.join(config.RESULTS, 'benchmark_eval')
OVERWRITE = False

ID_COLS = ['method', 'dataset', 'split', 'seed', 'atom_id', 'config_id', 'train_downsampling']


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def done(*paths):
    """True if all paths exist and OVERWRITE is False (the part is skipped and its outputs are read back)."""
    return not OVERWRITE and all(os.path.exists(p) for p in paths)


def load_manifest(in_dir):
    """All manifest rows of <in_dir>/manifests/*.csv (dicts of str), in array order."""
    rows = []
    for path in sorted(glob.glob(os.path.join(in_dir, 'manifests', '*.csv'))):
        rows += read_manifest(path)
    if not rows:
        raise SystemExit('no manifests in {}'.format(os.path.join(in_dir, 'manifests')))
    return rows


def index_logs(in_dir):
    """
    {atom_id: (log path, log text from the line 'start <atom_id>' on)} from <in_dir>/logs/*/*.out; if an atom was
    started in several logs, the latest one (highest job id, then task id) counts.
    """
    found = {}
    for path in glob.glob(os.path.join(in_dir, 'logs', '*', '*.out')):
        m = re.match(r'(\d+)_(\d+)\.out$', os.path.basename(path))
        key = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        with open(path, errors='replace') as f:
            text = f.read()
        for m_start in re.finditer(r' start (\S+)', text):
            aid = m_start.group(1)
            if aid not in found or key > found[aid][0]:
                found[aid] = (key, path, text[m_start.start():])
    return dict((aid, (p, t)) for aid, (_, p, t) in found.items())


def test_samples(dataset, split):
    sp = pd.read_csv(os.path.join(config.DATA_SPLITS, dataset + '.csv'))
    return sp.loc[sp['split_{}'.format(split)] == 'test', 'sample'].tolist()


_N_CELLS = {}


def n_cells(dataset, sample):
    """Number of events of a curated sample (from the .npy header, no data loaded)."""
    key = (dataset, sample)
    if key not in _N_CELLS:
        _N_CELLS[key] = int(np.load(os.path.join(config.DATA_CURATED, dataset, sample + '_y.npy'),
                                    mmap_mode='r').shape[0])
    return _N_CELLS[key]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------
# Part 1: fit table
# ---------------------------------------------------------------------------------------------------------------

def atom_table(manifest, in_dir):
    """
    One row per manifest atom: identity, status, failure type, fit resources, run info (columns: module docstring).
    """
    logs = index_logs(in_dir)
    rows = []
    for r in manifest:
        d = load_json(atom_paths(in_dir, r)[0])
        status = 'missing' if d is None else d['status']
        log_path, log_text = logs.get(r['atom_id'], (None, None))
        failure, error = failure_type(status, None if d is None else d.get('error'), log_text)
        row = dict((c, r[c]) for c in ID_COLS + ['param_source'])
        row.update({'split': int(r['split']), 'seed': int(r['seed']), 'atom_status': status,
                    'phase': None if d is None else d.get('phase'), 'failure': failure,
                    'failure_phase': '' if status in ('ok', 'missing') else (d.get('phase') or ''),
                    'error': error, 'log': os.path.relpath(log_path, in_dir) if log_path else '',
                    'n_test_done': None if d is None else d.get('n_test_done'),
                    'n_test': len(test_samples(r['dataset'], r['split']))})
        if d is not None:
            res = d.get('resources') or {}
            fit = res.get('fit') or {}
            meta = d.get('metadata') or {}
            row.update({
                'n_train_samples': len(d['train_samples']) if d.get('train_samples') else None,
                'n_train_events': d.get('n_train_events'),
                'fit_s': fit.get('wall_s'),
                'fit_peak_mb': fit.get('cpu_rss_peak_mb'),
                'fit_peak_increase_mb': fit.get('cpu_rss_peak_increase_mb'),
                'fit_gpu_peak_mb': fit.get('gpu_peak_mb'),
                'host': meta.get('host'),
                'gpu': ';'.join(meta.get('gpus') or []),
                'n_threads': d.get('n_threads'),
                'cap_hit': cap_hit(r['method'], d['params'], d.get('fit_info')),
            })
            for k, v in (d.get('fit_info') or {}).items():  # Scalars only
                if not isinstance(v, (list, dict)):
                    row['fit_info_' + k] = v
        rows.append(row)
    cols = (ID_COLS + ['param_source', 'atom_status', 'phase', 'failure', 'failure_phase', 'error', 'log',
                       'n_test_done', 'n_test', 'n_train_samples', 'n_train_events', 'fit_s', 'fit_peak_mb',
                       'fit_peak_increase_mb', 'fit_gpu_peak_mb', 'host', 'gpu', 'n_threads', 'cap_hit'])
    t = pd.DataFrame(rows)
    for c in cols:
        if c not in t:
            t[c] = None
    return t[cols + sorted(c for c in t.columns if c.startswith('fit_info_'))]


# ---------------------------------------------------------------------------------------------------------------
# Part 2: wide score tables (one per dataset)
# ---------------------------------------------------------------------------------------------------------------

def score_table(dataset, manifest, atoms, in_dir):
    """
    One row per manifest atom x test sample of the dataset (columns: module docstring).
    """
    labels = pd.read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_labels.csv'))['label'].tolist()
    k = len(labels)
    status = atoms.set_index('atom_id')
    rows = []
    for r in manifest:
        if r['dataset'] != dataset:
            continue
        json_path, counts_path, _ = atom_paths(in_dir, r)
        d = load_json(json_path)
        pred_res = ((d or {}).get('resources') or {}).get('predict') or {}
        counts = {}
        if pred_res and os.path.exists(counts_path):
            c = pd.read_csv(counts_path).sort_values(['sample', 'label'])
            for s, g in c.groupby('sample', sort=False):
                if g['label'].tolist() != labels:
                    raise ValueError('unexpected labels for {} in {}'.format(s, counts_path))
                counts[s] = g
        a = status.loc[r['atom_id']]
        for s in test_samples(dataset, r['split']):
            row = dict((col, r[col]) for col in ID_COLS)
            row.update({'split': int(r['split']), 'seed': int(r['seed']), 'sample': s,
                        'atom_status': a['atom_status'], 'failure': a['failure'],
                        'failure_phase': a['failure_phase'], 'n_cells': n_cells(dataset, s)})
            predicted = s in pred_res and s in counts
            row['predicted'] = predicted
            if predicted:
                g = counts[s]
                p = pred_res[s]
                n_true = g['n_true'].to_numpy()
                n_abst = int(g['n_abstained'].sum())
                # n_cells = curated sample size; the counts cover n_events_used events (= n_cells, except --smoke)
                n_pred = g['n_pred'].to_numpy()
                if (int(g['n_events_sample'].iloc[0]) != row['n_cells']
                        or int(n_true.sum()) != int(g['n_events_used'].iloc[0])
                        or int(n_pred.sum()) + n_abst != int(n_true.sum())
                        or not (n_pred == g['tp'].to_numpy() + g['fp'].to_numpy()).all()):
                    raise ValueError('event count mismatch for {} in {}'.format(s, counts_path))
                prec, rec, f1 = class_scores(g['tp'], g['fp'], g['fn'])
                row.update({'n_abstained': n_abst, 'abstention_rate': n_abst / float(n_true.sum()),
                            'pred_s': p.get('wall_s'), 'pred_peak_mb': p.get('cpu_rss_peak_mb'),
                            'pred_peak_increase_mb': p.get('cpu_rss_peak_increase_mb'),
                            'pred_gpu_peak_mb': p.get('gpu_peak_mb')})
                for i, l in enumerate(labels):
                    row.update({'n_{}'.format(l): int(n_true[i]), 'npred_{}'.format(l): int(n_pred[i]),
                                'prec_{}'.format(l): prec[i],
                                'rec_{}'.format(l): rec[i], 'f1_{}'.format(l): f1[i]})
            rows.append(row)
    cols = (['method', 'dataset', 'split', 'seed', 'atom_id', 'config_id', 'train_downsampling', 'sample',
             'atom_status', 'failure', 'failure_phase', 'predicted', 'n_cells', 'n_abstained', 'abstention_rate',
             'pred_s', 'pred_peak_mb', 'pred_peak_increase_mb', 'pred_gpu_peak_mb']
            + ['{}_{}'.format(m, l) for l in labels for m in ('n', 'npred', 'prec', 'rec', 'f1')])
    t = pd.DataFrame(rows)
    for c in cols:
        if c not in t:
            t[c] = np.nan
    t = t[cols]
    for c in ['n_abstained'] + ['{}_{}'.format(m, l) for l in labels for m in ('n', 'npred')]:
        t[c] = t[c].astype('Int64')  # Integer counts, NA where not predicted
    return t


# ---------------------------------------------------------------------------------------------------------------
# Part 3: failure counts
# ---------------------------------------------------------------------------------------------------------------

def failure_table(atoms):
    t = atoms.fillna({'failure': '', 'failure_phase': '', 'error': ''})
    return (t.groupby(['method', 'dataset', 'failure', 'failure_phase', 'error'], sort=False).size()
            .rename('n_atoms').reset_index())


# ---------------------------------------------------------------------------------------------------------------
# Part 4: population sizes (long) and Part 5: their agreement (Bland-Altman statistics)
# ---------------------------------------------------------------------------------------------------------------

def population_table(dataset, scores):
    """
    One row per predicted atom x test sample x class (columns: module docstring).

    Args:
        dataset (str): Dataset name.
        scores (pd.DataFrame): Score table of the dataset (Part 2).
    """
    t = scores[scores['predicted'].astype(bool)]
    names = pd.read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_labels.csv'))
    parts = []
    for l, name in zip(names['label'], names['name']):
        n_true = t['n_{}'.format(l)].astype(float)
        n_pred = t['npred_{}'.format(l)].astype(float)
        parts.append(pd.DataFrame({
            'method': t['method'], 'dataset': dataset, 'split': t['split'], 'seed': t['seed'],
            'atom_id': t['atom_id'], 'sample': t['sample'], 'label': l, 'name': name, 'n_cells': t['n_cells'],
            'true_pct': 100.0 * n_true / t['n_cells'], 'pred_pct': 100.0 * n_pred / t['n_cells'],
            'present': (n_true + n_pred) > 0,
        }))
    cols = ['method', 'dataset', 'split', 'seed', 'atom_id', 'sample', 'label', 'name', 'n_cells', 'true_pct',
            'pred_pct', 'd_pp', 'mean_pct', 'present']
    if not parts or t.empty:
        return pd.DataFrame(columns=cols)
    pop = pd.concat(parts).sort_index(kind='stable').reset_index(drop=True)  # Atom x sample, then classes
    pop['d_pp'] = pop['pred_pct'] - pop['true_pct']
    pop['mean_pct'] = (pop['pred_pct'] + pop['true_pct']) / 2.0
    return pop[cols]


def agreement_table(pops):
    """
    Bland-Altman statistics per method x dataset x class on the seed means, present points only.

    Args:
        pops (list of pd.DataFrame): Population tables (Part 4) of the datasets.
    """
    rows = []
    for pop in pops:
        if pop.empty:
            continue
        m = population_seed_mean(pop)
        m = m[m['present']]
        for (method, dataset, label, name), g in m.groupby(['method', 'dataset', 'label', 'name'], sort=False):
            row = {'method': method, 'dataset': dataset, 'label': label, 'name': name}
            row.update(ba_stats(g['d_pp']))
            rows.append(row)
    return pd.DataFrame(rows, columns=['method', 'dataset', 'label', 'name', 'n', 'bias_pp', 'sd_pp', 'mae_pp',
                                       'loa_low_pp', 'loa_high_pp'])


# ---------------------------------------------------------------------------------------------------------------
# Part 6: fold x seed means and the fold / seed variability
# ---------------------------------------------------------------------------------------------------------------

METRICS = ['f1', 'prec', 'rec']


def sample_metric(scores, metric):
    """
    Sample score of every row of a score table: 'macro' (NaN-mean of <metric>_<l>) or 'binary' (<metric>_1, datasets
    with 2 labels); NaN for unpredicted rows.

    Returns:
        (pd.Series, str): Scores (index of the table), 'macro' or 'binary'.
    """
    labels = [c[3:] for c in scores.columns if c.startswith('f1_')]
    average = 'binary' if len(labels) == 2 else 'macro'
    if average == 'binary':
        sample = scores['{}_1'.format(metric)]
    else:
        sample = scores[['{}_{}'.format(metric, l) for l in labels]].mean(axis=1, skipna=True)
    return sample.where(scores['predicted'].astype(bool)), average


def fold_seed_table(dataset, scores):
    """
    Mean score per fold x seed (atom) and metric (columns: module docstring).

    Args:
        dataset (str): Dataset name.
        scores (pd.DataFrame): Score table of the dataset (Part 2): one row per manifest atom x test sample.
    """
    rows = []
    for metric in METRICS:
        sample, average = sample_metric(scores, metric)
        g = sample.groupby([scores['method'], scores['split'], scores['seed']], sort=False)
        t = pd.DataFrame({'m': g.mean(), 'n_samples': g.size(), 'n_valid': g.count()}).reset_index()
        t.loc[t['n_valid'] < t['n_samples'], 'm'] = np.nan  # Missingness propagates
        t.insert(1, 'dataset', dataset)
        t.insert(4, 'metric', metric)
        t.insert(5, 'average', average)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def fold_seed_stats(m):
    """
    Overall mean and the fold / seed standard deviations of a fold x seed table of mean scores.

    With m_{f,s} the mean score over the test samples of fold f (f = 1..F) and seed s (s = 1..S):

        M        = 1/(F S) sum_f sum_s m_{f,s}                                  overall mean
        m_f      = 1/S sum_s m_{f,s}                                            fold mean (over the seeds)
        sd_fold  = sqrt( 1/(F-1) sum_f (m_f - M)^2 )                            fold SD
        sd_seed  = sqrt( 1/(F (S-1)) sum_f sum_s (m_{f,s} - m_f)^2 )            seed SD (pooled within-fold variance)

    sd_seed is NaN for S = 1 (deterministic methods). Any NaN m_{f,s} makes all three NaN.

    Args:
        m (np.ndarray): Shape (F, S).

    Returns:
        (float, float, float): M, sd_fold, sd_seed.
    """
    m = np.asarray(m, dtype=np.float64)
    f, s = m.shape
    if np.isnan(m).any():
        return np.nan, np.nan, np.nan
    mean = m.mean()
    m_f = m.mean(axis=1)
    sd_fold = np.sqrt(np.sum((m_f - mean) ** 2) / (f - 1)) if f > 1 else np.nan
    sd_seed = np.sqrt(np.sum((m - m_f[:, None]) ** 2) / (f * (s - 1))) if s > 1 else np.nan
    return float(mean), float(sd_fold), float(sd_seed)


def variability_table(fold_seed):
    """One row per method x dataset x metric with mean, fold_sd, seed_sd (fold_seed_stats)."""
    rows = []
    for (method, dataset, metric, average), g in fold_seed.groupby(['method', 'dataset', 'metric', 'average'],
                                                                     sort=False):
        mat = g.pivot(index='split', columns='seed', values='m').sort_index().sort_index(axis=1)
        mean, sd_fold, sd_seed = fold_seed_stats(mat.to_numpy())
        # Re-run range: overall mean per seed (all test samples = fold means weighted by their sample numbers)
        n = g.pivot(index='split', columns='seed', values='n_samples').loc[mat.index, mat.columns]
        per_seed = (mat * n).sum() / n.sum()
        if mat.isna().to_numpy().any() or mat.shape[1] < 2:
            rerun = np.nan
        else:
            rerun = float(per_seed.max() - per_seed.min())
        rows.append({'method': method, 'dataset': dataset, 'metric': metric, 'average': average, 'mean': mean,
                     'fold_sd': sd_fold, 'seed_sd': sd_seed, 'rerun_range': rerun, 'n_folds': mat.shape[0],
                     'n_seeds': mat.shape[1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Part 7: paired differences between methods
# ---------------------------------------------------------------------------------------------------------------

N_BOOT = 10000  # Bootstrap resamples for the CI of the mean paired delta


def boot_p(boot, c, side):
    """
    One-sided bootstrap p-value of H0: mean <= c (side 'le') or mean >= c (side 'ge') from the bootstrap means,
    (1 + #{boot on the H0 side of c}) / (N_BOOT + 1) (the usual +1 correction: p >= 1 / (N_BOOT + 1), never 0);
    consistent with the percentile CI: p < 0.025 if and only if c lies outside it on that side (up to the percentile
    interpolation).
    """
    k = np.sum(boot <= c) if side == 'le' else np.sum(boot >= c)
    return float((1 + k) / (len(boot) + 1.0))


def pairwise_table(dataset, scores):
    """
    Paired per-sample differences of all method pairs (columns: module docstring).

    Args:
        dataset (str): Dataset name.
        scores (pd.DataFrame): Score table of the dataset (Part 2).
    """
    rows = []
    for metric in METRICS:
        sample, average = sample_metric(scores, metric)
        g = sample.groupby([scores['method'], scores['split'], scores['sample']])
        wide = g.mean().unstack('method')  # Seed mean over the seeds that predicted the sample; rows: split x sample
        methods = [m for m in config.PLOT_METHODS if m in wide.columns] + sorted(set(wide.columns) -
                                                                                  set(config.PLOT_METHODS))
        n_total = len(wide)
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                d = (wide[a] - wide[b]).to_numpy()
                d = d[~np.isnan(d)]  # Samples predicted by both methods
                n = len(d)
                if n < 2:
                    st = dict(mean_delta=np.nan, sd_delta=np.nan, median_delta=np.nan, q1_delta=np.nan,
                              q3_delta=np.nan, ci_low=np.nan, ci_high=np.nan)
                    p_ab = p_ba = dict(p_superior=np.nan, p_noninferior=np.nan, p_equivalent=np.nan)
                else:
                    rng = np.random.default_rng(derive_seed('benchmark_eval', 'pairwise_deltas', dataset, metric,
                                                            a, b))
                    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    q1, med, q3 = np.percentile(d, [25, 50, 75])
                    st = dict(mean_delta=float(d.mean()), sd_delta=float(d.std(ddof=1)), median_delta=float(med),
                              q1_delta=float(q1), q3_delta=float(q3), ci_low=float(lo), ci_high=float(hi))
                    # One-sided p-values for a - b; the reversed row b - a has the bootstrap means -boot
                    p_eq = max(boot_p(boot, -config.DELTA, 'le'), boot_p(boot, config.DELTA, 'ge'))  # TOST
                    p_ab = dict(p_superior=boot_p(boot, 0, 'le'), p_noninferior=boot_p(boot, -config.DELTA, 'le'),
                                p_equivalent=p_eq)
                    p_ba = dict(p_superior=boot_p(boot, 0, 'ge'), p_noninferior=boot_p(boot, config.DELTA, 'ge'),
                                p_equivalent=p_eq)
                base = {'dataset': dataset, 'metric': metric, 'average': average, 'n_samples': n, 'n_total': n_total}
                rows.append(dict(base, method_a=a, method_b=b, **dict(st, **p_ab)))
                rows.append(dict(base, method_a=b, method_b=a, mean_delta=-st['mean_delta'], sd_delta=st['sd_delta'],
                                 median_delta=-st['median_delta'], q1_delta=-st['q3_delta'],
                                 q3_delta=-st['q1_delta'], ci_low=-st['ci_high'], ci_high=-st['ci_low'], **p_ba))
    t = pd.DataFrame(rows, columns=['dataset', 'metric', 'average', 'method_a', 'method_b', 'n_samples', 'n_total',
                                    'mean_delta', 'sd_delta', 'median_delta', 'q1_delta', 'q3_delta', 'ci_low',
                                    'ci_high', 'p_superior', 'p_noninferior', 'p_equivalent'])
    # Decisions from the 95 % CI (each one-sided at alpha = 0.025); empty where the CI is NaN
    valid = t['ci_low'].notna()
    t['margin'] = config.DELTA
    t['superior'] = (t['ci_low'] > 0).where(valid)
    t['noninferior'] = (t['ci_low'] > -config.DELTA).where(valid)
    t['equivalent'] = ((t['ci_low'] > -config.DELTA) & (t['ci_high'] < config.DELTA)).where(valid)
    cols = [c for c in t.columns if not c.startswith('p_')]  # Each p-value next to its decision
    for test in ['superior', 'noninferior', 'equivalent']:
        cols.insert(cols.index(test) + 1, 'p_' + test)
    return t[cols]


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------

def main(in_dir, out_dir):
    os.makedirs(os.path.join(out_dir, 'scores'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'populations'), exist_ok=True)
    manifest = load_manifest(in_dir)

    # --- Part 1: fit table
    p_atoms = os.path.join(out_dir, 'atoms.csv')
    if done(p_atoms):
        atoms = pd.read_csv(p_atoms, keep_default_na=False, na_values=[''])
        atoms[['failure', 'failure_phase', 'error']] = atoms[['failure', 'failure_phase', 'error']].fillna('')
    else:
        atoms = atom_table(manifest, in_dir)
        write_csv(atoms, p_atoms)
    print('atoms:', atoms['atom_status'].value_counts().to_dict())

    # --- Part 2: score tables
    datasets = [d for d in config.DATASETS if any(r['dataset'] == d for r in manifest)]
    for dataset in datasets:
        path = os.path.join(out_dir, 'scores', dataset + '.csv')
        if done(path):
            continue
        t = score_table(dataset, manifest, atoms, in_dir)
        write_csv(t, path)
        print('scores {}: {} rows, {} predicted'.format(dataset, len(t), int(t['predicted'].sum())))

    # --- Part 3: failure counts
    p_fail = os.path.join(out_dir, 'failures.csv')
    if not done(p_fail):
        write_csv(failure_table(atoms), p_fail)
    print('failures:', p_fail)

    # --- Part 4: population sizes
    pops = []
    for dataset in datasets:
        path = os.path.join(out_dir, 'populations', dataset + '.csv')
        if done(path):
            pops.append(pd.read_csv(path))
            continue
        pop = population_table(dataset, pd.read_csv(os.path.join(out_dir, 'scores', dataset + '.csv')))
        write_csv(pop, path)
        pops.append(pop)
        print('populations {}: {} rows'.format(dataset, len(pop)))

    # --- Part 5: agreement of the population sizes
    p_agree = os.path.join(out_dir, 'population_agreement.csv')
    if not done(p_agree):
        write_csv(agreement_table(pops), p_agree)
    print('population agreement:', p_agree)

    # --- Part 6: fold x seed means and their variability
    p_fs = os.path.join(out_dir, 'fold_seed_scores.csv')
    if done(p_fs):
        fold_seed = pd.read_csv(p_fs)
    else:
        fold_seed = pd.concat([fold_seed_table(d, pd.read_csv(os.path.join(out_dir, 'scores', d + '.csv')))
                               for d in datasets], ignore_index=True)
        write_csv(fold_seed, p_fs)
    p_var = os.path.join(out_dir, 'fold_seed_variability.csv')
    if not done(p_var):
        write_csv(variability_table(fold_seed), p_var)
    print('fold / seed variability:', p_var)

    # --- Part 7: paired differences between methods
    p_pair = os.path.join(out_dir, 'pairwise_deltas.csv')
    if not done(p_pair):
        write_csv(pd.concat([pairwise_table(d, pd.read_csv(os.path.join(out_dir, 'scores', d + '.csv')))
                             for d in datasets], ignore_index=True), p_pair)
    print('pairwise deltas:', p_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in', dest='in_dir', default=IN_DIR, help='benchmark output root (default: %(default)s)')
    parser.add_argument('--out', default=OUT_DIR, help='output directory (default: %(default)s)')
    args = parser.parse_args()
    main(os.path.abspath(args.in_dir), os.path.abspath(args.out))
