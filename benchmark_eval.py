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
from validation.core.scores import class_scores
from validation.eval_helpers import cap_hit, failure_type, write_csv

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
# Main
# ---------------------------------------------------------------------------------------------------------------

def main(in_dir, out_dir):
    os.makedirs(os.path.join(out_dir, 'scores'), exist_ok=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in', dest='in_dir', default=IN_DIR, help='benchmark output root (default: %(default)s)')
    parser.add_argument('--out', default=OUT_DIR, help='output directory (default: %(default)s)')
    args = parser.parse_args()
    main(os.path.abspath(args.in_dir), os.path.abspath(args.out))
