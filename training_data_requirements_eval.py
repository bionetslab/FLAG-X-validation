"""
Evaluation of the training data requirements (PLAN.md step 8, working_notes/training_data_requirements_eval_plan.md):
score curves over n_samples x n_events, the minimal training configuration per method x dataset (paired
non-inferiority to the full-data baseline of step 7), its variability over trials / folds, the sensitivity to the
criterion (delta sweep, legacy criterion), and a cross-fitted selection-bias check.

Env: fx_plot (pandas).

Usage:
    python training_data_requirements_eval.py [--in DIR] [--baseline DIR] [--out DIR]
        --in: step-8 output root (default results/training_data_requirements)
        --baseline: step-7 eval output (default results/benchmark_eval; reads scores/<dataset>.csv)
        --out: output directory (default results/training_data_requirements_eval)

Inputs: <in>/{manifests,atoms,counts}, <baseline>/scores/<dataset>.csv, data/splits/<dataset>.csv (test samples,
patient_id = bootstrap unit), data/curated/<dataset>/meta_labels.csv and sample_<NNN>_y.npy headers (training size of
the step-7 alias).

Compute budget (applied first): per method x dataset the cutoff x = smallest manifest n_train_events of any timeout /
oom atom (any fold / trial); every atom with n_train_events >= x is out of budget, whatever its status. Atoms without a
final JSON that the cutoff covers count as 'skipped'.

Cells across lanes: a cell is a grid point (n_samples, n_events) of the method x dataset (union over its lanes). The
manifest drops duplicates per lane, so per lane a cell is resolved to the atom with the same training data:
n_samples >= the split's training samples -> 'all'; a missing (n_samples, n_events) -> (n_samples, 'all');
('all', 'all') -> the step-7 model (score = baseline, d = 0; training size = all events of the split's training
samples). Lane = (split, trial).

Scores: per test sample macro F1 (NaN-mean of the per-class F1) or, for datasets with 2 labels, the F1 of label 1
(benchmark_eval.sample_metric). Per cell x test sample: trial mean over the trials with a defined score (NaN unless
every trial predicted the sample); baseline = step-7 seed mean over the seeds that predicted it; d = trial mean -
baseline. Samples with an undefined d are left out of the means (n_test vs n_test_total).

Scopes (the part of the data a statistic is computed on; a 'scope' column in curves.csv / minimal.csv, one table):
    pooled      all folds, trial means          main result (curves, minimal cell; step 9)
    trial_<t>   all folds, trial t only         minimal cell per trial (variability over sample orders, R4.5)
    fold_<k>    fold k only, trial means        minimal cell per fold (spread); evaluation of the cross-fitted cell
    excl_<k>    all folds but k, trial means    selection of the cross-fitted cell for fold k
A cell is eligible in a scope if all its lanes there are 'ok' and in budget (the step-7 alias: in budget if the
split's total training events are below the cutoff).

Bootstrap: per dataset and sample subset (all / fold k / all but k) one fixed matrix of N_BOOT resamples of the
patients (patient_id: donors for Imstat, samples elsewhere), shared by all methods and cells. Bootstrap mean = sum of
d over the drawn patients' samples / their number of samples with a defined d. Percentile CI (CI_LEVEL) of mean d;
the same resamples give the baseline CI (legacy criterion).

Criteria: 'noninf' (ci_low of mean d > -delta) for every delta in DELTAS; 'legacy_ci' (mean score >= base_ci_low).
Orders: 'greedy' (smallest (n_samples, n_events), 'all' = largest) and 'joint' (smallest mean n_train_events over the
scope's lanes, ties greedy). stable = True additionally requires every eligible cell with larger n_samples and the same
n_events to pass. Minimal cell = first eligible passing cell in that order.

Outputs (<out>):
    atoms.csv   one row per manifest atom: atom_id, method, dataset, split, trial, n_samples, n_events,
                n_train_events (manifest), status (ok / failed / timeout / oom / skipped / started / missing), phase,
                error (last traceback line), skipped_by, n_test_done, n_test, fit_s, fit_peak_mb, fit_gpu_peak_mb,
                pred_s (sum over the predicted test samples), worker_wall_s, host, cpu_model, gpu, in_budget,
                excluded_by (the cutoff atom: smallest timeout / oom atom of the method x dataset; '' if in budget).
    train_class_counts/<dataset>.csv   class counts of the training sets (for the minority count analysis), one row
                per dataset x split x trial x cell x training sample, for the cells that at least one method completed
                (status ok): dataset, split, trial, n_samples, n_events, sample, n_<l> per label (events of class l
                from this sample in the training set, after downsampling; atom JSON n_train_events_sample_per_class).
                No method column: the training data depend only on dataset, split, trial and cell (the data seeds do
                not contain the method); checked: all methods that completed a cell report the same counts (else
                ValueError). Per method: join with its ok atoms in atoms.csv on dataset, split, trial, n_samples,
                n_events.
    scores/<dataset>.csv   one row per atom x predicted test sample: method, dataset, split, trial, n_samples,
                n_events, atom_id, atom_status, sample, predicted (True), n_cells, n_abstained, abstention_rate, pred_s,
                pred_peak_mb, then per label l: n_<l>, npred_<l>, prec_<l>, rec_<l>, f1_<l> (as benchmark_eval.py).
    sample_scores.csv   one row per method x dataset x cell x test sample: method, dataset, n_samples, n_events,
                split, sample, patient_id, score_t<t> (per trial; NaN if not predicted or undefined), n_trials_pred,
                trial_mean, baseline, d.
    curves.csv  one row per method x dataset x cell x scope: method, dataset, n_samples, n_events, scope, n_lanes,
                n_alias (lanes resolved to another step-8 atom), n_baseline_alias (lanes resolved to step 7), n_ok
                (lanes ok and in budget), eligible, n_train_events_mean / _min / _max (over the lanes), n_test (samples
                with a defined d), n_test_total, mean_score, mean_baseline, mean_d, sd_d, ci_low, ci_high (CI of mean
                d), base_ci_low, base_ci_high (CI of the mean baseline over the same samples), score_ci_low,
                score_ci_high (CI of the mean score; same resamples).
    minimal.csv one row per method x dataset x criterion x delta x order x stable x scope: found, n_samples,
                n_events, n_train_events_mean, mean_d, ci_low, ci_high, mean_score, mean_baseline, n_test.
    crossfit.csv one row per method x dataset x criterion x delta x order x stable x fold (0..K-1, then 'all'):
                cf_n_samples / cf_n_events (selected in excl_<k>), cf_mean_d / cf_mean_score / cf_n_test (evaluated
                in fold_<k>); pooled_n_samples / pooled_n_events (selected in 'pooled'), in_mean_d / in_mean_score /
                in_n_test (the pooled cell evaluated in fold_<k>); same_cell. Row 'all': the n-weighted means over the
                folds (NaN if a fold is NaN); in_* there is the pooled cell's pooled mean.
    minimal_configs.json   per method x dataset: the primary criterion (noninf, PRIMARY_DELTA, greedy), with and
                without the stability rule: the pooled cell and the cross-fitted cell per fold. Read by step 9.
Each part is skipped if its outputs exist (OVERWRITE = False); the parts after it then read them.
"""

import argparse
import glob
import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import config
from benchmark_eval import sample_metric
from training_data_requirements import FINAL, N_TRIALS, atom_paths, sample_sizes, size_key
from validation.core.data import outer_splits
from validation.core.runs import read_manifest
from validation.core.scores import class_scores
from validation.core.seeding import derive_seed
from validation.eval_helpers import last_line, write_csv

IN_DIR = os.path.join(config.RESULTS, 'training_data_requirements')
BASELINE_DIR = os.path.join(config.RESULTS, 'benchmark_eval')
OUT_DIR = os.path.join(config.RESULTS, 'training_data_requirements_eval')
OVERWRITE = False

PRIMARY_DELTA = config.DELTA  # Non-inferiority margin (macro F1; F1 of label 1 for binary datasets)
# Delta sweep: R10 preferred numbers (log-spaced, 10 per decade), 0.001-0.1; contains 0.01, 0.02, 0.05
DELTAS = [0.001, 0.00125, 0.0016, 0.002, 0.0025, 0.00315, 0.004, 0.005, 0.0063, 0.008,
          0.01, 0.0125, 0.016, 0.02, 0.025, 0.0315, 0.04, 0.05, 0.063, 0.08,
          0.1]
ORDERS = ['greedy', 'joint']  # greedy: fewest samples, then fewest events; joint: fewest total training events
PRIMARY_ORDER = 'greedy'
N_BOOT = 10000
CI_LEVEL = 0.95

CELL = ['n_samples', 'n_events']
CUTOFF = ('timeout', 'oom')


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def done(*paths):
    """True if all paths exist and OVERWRITE is False (the part is skipped and its outputs are read back)."""
    return not OVERWRITE and all(os.path.exists(p) for p in paths)


def load_manifest(in_dir):
    """All manifest rows of <in_dir>/manifests/*.csv (dicts of str)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(in_dir, 'manifests', '*.csv'))):
        rows += read_manifest(path)
    if not rows:
        raise SystemExit('no manifests in {}'.format(os.path.join(in_dir, 'manifests')))
    return rows


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def grid_value(v):
    """Grid value as stored in the tables: 'all' or int (str from the manifest / CSV, int, or float from pandas)."""
    return 'all' if str(v) == 'all' else int(float(v))


def cell_key(ns, ne):
    """Sort key of a cell: (n_samples, n_events), 'all' = largest."""
    return (size_key(ns), size_key(ne))


def split_table(dataset):
    """Split CSV of the dataset: sample, patient_id, split_<i> columns."""
    return pd.read_csv(os.path.join(config.DATA_SPLITS, dataset + '.csv'), dtype={'patient_id': str})


def n_labels(dataset):
    return len(pd.read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_labels.csv')))


# ---------------------------------------------------------------------------------------------------------------
# Part 1: atom table (status, resources, compute budget)
# ---------------------------------------------------------------------------------------------------------------

def atom_table(manifest, in_dir):
    """One row per manifest atom (columns: module docstring), incl. the post-hoc budget (in_budget, excluded_by)."""
    rows = []
    for r in manifest:
        d = load_json(atom_paths(in_dir, r)[0])
        status = 'missing' if d is None else d.get('status')
        if status not in FINAL + ('missing',):
            status = 'started'
        row = OrderedDict((c, r[c]) for c in ['atom_id', 'method', 'dataset'])
        row.update([('split', int(r['split'])), ('trial', int(r['trial'])), ('n_samples', r['n_samples']),
                    ('n_events', r['n_events']), ('n_train_events', int(r['n_train_events'])), ('status', status)])
        d = d or {}
        res = d.get('resources') or {}
        fit = res.get('fit') or {}
        pred = res.get('predict') or {}
        meta = d.get('metadata') or {}
        row.update([
            ('phase', d.get('phase')), ('error', last_line(d.get('error'))), ('skipped_by', d.get('skipped_by') or ''),
            ('n_test_done', d.get('n_test_done')),
            ('n_test', len(d['test_samples']) if d.get('test_samples') is not None else None),
            ('fit_s', fit.get('wall_s')), ('fit_peak_mb', fit.get('cpu_rss_peak_mb')),
            ('fit_gpu_peak_mb', fit.get('gpu_peak_mb')),
            ('pred_s', sum(p.get('wall_s') or 0 for p in pred.values()) if pred else None),
            ('worker_wall_s', (d.get('worker') or {}).get('wall_s')), ('host', meta.get('host')),
            ('cpu_model', meta.get('cpu_model')), ('gpu', ';'.join(meta.get('gpus') or [])),
        ])
        rows.append(row)
    t = pd.DataFrame(rows)
    # Post-hoc budget: cutoff = smallest n_train_events of a timeout / oom atom per method x dataset
    t['in_budget'] = True
    t['excluded_by'] = ''
    for (method, dataset), g in t.groupby(['method', 'dataset'], sort=False):
        cut = g[g['status'].isin(CUTOFF)]
        if cut.empty:
            continue
        first = cut.sort_values(['n_train_events', 'atom_id']).iloc[0]
        out = g.index[g['n_train_events'] >= first['n_train_events']]
        t.loc[out, 'in_budget'] = False
        t.loc[out, 'excluded_by'] = first['atom_id']
        covered = out[t.loc[out, 'status'].isin(['missing', 'started'])]  # Never run because of the cutoff
        t.loc[covered, 'status'] = 'skipped'
        t.loc[covered, 'skipped_by'] = first['atom_id']
    return t


# ---------------------------------------------------------------------------------------------------------------
# Part 1b: class counts of the training sets (minority count analysis)
# ---------------------------------------------------------------------------------------------------------------

def train_class_count_table(dataset, manifest, atoms, in_dir):
    """
    Class counts per training sample of the cells completed by at least one method (columns: module docstring).
    Raises ValueError if two methods report different counts for the same cell.
    """
    labels = [str(l) for l in pd.read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_labels.csv'))['label']]
    ok = set(atoms.loc[(atoms['dataset'] == dataset) & (atoms['status'] == 'ok'), 'atom_id'])
    seen = OrderedDict()  # (split, trial, n_samples, n_events) -> (atom_id, {sample: {label: count}})
    for r in manifest:
        if r['dataset'] != dataset or r['atom_id'] not in ok:
            continue
        counts = (load_json(atom_paths(in_dir, r)[0]) or {}).get('n_train_events_sample_per_class')
        if counts is None:
            raise ValueError('no n_train_events_sample_per_class in ok atom {}'.format(r['atom_id']))
        key = (int(r['split']), int(r['trial']), r['n_samples'], r['n_events'])
        if key not in seen:
            seen[key] = (r['atom_id'], counts)
        elif counts != seen[key][1]:
            raise ValueError('training class counts differ between {} and {}'.format(seen[key][0], r['atom_id']))
    rows = []
    for (split, trial, ns, ne), (_, counts) in seen.items():
        for sample, c in counts.items():
            rows.append([dataset, split, trial, ns, ne, sample] + [int(c.get(l, 0)) for l in labels])
    return pd.DataFrame(rows, columns=['dataset', 'split', 'trial', 'n_samples', 'n_events', 'sample']
                        + ['n_{}'.format(l) for l in labels])


# ---------------------------------------------------------------------------------------------------------------
# Part 2: score tables (per atom x predicted test sample, per-class scores as in benchmark_eval.py)
# ---------------------------------------------------------------------------------------------------------------

def score_table(dataset, manifest, atoms, in_dir):
    """One row per atom x predicted test sample of the dataset (columns: module docstring)."""
    labels = pd.read_csv(os.path.join(config.DATA_CURATED, dataset, 'meta_labels.csv'))['label'].tolist()
    k = len(labels)
    status = atoms.set_index('atom_id')['status']
    parts = []
    for r in manifest:
        if r['dataset'] != dataset:
            continue
        json_path, counts_path, _ = atom_paths(in_dir, r)
        if not os.path.exists(counts_path):
            continue
        d = load_json(json_path) or {}
        pred_res = (d.get('resources') or {}).get('predict') or {}
        c = pd.read_csv(counts_path).sort_values(['sample', 'label'], kind='stable')
        c = c[c['sample'].isin(list(pred_res))]
        if c.empty:
            continue
        samples = c['sample'].iloc[::k].tolist()
        if len(c) != k * len(samples) or c['label'].tolist() != labels * len(samples):
            raise ValueError('unexpected labels in {}'.format(counts_path))

        def mat(col):
            return c[col].to_numpy().reshape(len(samples), k)

        n_true, n_pred, n_abst = mat('n_true'), mat('n_pred'), mat('n_abstained')
        if (not (n_true.sum(axis=1) == c['n_events_used'].to_numpy()[::k]).all()
                or not (n_pred.sum(axis=1) + n_abst.sum(axis=1) == n_true.sum(axis=1)).all()
                or not (n_pred == mat('tp') + mat('fp')).all()):
            raise ValueError('event count mismatch in {}'.format(counts_path))
        prec, rec, f1 = class_scores(mat('tp'), mat('fp'), mat('fn'))
        t = pd.DataFrame(OrderedDict([
            ('method', r['method']), ('dataset', dataset), ('split', int(r['split'])), ('trial', int(r['trial'])),
            ('n_samples', r['n_samples']), ('n_events', r['n_events']), ('atom_id', r['atom_id']),
            ('atom_status', status[r['atom_id']]), ('sample', samples), ('predicted', True),
            ('n_cells', c['n_events_sample'].to_numpy()[::k]), ('n_abstained', n_abst.sum(axis=1)),
            ('abstention_rate', n_abst.sum(axis=1) / n_true.sum(axis=1).astype(float)),
            ('pred_s', [pred_res[s].get('wall_s') for s in samples]),
            ('pred_peak_mb', [pred_res[s].get('cpu_rss_peak_mb') for s in samples]),
        ]))
        for i, l in enumerate(labels):
            t['n_{}'.format(l)] = n_true[:, i]
            t['npred_{}'.format(l)] = n_pred[:, i]
            t['prec_{}'.format(l)] = prec[:, i]
            t['rec_{}'.format(l)] = rec[:, i]
            t['f1_{}'.format(l)] = f1[:, i]
        parts.append(t)
    cols = (['method', 'dataset', 'split', 'trial', 'n_samples', 'n_events', 'atom_id', 'atom_status', 'sample',
             'predicted', 'n_cells', 'n_abstained', 'abstention_rate', 'pred_s', 'pred_peak_mb']
            + ['{}_{}'.format(m, l) for l in labels for m in ('n', 'npred', 'prec', 'rec', 'f1')])
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------------------------------------------
# Part 3: cells per lane, per-sample trial means, paired with the step-7 baseline
# ---------------------------------------------------------------------------------------------------------------

def baseline_scores(dataset, baseline_dir):
    """Step-7 score per method x split x test sample (seed mean over the seeds that predicted it)."""
    s = pd.read_csv(os.path.join(baseline_dir, 'scores', dataset + '.csv'))
    s['score'], _ = sample_metric(s, 'f1')
    return s.groupby(['method', 'split', 'sample'])['score'].mean().rename('baseline').reset_index()


def lane_table(dataset, manifest, atoms):
    """
    One row per method x cell x split x trial of the dataset: the resolved atom and its state.

    Columns: method, dataset, n_samples, n_events, split, trial, alias ('' own atom, 'atom' another step-8 atom,
    'baseline' the step-7 model), atom_id, status, ok (status ok and in budget), n_train_events.
    """
    a = atoms.set_index('atom_id')
    lanes, cells = OrderedDict(), OrderedDict()
    for r in manifest:
        if r['dataset'] != dataset:
            continue
        key = (r['method'], int(r['split']), int(r['trial']))
        lane = lanes.setdefault(key, {'n_train': int(r['n_train_samples_split']), 'atoms': {}})
        cell = (grid_value(r['n_samples']), grid_value(r['n_events']))
        lane['atoms'][cell] = r['atom_id']
        cells.setdefault(r['method'], set()).add(cell)
    sizes = sample_sizes(dataset)
    sp = split_table(dataset)
    full = dict((int(c.split('_')[1]), int(sum(sizes[s] for s in sp.loc[sp[c] == 'train', 'sample'])))
                for c in sp.columns if c.startswith('split_'))  # Training events of the step-7 model per split
    cut = atoms[(atoms['dataset'] == dataset) & atoms['status'].isin(CUTOFF)].groupby('method')['n_train_events'].min()
    rows = []
    for (method, split, trial), lane in lanes.items():
        x = cut.get(method, np.inf)
        for ns, ne in sorted(cells[method], key=lambda c: cell_key(*c)):
            ns_eff = 'all' if ns != 'all' and ns >= lane['n_train'] else ns
            if (ns_eff, ne) in lane['atoms']:
                aid = lane['atoms'][(ns_eff, ne)]
            elif (ns_eff, 'all') in lane['atoms']:
                aid = lane['atoms'][(ns_eff, 'all')]
            elif ns_eff == 'all':
                aid = None  # ('all', 'all'): the step-7 model
            else:
                raise ValueError('cannot resolve cell {} x {} in lane {} {} split {} trial {}'.format(
                    ns, ne, method, dataset, split, trial))
            if aid is None:
                n_train = full[split]
                rows.append([method, dataset, ns, ne, split, trial, 'baseline', '', 'ok', bool(n_train < x), n_train])
            else:
                row = a.loc[aid]
                rows.append([method, dataset, ns, ne, split, trial,
                             '' if aid == lane['atoms'].get((ns, ne)) else 'atom', aid, row['status'],
                             bool(row['status'] == 'ok' and row['in_budget']), int(row['n_train_events'])])
    return pd.DataFrame(rows, columns=['method', 'dataset', 'n_samples', 'n_events', 'split', 'trial', 'alias',
                                       'atom_id', 'status', 'ok', 'n_train_events'])


def sample_score_table(dataset, lanes, scores, baseline, n_trials=N_TRIALS):
    """One row per method x cell x test sample (columns: module docstring); n_trials: trials 0 .. n_trials - 1."""
    sp = split_table(dataset)
    if len(scores):
        sc = scores.copy()
        sc['score'], _ = sample_metric(sc, 'f1')
        by_atom = dict(zip(zip(sc['atom_id'], sc['sample']), sc['score']))
    else:
        by_atom = {}
    base = dict(zip(zip(baseline['method'], baseline['split'], baseline['sample']), baseline['baseline']))
    test = dict((int(c.split('_')[1]), sp.loc[sp[c] == 'test', ['sample', 'patient_id']])
                for c in sp.columns if c.startswith('split_'))
    rows = []
    for (method, ns, ne, split), g in lanes.groupby(['method', 'n_samples', 'n_events', 'split'], sort=False):
        ts = test[split]
        b = np.array([base.get((method, split, s), np.nan) for s in ts['sample']])
        per_trial = np.full((len(ts), n_trials), np.nan)
        predicted = np.zeros((len(ts), n_trials), dtype=bool)
        for _, lane in g.iterrows():
            if lane['alias'] == 'baseline':
                per_trial[:, lane['trial']] = b
                predicted[:, lane['trial']] = True
                continue
            for i, s in enumerate(ts['sample']):
                if (lane['atom_id'], s) in by_atom:
                    per_trial[i, lane['trial']] = by_atom[(lane['atom_id'], s)]
                    predicted[i, lane['trial']] = True
        n_pred = predicted.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)  # All trials undefined -> NaN (intended)
            mean = np.where(n_pred == n_trials, np.nanmean(per_trial, axis=1), np.nan)
        t = pd.DataFrame(OrderedDict([('method', method), ('dataset', dataset), ('n_samples', ns), ('n_events', ne),
                                      ('split', split), ('sample', ts['sample'].to_numpy()),
                                      ('patient_id', ts['patient_id'].to_numpy())]))
        for tr in range(n_trials):
            t['score_t{}'.format(tr)] = per_trial[:, tr]
        t['n_trials_pred'] = n_pred
        t['trial_mean'] = mean
        t['baseline'] = b
        t['d'] = mean - b
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Part 4: curves per scope with bootstrap CIs
# ---------------------------------------------------------------------------------------------------------------

def scopes(splits, n_trials=N_TRIALS):
    """[(scope, splits in the scope, trials in the scope, value column)]."""
    trials = list(range(n_trials))
    res = [('pooled', splits, trials, 'trial_mean')]
    res += [('trial_{}'.format(t), splits, [t], 'score_t{}'.format(t)) for t in trials]
    res += [('fold_{}'.format(k), [k], trials, 'trial_mean') for k in splits]
    res += [('excl_{}'.format(k), [s for s in splits if s != k], trials, 'trial_mean') for k in splits]
    return res


def boot_counts(dataset, subset, groups):
    """
    N_BOOT x G matrix: how often each patient (group) is drawn in each resample (G draws with replacement). Fixed per
    dataset and sample subset (seed from both), so all methods and cells share the resamples.
    """
    rng = np.random.default_rng(derive_seed('tdr_eval', dataset, subset))
    g = len(groups)
    idx = rng.integers(0, g, size=(N_BOOT, g))
    counts = np.zeros((N_BOOT, g))
    np.add.at(counts, (np.repeat(np.arange(N_BOOT), g), idx.ravel()), 1)
    return counts


def curve_table(dataset, lanes, samples, scope_list=None):
    """One row per method x cell x scope (columns: module docstring); scope_list: default scopes(splits)."""
    splits = sorted(samples['split'].unique())
    q = [100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2]
    cell_cols = ['method', 'n_samples', 'n_events']
    out = []
    for scope, sc_splits, sc_trials, col in (scopes(splits) if scope_list is None else scope_list):
        if len(sc_splits) == len(splits):
            subset = 'all'  # pooled and trial_<t> share the resamples
        else:
            subset = scope
        s = samples[samples['split'].isin(sc_splits)].reset_index(drop=True)
        groups = sorted(s['patient_id'].unique())
        counts = boot_counts(dataset, subset, groups)
        g_idx = s['patient_id'].map(dict((p, i) for i, p in enumerate(groups))).to_numpy()
        cid = s.groupby(cell_cols, sort=False).ngroup().to_numpy()
        keys = [tuple(r) for r in s[cell_cols].drop_duplicates().itertuples(index=False)]  # Order of ngroup
        valid = (s[col] - s['baseline']).notna().to_numpy()
        d = np.where(valid, s[col] - s['baseline'], 0.0)
        v = np.where(valid, s[col], 0.0)
        b = np.where(valid, s['baseline'], 0.0)

        def group_sums(x):  # patients x cells
            m = np.zeros((len(groups), len(keys)))
            np.add.at(m, (g_idx, cid), x)
            return m

        with np.errstate(invalid='ignore', divide='ignore'):
            boot_n = counts @ group_sums(valid.astype(float))
            boot_d = np.where(boot_n > 0, counts @ group_sums(d) / boot_n, np.nan)
            boot_b = np.where(boot_n > 0, counts @ group_sums(b) / boot_n, np.nan)
            boot_v = np.where(boot_n > 0, counts @ group_sums(v) / boot_n, np.nan)
        n = np.bincount(cid, weights=valid, minlength=len(keys))
        n_total = np.bincount(cid, minlength=len(keys))
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_d = np.where(n > 0, np.bincount(cid, weights=d, minlength=len(keys)) / n, np.nan)
            mean_v = np.where(n > 0, np.bincount(cid, weights=v, minlength=len(keys)) / n, np.nan)
            mean_b = np.where(n > 0, np.bincount(cid, weights=b, minlength=len(keys)) / n, np.nan)
            ss = np.bincount(cid, weights=np.where(valid, (d - mean_d[cid]) ** 2, 0.0), minlength=len(keys))
            sd_d = np.where(n >= 2, np.sqrt(ss / (n - 1)), np.nan)

        ln = lanes[lanes['split'].isin(sc_splits) & lanes['trial'].isin(sc_trials)]
        lstat = dict((k, g) for k, g in ln.groupby(cell_cols, sort=False))
        for j, key in enumerate(keys):
            g = lstat[key]
            if n[j] >= 2:
                ci, bci = np.nanpercentile(boot_d[:, j], q), np.nanpercentile(boot_b[:, j], q)
                sci = np.nanpercentile(boot_v[:, j], q)
            else:
                ci = bci = sci = [np.nan, np.nan]
            out.append(OrderedDict([
                ('method', key[0]), ('dataset', dataset), ('n_samples', key[1]), ('n_events', key[2]),
                ('scope', scope), ('n_lanes', len(g)), ('n_alias', int((g['alias'] == 'atom').sum())),
                ('n_baseline_alias', int((g['alias'] == 'baseline').sum())), ('n_ok', int(g['ok'].sum())),
                ('eligible', bool(g['ok'].all())), ('n_train_events_mean', float(g['n_train_events'].mean())),
                ('n_train_events_min', int(g['n_train_events'].min())),
                ('n_train_events_max', int(g['n_train_events'].max())),
                ('n_test', int(n[j])), ('n_test_total', int(n_total[j])), ('mean_score', mean_v[j]),
                ('mean_baseline', mean_b[j]), ('mean_d', mean_d[j]), ('sd_d', sd_d[j]),
                ('ci_low', float(ci[0])), ('ci_high', float(ci[1])),
                ('base_ci_low', float(bci[0])), ('base_ci_high', float(bci[1])),
                ('score_ci_low', float(sci[0])), ('score_ci_high', float(sci[1]))]))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------
# Part 5: minimal configurations (criteria x orders x stability x scopes)
# ---------------------------------------------------------------------------------------------------------------

def criteria():
    """[(criterion, delta)]: noninf for every delta of the sweep, then legacy_ci (delta NaN)."""
    return [('noninf', d) for d in DELTAS] + [('legacy_ci', np.nan)]


def passes(curve, criterion, delta):
    """Boolean array: the cells (rows of curve) that meet the criterion (eligibility not included)."""
    if criterion == 'noninf':
        return (curve['ci_low'] > -delta).to_numpy()
    if criterion == 'legacy_ci':
        return (curve['mean_score'] >= curve['base_ci_low']).to_numpy()
    raise ValueError(criterion)


def minimal_cell(curve, ok, order, stable):
    """
    Index (into curve) of the minimal cell, or None.

    Args:
        curve (pd.DataFrame): Rows of one method x dataset x scope (n_samples, n_events, eligible,
            n_train_events_mean).
        ok (np.ndarray): Passes the criterion (bool per row).
        order (str): 'greedy' or 'joint'.
        stable (bool): Require every eligible cell with larger n_samples and the same n_events to pass.
    """
    elig = curve['eligible'].to_numpy().astype(bool)
    ns = np.array([size_key(v) for v in curve['n_samples']])
    ne = np.array([size_key(v) for v in curve['n_events']])
    good = elig & ok
    if stable:
        fail = elig & ~ok
        for i in np.flatnonzero(good):
            if (fail & (ne == ne[i]) & (ns > ns[i])).any():
                good[i] = False
    idx = np.flatnonzero(good)
    if not len(idx):
        return None
    if order == 'greedy':
        keys = sorted(idx, key=lambda i: (ns[i], ne[i]))
    else:
        nt = curve['n_train_events_mean'].to_numpy()
        keys = sorted(idx, key=lambda i: (nt[i], ns[i], ne[i]))
    return keys[0]


def minimal_table(curves):
    """One row per method x dataset x criterion x delta x order x stable x scope (columns: module docstring)."""
    rows = []
    for (method, dataset, scope), c in curves.groupby(['method', 'dataset', 'scope'], sort=False):
        c = c.reset_index(drop=True)
        for criterion, delta in criteria():
            ok = passes(c, criterion, delta)
            for order in ORDERS:
                for stable in (False, True):
                    i = minimal_cell(c, ok, order, stable)
                    row = OrderedDict([('method', method), ('dataset', dataset), ('criterion', criterion),
                                       ('delta', delta), ('order', order), ('stable', stable), ('scope', scope),
                                       ('found', i is not None)])
                    for col in CELL + ['n_train_events_mean', 'mean_d', 'ci_low', 'ci_high', 'mean_score',
                                       'mean_baseline', 'n_test']:
                        row[col] = c.loc[i, col] if i is not None else (np.nan if col not in CELL else '')
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Part 6: cross-fitted selection (select on folds != k, evaluate on fold k)
# ---------------------------------------------------------------------------------------------------------------

def crossfit_table(minimal, curves):
    """One row per method x dataset x criterion x delta x order x stable x fold (columns: module docstring)."""
    key = ['method', 'dataset', 'criterion', 'delta', 'order', 'stable']
    cv = curves.set_index(['method', 'dataset', 'scope', 'n_samples', 'n_events'])
    mn = minimal.copy()
    mn['delta'] = mn['delta'].fillna(-1)  # NaN (legacy_ci) cannot be a groupby key
    rows = []
    for k_vals, g in mn.groupby(key, sort=False):
        method, dataset = k_vals[0], k_vals[1]
        sel = g.set_index('scope')
        folds = sorted(int(s.split('_')[1]) for s in sel.index if s.startswith('fold_'))
        pooled = sel.loc['pooled']

        def at(scope, cell):
            if cell is None:
                return np.nan, np.nan, 0
            r = cv.loc[(method, dataset, scope) + cell]
            return r['mean_d'], r['mean_score'], int(r['n_test'])

        p_cell = (pooled['n_samples'], pooled['n_events']) if pooled['found'] else None
        fold_rows = []
        for k in folds:
            e = sel.loc['excl_{}'.format(k)]
            cf_cell = (e['n_samples'], e['n_events']) if e['found'] else None
            cf = at('fold_{}'.format(k), cf_cell)
            ins = at('fold_{}'.format(k), p_cell)
            fold_rows.append(OrderedDict(
                list(zip(key, k_vals)) + [('fold', str(k)),
                                          ('cf_n_samples', cf_cell[0] if cf_cell else ''),
                                          ('cf_n_events', cf_cell[1] if cf_cell else ''),
                                          ('cf_mean_d', cf[0]), ('cf_mean_score', cf[1]), ('cf_n_test', cf[2]),
                                          ('pooled_n_samples', p_cell[0] if p_cell else ''),
                                          ('pooled_n_events', p_cell[1] if p_cell else ''),
                                          ('in_mean_d', ins[0]), ('in_mean_score', ins[1]), ('in_n_test', ins[2]),
                                          ('same_cell', cf_cell is not None and cf_cell == p_cell)]))
        f = pd.DataFrame(fold_rows)
        w = f['cf_n_test'].to_numpy(dtype=float)
        pin = at('pooled', p_cell)
        all_row = OrderedDict(list(zip(key, k_vals)) + [
            ('fold', 'all'), ('cf_n_samples', ''), ('cf_n_events', ''),
            ('cf_mean_d', float(np.sum(w * f['cf_mean_d']) / w.sum()) if w.sum() else np.nan),
            ('cf_mean_score', float(np.sum(w * f['cf_mean_score']) / w.sum()) if w.sum() else np.nan),
            ('cf_n_test', int(w.sum())),
            ('pooled_n_samples', p_cell[0] if p_cell else ''), ('pooled_n_events', p_cell[1] if p_cell else ''),
            ('in_mean_d', pin[0]), ('in_mean_score', pin[1]), ('in_n_test', pin[2]),
            ('same_cell', int(f['same_cell'].sum()))])
        rows += fold_rows + [all_row]
    t = pd.DataFrame(rows)
    t.loc[t['delta'] == -1, 'delta'] = np.nan
    return t


# ---------------------------------------------------------------------------------------------------------------
# Part 7: minimal configurations for step 9
# ---------------------------------------------------------------------------------------------------------------

def minimal_configs(minimal):
    """{method: {dataset: {'stable' | 'unstable': {'pooled': cell, 'crossfit': {fold: cell}}}}} (primary criterion)."""
    m = minimal[(minimal['criterion'] == 'noninf') & np.isclose(minimal['delta'], PRIMARY_DELTA)
                & (minimal['order'] == PRIMARY_ORDER)]

    def cell(r):
        if not r['found']:
            return None
        return OrderedDict([('n_samples', grid_value(r['n_samples'])), ('n_events', grid_value(r['n_events'])),
                            ('n_train_events_mean', float(r['n_train_events_mean'])),
                            ('mean_d', float(r['mean_d'])), ('ci_low', float(r['ci_low']))])

    res = OrderedDict([('criterion', 'noninf'), ('delta', PRIMARY_DELTA), ('order', PRIMARY_ORDER),
                       ('methods', OrderedDict())])
    for (method, dataset, stable), g in m.groupby(['method', 'dataset', 'stable'], sort=False):
        sel = g.set_index('scope')
        folds = sorted(int(s.split('_')[1]) for s in sel.index if s.startswith('excl_'))
        entry = OrderedDict([('pooled', cell(sel.loc['pooled'])),
                             ('crossfit', OrderedDict((str(k), cell(sel.loc['excl_{}'.format(k)])) for k in folds))])
        res['methods'].setdefault(method, OrderedDict()).setdefault(dataset, OrderedDict())[
            'stable' if stable else 'unstable'] = entry
    return res


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------

def read_cells(path):
    """Reads a table with n_samples / n_events columns back with grid values ('all' or int; '' = no cell)."""
    t = pd.read_csv(path, dtype={'n_samples': str, 'n_events': str, 'patient_id': str}, keep_default_na=False,
                    na_values=[''])
    for c in CELL:
        t[c] = [grid_value(v) if isinstance(v, str) else '' for v in t[c]]
    return t


def main(in_dir, baseline_dir, out_dir):
    os.makedirs(os.path.join(out_dir, 'scores'), exist_ok=True)
    manifest = load_manifest(in_dir)
    datasets = [d for d in config.DATASETS if any(r['dataset'] == d for r in manifest)]

    # --- Part 1: atom table with the post-hoc budget
    p_atoms = os.path.join(out_dir, 'atoms.csv')
    if done(p_atoms):
        atoms = pd.read_csv(p_atoms, dtype={'n_samples': str, 'n_events': str}, keep_default_na=False,
                            na_values=[''])
        atoms[['skipped_by', 'excluded_by', 'error']] = atoms[['skipped_by', 'excluded_by', 'error']].fillna('')
    else:
        atoms = atom_table(manifest, in_dir)
        write_csv(atoms, p_atoms)
    print('atoms:', atoms['status'].value_counts().to_dict(), '| out of budget:', int((~atoms['in_budget']).sum()))

    # --- Part 1b: class counts of the training sets (minority count analysis)
    os.makedirs(os.path.join(out_dir, 'train_class_counts'), exist_ok=True)
    for dataset in datasets:
        path = os.path.join(out_dir, 'train_class_counts', dataset + '.csv')
        if done(path):
            continue
        t = train_class_count_table(dataset, manifest, atoms, in_dir)
        write_csv(t, path)
        print('train class counts {}: {} rows'.format(dataset, len(t)))

    # --- Part 2: score tables
    for dataset in datasets:
        path = os.path.join(out_dir, 'scores', dataset + '.csv')
        if done(path):
            continue
        t = score_table(dataset, manifest, atoms, in_dir)
        write_csv(t, path)
        print('scores {}: {} rows'.format(dataset, len(t)))

    # --- Part 3: per-sample trial means and the paired difference to the step-7 baseline
    p_samples = os.path.join(out_dir, 'sample_scores.csv')
    lanes = dict((d, lane_table(d, manifest, atoms)) for d in datasets)
    if done(p_samples):
        samples = read_cells(p_samples)
    else:
        parts = []
        for dataset in datasets:
            scores = pd.read_csv(os.path.join(out_dir, 'scores', dataset + '.csv'),
                                 usecols=lambda c: c in ('atom_id', 'sample', 'predicted') or c.startswith('f1_'))
            parts.append(sample_score_table(dataset, lanes[dataset], scores,
                                            baseline_scores(dataset, baseline_dir)))
        samples = pd.concat(parts, ignore_index=True)
        write_csv(samples, p_samples)
    print('sample scores:', p_samples)

    # --- Part 4: curves per scope
    p_curves = os.path.join(out_dir, 'curves.csv')
    if done(p_curves):
        curves = read_cells(p_curves)
    else:
        curves = pd.concat([curve_table(d, lanes[d], samples[samples['dataset'] == d]) for d in datasets],
                           ignore_index=True)
        write_csv(curves, p_curves)
    print('curves:', p_curves)

    # --- Part 5: minimal cells
    p_min = os.path.join(out_dir, 'minimal.csv')
    if done(p_min):
        minimal = read_cells(p_min)
    else:
        minimal = minimal_table(curves)
        write_csv(minimal, p_min)
    print('minimal:', p_min)

    # --- Part 6: cross-fitting
    p_cf = os.path.join(out_dir, 'crossfit.csv')
    if not done(p_cf):
        write_csv(crossfit_table(minimal, curves), p_cf)
    print('crossfit:', p_cf)

    # --- Part 7: minimal configurations for step 9
    p_json = os.path.join(out_dir, 'minimal_configs.json')
    if not done(p_json):
        tmp = p_json + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(minimal_configs(minimal), f, indent=2)
        os.replace(tmp, p_json)
    print('minimal configs:', p_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in', dest='in_dir', default=IN_DIR, help='step-8 output root (default: %(default)s)')
    parser.add_argument('--baseline', default=BASELINE_DIR, help='step-7 eval output (default: %(default)s)')
    parser.add_argument('--out', default=OUT_DIR, help='output directory (default: %(default)s)')
    args = parser.parse_args()
    main(os.path.abspath(args.in_dir), os.path.abspath(args.baseline), os.path.abspath(args.out))
