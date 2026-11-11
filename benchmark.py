"""
Main benchmark (PLAN.md step 7, working_notes/benchmark_plan.md): every method on every dataset with its tuned
parameters.

One atom = (method, dataset, outer split, model seed): fit on all training samples of the outer split (all events,
unless TRAIN_DOWNSAMPLING has an entry for the method and dataset), predict every test sample in full, store per test
sample the evaluation counts, the predicted labels (SAVE_LABELS) and the time / memory of fit and predict. Scores and
aggregation are computed later from the atoms (benchmark_eval.py).

Env: the env of the array (ARRAYS below), e.g. fx_sklearn for 'sklearn_cpu'. Python 3.5-compatible (fx_deepcytof).

Hyperparameters (D6): Imstat atoms use the best config of their outer split
(params_outer_<method>.json['splits'][str(split)]), all other datasets the consensus config
(params_overall_<method>.json), both from results/tune_hyperparameters_eval/best_configs/. They are read when the
manifest is written and frozen into it.

Seeds: N_SEEDS model seeds per method (3 for stochastic, 1 for deterministic methods). Model seed
derive_seed('benchmark', method, dataset, split, seed); data seeds (downsampling per training sample, join shuffle)
derive_seed('benchmark', dataset, split, seed, <sample> | 'join'), the same for all methods.

Arrays:
    One SLURM array per (env, cpu/gpu) combination, as in the tuning (ARRAYS below): sklearn_cpu, flagx_cpu,
    gatemeclass_cpu, lda_cpu, deepcytof_cpu (Woody) and flagx_gpu (TinyGPU). Each array has its own manifest (task list,
    numbered from 0) and SLURM script (slurm_scripts/bench_<array>.sh). Array task i runs row i of
    manifests/<array>.csv.

Commands:
    python benchmark.py manifest <array> [--out DIR]
        Writes the frozen task list <out>/manifests/<array>.csv (index, atom_id, method, dataset, split, seed,
        config_id, param_source, params, train_downsampling), creates the SLURM log dir and prints the number of atoms
        N (SLURM: --array=0-<N-1>). Any env with numpy works (no method code is imported).
    python benchmark.py run <array> [<index> | all] [--out DIR] [--smoke]
        Runs atom <index> of the manifest (default: $SLURM_ARRAY_TASK_ID), or all atoms sequentially ('all').
        Must run in the array's env. --smoke: 3 train / 2 test samples, 1 % of the events of each (only with --out,
        for testing).
    python benchmark.py missing <array> [--out DIR]
        Manifest indices whose atom has no JSON yet (never started), as a SLURM --array range.
    python benchmark.py status <array> [--out DIR]
        Table method x dataset with the number of atoms ok / failed / started (killed: timeout, OOM, node failure) /
        missing, and the SLURM range of all atoms that are not ok. Input for the decision on a second run.

Second run with downsampling (after OOM / timeout): add entries to TRAIN_DOWNSAMPLING, write the manifest again and
submit the printed ranges (status). A downsampled atom has its own atom_id and files (suffix __ds_<tag>, e.g.
__ds_n20000), so the failed full-data atom stays on disk as a record; the manifest defines which atoms are current.
The row order of a manifest does not depend on TRAIN_DOWNSAMPLING, so the indices stay the same.

Outputs (<out> = results/benchmark by default; <stem> = split_<i>_seed_<r>[__ds_<tag>]):
    atoms/<method>/<dataset>/<stem>.json: everything about the atom (identity, params, seeds, samples, downsampling,
        preprocessing, resources of fit and of each test sample's predict, run metadata) and its status: 'started'
        (written first; stays if the job was killed), 'ok', 'failed' (+ traceback).
    counts/<method>/<dataset>/<stem>.csv (only if ok): one row per test sample x true class with the identifiers,
        per-class counts (n_true, n_pred, n_abstained, tp, fp, fn, tn) and the confusion row (pred_<label>...,
        pred_abstained), as in the tuning. Concatenates directly.
    predictions/<method>/<dataset>/<stem>/<sample>.npz (SAVE_LABELS; git-ignored): 'y_pred' (int16, original labels,
        -1 = abstention) in the event order of the curated sample (test samples are not downsampled; except --smoke);
        'proba' (float32, columns in the class order of the fitted model, see the runner) if SAVE_PROBA.
An atom with an existing JSON is skipped (OVERWRITE = False); delete the JSON to rerun it.
"""

import argparse
import importlib
import json
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np

import config
from validation.core.data import load_meta_labels, load_sample, load_split
from validation.core.evaluation import class_counts, confusion_counts
from validation.core.preprocessing import preprocessor_state, shared_preprocessor
from validation.core.resources import track_resources
from validation.core.runs import index_ranges, n_threads, now, read_manifest, run_metadata, save_csv, save_json
from validation.core.sampling import downsample_indices, join_samples
from validation.core.seeding import derive_seed

# ### Parameters #######################################################################################################
DATASETS = config.DATASETS
OUT = os.path.join(config.RESULTS, 'benchmark')
BEST_CONFIGS = os.path.join(config.RESULTS, 'tune_hyperparameters_eval', 'best_configs')
OVERWRITE = False

TUNING_DATASET = 'imstat'  # Uses the per-outer-split best configs (nested CV); all others the consensus config (D6)

SAVE_LABELS = True  # Per-cell predicted labels per test sample
SAVE_PROBA = False  # Also the probabilities, for methods that have them
STRATIFIED = True  # Class-stratified training downsampling, only with TRAIN_DOWNSAMPLING

# Training downsampling per (method, dataset), applied to every training sample (seeded, on the fly):
# {'n': 20000} = at most 20000 events per sample, {'frac': 0.1} = 10 % of the events per sample.
# Empty = all events (first run). Entries are added for atoms that hit OOM / timeout, e.g.
# ('som', 'blast110_p1'): {'n': 20000}
TRAIN_DOWNSAMPLING = {}

SMOKE = {'n_train': 3, 'n_test': 2, 'frac': 0.01}  # --smoke

# Array = one SLURM array job per (env, cpu/gpu): env, device, methods (same as tune_hyperparameters.py)
ARRAYS = OrderedDict([
    ('sklearn_cpu', ('fx_sklearn', 'cpu', ['rf', 'hgb', 'logreg', 'knn', 'lda_sklearn', 'gmm_mocking'])),
    ('flagx_cpu', ('fx_flagx_cpu', 'cpu', ['som'])),
    ('flagx_gpu', ('fx_flagx_gpu', 'gpu', ['mlp', 'dgcytof'])),
    ('gatemeclass_cpu', ('fx_gatemeclass', 'cpu', ['gatemeclass'])),
    ('lda_cpu', ('fx_lda', 'cpu', ['lda_abdelaal'])),
    ('deepcytof_cpu', ('fx_deepcytof', 'cpu', ['deepcytof'])),
])

# Model seeds per method: 3 for stochastic methods (initialization, sampling, random splits), 1 for deterministic ones
N_SEEDS = OrderedDict([
    ('som', 3), ('mlp', 3), ('dgcytof', 3), ('deepcytof', 3), ('gatemeclass', 3), ('rf', 3), ('hgb', 3),
    ('gmm_mocking', 3), ('logreg', 1), ('knn', 1), ('lda_sklearn', 1), ('lda_abdelaal', 1),
])
########################################################################################################################


# ### Manifest #########################################################################################################

def best_config(method, dataset, split):
    """(config_id, params, param_source) of the tuned config for an atom (D6)."""
    if dataset == TUNING_DATASET:
        with open(os.path.join(BEST_CONFIGS, 'params_outer_{}.json'.format(method))) as f:
            entry = json.load(f)['splits'][str(split)]
        return entry['config_id'], entry['params'], 'outer'
    with open(os.path.join(BEST_CONFIGS, 'params_overall_{}.json'.format(method))) as f:
        entry = json.load(f)
    return entry['config_id'], entry['params'], 'overall'


def ds_tag(spec):
    """Downsampling spec -> file name tag: {'n': 20000} -> 'n20000', {'frac': 0.1} -> 'frac0.1'; None -> None."""
    if not spec:
        return None
    if sorted(spec) not in (['n'], ['frac']):
        raise ValueError("downsampling spec must be {'n': int} or {'frac': float}, got {!r}".format(spec))
    key = list(spec)[0]
    return '{}{}'.format(key, spec[key])


def atom_stem(split, seed, spec):
    stem = 'split_{}_seed_{}'.format(split, seed)
    tag = ds_tag(spec)
    return stem if tag is None else '{}__ds_{}'.format(stem, tag)


def atom_id(method, dataset, split, seed, spec):
    return '{}__{}__{}'.format(method, dataset, atom_stem(split, seed, spec))


def atom_paths(out, row):
    """(json, counts csv, predictions dir) of a manifest row."""
    spec = json.loads(row['train_downsampling'])
    stem = atom_stem(row['split'], row['seed'], spec)
    rel = os.path.join(row['method'], row['dataset'])
    return (os.path.join(out, 'atoms', rel, stem + '.json'), os.path.join(out, 'counts', rel, stem + '.csv'),
            os.path.join(out, 'predictions', rel, stem))


def outer_splits(dataset):
    """Outer split indices from the columns of data/splits/<dataset>.csv (split_<i>), the single source of truth."""
    with open(os.path.join(config.DATA_SPLITS, dataset + '.csv')) as f:
        header = f.readline().strip().split(',')
    return sorted(int(c.split('_')[1]) for c in header if c.startswith('split_'))


def write_manifest(array, out):
    for key in TRAIN_DOWNSAMPLING:
        if key[0] not in N_SEEDS or key[1] not in DATASETS:
            raise ValueError('unknown (method, dataset) in TRAIN_DOWNSAMPLING: {!r}'.format(key))
    env, device, methods = ARRAYS[array]
    rows, index = [], 0
    for method in methods:
        for dataset in DATASETS:
            spec = TRAIN_DOWNSAMPLING.get((method, dataset))
            ds_tag(spec)  # Validates the spec
            for split in outer_splits(dataset):
                cid, params, source = best_config(method, dataset, split)
                for seed in range(N_SEEDS[method]):
                    rows.append([index, atom_id(method, dataset, split, seed, spec), method, dataset, split, seed,
                                 cid, source, json.dumps(params, sort_keys=True),
                                 json.dumps(spec, sort_keys=True)])
                    index += 1
    path = os.path.join(out, 'manifests', array + '.csv')
    save_csv(path, ['index', 'atom_id', 'method', 'dataset', 'split', 'seed', 'config_id', 'param_source', 'params',
                    'train_downsampling'], rows)
    # Log dir of the SLURM script (--output=results/benchmark/logs/%x/...; must exist at submission)
    log_dir = os.path.join(out, 'logs', 'bench_' + array)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print('{}: {} atoms ({}), manifest {}'.format(array, len(rows), ', '.join(
        '{} {}'.format(m, sum(1 for r in rows if r[2] == m)) for m in methods), path))
    n_ds = sum(1 for r in rows if r[-1] != 'null')
    if n_ds:
        print('  {} atoms with training downsampling'.format(n_ds))
    print('SLURM: --array=0-{}'.format(len(rows) - 1))


def manifest_rows(array, out):
    return read_manifest(os.path.join(out, 'manifests', array + '.csv'))


def atom_status(out, row):
    path = atom_paths(out, row)[0]
    if not os.path.exists(path):
        return 'missing'
    with open(path) as f:
        return json.load(f)['status']


def print_missing(array, out):
    manifest = manifest_rows(array, out)
    missing = [int(r['index']) for r in manifest if not os.path.exists(atom_paths(out, r)[0])]
    print('{}: {} of {} atoms missing'.format(array, len(missing), len(manifest)))
    if missing:
        print('SLURM: --array={}'.format(index_ranges(missing)))


def print_status(array, out):
    manifest = manifest_rows(array, out)
    states = ('ok', 'failed', 'started', 'missing')
    table, not_ok = OrderedDict(), []
    for r in manifest:
        st = atom_status(out, r)
        counts = table.setdefault((r['method'], r['dataset']), OrderedDict((s, 0) for s in states))
        counts[st] += 1
        if st != 'ok':
            not_ok.append(int(r['index']))
    print('{:<14} {:<13} {:>4} {:>7} {:>8} {:>8}  {}'.format('method', 'dataset', *(states + ('downsampling',))))
    for (method, dataset), c in table.items():
        spec = TRAIN_DOWNSAMPLING.get((method, dataset))
        print('{:<14} {:<13} {:>4} {:>7} {:>8} {:>8}  {}'.format(method, dataset, *(list(c.values()) + [
            ds_tag(spec) or '-'])))
    print('{}: {} of {} atoms ok'.format(array, len(manifest) - len(not_ok), len(manifest)))
    if not_ok:
        print('not ok: --array={}'.format(index_ranges(not_ok)))


# ### One atom #########################################################################################################

def save_npz(path, **arrays):
    """np.savez_compressed atomically (temporary file, then os.replace)."""
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = '{}.tmp{}'.format(path, os.getpid())
    try:
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_atom(array, row, out, smoke=False):
    env, device, _ = ARRAYS[array]
    method, dataset = row['method'], row['dataset']
    split, seed = int(row['split']), int(row['seed'])
    params = json.loads(row['params'])
    spec = json.loads(row['train_downsampling'])
    json_path, counts_path, pred_dir = atom_paths(out, row)
    if os.path.exists(json_path) and not OVERWRITE:
        print('skip {} (exists)'.format(row['atom_id']))
        return

    if smoke:
        spec = {'frac': SMOKE['frac']}
    column = 'split_{}'.format(split)
    model_seed = derive_seed('benchmark', method, dataset, split, seed)
    info = OrderedDict([
        ('atom_id', row['atom_id']), ('method', method), ('array', array), ('env', env), ('device', device),
        ('dataset', dataset), ('split', split), ('seed', seed), ('split_column', column),
        ('config_id', row['config_id']), ('param_source', row['param_source']), ('params', params),
        ('status', 'started'), ('error', None),
        ('train_downsampling', spec), ('stratified', STRATIFIED), ('smoke', smoke),
        ('save_labels', SAVE_LABELS), ('save_proba', SAVE_PROBA),
        ('seeds', {'model': model_seed, 'data': "derive_seed('benchmark', dataset, split, seed, sample)",
                   'join': "derive_seed('benchmark', dataset, split, seed, 'join')"}),
        ('start', now()), ('end', None),
    ])
    save_json(json_path, info)
    print('{} start {}'.format(now(), row['atom_id']))
    sys.stdout.flush()

    try:
        runner = importlib.import_module('validation.runners.' + method)
        info['metadata'] = run_metadata(runner.MODULES)
        if info['metadata']['conda_env'] != env:
            print('WARNING: running in env {!r}, array {!r} expects {!r}'.format(
                info['metadata']['conda_env'], array, env))
        threads = n_threads()
        gpu = device == 'gpu'

        roles = load_split(dataset, column)
        train, test = roles['train'], roles['test']
        if smoke:
            train, test = train[:SMOKE['n_train']], test[:SMOKE['n_test']]
        labels_meta = load_meta_labels(dataset)
        labels = [r['label'] for r in labels_meta]
        info.update([('train_samples', train), ('test_samples', test), ('labels', labels_meta)])

        # Training data: optional downsampling per sample (seed per sample, the same for all methods), join, shuffle
        Xs, ys, n_train_sample = [], [], OrderedDict()
        for s in train:
            X, y = load_sample(dataset, s)
            n_train_sample[s] = int(y.shape[0])
            if spec:
                idx = downsample_indices(y, n=spec.get('n'), frac=spec.get('frac'), stratified=STRATIFIED,
                                         seed=derive_seed('benchmark', dataset, split, seed, s))
                X, y = X[idx], y[idx]
            Xs.append(X)
            ys.append(y)
        X, y, sample_ids = join_samples(Xs, ys, train, shuffle=True,
                                        seed=derive_seed('benchmark', dataset, split, seed, 'join'))
        del Xs, ys
        info['n_train_events_sample'] = n_train_sample  # Before downsampling
        info['n_train_events'] = int(y.shape[0])  # Used for training
        info['n_train_events_per_class'] = OrderedDict((str(l), int(np.sum(y == l))) for l in labels)

        arm = runner.input_arm(params)
        info.update([('input_arm', arm), ('n_threads', threads)])
        pre = shared_preprocessor(dataset, arm)
        with track_resources(gpu=gpu) as res_fit:
            Xt = pre.fit_transform(X)
            model = runner.build(params, model_seed, dataset, threads)
            model = runner.fit(model, Xt, y, sample_ids)
        if hasattr(runner, 'fit_info'):
            info['fit_info'] = runner.fit_info(model)
        del X, Xt, y, sample_ids
        info['preprocessor'] = preprocessor_state(pre)
        print('{} fit done ({:.1f} s)'.format(now(), res_fit['wall_s']))
        sys.stdout.flush()

        # Test samples: predict in full (smoke: downsampled), counts, optional per-cell outputs
        header = (['method', 'dataset', 'split', 'seed', 'config_id', 'sample', 'n_events_sample', 'n_events_used',
                   'label', 'n_true', 'n_pred', 'n_abstained', 'tp', 'fp', 'fn', 'tn']
                  + ['pred_{}'.format(l) for l in labels] + ['pred_abstained'])
        rows, res_pred = [], OrderedDict()
        for s in test:
            Xv, yv = load_sample(dataset, s)
            n_sample = int(yv.shape[0])
            if smoke:
                idx = downsample_indices(yv, frac=SMOKE['frac'], stratified=True,
                                         seed=derive_seed('benchmark', dataset, split, seed, s))
                Xv, yv = Xv[idx], yv[idx]
            with track_resources(gpu=gpu) as r:
                y_pred, proba = runner.predict(model, pre.transform(Xv), proba=SAVE_PROBA)
            res_pred[s] = r
            if SAVE_LABELS:
                arrays = {'y_pred': np.asarray(y_pred).astype(np.int16)}
                if proba is not None:
                    arrays['proba'] = np.asarray(proba, dtype=np.float32)
                save_npz(os.path.join(pred_dir, s + '.npz'), **arrays)
            cm = confusion_counts(yv, y_pred, labels)
            cc = class_counts(cm)
            for i, l in enumerate(labels):
                rows.append([method, dataset, split, seed, row['config_id'], s, n_sample, int(yv.shape[0]), l]
                            + [int(cc[k][i]) for k in ('n_true', 'n_pred', 'n_abstained', 'tp', 'fp', 'fn', 'tn')]
                            + [int(v) for v in cm[i]])
            print('{} predicted {} ({:.1f} s)'.format(now(), s, r['wall_s']))
            sys.stdout.flush()
        info['resources'] = {'fit': res_fit, 'predict': res_pred}

        save_csv(counts_path, header, rows)
        info['status'] = 'ok'
    except Exception:
        info['status'] = 'failed'
        info['error'] = traceback.format_exc()
        print(info['error'])
    info['end'] = now()
    save_json(json_path, info)  # Written last: status ok only after the counts (and predictions) exist
    print('{} {} {}'.format(now(), info['status'], row['atom_id']))
    sys.stdout.flush()


# ### Main #############################################################################################################

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['manifest', 'run', 'missing', 'status'])
    ap.add_argument('array', choices=list(ARRAYS))
    ap.add_argument('index', nargs='?', default=None, help="atom index, 'all', default $SLURM_ARRAY_TASK_ID")
    ap.add_argument('--out', default=OUT, help='output root (default results/benchmark)')
    ap.add_argument('--smoke', action='store_true', help='tiny test run (requires --out)')
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    if args.command == 'manifest':
        write_manifest(args.array, out)
        return
    if args.command == 'missing':
        print_missing(args.array, out)
        return
    if args.command == 'status':
        print_status(args.array, out)
        return

    if args.smoke and out == os.path.abspath(OUT):
        raise SystemExit('--smoke requires --out with a directory other than {} (smoke atoms there would count as '
                         'finished and be skipped by the real runs); run manifest with the same --out first'.format(
                             OUT))
    manifest = manifest_rows(args.array, out)
    index = args.index if args.index is not None else os.environ.get('SLURM_ARRAY_TASK_ID')
    if index is None:
        raise SystemExit("give an atom index or 'all' (or set SLURM_ARRAY_TASK_ID)")
    rows = manifest if index == 'all' else [manifest[int(index)]]
    for row in rows:
        run_atom(args.array, row, out, smoke=args.smoke)


if __name__ == '__main__':
    main()
