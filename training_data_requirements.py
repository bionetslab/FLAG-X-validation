"""
Training data requirements (PLAN.md step 8, working_notes/training_data_requirements_plan.md): score, time and memory
as a function of the number of training samples x events per training sample, under a fixed small budget (4 CPUs or
1 GPU, ATOM_TIME_LIMIT_S per atom).

One atom = (method, dataset, outer split, trial, n_samples, n_events): fit on the first n_samples of the trial's
sample order (a per-trial permutation of the outer training samples; nested over n_samples, the same for all methods),
each downsampled to n_events events (on the fly, stratified; the first n_events of a fixed per-sample event order, so
nested over n_events, see event_order), predict every test sample in full, store per test sample
the evaluation counts and the time / memory of fit and predict. No predictions are stored. Scores and aggregation are
computed later from the atoms (eval script, to be written).

Env: the env of the array (ARRAYS below), e.g. fx_sklearn for 'sklearn_cpu'. Python 3.5-compatible like
benchmark.py.

Hyperparameters (D6, as in benchmark.py): Imstat atoms use the best config of their outer split, all other datasets the
consensus config (results/tune_hyperparameters_eval/best_configs/); frozen into the manifest.

Seeds: sample order derive_seed('train_req', dataset, split, trial); event order per training sample
derive_seed('train_req', dataset, split, trial, sample) (the same subset of a sample for every n_samples, nested over
n_events); join shuffle
derive_seed('train_req', dataset, split, trial, n_samples, n_events, 'join'); model derive_seed('train_req', method,
dataset, split, trial). The data seeds do not depend on the method, so all methods see the same training data.

Grid: N_SAMPLES (per dataset) x N_EVENTS (same for all datasets) per lane, without duplicates: n_samples >= number of
training samples is the same as 'all'; an event cap >= the largest selected sample is the same as 'all' events;
('all', 'all') is the main benchmark (step 7) and is left out.

Execution (lane workers):
    A lane = (method, dataset, outer split, trial) = one SLURM array task. The worker ('run') walks the lane's atoms
    from small to large (n_samples ascending, within it n_events ascending) and runs each as a child process ('atom')
    with a time limit of ATOM_TIME_LIMIT_S (whole atom: loading, fit, predict). Per atom:
    1. Finished (JSON status ok / failed / timeout / oom / skipped) -> next.
    2. Dominated: an atom of the same method x dataset (any split, any trial) with at most as many total training
       events (n_train_events, from the manifest) ended with timeout / oom (marker in cutoffs/<method>/<dataset>/) -> JSON
       status 'skipped' (skipped_by), no run.
    3. Not enough time left in the lane (now + ATOM_TIME_LIMIT_S + WRAPUP_BUFFER_S > deadline) -> the worker stops
       (exit 0). Deadline: under SLURM the end of the job (time left from squeue at the worker's start; the limit
       varies with --time-min), otherwise lane start + LANE_TIME_LIMIT_S; 'missing' lists the lane for a resubmission, which continues where it stopped.
    4. Run the child (own session; stdout / stderr -> logs/<method>/<dataset>/...log). Over the time limit, the whole
       process group is killed -> 'timeout'. Killed by a signal (cgroup OOM killer) or a memory error in the traceback
       -> 'oom'. timeout / oom write a cutoff marker. Other Python errors -> 'failed' (not a cutoff).
    The worker never loads data, so its own memory stays small.

Arrays: sklearn_cpu (rf, hgb), flagx_cpu (som), flagx_gpu (mlp); METHODS selects the methods. One manifest and one
SLURM script (slurm_scripts/tdr_<array>.sh) per array; array task i runs lane i.

Commands:
    python training_data_requirements.py manifest <array> [--out DIR]
        Writes the frozen atom list <out>/manifests/<array>.csv (lane order split -> dataset -> method -> trial, atoms
        small -> large within a lane), creates the SLURM log dir, prints the number of lanes L (--array=0-<L-1>).
    python training_data_requirements.py run <array> [<lane> | all] [--out DIR] [--smoke] [--atom-limit S]
                                         [--lane-limit S]
        Lane worker for lane <lane> (default: $SLURM_ARRAY_TASK_ID) or all lanes sequentially. --smoke: only atoms
        with <= SMOKE['max_samples'] samples and <= SMOKE['max_events'] events, test samples reduced (only with
        --out). --atom-limit / --lane-limit override the time limits (only with --out, for tests; --lane-limit also
        overrides the SLURM time left).
    python training_data_requirements.py atom <array> <index> [--out DIR] [--smoke]
        One atom in this process (the worker's child; also for debugging). No time limit, no cutoff marker.
    python training_data_requirements.py status <array> [--out DIR]
        Per method x dataset: atoms per status, max fit time / RAM, the cutoff (smallest n_train_events of a timeout /
        oom atom) and the number of 'ok' atoms at or above it (finished before the cutoff was known; dropped in the eval);
        lanes not finished.
    python training_data_requirements.py missing <array> [--out DIR]
        Lanes with atoms that are neither finished nor dominated, as a SLURM --array range.

Outputs (<out> = results/training_data_requirements by default; <stem> = split_<i>/trial_<t>/ns_<s>__ne_<e>):
    manifests/<array>.csv
    atoms/<method>/<dataset>/<stem>.json: identity, params, seeds, training samples, n_train_events (total, per class,
        per sample x class), preprocessing, resources of the fit and of each test sample's predict, run metadata,
        status (started / ok / failed / timeout / oom / skipped), phase (fit / predict / done), and the worker's
        record (return code, wall time, timed_out).
    counts/<method>/<dataset>/<stem>.csv: one row per test sample x true class (as in benchmark.py).
    cutoffs/<method>/<dataset>/split_<i>__trial_<t>__ns_<s>__ne_<e>.json: one marker per timeout / oom atom (with its
        n_train_events).
    logs/<method>/<dataset>/<stem>.log: stdout / stderr of the atom; logs/tdr_<array>/: SLURM logs of the workers.
"""

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np

import config
from validation.core.data import load_meta_labels, load_sample, load_split, outer_splits
from validation.core.evaluation import class_counts, confusion_counts
from validation.core.preprocessing import preprocessor_state, shared_preprocessor
from validation.core.resources import track_resources
from validation.core.runs import (best_config, index_ranges, n_threads, now, read_manifest, run_metadata, save_csv,
                                  save_json)
from validation.core.sampling import downsample_indices, join_samples
from validation.core.seeding import derive_seed

# ### Parameters #######################################################################################################
DATASETS = config.DATASETS
OUT = os.path.join(config.RESULTS, 'training_data_requirements')
BEST_CONFIGS = os.path.join(config.RESULTS, 'tune_hyperparameters_eval', 'best_configs')
TUNING_DATASET = 'imstat'  # Uses the per-outer-split best configs (nested CV); all others the consensus config (D6)
OVERWRITE = False  # False: finished atoms (JSON with a final status) are kept

# Top performers of the main benchmark: 'rf', 'hgb', 'mlp'; and 'som' for continuity
METHODS = ['rf', 'hgb', 'mlp', 'som']

# Trial = sample order + event order + model seed
N_TRIALS = 3

# Grid. n_samples per dataset ('default' unless listed), n_events the same for all datasets
# (a sample with fewer events than the cap is used completely). Values that duplicate 'all' are dropped per lane.
N_SAMPLES = {
    'default': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 'all'],
    'flowcyt': [1, 2, 3, 4, 6, 8, 12, 16, 'all'],
}
N_EVENTS = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 'all']

STRATIFIED = True  # Class-stratified event order of the training samples (event_order; benchmark.py also stratifies)

# Time budget per atom (whole atom: loading, fit, predict; the same for all lanes; reasoning: working note).
# Lane time: under SLURM the worker reads the time left of its own job (squeue %L; the limit varies with --time-min);
# LANE_TIME_LIMIT_S is only the fallback outside SLURM. WRAPUP_BUFFER_S is kept free at the end of a lane.
ATOM_TIME_LIMIT_S = 3600
LANE_TIME_LIMIT_S = 24 * 3600
WRAPUP_BUFFER_S = 600

# Failed atoms whose traceback contains one of these count as 'oom' (cutoff) instead of 'failed'
OOM_MARKERS = ('MemoryError', 'TerminatedWorkerError', 'OutOfMemoryError', 'CUDA out of memory')

SMOKE = {'max_samples': 2, 'max_events': 1000, 'n_test': 2, 'test_frac': 0.01}  # --smoke

# Array = one SLURM array job per (env, cpu/gpu): env, device, methods (subset of benchmark.py's arrays)
ARRAYS = OrderedDict([
    ('sklearn_cpu', ('fx_sklearn', 'cpu', ['rf', 'hgb'])),
    ('flagx_cpu', ('fx_flagx_cpu', 'cpu', ['som'])),
    ('flagx_gpu', ('fx_flagx_gpu', 'gpu', ['mlp'])),
])
########################################################################################################################

FINAL = ('ok', 'failed', 'timeout', 'oom', 'skipped')
CUTOFF = ('timeout', 'oom')
INF = float('inf')


# ### Grid, names and paths ############################################################################################

def size_key(v):
    """Grid value ('all' or int, also as str from the manifest) -> number for comparisons ('all' = inf)."""
    return INF if str(v) == 'all' else int(v)


def cell_stem(split, trial, n_samples, n_events):
    return os.path.join('split_{}'.format(split), 'trial_{}'.format(trial),
                        'ns_{}__ne_{}'.format(n_samples, n_events))


def flat_stem(split, trial, n_samples, n_events):
    return 'split_{}__trial_{}__ns_{}__ne_{}'.format(split, trial, n_samples, n_events)


def atom_id(method, dataset, split, trial, n_samples, n_events):
    return '{}__{}__{}'.format(method, dataset, flat_stem(split, trial, n_samples, n_events))


def atom_paths(out, row):
    """(json, counts csv, log) of a manifest row."""
    stem = cell_stem(row['split'], row['trial'], row['n_samples'], row['n_events'])
    rel = os.path.join(row['method'], row['dataset'])
    return (os.path.join(out, 'atoms', rel, stem + '.json'), os.path.join(out, 'counts', rel, stem + '.csv'),
            os.path.join(out, 'logs', rel, stem + '.log'))


def cutoff_dir(out, method, dataset):
    return os.path.join(out, 'cutoffs', method, dataset)


def sample_order(dataset, split, trial, train):
    """Per-trial permutation of the outer training samples (the first n are the training samples of n_samples=n)."""
    rng = np.random.RandomState(derive_seed('train_req', dataset, split, trial))
    return [train[i] for i in rng.permutation(len(train))]


def event_order(y, seed, stratified=STRATIFIED):
    """
    Fixed random order of the events of one sample; n_events = the first n of it, so the event subsets are nested
    over n_events (validation.core.sampling.downsample_indices is not: its random stream shifts with n).

    Stratified: a random permutation of the events; within each class (in permutation order) the j-th of m_c events
    gets the rank (j + 0.5) / m_c, and the permutation is stably sorted by rank (ties keep the random order). The first
    n events then hold n * m_c / m events per class up to ~1 event (a class appears once its quota reaches ~0.5, as
    with downsample_indices). Not stratified: the random permutation itself.
    """
    y = np.asarray(y)
    perm = np.random.RandomState(seed).permutation(y.shape[0])
    if not stratified:
        return perm
    yp = y[perm]  # Rearange events into permutatioun order
    by_class = np.argsort(yp, kind='mergesort')  # Indices that sort by class label, random order within a class block
    _, counts = np.unique(yp, return_counts=True)  # Class counts
    rank = np.empty(y.shape[0])
    start = 0
    for m in counts:  # Iterate over class sizes
        # by_class[start:start + m] selects part of sorted indices belonging to current class
        # each event gets a normalized rank within its class members, store in sorted block order 
        # -> can use by_class to access
        rank[by_class[start:start + m]] = (np.arange(m) + 0.5) / m
        start += m
    # By construction, perm, yp, and rank use the same event order
    # np.argsort(rank, kind='mergesort') sort all events by rank, this gives a stratified reordering
    # reorder perm to the stratified order
    return perm[np.argsort(rank, kind='mergesort')]


def sample_sizes(dataset):
    """Number of events per sample (from the .npy headers, nothing is read)."""
    d = os.path.join(config.DATA_CURATED, dataset)
    return dict((f[:-len('_y.npy')], int(np.load(os.path.join(d, f), mmap_mode='r').shape[0]))
                for f in os.listdir(d) if f.endswith('_y.npy'))


def lane_cells(dataset, order, sizes):
    """
    Grid cells (n_samples, n_events, n_train_events) of one lane, small -> large, without duplicates of 'all' and
    without ('all', 'all') (main benchmark).
    """
    cells = []
    for ns in N_SAMPLES.get(dataset, N_SAMPLES['default']):
        if ns != 'all' and ns >= len(order):
            continue  # Same as 'all'
        selected = order if ns == 'all' else order[:ns]
        largest = max(sizes[s] for s in selected)
        for ne in N_EVENTS:
            if ne != 'all' and ne >= largest:
                continue  # Same as 'all' events
            if ns == 'all' and ne == 'all':
                continue  # Main benchmark (step 7)
            n_train = sum(sizes[s] if ne == 'all' else min(sizes[s], ne) for s in selected)
            cells.append((ns, ne, n_train))
    return cells


# ### Manifest #########################################################################################################

def methods_of(array):
    return [m for m in ARRAYS[array][2] if m in METHODS]


def write_manifest(array, out):
    methods = methods_of(array)
    sizes = dict((d, sample_sizes(d)) for d in DATASETS)
    rows, index, lane = [], 0, 0
    # Lane order: split -> dataset -> method -> trial (fold 0 first); atoms small -> large within a lane
    for split in range(max(max(outer_splits(d)) for d in DATASETS) + 1):
        for dataset in DATASETS:
            if split not in outer_splits(dataset):
                continue
            train = load_split(dataset, 'split_{}'.format(split))['train']
            for method in methods:
                cid, params, source = best_config(method, dataset, split, BEST_CONFIGS, TUNING_DATASET)
                for trial in range(N_TRIALS):
                    order = sample_order(dataset, split, trial, train)
                    for ns, ne, n_train in lane_cells(dataset, order, sizes[dataset]):
                        rows.append([index, lane, atom_id(method, dataset, split, trial, ns, ne), method, dataset,
                                     split, trial, ns, ne, len(train), n_train, cid, source,
                                     json.dumps(params, sort_keys=True)])
                        index += 1
                    lane += 1
    path = os.path.join(out, 'manifests', array + '.csv')
    save_csv(path, ['index', 'lane', 'atom_id', 'method', 'dataset', 'split', 'trial', 'n_samples', 'n_events',
                    'n_train_samples_split', 'n_train_events', 'config_id', 'param_source', 'params'], rows)
    log_dir = os.path.join(out, 'logs', 'tdr_' + array)  # --output of the SLURM script; must exist at submission
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print('{}: {} lanes, {} atoms ({}), manifest {}'.format(array, lane, len(rows), ', '.join(
        '{} {}'.format(m, sum(1 for r in rows if r[3] == m)) for m in methods), path))
    print('SLURM: --array=0-{}'.format(lane - 1))


def manifest_rows(array, out):
    return read_manifest(os.path.join(out, 'manifests', array + '.csv'))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cutoffs(out, method, dataset):
    """[(n_train_events, flat stem)] of the timeout / oom atoms of a method x dataset (marker files), smallest first."""
    d = cutoff_dir(out, method, dataset)
    if not os.path.isdir(d):
        return []
    res = []
    for f in os.listdir(d):
        if f.endswith('.json'):
            res.append((int(load_json(os.path.join(d, f))['n_train_events']), f[:-len('.json')]))
    return sorted(res)


def dominated_by(out, row):
    """
    Atom id of a timeout / oom atom (same method x dataset, any split / trial) with at most as many total training
    events as this atom, else None. Ties are dominated: the same cell in the other lanes is skipped (a cell with any
    timeout is not eligible in the eval anyway).
    """
    for n_cut, stem in cutoffs(out, row['method'], row['dataset']):
        if int(row['n_train_events']) >= n_cut:
            return '{}__{}__{}'.format(row['method'], row['dataset'], stem)
        break  # Sorted: the smallest cutoff decides
    return None


def in_smoke(row):
    return (size_key(row['n_samples']) <= SMOKE['max_samples'] and size_key(row['n_events']) <= SMOKE['max_events'])


# ### One atom (child process) #########################################################################################

def run_atom(array, row, out, smoke=False):
    env, device, _ = ARRAYS[array]
    method, dataset = row['method'], row['dataset']
    split, trial = int(row['split']), int(row['trial'])
    ns, ne = row['n_samples'], row['n_events']
    params = json.loads(row['params'])
    json_path, counts_path, _ = atom_paths(out, row)

    column = 'split_{}'.format(split)
    model_seed = derive_seed('train_req', method, dataset, split, trial)
    info = OrderedDict([
        ('atom_id', row['atom_id']), ('method', method), ('array', array), ('env', env), ('device', device),
        ('dataset', dataset), ('split', split), ('trial', trial), ('n_samples', size_value(ns)),
        ('n_events', size_value(ne)), ('split_column', column),
        ('config_id', row['config_id']), ('param_source', row['param_source']), ('params', params),
        ('status', 'started'), ('phase', 'fit'), ('error', None), ('stratified', STRATIFIED), ('smoke', smoke),
        ('seeds', {'model': model_seed, 'order': derive_seed('train_req', dataset, split, trial),
                   'data': "derive_seed('train_req', dataset, split, trial, sample)",
                   'join': derive_seed('train_req', dataset, split, trial, size_value(ns), size_value(ne), 'join')}),
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
        order = sample_order(dataset, split, trial, roles['train'])
        train = order if ns == 'all' else order[:int(ns)]
        test = roles['test'][:SMOKE['n_test']] if smoke else roles['test']
        labels_meta = load_meta_labels(dataset)
        labels = [r['label'] for r in labels_meta]
        info.update([('train_samples', train), ('test_samples', test), ('labels', labels_meta)])

        # Training data: first n samples of the trial's order, each downsampled to n_events (memory-mapped, only the
        # selected rows are read), joined and shuffled
        Xs, ys, n_train_sample, n_train_sample_class = [], [], OrderedDict(), OrderedDict()
        for s in train:
            X, y = load_sample(dataset, s, mmap=True)
            y = np.asarray(y)
            n_train_sample[s] = int(y.shape[0])
            if ne == 'all':
                X = np.asarray(X)
            else:
                idx = np.sort(event_order(y, derive_seed('train_req', dataset, split, trial, s))[:int(ne)])
                X, y = np.asarray(X[idx]), y[idx]
            n_train_sample_class[s] = OrderedDict((str(l), int(np.sum(y == l))) for l in labels)
            Xs.append(X)
            ys.append(y)
        X, y, sample_ids = join_samples(Xs, ys, train, shuffle=True, seed=info['seeds']['join'])
        del Xs, ys
        info['n_train_events_sample'] = n_train_sample  # Before downsampling
        info['n_train_events'] = int(y.shape[0])  # Used for training
        info['n_train_events_per_class'] = OrderedDict((str(l), int(np.sum(y == l))) for l in labels)
        info['n_train_events_sample_per_class'] = n_train_sample_class  # Used for training, per sample

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
        rows, res_pred = [], OrderedDict()
        info.update([('phase', 'predict'), ('n_test_done', 0),
                     ('resources', OrderedDict([('fit', res_fit), ('predict', res_pred)]))])
        save_json(json_path, info)
        print('{} fit done ({:.1f} s)'.format(now(), res_fit['wall_s']))
        sys.stdout.flush()

        # Test samples: predicted in full (smoke: downsampled); counts and JSON written after each sample
        header = (['method', 'dataset', 'split', 'trial', 'n_samples', 'n_events', 'config_id', 'sample',
                   'n_events_sample', 'n_events_used', 'label', 'n_true', 'n_pred', 'n_abstained', 'tp', 'fp', 'fn',
                   'tn'] + ['pred_{}'.format(l) for l in labels] + ['pred_abstained'])
        for s in test:
            Xv, yv = load_sample(dataset, s)
            n_sample = int(yv.shape[0])
            if smoke:
                idx = downsample_indices(yv, frac=SMOKE['test_frac'], stratified=True,
                                         seed=derive_seed('train_req', dataset, split, trial, s))
                Xv, yv = Xv[idx], yv[idx]
            with track_resources(gpu=gpu) as r:
                y_pred, _ = runner.predict(model, pre.transform(Xv), proba=False)
            res_pred[s] = r
            cm = confusion_counts(yv, y_pred, labels)
            cc = class_counts(cm)
            for i, l in enumerate(labels):
                rows.append([method, dataset, split, trial, ns, ne, row['config_id'], s, n_sample, int(yv.shape[0]), l]
                            + [int(cc[k][i]) for k in ('n_true', 'n_pred', 'n_abstained', 'tp', 'fp', 'fn', 'tn')]
                            + [int(v) for v in cm[i]])
            save_csv(counts_path, header, rows)
            info['n_test_done'] = len(res_pred)
            save_json(json_path, info)
            print('{} predicted {} ({:.1f} s)'.format(now(), s, r['wall_s']))
            sys.stdout.flush()

        save_csv(counts_path, header, rows)  # Also with 0 test samples
        info.update([('status', 'ok'), ('phase', 'done')])
    except Exception:
        info['status'] = 'failed'
        info['error'] = traceback.format_exc()
        print(info['error'])
    info['end'] = now()
    save_json(json_path, info)  # Written last: status ok only after the counts exist
    print('{} {} {}'.format(now(), info['status'], row['atom_id']))
    sys.stdout.flush()


def size_value(v):
    """Manifest value -> 'all' or int (for the JSON and seeds)."""
    return 'all' if str(v) == 'all' else int(v)


# ### Lane worker ######################################################################################################

def kill_group(proc):
    """Kills the child's whole process group (child + its workers, e.g. loky / DataLoader)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:  # Already gone
        pass


def run_child(array, row, out, smoke, atom_limit):
    """Runs one atom as a child process with a time limit. Returns (returncode, timed_out, wall_s)."""
    log_path = atom_paths(out, row)[2]
    if not os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    cmd = [sys.executable, os.path.abspath(__file__), 'atom', array, row['index'], '--out', out]
    if smoke:
        cmd.append('--smoke')
    t0 = time.time()
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=config.ROOT, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=atom_limit)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:  # Also on timeout, Ctrl-C or SIGTERM of the worker: no orphaned processes
            kill_group(proc)
            proc.wait()
    return proc.returncode, timed_out, time.time() - t0


def finish_atom(out, row, returncode, timed_out, wall_s, atom_limit):
    """Final status of an atom from the child's JSON and the way the child ended; cutoff marker for timeout / oom."""
    json_path = atom_paths(out, row)[0]
    info = load_json(json_path) or OrderedDict([('atom_id', row['atom_id']), ('status', 'started'), ('phase', None),
                                                ('error', 'child wrote no JSON')])
    error = info.get('error') or ''
    if info.get('status') == 'ok':
        status = 'ok'  # Counts and final JSON written (a kill during interpreter shutdown does not matter)
    elif timed_out:
        status = 'timeout'
    elif returncode is not None and returncode < 0:
        status = 'oom'  # Killed by a signal (SIGKILL of the cgroup OOM killer)
    elif info.get('status') == 'failed' and any(m in error for m in OOM_MARKERS):
        status = 'oom'
    else:
        status = 'failed'
    info['status'] = status
    info['end'] = info.get('end') or now()
    info['worker'] = OrderedDict([('returncode', returncode), ('timed_out', timed_out), ('wall_s', round(wall_s, 2)),
                                  ('atom_time_limit_s', atom_limit), ('log', os.path.relpath(atom_paths(out, row)[2],
                                                                                              out))])
    save_json(json_path, info)
    if status in CUTOFF:
        save_json(os.path.join(cutoff_dir(out, row['method'], row['dataset']),
                               flat_stem(row['split'], row['trial'], row['n_samples'], row['n_events']) + '.json'),
                  OrderedDict([('atom_id', row['atom_id']), ('status', status), ('phase', info.get('phase')),
                               ('n_train_events', int(row['n_train_events'])), ('time', now())]))
    return status


def slurm_time_left():
    """Seconds left of the current SLURM job (squeue %L: [D-]HH:MM:SS or MM:SS), None outside SLURM or on failure."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return None
    try:
        s = subprocess.check_output(['squeue', '-j', job, '-h', '-o', '%L'], universal_newlines=True,
                                    timeout=60).strip()  # The controller can hang
        days = 0
        if '-' in s:
            d, s = s.split('-', 1)
            days = int(d)
        parts = [int(p) for p in s.split(':')]
        parts = [0] * (3 - len(parts)) + parts
        return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print('{} could not read the SLURM time left ({!r}); using LANE_TIME_LIMIT_S'.format(now(), e))
        return None


def run_lane(array, lane_rows, out, smoke, atom_limit, deadline):
    """Runs the lane's atoms; starts an atom only if it can finish (incl. WRAPUP_BUFFER_S) before deadline (epoch s)."""
    lane = lane_rows[0]['lane']
    print('{} lane {} ({} atoms): {} {} split {} trial {}'.format(
        now(), lane, len(lane_rows), lane_rows[0]['method'], lane_rows[0]['dataset'], lane_rows[0]['split'],
        lane_rows[0]['trial']))
    sys.stdout.flush()
    for row in lane_rows:
        if smoke and not in_smoke(row):
            continue
        json_path = atom_paths(out, row)[0]
        info = load_json(json_path)
        if info is not None and info.get('status') in FINAL and not OVERWRITE:
            continue
        dom = dominated_by(out, row)
        if dom is not None:
            save_json(json_path, OrderedDict([('atom_id', row['atom_id']), ('status', 'skipped'),
                                              ('skipped_by', dom), ('time', now())]))
            print('{} skipped {} (cutoff {})'.format(now(), row['atom_id'], dom))
            continue
        if time.time() + atom_limit + WRAPUP_BUFFER_S > deadline:
            print('{} lane {}: not enough time left for {}; stopping (resubmit via "missing")'.format(
                now(), lane, row['atom_id']))
            return
        rc, timed_out, wall = run_child(array, row, out, smoke, atom_limit)
        status = finish_atom(out, row, rc, timed_out, wall, atom_limit)
        print('{} {} {} ({:.1f} s, rc {})'.format(now(), status, row['atom_id'], wall, rc))
        sys.stdout.flush()
    print('{} lane {} done'.format(now(), lane))


def lanes(manifest):
    res = OrderedDict()
    for r in manifest:
        res.setdefault(int(r['lane']), []).append(r)
    return res


# ### Status ###########################################################################################################

def atom_state(out, row):
    """Final status, 'dominated' (no JSON / started, but a cutoff covers it), 'started' or 'missing'."""
    info = load_json(atom_paths(out, row)[0])
    if info is not None and info.get('status') in FINAL:
        return info['status'], info
    if dominated_by(out, row) is not None:
        return 'dominated', info
    return ('missing' if info is None else 'started'), info


def print_missing(array, out):
    manifest = manifest_rows(array, out)
    todo = sorted(set(int(r['lane']) for r in manifest if atom_state(out, r)[0] in ('started', 'missing')))
    print('{}: {} of {} lanes not finished'.format(array, len(todo), len(lanes(manifest))))
    if todo:
        print('SLURM: --array={}'.format(index_ranges(todo)))


def print_status(array, out):
    manifest = manifest_rows(array, out)
    states = ('ok', 'failed', 'timeout', 'oom', 'skipped', 'dominated', 'started', 'missing')
    table, todo = OrderedDict(), set()
    for r in manifest:
        st, info = atom_state(out, r)
        e = table.setdefault((r['method'], r['dataset']), {'n': OrderedDict((s, 0) for s in states), 'fit_h': [],
                                                           'fit_gb': [], 'ok_events': []})
        e['n'][st] += 1
        if st == 'ok':
            e['ok_events'].append(int(r['n_train_events']))
        if st in ('started', 'missing'):
            todo.add(int(r['lane']))
        res = (info or {}).get('resources') or {}
        if res.get('fit'):
            e['fit_h'].append(res['fit']['wall_s'] / 3600.)
            e['fit_gb'].append(res['fit']['cpu_rss_peak_mb'] / 1e3)

    def mx(v, fmt):
        return fmt.format(max(v)) if v else '-'

    line = '{:<6} {:<12}' + ' {:>7}' * len(states) + ' {:>6} {:>7} {:>11} {:>8}'
    print(line.format('method', 'dataset', *(states + ('fit_h', 'fit_GB', 'cutoff_ev', 'ok>cut'))))
    for (method, dataset), e in table.items():
        cut = cutoffs(out, method, dataset)
        n_cut = cut[0][0] if cut else None
        above = sum(1 for n in e['ok_events'] if n_cut is not None and n >= n_cut)  # Post-hoc check
        print(line.format(method, dataset, *(list(e['n'].values()) + [
            mx(e['fit_h'], '{:.2f}'), mx(e['fit_gb'], '{:.1f}'), n_cut if n_cut is not None else '-',
            above if n_cut is not None else '-'])))
    print('{}: {} atoms, {} of {} lanes not finished'.format(array, len(manifest), len(todo), len(lanes(manifest))))
    if todo:
        print('not finished: --array={}'.format(index_ranges(sorted(todo))))


# ### Main #############################################################################################################

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command')
    for cmd in ('manifest', 'run', 'atom', 'status', 'missing'):
        p = sub.add_parser(cmd)
        p.add_argument('array', choices=list(ARRAYS))
        p.add_argument('--out', default=OUT, help='output root (default results/training_data_requirements)')
        if cmd == 'run':
            p.add_argument('lane', nargs='?', default=None, help="lane index, 'all', default $SLURM_ARRAY_TASK_ID")
            p.add_argument('--atom-limit', type=float, default=None, help='seconds per atom (tests, requires --out)')
            p.add_argument('--lane-limit', type=float, default=None, help='seconds per lane (tests, requires --out)')
        if cmd == 'atom':
            p.add_argument('index', type=int, help='atom index (manifest row)')
        if cmd in ('run', 'atom'):
            p.add_argument('--smoke', action='store_true', help='tiny test run (requires --out)')
    args = ap.parse_args()
    if args.command is None:
        ap.error('give a command')

    out = os.path.abspath(args.out)
    if args.command == 'manifest':
        write_manifest(args.array, out)
        return
    if args.command == 'status':
        print_status(args.array, out)
        return
    if args.command == 'missing':
        print_missing(args.array, out)
        return

    scratch_only = args.smoke or getattr(args, 'atom_limit', None) is not None or \
        getattr(args, 'lane_limit', None) is not None
    if scratch_only and out == os.path.abspath(OUT):
        raise SystemExit('--smoke / --atom-limit / --lane-limit require --out with a directory other than {}'.format(
            OUT))
    manifest = manifest_rows(args.array, out)
    if args.command == 'atom':
        run_atom(args.array, manifest[args.index], out, smoke=args.smoke)
        return

    # Worker: SIGTERM (e.g. scancel) -> SystemExit, so the running child's process group is killed in run_child
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    lane = args.lane if args.lane is not None else os.environ.get('SLURM_ARRAY_TASK_ID')
    if lane is None:
        raise SystemExit("give a lane index or 'all' (or set SLURM_ARRAY_TASK_ID)")
    by_lane = lanes(manifest)
    selected = list(by_lane) if lane == 'all' else [int(lane)]
    atom_limit = args.atom_limit if args.atom_limit is not None else ATOM_TIME_LIMIT_S
    # Deadline: --lane-limit (tests) per lane; under SLURM the end of the job (shared by all selected lanes);
    # otherwise LANE_TIME_LIMIT_S per lane
    left = None if args.lane_limit is not None else slurm_time_left()
    if left is not None:
        job_end = time.time() + left
        print('{} SLURM job {}: {:.2f} h left'.format(now(), os.environ.get('SLURM_JOB_ID'), left / 3600.))
    for l in selected:
        if left is not None:
            deadline = job_end
        else:
            deadline = time.time() + (args.lane_limit if args.lane_limit is not None else LANE_TIME_LIMIT_S)
        run_lane(args.array, by_lane[l], out, args.smoke, atom_limit, deadline)


if __name__ == '__main__':
    main()
