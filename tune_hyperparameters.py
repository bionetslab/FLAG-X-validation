"""
Hyperparameter tuning on the Imstat inner splits (PLAN.md step 6, working_notes/tuning_plan.md).

One atom = (method, config, outer split, inner split): fit on the inner train samples (all events, TRAIN_FRAC = 1.0),
predict every inner val sample, store per val sample the evaluation counts and the time / memory of fit and predict.
Scores, selection (per fold / consensus, D6) and plots are computed later from the atoms.

Env: the env of the array (ARRAYS below), e.g. fx_sklearn for 'sklearn_cpu'. Python 3.5-compatible (fx_deepcytof).

Arrays:
    The methods need different conda envs and some a GPU, so the atoms are split into arrays, one per (env, cpu/gpu)
    combination (ARRAYS below): sklearn_cpu, flagx_cpu, gatemeclass_cpu, lda_cpu, deepcytof_cpu (Woody) and
    flagx_gpu (TinyGPU). Each array has its own manifest (task list, numbered from 0) and its own SLURM script
    (slurm_scripts/tune_<array>.sh), which activates the array's env and runs 'run <array>'. Array task i runs row i
    of manifests/<array>.csv. The arrays are independent: submit them in any order, also in parallel.

Commands:
    python tune_hyperparameters.py manifest <array> [--out DIR]
        Expands the grids of the array's methods, writes <out>/configs/<method>.csv (config_id, grid position,
        is_default, params) and the frozen task list <out>/manifests/<array>.csv (index, atom_id, method, config_id,
        split, inner, params), creates the SLURM log dir and prints the number of atoms N (SLURM: --array=0-<N-1>).
        Any env with numpy works (no method code is imported).
    python tune_hyperparameters.py run <array> [<index> | all] [--out DIR] [--smoke]
        Runs atom <index> of the manifest (default: $SLURM_ARRAY_TASK_ID), or all atoms sequentially ('all').
        Must run in the array's env. --smoke: 3 train / 2 val samples, 1 % of the events (only with --out, for
        testing).

Workflow (from the repo root, with the code state the jobs should run committed):
    1. python tune_hyperparameters.py manifest sklearn_cpu      # prints N, e.g. 3405
    2. sbatch --array=0-3404%50 slurm_scripts/tune_sklearn_cpu.sh                  # Woody (%50: max. running tasks)
       sbatch.tinygpu --array=0-1889%20 slurm_scripts/tune_flagx_gpu.sh            # TinyGPU, for 'flagx_gpu'
    3. Resume / rerun missing atoms: submit the same command again; atoms with a JSON are skipped.
    Repeat 1-2 for every array. Write the manifest right before submitting and do not rewrite it while the array
    runs: the task indices refer to the frozen file, a changed grid would shift them. After a grid change, write a new
    manifest and submit again; finished atoms are kept, because config_id is a hash of the params.
    Local test: python tune_hyperparameters.py manifest lda_cpu --out <scratch_dir> && \\
                python tune_hyperparameters.py run lda_cpu 0 --out <scratch_dir> --smoke
    Details and cluster setup: slurm_scripts/README.md.

Outputs (<out> = results/tune_hyperparameters by default):
    atoms/<method>/<config_id>/split_<i>_inner_<j>.json: everything about the atom (identity, params, data, seeds,
        preprocessing, resources of fit and of each val sample's predict, run metadata) and its status:
        'started' (written first; stays if the job was killed: timeout, OOM, node failure), 'ok', 'failed' (+
        traceback). Any status other than 'ok' puts the config out (working_notes/tuning_plan.md).
    atoms/<method>/<config_id>/split_<i>_inner_<j>_counts.csv (only if ok): one row per val sample x true class with
        the identifiers, per-class counts (n_true, n_pred, n_abstained, tp, fp, fn, tn), the confusion row
        (pred_<label>..., pred_abstained). Prediction performance only; all time / memory values are in the JSON.
        Concatenates directly.
An atom with an existing JSON is skipped (OVERWRITE = False); delete the JSON to rerun it.
"""

import argparse
import hashlib
import importlib
import itertools
import json
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np

import config
from validation.core.data import load_meta_labels, load_sample, load_split
from validation.core.evaluation import class_counts, confusion_counts
from validation.core.preprocessing import Preprocessor
from validation.core.resources import track_resources
from validation.core.runs import n_threads, now, run_metadata, save_csv, save_json
from validation.core.sampling import downsample_indices, join_samples
from validation.core.seeding import derive_seed

# ### Parameters #######################################################################################################
DATASET = 'imstat'
OUT = os.path.join(config.RESULTS, 'tune_hyperparameters')
OVERWRITE = False

TRAIN_FRAC = 1.0  # Fraction of the events kept per training sample (class-stratified); 1.0 = all, as in the benchmark
VAL_FRAC = 1.0  # Fraction of the events kept per val sample (class-stratified); 1.0 = no downsampling
STRATIFIED = True

SMOKE = {'n_train': 3, 'n_val': 2, 'train_frac': 0.01, 'val_frac': 0.01}  # --smoke

# Array = one SLURM array job per (env, cpu/gpu): env, device, methods
ARRAYS = OrderedDict([
    ('sklearn_cpu', ('fx_sklearn', 'cpu', ['rf', 'hgb', 'logreg', 'knn', 'lda_sklearn', 'gmm_mocking'])),
    ('flagx_cpu', ('fx_flagx_cpu', 'cpu', ['som'])),
    ('flagx_gpu', ('fx_flagx_gpu', 'gpu', ['mlp', 'dgcytof'])),
    ('gatemeclass_cpu', ('fx_gatemeclass', 'cpu', ['gatemeclass'])),
    ('lda_cpu', ('fx_lda', 'cpu', ['lda_abdelaal'])),
    ('deepcytof_cpu', ('fx_deepcytof', 'cpu', ['deepcytof'])),
])

# Grids: per method a list of blocks; a block is a list of (param, values) pairs whose product gives configs (blocks
# allow conditional params, e.g. different fixed values per training schedule). FIXED params are added to every
# config. DEFAULT = the default / published config (must be in the grid; listed first in the config table).
# Param names are the runner's (validation/runners/<method>.py), mostly the wrapper / sklearn argument names.
# Justification of every grid (values, fixed params, defaults, sources): working_notes/tuning_plan.md, section 6.
GRIDS = OrderedDict([
    ('som', {
        'blocks': [
            [('som_dimensions', [[15, 15], [20, 20], [25, 25], [30, 30]]),
             ('n_epochs', [50, 100, 250, 500, 750, 1000, 2000])],
        ],
        'fixed': {'som_topology': 'planar', 'som_grid_type': 'rectangular', 'neighborhood': 'gaussian',
                  'gaussian_neighborhood_sigma': 0.1, 'initialization': 'pca', 'radius_cooling': 'exponential',
                  'learning_rate_decay': 'exponential', 'learning_rate_0': 0.1, 'learning_rate_n': 0.001,
                  'radius_0': -0.5, 'radius_n': 0.1},
        # Paper (legacy/main.py L928-943)
        'default': {'som_dimensions': [25, 25], 'n_epochs': 1000},
    }),
    ('mlp', {
        'blocks': [
            [('layer_sizes', [[64, 32, 16], [128, 64, 32], [256, 128, 64]]), ('lr', [1e-4, 1e-3, 1e-2]),
             ('batch_size', [128, 1024]),
             ('n_epochs', [20]), ('early_stopping', [False]),
             ('validation_fraction', [0.0])],
            [('layer_sizes', [[64, 32, 16], [128, 64, 32], [256, 128, 64]]), ('lr', [1e-4, 1e-3, 1e-2]),
             ('batch_size', [128, 1024]),
             ('n_epochs', [1000]), ('early_stopping', [True]),
             ('validation_fraction', [0.1])],
        ],
        'fixed': {'n_iter_no_change': 5},
        # Paper (legacy/main.py main_fcnn)
        'default': {'layer_sizes': [128, 64, 32], 'lr': 1e-3, 'batch_size': 128, 'n_epochs': 20,
                    'early_stopping': False, 'validation_fraction': 0.0},
    }),
    ('dgcytof', {
        'blocks': [[('layer_sizes', [[64, 32, 16], [128, 64, 32], [256, 128, 64]]), ('n_epochs', [10, 20, 50]),
                    ('lr', [1e-4, 1e-3, 1e-2]), ('batch_size', [128, 1024])]],
        'fixed': {'val_size': 0.2},
        'default': {'layer_sizes': [128, 64, 32], 'n_epochs': 20, 'lr': 1e-3, 'batch_size': 128},
    }),
    ('gatemeclass', {
        'blocks': [[('gmc_gmm_parameterization', ['E', 'V']), ('gmc_k', [10, 20, 40]),
                    ('gmc_sampling', [0.05, 0.1, 0.2])]],
        'fixed': {'gmc_sampling_imp_vars': 0.05},
        'default': {'gmc_gmm_parameterization': 'E', 'gmc_k': 20, 'gmc_sampling': 0.1},
    }),
    ('lda_abdelaal', {
        'blocks': [[('transformation', [False])]],
        'fixed': {},
        'default': {'transformation': False},
    }),
    ('deepcytof', {
        'blocks': [[('preprocess_log', [False, True]), ('calibrate', [True, False]),
                    ('hidden_layer_sizes', [[12, 6, 3], [128, 64, 32]]),
                    ('lr_initial', [1e-4, 1e-3, 1e-2])]],
        'fixed': {'denoise': False},
        'default': {'preprocess_log': True, 'calibrate': True, 'hidden_layer_sizes': [12, 6, 3], 'lr_initial': 1e-3},
    }),
    ('rf', {
        'blocks': [[('max_depth', [10, 20, None]), ('min_samples_leaf', [1, 10, 100]),
                    ('max_features', ['sqrt', 0.5, None]), ('criterion', ['gini', 'entropy'])]],
        'fixed': {'n_estimators': 100, 'class_weight': None},
        'default': {'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'criterion': 'gini'},
    }),
    ('hgb', {
        'blocks': [[('learning_rate', [0.01, 0.1, 0.3]), ('max_leaf_nodes', [15, 31, 63]),
                    ('l2_regularization', [0.0, 1.0, 10.0])]],
        'fixed': {'max_iter': 1000, 'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 10,
                  'class_weight': None},
        'default': {'learning_rate': 0.1, 'max_leaf_nodes': 31, 'l2_regularization': 0.0},
    }),
    ('logreg', {
        # C=None: no penalty (runner: C=inf)
        'blocks': [[('C', [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, None])]],
        'fixed': {'max_iter': 1000, 'class_weight': None},
        'default': {'C': 1.0},
    }),
    ('knn', {
        'blocks': [[('n_neighbors', [1, 5, 15, 30, 50]), ('weights', ['uniform', 'distance']), ('p', [1, 2])]],
        'fixed': {},
        'default': {'n_neighbors': 5, 'weights': 'uniform', 'p': 2},
    }),
    ('lda_sklearn', {
        'blocks': [[('shrinkage', [None, 'auto', 0.1, 0.5, 0.9, 1.0])]],
        'fixed': {'solver': 'lsqr'},
        'default': {'shrinkage': None},
    }),
    ('gmm_mocking', {
        'blocks': [[('n_components', [1, 2, 5, 10, 20]), ('covariance_type', ['full', 'tied', 'diag', 'spherical'])]],
        'fixed': {},
        # Original GMMClassifier default (1 component per class, full covariance)
        'default': {'n_components': 1, 'covariance_type': 'full'},
    }),
])
########################################################################################################################


# ### Grids and manifest ###############################################################################################

def config_id(params):
    """Short hash of the canonical params JSON; stable when other grid points change."""
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode('utf-8')).hexdigest()[:10]


def expand_grid(method):
    """List of (config_id, params, is_default), default first, then grid order; duplicates removed."""
    g = GRIDS[method]
    default = dict(g['fixed'], **g['default'])
    out, seen = [], set()
    configs = []
    for block in g['blocks']:
        keys = [k for k, _ in block]
        for values in itertools.product(*[v for _, v in block]):
            configs.append(dict(g['fixed'], **dict(zip(keys, values))))
    if default not in configs:
        raise ValueError('default config of {} is not in its grid: {}'.format(method, default))
    for params in [default] + configs:
        cid = config_id(params)
        if cid not in seen:
            seen.add(cid)
            out.append((cid, params, params == default))
    return out


def inner_splits():
    """[(outer split, inner split)] from the columns of <DATASET>_inner.csv."""
    with open(os.path.join(config.DATA_SPLITS, DATASET + '_inner.csv')) as f:
        header = f.readline().strip().split(',')
    pairs = []
    for c in header[1:]:
        parts = c.split('_')  # split_<i>_inner_<j>
        pairs.append((int(parts[1]), int(parts[3])))
    return sorted(pairs)


def atom_id(method, cid, split, inner):
    return '{}__{}__split_{}_inner_{}'.format(method, cid, split, inner)


def write_manifest(array, out):
    env, device, methods = ARRAYS[array]
    rows, index = [], 0
    for method in methods:
        configs = expand_grid(method)
        save_csv(os.path.join(out, 'configs', method + '.csv'), ['config_id', 'position', 'is_default', 'params'],
                 [[cid, i, d, json.dumps(p, sort_keys=True)] for i, (cid, p, d) in enumerate(configs)])
        for cid, params, _ in configs:
            for split, inner in inner_splits():
                rows.append([index, atom_id(method, cid, split, inner), method, cid, split, inner,
                             json.dumps(params, sort_keys=True)])
                index += 1
    path = os.path.join(out, 'manifests', array + '.csv')
    save_csv(path, ['index', 'atom_id', 'method', 'config_id', 'split', 'inner', 'params'], rows)
    # Log dir of the SLURM script (--output=results/tune_hyperparameters/logs/%x/...; must exist at submission)
    log_dir = os.path.join(out, 'logs', 'tune_' + array)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print('{}: {} atoms ({}), manifest {}'.format(array, len(rows), ', '.join(
        '{} {}'.format(m, sum(1 for r in rows if r[2] == m)) for m in methods), path))
    print('SLURM: --array=0-{}'.format(len(rows) - 1))


def read_manifest(array, out):
    import csv
    path = os.path.join(out, 'manifests', array + '.csv')
    if not os.path.exists(path):
        raise SystemExit('no manifest {}: run "python tune_hyperparameters.py manifest {}" first'.format(path, array))
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# ### One atom #########################################################################################################

def atom_paths(out, method, cid, split, inner):
    base = os.path.join(out, 'atoms', method, cid, 'split_{}_inner_{}'.format(split, inner))
    return base + '.json', base + '_counts.csv'


def make_preprocessor(arm):
    if arm == 'shared':
        return Preprocessor(config.CHANNELS[DATASET], 'log10_cutoff', {'cutoff': config.LOG10_CUTOFFS[DATASET]})
    if arm == 'native':
        return Preprocessor(config.CHANNELS[DATASET], 'none')
    raise ValueError('unknown input arm {!r}'.format(arm))


def preprocessor_state(p):
    return {'trafo': p.trafo, 'trafo_params': p.trafo_params, 'scaling': p.scaling,
            'scaling_params': p.scaling_params,
            'center_': None if p.center_ is None else p.center_.tolist(),
            'scale_': None if p.scale_ is None else p.scale_.tolist()}


def run_atom(array, row, out, smoke=False):
    env, device, _ = ARRAYS[array]
    method, cid = row['method'], row['config_id']
    split, inner = int(row['split']), int(row['inner'])
    params = json.loads(row['params'])
    json_path, counts_path = atom_paths(out, method, cid, split, inner)
    if os.path.exists(json_path) and not OVERWRITE:
        print('skip {} (exists)'.format(row['atom_id']))
        return

    train_frac = SMOKE['train_frac'] if smoke else TRAIN_FRAC
    val_frac = SMOKE['val_frac'] if smoke else VAL_FRAC
    column = 'split_{}_inner_{}'.format(split, inner)
    model_seed = derive_seed('tune', method, split, inner)
    info = OrderedDict([
        ('atom_id', row['atom_id']), ('method', method), ('array', array), ('env', env), ('device', device),
        ('config_id', cid), ('params', params), ('status', 'started'), ('error', None),
        ('dataset', DATASET), ('split', split), ('inner', inner), ('split_column', column),
        ('train_frac', train_frac), ('val_frac', val_frac), ('stratified', STRATIFIED), ('smoke', smoke),
        ('seeds', {'model': model_seed, 'data': "derive_seed('tune', dataset, split, inner, sample)",
                   'join': "derive_seed('tune', dataset, split, inner, 'join')"}),
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

        roles = load_split(DATASET, column, inner=True)
        train, val = roles['train'], roles['val']
        if smoke:
            train, val = train[:SMOKE['n_train']], val[:SMOKE['n_val']]
        labels_meta = load_meta_labels(DATASET)
        labels = [r['label'] for r in labels_meta]
        info.update([('train_samples', train), ('val_samples', val), ('labels', labels_meta)])

        # Training data: downsample each sample (seed per sample, the same for all methods / configs), join, shuffle
        Xs, ys = [], []
        for s in train:
            X, y = load_sample(DATASET, s)
            idx = downsample_indices(y, frac=train_frac, stratified=STRATIFIED,
                                     seed=derive_seed('tune', DATASET, split, inner, s))
            Xs.append(X[idx])
            ys.append(y[idx])
        X, y, sample_ids = join_samples(Xs, ys, train, shuffle=True,
                                        seed=derive_seed('tune', DATASET, split, inner, 'join'))
        del Xs, ys
        info['n_train_events'] = int(y.shape[0])
        # Training events per class (all classes of the dataset, 0 if absent), e.g. to spot classes lost by downsampling
        info['n_train_events_per_class'] = OrderedDict((str(l), int(np.sum(y == l))) for l in labels)

        arm = runner.input_arm(params)
        info.update([('input_arm', arm), ('n_threads', threads)])
        pre = make_preprocessor(arm)
        with track_resources(gpu=gpu) as res_fit:
            Xt = pre.fit_transform(X)
            model = runner.build(params, model_seed, DATASET, threads)
            model = runner.fit(model, Xt, y, sample_ids)
        if hasattr(runner, 'fit_info'):
            info['fit_info'] = runner.fit_info(model)
        del X, Xt
        info['preprocessor'] = preprocessor_state(pre)
        print('{} fit done ({:.1f} s)'.format(now(), res_fit['wall_s']))
        sys.stdout.flush()

        # Val samples: downsample (val_frac), predict, counts
        header = (['method', 'config_id', 'split', 'inner', 'sample', 'n_events_sample', 'n_events_used', 'label',
                   'n_true', 'n_pred', 'n_abstained', 'tp', 'fp', 'fn', 'tn']
                  + ['pred_{}'.format(l) for l in labels] + ['pred_abstained'])
        rows, res_pred = [], OrderedDict()
        for s in val:
            Xv, yv = load_sample(DATASET, s)
            idx = downsample_indices(yv, frac=val_frac, stratified=STRATIFIED,
                                     seed=derive_seed('tune', DATASET, split, inner, s))
            n_sample = int(yv.shape[0])
            Xv, yv = Xv[idx], yv[idx]
            with track_resources(gpu=gpu) as r:
                y_pred, _ = runner.predict(model, pre.transform(Xv), proba=False)
            res_pred[s] = r
            cm = confusion_counts(yv, y_pred, labels)
            cc = class_counts(cm)
            for i, l in enumerate(labels):
                rows.append([method, cid, split, inner, s, n_sample, int(yv.shape[0]), l]
                            + [int(cc[k][i]) for k in ('n_true', 'n_pred', 'n_abstained', 'tp', 'fp', 'fn', 'tn')]
                            + [int(v) for v in cm[i]])
        info['resources'] = {'fit': res_fit, 'predict': res_pred}

        save_csv(counts_path, header, rows)
        info['status'] = 'ok'
    except Exception:
        info['status'] = 'failed'
        info['error'] = traceback.format_exc()
        print(info['error'])
    info['end'] = now()
    save_json(json_path, info)  # Written last: status ok only after the counts file exists
    print('{} {} {}'.format(now(), info['status'], row['atom_id']))
    sys.stdout.flush()


# ### Main #############################################################################################################

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['manifest', 'run'])
    ap.add_argument('array', choices=list(ARRAYS))
    ap.add_argument('index', nargs='?', default=None, help="atom index, 'all', default $SLURM_ARRAY_TASK_ID")
    ap.add_argument('--out', default=OUT, help='output root (default results/tune_hyperparameters)')
    ap.add_argument('--smoke', action='store_true', help='tiny test run (requires --out)')
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    if args.command == 'manifest':
        write_manifest(args.array, out)
        return

    if args.smoke and out == os.path.abspath(OUT):
        raise SystemExit('--smoke requires --out with a directory other than {} (smoke atoms there would count as '
                         'finished and be skipped by the real runs); e.g. --out results/tune_smoke, after running '
                         'manifest with the same --out'.format(OUT))
    manifest = read_manifest(args.array, out)
    index = args.index if args.index is not None else os.environ.get('SLURM_ARRAY_TASK_ID')
    if index is None:
        raise SystemExit("give an atom index or 'all' (or set SLURM_ARRAY_TASK_ID)")
    rows = manifest if index == 'all' else [manifest[int(index)]]
    for row in rows:
        run_atom(args.array, row, out, smoke=args.smoke)


if __name__ == '__main__':
    main()
