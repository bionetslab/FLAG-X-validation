"""
Robustness of the minimal training set to the sample order (PLAN.md step 9,
working_notes/sample_order_robustness_plan.md): many new sample orders (trials) with 1 .. n_min training samples at the
events per sample of the step-8 minimal cell (n_min, ne_min) of every method x dataset. The eval gives the share of
trials that pass the step-8 criterion at the minimal cell and the per-trial minimal number of samples.

Mirrors training_data_requirements.py (step 8) on a smaller grid and imports its machinery (atom, lane worker, cutoffs,
status); see its docstring for the execution details. The trials stand on their own: numbered from 0 with their own
seeds, the step-8 trials are not reused.

Order schemes (one output root each, <out>/<scheme>/):
    random      a random permutation of the outer training samples per trial (as step 8, new seeds).
    stratified  round-robin over strata (STRATA; a meta_samples.csv column): the samples are shuffled within each
                stratum; every round takes the next sample of each stratum that has samples left, in a random stratum
                order drawn anew per round (so with n not a multiple of the number of strata, different strata are
                left out). Datasets without strata (STRATA None: Flowcyt) are not run in this arm (it would only
                repeat the random arm).

Atom = (method, dataset, outer split, trial, n_samples) with n_samples = 1 .. n_min (every integer; values >= the
split's training samples are dropped) and n_events = ne_min. Lane = (method, dataset, outer split, trial), order
split -> dataset -> method -> trial, atoms small -> large. Method x dataset without a minimal cell or with
n_min = 'all' are skipped (printed by 'manifest').

Minimal cell: MINIMAL_CONFIGS (step-8 eval, minimal_configs.json: primary criterion), the pooled cell with or without
the stability rule (STABLE).

Env: the env of the array (training_data_requirements.ARRAYS), e.g. fx_sklearn for 'sklearn_cpu'. Python
3.5-compatible.

Hyperparameters (D6): as step 8 (Imstat per outer split, the others consensus), frozen into the manifest.

Seeds (key = ('order_robust', scheme)): sample order derive_seed(*key, dataset, split, trial); event order per
training sample derive_seed(*key, dataset, split, trial, sample) (nested over n_samples); join derive_seed(*key,
dataset, split, trial, n_samples, n_events, 'join'); model derive_seed(*key, method, dataset, split, trial). The data
seeds do not depend on the method, so all methods see the same training data.

Commands (<scheme> = random | stratified):
    python sample_order_robustness.py manifest <scheme> <array> [--out DIR] [--configs JSON]
        Writes <out>/<scheme>/manifests/<array>.csv, creates the SLURM log dir, prints the number of lanes L.
    python sample_order_robustness.py run <scheme> <array> [<lane> | all] [--out DIR] [--smoke] [--atom-limit S]
                                      [--lane-limit S]
    python sample_order_robustness.py atom <scheme> <array> <index> [--out DIR] [--smoke]
    python sample_order_robustness.py status <scheme> <array> [--out DIR]
    python sample_order_robustness.py missing <scheme> <array> [--out DIR]
        As in training_data_requirements.py.

Outputs (<out> = results/sample_order_robustness by default): <out>/<scheme>/ with the layout of step 8 (manifests/,
atoms/, counts/, cutoffs/, logs/); the atom JSON also holds order_scheme. SLURM logs: <out>/logs/sor_<array>/.
"""
# TODO: before the run, estimate the lane time from pred_s in the step-8 atoms.csv (the full test-set prediction
#  dominates, e.g. BLAST110) and set the SLURM --time / the lane split accordingly.

import argparse
import json
import os
import signal
import sys
import time
from collections import OrderedDict

import numpy as np

import config
import training_data_requirements as tdr
from validation.core.data import load_meta_samples, load_split, outer_splits
from validation.core.runs import best_config, now, save_csv
from validation.core.seeding import derive_seed

# ### Parameters #######################################################################################################
DATASETS = config.DATASETS
OUT = os.path.join(config.RESULTS, 'sample_order_robustness')
MINIMAL_CONFIGS = os.path.join(config.RESULTS, 'training_data_requirements_eval', 'minimal_configs.json')
OVERWRITE = False  # False: finished atoms (JSON with a final status) are kept

# TODO: decide after the real step-8 eval: minimal cell with the stability rule (True) or the naive minimum (False)
STABLE = True

N_TRIALS = 30  # TODO: final number of trials (30 for now)

SCHEMES = ['random', 'stratified']

# Strata of the stratified arm (meta_samples.csv column); None = no strata (dataset not run in the stratified arm)
STRATA = {
    'imstat': 'lysis',
    'lt1': 'subtype', 'lt2': 'subtype', 'lt1b': 'subtype', 'lt2b': 'subtype',
    'flowcyt': None,
    'blast110_p1': 'sample_type', 'blast110_p2': 'sample_type', 'blast110_p3': 'sample_type',
    'blast110_p4': 'sample_type',
}

# As step 8: methods, arrays, per-atom time limit, lane fallback time
METHODS = tdr.METHODS
ARRAYS = tdr.ARRAYS
ATOM_TIME_LIMIT_S = tdr.ATOM_TIME_LIMIT_S
LANE_TIME_LIMIT_S = tdr.LANE_TIME_LIMIT_S
########################################################################################################################

SEED_TAG = 'order_robust'


def seed_key(scheme):
    return (SEED_TAG, scheme)


def scheme_out(out, scheme):
    return os.path.join(out, scheme)


# ### Sample orders ####################################################################################################

def random_order(scheme, dataset, split, trial, train):
    """Random permutation of the outer training samples."""
    rng = np.random.RandomState(derive_seed(*(seed_key(scheme) + (dataset, split, trial))))
    return [train[i] for i in rng.permutation(len(train))]


def stratified_order(scheme, dataset, split, trial, train):
    """
    Round-robin over the strata (STRATA[dataset]): samples shuffled within each stratum; per round the next sample of
    every stratum with samples left, in a random stratum order drawn anew per round.
    """
    rng = np.random.RandomState(derive_seed(*(seed_key(scheme) + (dataset, split, trial))))
    column = STRATA[dataset]
    stratum = dict((r['sample'], r[column]) for r in load_meta_samples(dataset))
    groups = OrderedDict()
    for s in train:  # Split file order; strata sorted by name below, so the result does not depend on dict order
        groups.setdefault(stratum[s], []).append(s)
    queues = []
    for k in sorted(groups):
        g = groups[k]
        queues.append([g[i] for i in rng.permutation(len(g))])
    order = []
    while any(queues):
        active = [q for q in queues if q]
        for i in rng.permutation(len(active)):
            order.append(active[i].pop(0))
    return order


def lane_order(scheme, dataset, split, trial, train):
    if scheme == 'random':
        return random_order(scheme, dataset, split, trial, train)
    return stratified_order(scheme, dataset, split, trial, train)


def scheme_datasets(scheme):
    return [d for d in DATASETS if scheme == 'random' or STRATA[d] is not None]


# ### Manifest #########################################################################################################

def minimal_cells(path, stable):
    """{(method, dataset): (n_samples, n_events)} of the pooled minimal cell (primary criterion), None if not found."""
    with open(path) as f:
        res = json.load(f)
    key = 'stable' if stable else 'unstable'
    cells = {}
    for method, by_ds in res['methods'].items():
        for dataset, entry in by_ds.items():
            c = entry[key]['pooled']
            cells[(method, dataset)] = None if c is None else (c['n_samples'], c['n_events'])
    print('minimal cells: {} (criterion {}, delta {}, order {}, {})'.format(
        path, res['criterion'], res['delta'], res['order'], key))
    return cells


def write_manifest(scheme, array, out, configs_path):
    methods = tdr.methods_of(array)
    datasets = scheme_datasets(scheme)
    cells = minimal_cells(configs_path, STABLE)
    sizes = dict((d, tdr.sample_sizes(d)) for d in datasets)
    skipped = OrderedDict()
    for dataset in datasets:
        for method in methods:
            c = cells.get((method, dataset))
            if c is None or str(c[0]) == 'all':
                skipped[(method, dataset)] = 'no minimal cell' if c is None else 'n_min = all'
    rows, index, lane = [], 0, 0
    # Lane order: split -> dataset -> method -> trial (fold 0 first); atoms small -> large within a lane
    for split in range(max(max(outer_splits(d)) for d in datasets) + 1):
        for dataset in datasets:
            if split not in outer_splits(dataset):
                continue
            train = load_split(dataset, 'split_{}'.format(split))['train']
            for method in methods:
                if (method, dataset) in skipped:
                    continue
                n_min, ne = cells[(method, dataset)]
                ne = tdr.size_value(ne)
                cid, params, source = best_config(method, dataset, split, tdr.BEST_CONFIGS, tdr.TUNING_DATASET)
                for trial in range(N_TRIALS):
                    order = lane_order(scheme, dataset, split, trial, train)
                    for ns in range(1, int(n_min) + 1):
                        if ns >= len(order):
                            break  # Same as 'all' samples: the order does not matter
                        n_train = sum(sizes[dataset][s] if ne == 'all' else min(sizes[dataset][s], ne)
                                      for s in order[:ns])
                        rows.append([index, lane, tdr.atom_id(method, dataset, split, trial, ns, ne), method, dataset,
                                     split, trial, ns, ne, len(train), n_train, cid, source,
                                     json.dumps(params, sort_keys=True), scheme, n_min])
                        index += 1
                    lane += 1
    path = os.path.join(scheme_out(out, scheme), 'manifests', array + '.csv')
    save_csv(path, ['index', 'lane', 'atom_id', 'method', 'dataset', 'split', 'trial', 'n_samples', 'n_events',
                    'n_train_samples_split', 'n_train_events', 'config_id', 'param_source', 'params', 'order_scheme',
                    'n_min'], rows)
    log_dir = os.path.join(out, 'logs', 'sor_' + array)  # --output of the SLURM script; must exist at submission
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    for (method, dataset), why in skipped.items():
        print('skipped {} {}: {}'.format(method, dataset, why))
    print('{} {}: {} lanes, {} atoms ({}), manifest {}'.format(scheme, array, lane, len(rows), ', '.join(
        '{} {}'.format(m, sum(1 for r in rows if r[3] == m)) for m in methods), path))
    print('SLURM: --array=0-{}'.format(lane - 1))


# ### Main #############################################################################################################

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command')
    for cmd in ('manifest', 'run', 'atom', 'status', 'missing'):
        p = sub.add_parser(cmd)
        p.add_argument('scheme', choices=SCHEMES)
        p.add_argument('array', choices=list(ARRAYS))
        p.add_argument('--out', default=OUT, help='output root (default results/sample_order_robustness)')
        if cmd == 'manifest':
            p.add_argument('--configs', default=MINIMAL_CONFIGS, help='minimal_configs.json of the step-8 eval')
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
    sout = scheme_out(out, args.scheme)
    if args.command == 'manifest':
        write_manifest(args.scheme, args.array, out, args.configs)
        return
    if args.command == 'status':
        tdr.print_status(args.array, sout)
        return
    if args.command == 'missing':
        tdr.print_missing(args.array, sout)
        return

    scratch_only = args.smoke or getattr(args, 'atom_limit', None) is not None or \
        getattr(args, 'lane_limit', None) is not None
    if scratch_only and out == os.path.abspath(OUT):
        raise SystemExit('--smoke / --atom-limit / --lane-limit require --out with a directory other than {}'.format(
            OUT))
    manifest = tdr.manifest_rows(args.array, sout)
    if args.command == 'atom':
        row = manifest[args.index]
        split, trial = int(row['split']), int(row['trial'])
        train = load_split(row['dataset'], 'split_{}'.format(split))['train']
        order = lane_order(args.scheme, row['dataset'], split, trial, train)
        tdr.run_atom(args.array, row, sout, smoke=args.smoke, order=order, seed_key=seed_key(args.scheme),
                     order_scheme=args.scheme)
        return

    # Worker: SIGTERM (e.g. scancel) -> SystemExit, so the running child's process group is killed in run_child
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    lane = args.lane if args.lane is not None else os.environ.get('SLURM_ARRAY_TASK_ID')
    if lane is None:
        raise SystemExit("give a lane index or 'all' (or set SLURM_ARRAY_TASK_ID)")
    by_lane = tdr.lanes(manifest)
    selected = list(by_lane) if lane == 'all' else [int(lane)]
    atom_limit = args.atom_limit if args.atom_limit is not None else ATOM_TIME_LIMIT_S
    atom_cmd = [sys.executable, os.path.abspath(__file__), 'atom', args.scheme, args.array, '--out', out]
    # Deadline: --lane-limit (tests) per lane; under SLURM the end of the job (shared by all selected lanes);
    # otherwise LANE_TIME_LIMIT_S per lane
    left = None if args.lane_limit is not None else tdr.slurm_time_left()
    if left is not None:
        job_end = time.time() + left
        print('{} SLURM job {}: {:.2f} h left'.format(now(), os.environ.get('SLURM_JOB_ID'), left / 3600.))
    for l in selected:
        if left is not None:
            deadline = job_end
        else:
            deadline = time.time() + (args.lane_limit if args.lane_limit is not None else LANE_TIME_LIMIT_S)
        tdr.run_lane(args.array, by_lane[l], sout, args.smoke, atom_limit, deadline, atom_cmd=atom_cmd,
                     overwrite=OVERWRITE)


if __name__ == '__main__':
    main()
