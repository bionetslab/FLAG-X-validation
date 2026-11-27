"""
Evaluation of the sample order robustness (PLAN.md step 9, working_notes/sample_order_robustness_plan.md): per order
scheme x method x dataset, the share of trials (sample orders) that meet the step-8 criterion at the step-8 minimal
cell, and the per-trial minimal number of training samples. Aggregation per scheme only; the schemes are compared in
the figures.

Env: fx_plot (pandas).

Usage:
    python sample_order_robustness_eval.py [--in DIR] [--baseline DIR] [--step8 DIR] [--out DIR]
        --in: step-9 output root with one folder per scheme (default results/sample_order_robustness)
        --baseline: step-7 eval output (default results/benchmark_eval; reads scores/<dataset>.csv)
        --step8: step-8 eval output (default results/training_data_requirements_eval; reads curves.csv for the
                 context columns of summary.csv; NaN if missing)
        --out: output directory (default results/sample_order_robustness_eval)

Reuses the step-8 eval (training_data_requirements_eval.py): atom table with the post-hoc budget, scores, the paired
per-sample difference d to the step-7 baseline, the bootstrap curves and the minimal-cell rule. See its docstring.

Cells: n_samples = 1 .. n_min at n_events = ne_min (manifest columns n_min / n_events; the variant of the step-8
minimal cell is the manifest column 'stable'). A lane = (method, split, trial); n_samples larger than the split's
training samples is resolved to the atom with all of them (alias 'atom').

Trial criterion: scope trial_<t> = trial t over all folds (one order per fold): mean d over the test samples of all
folds, percentile CI from the bootstrap over patients (donors for Imstat); the resamples are those of the step-8
pooled / trial_<t> scopes (same seed), shared by all trials, n and schemes. A trial passes at n if the cell is eligible
(all 5 fold lanes 'ok' and in budget) and meets the criterion: 'noninf' (ci_low > -delta) for every delta of the step-8
sweep, or 'legacy_ci'. A trial with a failed / timed-out / out-of-budget fold lane does not pass (counted in
n_failed). Scope 'pooled' = the mean over all trials (sanity check against step 8).

Per-trial minimum: the smallest passing n (minimal_cell of the step-8 eval, greedy, n_events fixed), without and with
the stability rule (every eligible larger tested n also passes; only n <= n_min are tested). Not found = above n_min
(inf in the quantiles of summary.csv).

Shares: counts and shares over the trials only, no CI (a binomial CI, e.g. Clopper-Pearson, can be computed later from
n_pass / n_trials and n_min_le_* / n_trials). The trials share the test data, so the share describes the order
variability given this test set. A single trial is noisier than the 3-trial mean that selected n_min in
step 8, so a share below 1 at n_min is expected even without an order effect.

Outputs (<out>):
    atoms.csv   one row per manifest atom: scheme + the columns of the step-8 atoms.csv (status, resources, in_budget,
                excluded_by).
    scores/<scheme>/<dataset>.csv   one row per atom x predicted test sample (as the step-8 scores).
    sample_scores.csv   one row per scheme x method x dataset x n_samples x test sample: scheme, method, dataset,
                n_samples, n_events, split, sample, patient_id, score_t<t> (t = 0 .. N-1), n_trials_pred, trial_mean,
                baseline, d.
    trial_curves.csv    one row per scheme x method x dataset x n_samples x scope (pooled, trial_<t>): scheme + the
                columns of the step-8 curves.csv (eligible, mean_d, ci_low, ci_high, base_ci_*, score_ci_*, ...).
    trial_minimal.csv   one row per scheme x method x dataset x trial x criterion x delta x stable: n_min, n_events,
                eligible_all (all tested n eligible), found, n_samples, mean_d, ci_low, mean_score.
    share.csv   one row per scheme x method x dataset x n_samples x criterion x delta: n_min, n_events, n_trials,
                n_eligible, n_failed, n_pass, share_pass (n_pass / n_trials), and per stability variant (suffix _naive /
                _stable) n_min_le, share_min_le (trials whose per-trial minimum is <= n_samples).
    summary.csv one row per scheme x method x dataset (noninf, PRIMARY_DELTA, the manifest's stability variant): n_min,
                n_events, stable, n_trials, n_failed, n_pass, share_pass at n_min; per-trial minimum: n_found,
                min_q0 / q25 / q50 / q75 / q100 (order statistics, inf = not found); pooled_mean_d, pooled_ci_low at
                n_min (mean of all trials); step-8 context at the minimal cell: step8_mean_d, step8_ci_low (pooled),
                step8_n_trials, step8_n_pass (its trial_<t> scopes).
Each part is skipped if its outputs exist (OVERWRITE = False); the parts after it then read them.
"""

import argparse
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import config
import training_data_requirements_eval as tdre
from sample_order_robustness import SCHEMES
from validation.eval_helpers import write_csv

IN_DIR = os.path.join(config.RESULTS, 'sample_order_robustness')
BASELINE_DIR = os.path.join(config.RESULTS, 'benchmark_eval')
STEP8_DIR = os.path.join(config.RESULTS, 'training_data_requirements_eval')
OUT_DIR = os.path.join(config.RESULTS, 'sample_order_robustness_eval')
OVERWRITE = False

PRIMARY_DELTA = tdre.PRIMARY_DELTA  # config.DELTA


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def done(*paths):
    """True if all paths exist and OVERWRITE is False (the part is skipped and its outputs are read back)."""
    return not OVERWRITE and all(os.path.exists(p) for p in paths)


def delta_key(delta):
    """Hashable key of a delta (NaN for legacy_ci)."""
    return 'nan' if pd.isna(delta) else round(float(delta), 10)


def datasets_of(manifest):
    present = set(r['dataset'] for r in manifest)
    return [d for d in config.DATASETS if d in present]


def n_trials_of(manifest):
    trials = sorted(set(int(r['trial']) for r in manifest))
    if trials != list(range(len(trials))):
        raise ValueError('trials are not 0 .. N-1: {}'.format(trials))
    return len(trials)


def manifest_cells(manifest):
    """{(method, dataset): (n_min, n_events, stable)} from the manifest."""
    res = {}
    for r in manifest:
        c = (int(r['n_min']), tdre.grid_value(r['n_events']), r['stable'] == 'True')
        if res.setdefault((r['method'], r['dataset']), c) != c:
            raise ValueError('inconsistent n_min / n_events / stable for {} {}'.format(r['method'], r['dataset']))
    return res


# ---------------------------------------------------------------------------------------------------------------
# Lanes (cells resolved per lane)
# ---------------------------------------------------------------------------------------------------------------

def lane_table(dataset, manifest, atoms):
    """
    One row per method x n_samples x split x trial of the dataset (the step-8 lane_table columns): the resolved atom
    (n_samples > the split's training samples -> the atom with all of them, alias 'atom') and its state.
    """
    a = atoms.set_index('atom_id')
    lanes = OrderedDict()
    cells = manifest_cells([r for r in manifest if r['dataset'] == dataset])
    for r in manifest:
        if r['dataset'] != dataset:
            continue
        lane = lanes.setdefault((r['method'], int(r['split']), int(r['trial'])),
                                {'n_train': int(r['n_train_samples_split']), 'atoms': {}})
        lane['atoms'][int(r['n_samples'])] = r['atom_id']
    rows = []
    for (method, split, trial), lane in lanes.items():
        n_min, ne, _ = cells[(method, dataset)]
        for ns in range(1, n_min + 1):
            ns_eff = min(ns, lane['n_train'])
            if ns_eff not in lane['atoms']:
                raise ValueError('cannot resolve n_samples {} in lane {} {} split {} trial {}'.format(
                    ns, method, dataset, split, trial))
            aid = lane['atoms'][ns_eff]
            row = a.loc[aid]
            rows.append([method, dataset, ns, ne, split, trial, '' if ns_eff == ns else 'atom', aid, row['status'],
                         bool(row['status'] == 'ok' and row['in_budget']), int(row['n_train_events'])])
    return pd.DataFrame(rows, columns=['method', 'dataset', 'n_samples', 'n_events', 'split', 'trial', 'alias',
                                       'atom_id', 'status', 'ok', 'n_train_events'])


def scope_list(splits, n_trials):
    """Scopes of the trial curves: 'pooled' (trial means) and trial_<t> (all folds, one trial)."""
    trials = list(range(n_trials))
    return [('pooled', splits, trials, 'trial_mean')] + \
        [('trial_{}'.format(t), splits, [t], 'score_t{}'.format(t)) for t in trials]


# ---------------------------------------------------------------------------------------------------------------
# Per-trial minimum and shares
# ---------------------------------------------------------------------------------------------------------------

def trial_minimal_table(curves):
    """One row per scheme x method x dataset x trial x criterion x delta x stable (columns: module docstring)."""
    rows = []
    tc = curves[curves['scope'].str.startswith('trial_')]
    for (scheme, method, dataset, scope), c in tc.groupby(['scheme', 'method', 'dataset', 'scope'], sort=False):
        c = c.sort_values('n_samples').reset_index(drop=True)
        trial = int(scope.split('_')[1])
        for criterion, delta in tdre.criteria():
            ok = tdre.passes(c, criterion, delta)
            for stable in (False, True):
                i = tdre.minimal_cell(c, ok, 'greedy', stable)
                r = c.loc[i] if i is not None else None
                rows.append(OrderedDict([
                    ('scheme', scheme), ('method', method), ('dataset', dataset), ('trial', trial),
                    ('criterion', criterion), ('delta', delta), ('stable', stable),
                    ('n_min', int(c['n_samples'].max())), ('n_events', c['n_events'].iloc[0]),
                    ('eligible_all', bool(c['eligible'].all())), ('found', r is not None),
                    ('n_samples', int(r['n_samples']) if r is not None else ''),
                    ('mean_d', r['mean_d'] if r is not None else np.nan),
                    ('ci_low', r['ci_low'] if r is not None else np.nan),
                    ('mean_score', r['mean_score'] if r is not None else np.nan)]))
    return pd.DataFrame(rows)


def share_table(curves, minimal):
    """One row per scheme x method x dataset x n_samples x criterion x delta (columns: module docstring)."""
    rows = []
    tc = curves[curves['scope'].str.startswith('trial_')]
    mins = {}  # (scheme, method, dataset, criterion, delta key, stable) -> per-trial minimum (inf = not found)
    for r in minimal.itertuples(index=False):
        key = (r.scheme, r.method, r.dataset, r.criterion, delta_key(r.delta), bool(r.stable))
        mins.setdefault(key, []).append(int(r.n_samples) if r.found else np.inf)
    for (scheme, method, dataset), c in tc.groupby(['scheme', 'method', 'dataset'], sort=False):
        n_trials = c['scope'].nunique()
        n_min = int(c['n_samples'].max())
        for criterion, delta in tdre.criteria():
            ok = tdre.passes(c, criterion, delta) & c['eligible'].to_numpy().astype(bool)
            per_min = {}
            for stable in (False, True):
                per_min[stable] = np.array(mins[(scheme, method, dataset, criterion, delta_key(delta), stable)])
                if len(per_min[stable]) != n_trials:
                    raise ValueError('trials of trial_minimal and curves differ')
            for ns in range(1, n_min + 1):
                at = (c['n_samples'] == ns).to_numpy()
                n_elig = int(c.loc[at, 'eligible'].astype(bool).sum())
                n_pass = int(ok[at].sum())
                row = OrderedDict([
                    ('scheme', scheme), ('method', method), ('dataset', dataset), ('n_samples', ns),
                    ('n_events', c['n_events'].iloc[0]), ('criterion', criterion), ('delta', delta),
                    ('n_min', n_min), ('n_trials', n_trials), ('n_eligible', n_elig), ('n_failed', n_trials - n_elig),
                    ('n_pass', n_pass), ('share_pass', n_pass / float(n_trials))])
                for stable, tag in ((False, 'naive'), (True, 'stable')):
                    k = int((per_min[stable] <= ns).sum())
                    row.update([('n_min_le_' + tag, k), ('share_min_le_' + tag, k / float(n_trials))])
                rows.append(row)
    return pd.DataFrame(rows)


def step8_context(step8_dir):
    """{(method, dataset, n_samples, n_events): (mean_d, ci_low, n_trials, n_pass)} from the step-8 curves.csv."""
    path = os.path.join(step8_dir, 'curves.csv')
    if not os.path.exists(path):
        print('no step-8 curves at {}: step-8 context columns are NaN'.format(path))
        return {}
    c = tdre.read_cells(path)
    res = {}
    for key, g in c.groupby(['method', 'dataset', 'n_samples', 'n_events'], sort=False):
        pooled = g[g['scope'] == 'pooled']
        tr = g[g['scope'].str.startswith('trial_')]
        n_pass = int(((tr['ci_low'] > -PRIMARY_DELTA) & tr['eligible'].astype(bool)).sum())
        res[key] = (float(pooled['mean_d'].iloc[0]) if len(pooled) else np.nan,
                    float(pooled['ci_low'].iloc[0]) if len(pooled) else np.nan, len(tr), n_pass)
    return res


def summary_table(curves, minimal, share, cells, step8):
    """One row per scheme x method x dataset (columns: module docstring)."""
    rows = []
    prim = share[(share['criterion'] == 'noninf') & np.isclose(share['delta'], PRIMARY_DELTA)]
    mprim = minimal[(minimal['criterion'] == 'noninf') & np.isclose(minimal['delta'], PRIMARY_DELTA)]
    pooled = curves[curves['scope'] == 'pooled'].set_index(['scheme', 'method', 'dataset', 'n_samples'])
    for (scheme, method, dataset), g in prim.groupby(['scheme', 'method', 'dataset'], sort=False):
        n_min, ne, stable = cells[scheme][(method, dataset)]
        s = g[g['n_samples'] == n_min].iloc[0]
        m = mprim[(mprim['scheme'] == scheme) & (mprim['method'] == method) & (mprim['dataset'] == dataset)
                  & (mprim['stable'] == stable)]
        vals = np.array([int(v) if f else np.inf for v, f in zip(m['n_samples'], m['found'])])
        q = np.percentile(vals, [0, 25, 50, 75, 100], method='inverted_cdf')
        p = pooled.loc[(scheme, method, dataset, n_min)]
        s8 = step8.get((method, dataset, n_min, ne), (np.nan, np.nan, np.nan, np.nan))
        rows.append(OrderedDict([
            ('scheme', scheme), ('method', method), ('dataset', dataset), ('n_min', n_min), ('n_events', ne),
            ('stable', stable), ('n_trials', int(s['n_trials'])), ('n_failed', int(s['n_failed'])),
            ('n_pass', int(s['n_pass'])), ('share_pass', s['share_pass']), ('n_found', int(np.isfinite(vals).sum())),
            ('min_q0', q[0]), ('min_q25', q[1]), ('min_q50', q[2]), ('min_q75', q[3]), ('min_q100', q[4]),
            ('pooled_mean_d', p['mean_d']), ('pooled_ci_low', p['ci_low']),
            ('step8_mean_d', s8[0]), ('step8_ci_low', s8[1]), ('step8_n_trials', s8[2]), ('step8_n_pass', s8[3])]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------

def main(in_dir, baseline_dir, step8_dir, out_dir):
    schemes = [s for s in SCHEMES if os.path.isdir(os.path.join(in_dir, s, 'manifests'))]
    if not schemes:
        raise SystemExit('no <scheme>/manifests in {}'.format(in_dir))
    manifests = OrderedDict((s, tdre.load_manifest(os.path.join(in_dir, s))) for s in schemes)
    for s in schemes:
        d = os.path.join(out_dir, 'scores', s)
        if not os.path.isdir(d):
            os.makedirs(d)
    cells = dict((s, manifest_cells(m)) for s, m in manifests.items())
    n_trials = dict((s, n_trials_of(m)) for s, m in manifests.items())

    # --- Part 1: atom table (status, resources, post-hoc budget per scheme)
    p_atoms = os.path.join(out_dir, 'atoms.csv')
    if done(p_atoms):
        atoms = pd.read_csv(p_atoms, dtype={'n_samples': str, 'n_events': str})
        print('read {}'.format(p_atoms))
    else:
        parts = []
        for s, m in manifests.items():
            t = tdre.atom_table(m, os.path.join(in_dir, s))
            t.insert(0, 'scheme', s)
            parts.append(t)
        atoms = pd.concat(parts, ignore_index=True)
        write_csv(atoms, p_atoms)
        print('wrote {} ({} atoms: {})'.format(p_atoms, len(atoms), dict(atoms.groupby('scheme')['status']
                                                                         .value_counts())))

    # --- Part 2: scores per scheme x dataset
    scores = {}
    for s, m in manifests.items():
        a = atoms[atoms['scheme'] == s]
        for dataset in datasets_of(m):
            path = os.path.join(out_dir, 'scores', s, dataset + '.csv')
            if done(path):
                scores[(s, dataset)] = pd.read_csv(path, dtype={'n_samples': str, 'n_events': str})
            else:
                scores[(s, dataset)] = tdre.score_table(dataset, m, a, os.path.join(in_dir, s))
                write_csv(scores[(s, dataset)], path)
                print('wrote {} ({} rows)'.format(path, len(scores[(s, dataset)])))

    # --- Part 3: per-sample scores per trial, paired with the step-7 baseline
    lanes = {}
    for s, m in manifests.items():
        a = atoms[atoms['scheme'] == s]
        for dataset in datasets_of(m):
            lanes[(s, dataset)] = lane_table(dataset, m, a)
    p_samples = os.path.join(out_dir, 'sample_scores.csv')
    if done(p_samples):
        samples = tdre.read_cells(p_samples)
        print('read {}'.format(p_samples))
    else:
        parts = []
        for (s, dataset), ln in lanes.items():
            t = tdre.sample_score_table(dataset, ln, scores[(s, dataset)], tdre.baseline_scores(dataset, baseline_dir),
                                        n_trials=n_trials[s])
            t.insert(0, 'scheme', s)
            parts.append(t)
        samples = pd.concat(parts, ignore_index=True)
        write_csv(samples, p_samples)
        print('wrote {} ({} rows)'.format(p_samples, len(samples)))

    # --- Part 4: curves per trial (and pooled) with bootstrap CIs
    p_curves = os.path.join(out_dir, 'trial_curves.csv')
    if done(p_curves):
        curves = tdre.read_cells(p_curves)
        print('read {}'.format(p_curves))
    else:
        parts = []
        for (s, dataset), ln in lanes.items():
            sm = samples[(samples['scheme'] == s) & (samples['dataset'] == dataset)].drop(columns='scheme')
            splits = sorted(sm['split'].unique())
            t = tdre.curve_table(dataset, ln, sm.reset_index(drop=True), scope_list(splits, n_trials[s]))
            t.insert(0, 'scheme', s)
            parts.append(t)
            print('curves {} {}'.format(s, dataset))
        curves = pd.concat(parts, ignore_index=True)
        write_csv(curves, p_curves)
        print('wrote {} ({} rows)'.format(p_curves, len(curves)))

    # --- Part 5: per-trial minimum, shares, summary
    p_min = os.path.join(out_dir, 'trial_minimal.csv')
    p_share = os.path.join(out_dir, 'share.csv')
    p_sum = os.path.join(out_dir, 'summary.csv')
    if done(p_min, p_share, p_sum):
        print('{}, {}, {} exist'.format(p_min, p_share, p_sum))
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        minimal = trial_minimal_table(curves)
    share = share_table(curves, minimal)
    summary = summary_table(curves, minimal, share, cells, step8_context(step8_dir))
    for t, p in ((minimal, p_min), (share, p_share), (summary, p_sum)):
        write_csv(t, p)
        print('wrote {} ({} rows)'.format(p, len(t)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='in_dir', default=IN_DIR)
    ap.add_argument('--baseline', default=BASELINE_DIR)
    ap.add_argument('--step8', default=STEP8_DIR)
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()
    main(os.path.abspath(args.in_dir), os.path.abspath(args.baseline), os.path.abspath(args.step8),
         os.path.abspath(args.out))
