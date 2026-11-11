"""
Evaluation of the hyperparameter tuning on Imstat (PLAN.md step 6, working_notes/tuning_plan.md section 8).

Core: a chain of aggregation tables from the per-val-sample counts of the tuning atoms (tune_hyperparameters.py):
    sample        score of one val sample: macro F1 (binary datasets: F1 of label 1), scoring rule D8
    donor         over the val samples of one donor within one atom      (method, config_id, split, inner, donor_id)
    inner         over the donors of one inner val set                    (method, config_id, split, inner)
    outer         over the 3 inner folds of one outer split               (method, config_id, split)
    overall       over the 5 outer splits                                 (method, config_id)
Every level holds mean, sd (ddof=1, NaN if n = 1) and n (number of units of the level below that enter mean and sd)
of the level below's 'mean' (sample level: 'score'). A non-ok atom (failed / started / missing) gives NaN scores for
its val samples; the NaN propagates upwards (no nanmean), so a config is NaN at every level its failed atom enters.
The inner level also carries the atom's wall-clock runtimes, kept separate: fit_s (preprocessing fit + build + fit)
and predict_s (transform + predict, summed over the atom's val samples); NaN unless the atom is ok. They are
aggregated upwards the same way (outer, overall: fit_mean, fit_sd, predict_mean, predict_sd).

Best configs (D6): per method, the config with the highest mean at the outer level (per outer split, for the Imstat
benchmark) and at the overall level (consensus, for all other datasets). Configs with a NaN mean (incomplete) are
excluded; ties go to the default config, then to the lowest config_id (SELECTION_RULE). Stored as JSON, so the
benchmark (step 7) loads the params dict directly.

Env: fx_plot (pandas).

Usage:
    python tune_hyperparameters_eval.py [--out DIR]

Inputs: results/tune_hyperparameters/{manifests,configs,atoms}, data/splits/imstat_inner.csv, the curated
meta_samples.csv (donor_id) and meta_labels.csv.

Outputs (<out> = results/tune_hyperparameters_eval by default):
    atoms.csv          one row per manifest atom: status (ok / failed / started / missing), last error line, fit and
                       summed predict wall time, peak CPU RSS / GPU memory, host, GPU, flattened fit_info scalars,
                       cap_hit (HGB n_iter reached max_iter; GMM a class not converged; MLP with early stopping
                       reached n_epochs; empty if not applicable)
    failures.csv       atom counts per method x status x error line
    scores/scores_<level>.csv   the aggregation chain (sample, donor, inner, outer, overall);
                       is_default in all tables, the params JSON from the inner level upwards
    best_configs/params_overall_<method>.json   consensus config: {method, level, rule, config_id, params, mean, sd,
                       n, is_default, n_candidates}; params / config_id are null if no config is complete
    best_configs/params_outer_<method>.json     per outer split: {method, level, rule, splits: {"<split>": {config_id,
                       params, mean, sd, n, is_default, n_candidates}}}
                       Load: json.load(f)['params'] or json.load(f)['splits'][str(split)]['params'].
Each part is skipped if its outputs exist (OVERWRITE = False); the parts after it then read them.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import config
from validation.core.runs import save_json
from validation.core.scores import abstention_rate, f1_from_counts


DATASET = 'imstat'
TUNE_DIR = os.path.join(config.RESULTS, 'tune_hyperparameters')
OUT_DIR = os.path.join(config.RESULTS, 'tune_hyperparameters_eval')
OVERWRITE = False

LEVELS = ['sample', 'donor', 'inner', 'outer', 'overall']
SELECTION_RULE = 'argmax mean; configs with NaN mean excluded; ties: is_default first, then lowest config_id'
CONFIG_KEYS = ['method', 'config_id']


# ---------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------

def write_csv(df, path):
    """
    Writes a table atomically: first to <path>.tmp, then os.replace, so a killed run never leaves a half-written file.

    Args:
        df (pd.DataFrame): Table to write (the index is not written).
        path (str): Output CSV file; its directory must exist.
    """
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def done(*paths):
    """
    Skip check of one part of main.

    Args:
        *paths (str): The output files of the part.

    Returns:
        bool: True if all paths exist and OVERWRITE is False (the part is skipped and its outputs are read back).
    """
    return not OVERWRITE and all(os.path.exists(p) for p in paths)


def aggregate(df, by, value, prefix=''):
    """
    One level of the aggregation chain: mean, SD and count of one column per group.

    Args:
        df (pd.DataFrame): Table of the level below, one row per unit (e.g. the sample table, one row per val sample).
        by (list of str): Key columns of the new level (e.g. method, config_id, split, inner, donor_id); one output
            row per distinct combination.
        value (str): Column to aggregate ('score' at the sample level, 'mean' above, or a runtime column).
        prefix (str): Prefix of the output columns (e.g. 'fit_' -> fit_mean, fit_sd, fit_n).

    Returns:
        pd.DataFrame: The by columns plus <prefix>mean, <prefix>sd (ddof=1, NaN if n = 1) and <prefix>n (number of
        rows of the group, NaN rows included). A NaN in a group makes its mean and sd NaN (no nanmean), so a failed
        atom propagates upwards.
    """
    g = df.groupby(by, sort=True)[value]
    out = pd.DataFrame({
        prefix + 'mean': g.mean(), prefix + 'sd': g.std(ddof=1), prefix + 'n': g.size(),
    })
    has_nan = df[value].isna().groupby([df[c] for c in by], sort=True).any()
    out.loc[has_nan.values, [prefix + 'mean', prefix + 'sd']] = np.nan
    return out.reset_index()


def last_line(text):
    """
    Args:
        text (str or None): A traceback (the 'error' field of the atom results JSON).

    Returns:
        str: Its last non-empty line (the exception, e.g. "ValueError: ..."), or '' if there is none.
    """
    lines = [l for l in (text or '').strip().splitlines() if l.strip()]
    return lines[-1] if lines else ''


def cap_hit(method, params, fit_info):
    """
    Whether training stopped at the iteration cap instead of converging / early stopping.

    Args:
        method (str): Method name.
        params (dict): Params of the atom's config (atom results JSON 'params').
        fit_info (dict or None): The atom results JSON 'fit_info' (only HGB, GMM-Mocking and MLP store one).

    Returns:
        bool or None: HGB: n_iter >= max_iter; GMM-Mocking: a per-class mixture did not converge; MLP with early
        stopping: epochs_trained >= n_epochs. None for every other method / config (no cap or nothing stored).
    """
    if not fit_info:
        return None
    if method == 'hgb':
        return bool(fit_info['n_iter'] >= params['max_iter'])
    if method == 'gmm_mocking':
        return not all(fit_info['converged'])
    if method == 'mlp':
        return bool(fit_info['epochs_trained'] >= params['n_epochs']) if params['early_stopping'] else None
    return None


# ---------------------------------------------------------------------------------------------------------------
# Part 1: atoms (status, resources, fit_info)
# ---------------------------------------------------------------------------------------------------------------

def load_atoms():
    """
    Reads every atom of the frozen manifests (results/tune_hyperparameters/manifests/*.csv) and its atom results JSON
    (atoms/<method>/<config_id>/split_<i>_inner_<j>.json, written by tune_hyperparameters.py).

    Only the manifest atoms count: atom folders of configs no longer in the grids are ignored.

    Returns:
        pd.DataFrame: One row per manifest atom with atom_id, method, config_id, split, inner, status ('ok',
        'failed', 'started', or 'missing' if there is no atom results JSON), error (last line of the traceback),
        fit_wall_s, predict_wall_s (summed over the val samples), n_predict, peak CPU RSS / GPU memory, host, gpu, n_threads,
        cap_hit, fit_info_<key> (scalar fit_info entries), and fit_s / predict_s (= the wall times if status is 'ok',
        else NaN; these enter the aggregation chain).
    """
    manifest = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(TUNE_DIR, 'manifests', '*.csv')))],
                         ignore_index=True)
    rows = []
    for a in manifest.itertuples(index=False):
        path = os.path.join(TUNE_DIR, 'atoms', a.method, a.config_id, 'split_{}_inner_{}.json'.format(a.split, a.inner))
        row = {'atom_id': a.atom_id, 'method': a.method, 'config_id': a.config_id, 'split': a.split, 'inner': a.inner}
        if not os.path.exists(path):
            row['status'] = 'missing'
            rows.append(row)
            continue
        d = json.load(open(path))
        res = d.get('resources') or {}
        fit = res.get('fit') or {}
        pred = list((res.get('predict') or {}).values())
        meta = d.get('metadata') or {}
        row.update({
            'status': d['status'],
            'error': last_line(d.get('error')),
            'fit_wall_s': fit.get('wall_s'),
            'predict_wall_s': sum(p['wall_s'] for p in pred) if pred else None,
            'n_predict': len(pred),
            'fit_cpu_rss_peak_mb': fit.get('cpu_rss_peak_mb'),
            'fit_gpu_peak_mb': fit.get('gpu_peak_mb'),
            'predict_cpu_rss_peak_mb': max([p['cpu_rss_peak_mb'] for p in pred if p.get('cpu_rss_peak_mb') is not None],
                                           default=None),
            'host': meta.get('host'),
            'gpu': ';'.join(meta.get('gpus') or []),
            'n_threads': d.get('n_threads'),
            'cap_hit': cap_hit(a.method, d['params'], d.get('fit_info')),
        })
        for k, v in (d.get('fit_info') or {}).items():  # Scalars only (loss curves, per-class lists are skipped)
            if not isinstance(v, (list, dict)):
                row['fit_info_' + k] = v
        rows.append(row)
    atoms = pd.DataFrame(rows)
    ok = atoms['status'] == 'ok'
    atoms['fit_s'] = atoms['fit_wall_s'].where(ok)
    atoms['predict_s'] = atoms['predict_wall_s'].where(ok)
    return atoms


def failure_table(atoms):
    """
    Args:
        atoms (pd.DataFrame): Output of load_atoms.

    Returns:
        pd.DataFrame: Columns method, status, error, n_atoms: number of atoms per method x status x error line.
    """
    t = atoms.assign(error=atoms['error'].fillna(''))
    return t.groupby(['method', 'status', 'error']).size().rename('n_atoms').reset_index()


# ---------------------------------------------------------------------------------------------------------------
# Part 2: aggregation chain
# ---------------------------------------------------------------------------------------------------------------

def load_configs():
    """
    Reads the config tables (results/tune_hyperparameters/configs/<method>.csv).

    Returns:
        pd.DataFrame: One row per config with method, config_id, is_default (the method's default / published
        config) and params (JSON string).
    """
    t = [pd.read_csv(f, dtype={'config_id': str}).assign(method=os.path.basename(f)[:-4])
         for f in sorted(glob.glob(os.path.join(TUNE_DIR, 'configs', '*.csv')))]
    return pd.concat(t, ignore_index=True)[['method', 'config_id', 'is_default', 'params']]


def score_samples(atoms):
    """
    Sample level of the chain: one score per (atom, val sample).

    For an ok atom, its _counts.csv (one row per val sample x class) is reshaped to per-sample arrays of length K
    (label order of meta_labels.csv, checked) and scored with validation.core.scores. For a non-ok atom, its val
    samples are taken from the split table (data/splits/imstat_inner.csv) and get NaN, so the failure enters the
    chain.

    Args:
        atoms (pd.DataFrame): Output of load_atoms (uses method, config_id, split, inner, status).

    Returns:
        pd.DataFrame: Columns method, config_id, split, inner, sample, score (macro F1; binary datasets: F1 of
        label 1), abstention_rate, donor_id (from meta_samples.csv).
    """
    labels = pd.read_csv(os.path.join(config.DATA_CURATED, DATASET, 'meta_labels.csv'))['label'].tolist()
    binary = len(labels) == 2
    k = len(labels)
    splits = pd.read_csv(os.path.join(config.DATA_SPLITS, DATASET + '_inner.csv'))
    rows = []
    for a in atoms.itertuples(index=False):
        if a.status == 'ok':
            path = os.path.join(TUNE_DIR, 'atoms', a.method, a.config_id,
                                'split_{}_inner_{}_counts.csv'.format(a.split, a.inner))
            c = pd.read_csv(path).sort_values(['sample', 'label'])
            samples = c['sample'].unique()
            if len(c) != len(samples) * k or not (c['label'].to_numpy().reshape(len(samples), k) == labels).all():
                raise ValueError('unexpected labels in ' + path)
            arr = dict((col, c[col].to_numpy().reshape(len(samples), k))
                       for col in ['tp', 'fp', 'fn', 'n_true', 'n_abstained'])
            for i, s in enumerate(samples):
                rows.append((a.method, a.config_id, a.split, a.inner, s,
                             f1_from_counts(arr['tp'][i], arr['fp'][i], arr['fn'][i], binary=binary),
                             abstention_rate(arr['n_abstained'][i], arr['n_true'][i])))
        else:
            col = 'split_{}_inner_{}'.format(a.split, a.inner)
            for s in splits.loc[splits[col] == 'val', 'sample']:
                rows.append((a.method, a.config_id, a.split, a.inner, s, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=['method', 'config_id', 'split', 'inner', 'sample', 'score', 'abstention_rate'])
    donors = pd.read_csv(os.path.join(config.DATA_CURATED, DATASET, 'meta_samples.csv'))[['sample', 'donor_id']]
    return df.merge(donors, on='sample', how='left', validate='many_to_one')


def build_chain(sample, atoms, configs):
    """
    Builds the aggregation chain from the sample scores (one aggregate call per level, each on the 'mean' of the
    level below):
        donor        (method, config_id, split, inner, donor_id)  over the donor's val samples in one atom
        inner        (method, config_id, split, inner)            over the donors of the inner val set; + the
                                                                   atom's status, fit_s, predict_s
        outer        (method, config_id, split)                   over the 3 inner folds; + fit_/predict_ mean, sd
        overall      (method, config_id)                          over the 5 outer splits; + fit_/predict_ mean, sd

    Args:
        sample (pd.DataFrame): Output of score_samples.
        atoms (pd.DataFrame): Output of load_atoms (status, fit_s, predict_s are joined to the inner level).
        configs (pd.DataFrame): Output of load_configs.

    Returns:
        dict: level name ('sample', 'donor', 'inner', 'outer', 'overall') -> pd.DataFrame with the key columns,
        mean, sd, n (sample level: score, abstention_rate), is_default, and params from the inner level upwards.
    """
    cfg = CONFIG_KEYS
    donor = aggregate(sample, cfg + ['split', 'inner', 'donor_id'], 'score')
    inner = aggregate(donor, cfg + ['split', 'inner'], 'mean')
    inner = inner.merge(atoms[cfg + ['split', 'inner', 'status', 'fit_s', 'predict_s']], on=cfg + ['split', 'inner'],
                        how='left', validate='one_to_one')
    outer = aggregate(inner, cfg + ['split'], 'mean')
    overall = aggregate(outer, cfg, 'mean')
    for t in ('fit', 'predict'):  # Runtimes: same chain, mean and sd (n equals the score's n)
        outer = outer.merge(aggregate(inner, cfg + ['split'], t + '_s', t + '_').drop(columns=t + '_n'),
                            on=cfg + ['split'])
        overall = overall.merge(aggregate(outer, cfg, t + '_mean', t + '_').drop(columns=t + '_n'), on=cfg)
    tables = {'sample': sample, 'donor': donor, 'inner': inner, 'outer': outer, 'overall': overall}
    for level, t in tables.items():
        cols = ['is_default'] if level in ('sample', 'donor') else ['is_default', 'params']
        tables[level] = t.merge(configs[cfg + cols], on=cfg, how='left', validate='many_to_one')
    return tables


# ---------------------------------------------------------------------------------------------------------------
# Part 3: best configs
# ---------------------------------------------------------------------------------------------------------------

def select_best(table):
    """
    The best config of one method in one group (one outer split, or overall).

    Args:
        table (pd.DataFrame): Rows of one method (and one outer split at the outer level) of an aggregation table:
            one row per config with config_id, mean, sd, n, is_default, params (JSON string).

    Returns:
        dict: config_id, params (dict), mean, sd, n, is_default of the config with the highest mean (SELECTION_RULE:
        NaN means excluded; ties: is_default first, then lowest config_id), and n_candidates (number of configs with
        a non-NaN mean). config_id, params, mean, sd, n, is_default are None if no config is complete.
    """
    # Sort by mean F1, is_default, and config_id
    cand = table.dropna(subset=['mean']).sort_values(['mean', 'is_default', 'config_id'],
                                                     ascending=[False, False, True])
    # If no candidates remain if NaN entries were removed, return None config
    if not len(cand):
        best = dict((k, None) for k in ['config_id', 'params', 'mean', 'sd', 'n', 'is_default'])
    # Select best config and save to dict
    else:
        r = cand.iloc[0]
        best = {'config_id': r['config_id'], 'params': json.loads(r['params']), 'mean': float(r['mean']),
                'sd': None if pd.isna(r['sd']) else float(r['sd']), 'n': int(r['n']),
                'is_default': bool(r['is_default'])}
    best['n_candidates'] = int(len(cand))
    return best


def best_configs(tables):
    """
    Best configs of every method at the outer level (per outer split) and the overall level (consensus).

    Args:
        tables (dict): Output of build_chain (uses 'outer' and 'overall').

    Returns:
        dict: (level, method) -> JSON-ready dict. overall: {method, level, rule, **select_best}; outer: {method,
        level, rule, splits: {"<split>": select_best of that outer split}}.
    """
    out = {}
    for method, g in tables['overall'].groupby('method', sort=True):
        out[('overall', method)] = dict([('method', method), ('level', 'overall'), ('rule', SELECTION_RULE)],
                                        **select_best(g))
    for method, g in tables['outer'].groupby('method', sort=True):
        splits = dict((str(sp), select_best(gs)) for sp, gs in g.groupby('split', sort=True))
        out[('outer', method)] = {'method': method, 'level': 'outer', 'rule': SELECTION_RULE, 'splits': splits}
    return out


# ---------------------------------------------------------------------------------------------------------------

def main(out_dir):
    """
    Runs the parts in order; a part whose outputs exist (and OVERWRITE is False) is skipped and its outputs are read
    back for the next part.

    Args:
        out_dir (str): Output directory (created if needed).
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Part 1: atoms
    # Define path to atoms.csv and failures.csv
    p_atoms, p_fail = os.path.join(out_dir, 'atoms.csv'), os.path.join(out_dir, 'failures.csv')
    if done(p_atoms, p_fail):  # If already on disk, load
        atoms = pd.read_csv(p_atoms, dtype={'config_id': str})
    else:
        # Load atom-JSONs and create joint data frame; save to CSV
        atoms = load_atoms()
        write_csv(atoms, p_atoms)
        # Create failure table from atoms; save to CSV
        write_csv(failure_table(atoms), p_fail)
    print('atoms:', atoms['status'].value_counts().to_dict())

    # --- Part 2: aggregation chain
    # Create dict of {level: save_path}
    os.makedirs(os.path.join(out_dir, 'scores'), exist_ok=True)
    paths = dict((l, os.path.join(out_dir, 'scores', 'scores_{}.csv'.format(l))) for l in LEVELS)
    if done(*paths.values()):  # If already on disk, load
        tables = dict((l, pd.read_csv(p, dtype={'config_id': str})) for l, p in paths.items())
    else:
        # Create data frame with sample-level scores
        scores_sample = score_samples(atoms)
        # Aggregate
        tables = build_chain(scores_sample, atoms, load_configs())
        for l in LEVELS:
            write_csv(tables[l], paths[l])
    print('levels:', dict((l, len(t)) for l, t in tables.items()))

    # --- Part 3: best configs (one JSON per level x method)
    best = best_configs(tables)
    for (level, method), d in sorted(best.items()):
        path = os.path.join(out_dir, 'best_configs', 'params_{}_{}.json'.format(level, method))
        if done(path):
            continue
        save_json(path, d)  # Atomic, creates the directory
        entries = d['splits'].values() if level == 'outer' else [d]
        if any(e['config_id'] is None for e in entries):
            print('WARNING: no complete config for', method, level)
    print('best configs:', len(best), 'files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', default=OUT_DIR, help='output directory (default: %(default)s)')
    main(parser.parse_args().out)
