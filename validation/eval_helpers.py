"""
Helpers of the evaluation and plot scripts (tune_hyperparameters_eval.py, benchmark_eval.py, plot_benchmark.py):
atomic CSV writes, the last line of a traceback, the iteration-cap check, the failure type of an atom, and the
population-size agreement (seed means, Bland-Altman statistics).

Env: fx_plot (the eval scripts; write_csv / population_seed_mean take pandas DataFrames, the module imports os and
numpy). Not part of
validation/core/: cap_hit knows method parameters (HGB, GMM-Mocking, MLP), while the core helpers are method-agnostic,
and core is the code that runs inside the experiment atoms; these helpers are used only by the eval scripts.
"""

import os

import numpy as np


# Failure types of an atom (benchmark_eval.py). Patterns are matched case-insensitively.
FAILURE_PATTERNS_ERROR = [  # In the Python traceback of a 'failed' atom
    ('oom', ['MemoryError', 'out of memory', 'Unable to allocate', 'cannot allocate memory']),
]
FAILURE_PATTERNS_LOG = [  # In the SLURM log of a 'started' (killed) atom
    ('timeout', ['DUE TO TIME LIMIT']),
    ('oom', ['oom-kill', 'oom_kill', 'Out Of Memory', 'OUT_OF_MEMORY']),
    ('node_failure', ['NODE FAIL']),
]


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


def failure_type(status, error, log_text):
    """
    Failure type of an atom.

    Args:
        status (str): 'ok', 'failed', 'started' (killed: no final JSON) or 'missing' (no JSON).
        error (str or None): Traceback of a failed atom (atom results JSON 'error').
        log_text (str or None): SLURM log of the job that ran the atom (stdout + stderr), if found.

    Returns:
        (str, str): (failure, message). failure is '' (ok), 'missing', 'python_error', 'oom', 'timeout',
        'node_failure' or 'killed_unknown'; message is the traceback's last line or the matching log line ('' if
        none).
    """
    if status == 'ok':
        return '', ''
    if status == 'missing':
        return 'missing', ''
    if status == 'failed':
        msg = last_line(error)
        for failure, patterns in FAILURE_PATTERNS_ERROR:
            if any(p.lower() in (error or '').lower() for p in patterns):
                return failure, msg
        return 'python_error', msg
    lines = (log_text or '').splitlines()
    for failure, patterns in FAILURE_PATTERNS_LOG:  # In the order of FAILURE_PATTERNS_LOG
        for line in lines:
            if any(p.lower() in line.lower() for p in patterns):
                return failure, line.strip()
    return 'killed_unknown', ''


def population_seed_mean(pop):
    """
    Population sizes averaged over the model seeds (one point per method x test sample x class, as in the figures).

    Args:
        pop (pd.DataFrame): Long population table of benchmark_eval.py (populations/<dataset>.csv): one row per atom x
            test sample x class with method, dataset, split, seed, sample, label, name, true_pct, pred_pct.

    Returns:
        pd.DataFrame: One row per method x dataset x split x sample x label with name, true_pct, pred_pct (means over
        the seeds that predicted the sample), d_pp = pred_pct - true_pct, mean_pct = (pred_pct + true_pct) / 2 and
        present = true_pct > 0 or pred_pct > 0.
    """
    m = (pop.groupby(['method', 'dataset', 'split', 'sample', 'label', 'name'], sort=False)[['true_pct', 'pred_pct']]
         .mean().reset_index())
    m['d_pp'] = m['pred_pct'] - m['true_pct']
    m['mean_pct'] = (m['pred_pct'] + m['true_pct']) / 2.0
    m['present'] = (m['true_pct'] > 0) | (m['pred_pct'] > 0)
    return m


def ba_stats(d):
    """
    Bland-Altman statistics of prediction errors in percentage points.

    Args:
        d (array-like): Errors pred - true (pp).

    Returns:
        dict: n, bias_pp (mean d), sd_pp (SD of d, ddof 1), mae_pp (mean |d|), loa_low_pp / loa_high_pp (95 % limits
        of agreement, bias -/+ 1.96 SD). NaN where undefined (n = 0; SD and limits need n >= 2).
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    bias = float(np.mean(d)) if n else np.nan
    sd = float(np.std(d, ddof=1)) if n > 1 else np.nan
    return {'n': n, 'bias_pp': bias, 'sd_pp': sd, 'mae_pp': float(np.mean(np.abs(d))) if n else np.nan,
            'loa_low_pp': bias - 1.96 * sd, 'loa_high_pp': bias + 1.96 * sd}
