"""
Helpers of the evaluation scripts (tune_hyperparameters_eval.py, benchmark_eval.py): atomic CSV writes, the last line
of a traceback, the iteration-cap check, and the failure type of an atom.

Env: fx_plot (the eval scripts; write_csv takes a pandas DataFrame, the module itself imports only os). Not part of
validation/core/: cap_hit knows method parameters (HGB, GMM-Mocking, MLP), while the core helpers are method-agnostic,
and core is the code that runs inside the experiment atoms; these helpers are used only by the eval scripts.
"""

import os


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
