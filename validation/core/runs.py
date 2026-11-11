"""
Output and run helpers: atomic writes, run metadata, thread count, manifests (task lists) and SLURM index ranges.

stdlib only (numpy values are converted if present), Python 3.5-compatible (runs in every method env): no f-strings,
no variable annotations.
"""

import csv
import datetime
import importlib
import json
import os
import platform
import socket
import subprocess
import sys

import config


def save_json(path, obj):
    """
    Writes obj as JSON atomically (temporary file in the same directory, then os.replace), so a killed job never
    leaves a half-written file. numpy scalars / arrays are converted to Python types. Creates the directory.
    """
    _atomic_write(path, lambda f: json.dump(obj, f, indent=2, default=_to_builtin))


def save_csv(path, header, rows):
    """
    Writes a CSV atomically (see save_json).

    Args:
        path (str): Output file.
        header (list of str): Column names.
        rows (list of list): Rows, in the order of header.
    """
    def write(f):
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_to_builtin(v) if _is_numpy(v) else v for v in r])
    _atomic_write(path, write, newline='')


def now():
    """Current local time as ISO string (seconds)."""
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def n_threads():
    """
    Number of CPU threads for the job: SLURM_CPUS_PER_TASK if set, else the number of cores this process may run on
    (CPU affinity, e.g. restricted by taskset or a cgroup; os.cpu_count() would return all cores of the node).
    """
    v = os.environ.get('SLURM_CPUS_PER_TASK')
    if v:
        try:
            return int(v)
        except ValueError:
            pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # Not available on every platform
        return os.cpu_count() or 1


def run_metadata(modules=()):
    """
    Metadata of the current run.

    Args:
        modules (iterable of str): Module names whose __version__ is recorded (e.g. ['numpy', 'sklearn']); modules
            that cannot be imported are recorded as None.

    Returns:
        dict: python version, module versions, conda env, git commit + dirty flag (uncommitted changes to tracked files), host, platform, CPU model,
        number of CPUs (os.cpu_count and n_threads()), GPU names (nvidia-smi, empty list if none), SLURM ids.
    """
    # Uncommitted changes to tracked files (untracked files are ignored); None if git is not available
    status = _git(['status', '--porcelain', '--untracked-files=no'])
    versions = {}
    for m in modules:
        try:
            versions[m] = getattr(importlib.import_module(m), '__version__', None)
        except Exception:
            versions[m] = None
    return {
        'python': sys.version.split()[0],
        'versions': versions,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'git_commit': _git(['rev-parse', 'HEAD']),
        'git_dirty': None if status is None else bool(status),
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'cpu_model': _cpu_model(),
        'cpu_count': os.cpu_count(),
        'n_threads': n_threads(),
        'gpus': _gpu_names(),
        'slurm': dict((k, os.environ.get(k)) for k in (
            'SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_CLUSTER_NAME', 'SLURM_JOB_PARTITION',
            'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'CUDA_VISIBLE_DEVICES')),
    }


def read_manifest(path):
    """
    Rows of a manifest (frozen task list CSV of an experiment script) as a list of dicts (all values str).
    Exits with a hint if the file does not exist.
    """
    if not os.path.exists(path):
        raise SystemExit('no manifest {}: run the "manifest" command of the script first'.format(path))
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def index_ranges(indices):
    """Sorted ints -> SLURM --array range string, e.g. [1, 2, 3, 7] -> '1-3,7'."""
    parts, start = [], None
    for k, i in enumerate(indices):
        if start is None:
            start = i
        if k + 1 == len(indices) or indices[k + 1] != i + 1:
            parts.append(str(i) if i == start else '{}-{}'.format(start, i))
            start = None
    return ','.join(parts)


# ### Internals ########################################################################################################

def _atomic_write(path, write, newline=None):
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    tmp = '{}.tmp{}'.format(path, os.getpid())
    try:
        with open(tmp, 'w', newline=newline) as f:
            write(f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _is_numpy(v):
    return type(v).__module__ == 'numpy'


def _to_builtin(obj):
    # JSON / CSV fallback for numpy scalars and arrays
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError('{!r} is not JSON serializable'.format(obj))


def _run(cmd, cwd=None):
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=30)
        return out.strip()
    except Exception:
        return None


def _git(args):
    return _run(['git'] + args, cwd=config.ROOT)


def _cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (IOError, OSError):
        pass
    return platform.processor() or None


def _gpu_names():
    out = _run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    return [] if not out else [l.strip() for l in out.splitlines() if l.strip()]
