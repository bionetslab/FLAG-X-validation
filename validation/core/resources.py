"""
Wall-clock time and peak memory (CPU RSS, GPU device memory) of one block of code.

Port of scalability_wrapper / track_memory_cpu / track_memory_gpu (legacy/validation/utils/val_utils.py L799-L1000)
without psutil: stdlib only (time.perf_counter, /proc, nvidia-smi via subprocess), Python 3.5-compatible (runs in
every method env): no f-strings, no variable annotations. Linux only (/proc).

CPU: a polling thread sums VmRSS of the process and all its descendants (e.g. data loader workers; R runs in-process
via rpy2) every interval, plus one sample at the start and one at the end of the block. Peaks shorter than the
interval can be missed; blocks shorter than the interval are measured only at their start and end. One poll costs about
1.4 ms (process with 96 threads), i.e. ~3 % of one core at the default interval of 0.05 s.
GPU: nvidia-smi polls memory.used of the first visible GPU (CUDA_VISIBLE_DEVICES, else GPU 0). This is device memory
of all processes on that GPU (GPUs are allocated exclusively on the clusters used).
"""

import os
import subprocess
import threading
import time
from contextlib import contextmanager


@contextmanager
def track_resources(gpu=False, interval=0.05):
    """
    Context manager measuring one block.

    Usage:
        with track_resources(gpu=False) as res:
            model.fit(X, y)
        res['wall_s'], res['cpu_rss_peak_mb'], ...

    Args:
        gpu (bool): Also track GPU memory with nvidia-smi (gpu_* entries are None otherwise or if it fails).
        interval (float): Polling interval in seconds.

    Yields:
        dict: Filled when the block exits (also if it raises):
            'wall_s': wall-clock time of the block,
            'cpu_rss_baseline_mb': RSS of process + descendants before the block,
            'cpu_rss_peak_mb': peak RSS of process + descendants during the block,
            'cpu_rss_peak_increase_mb': peak - baseline,
            'gpu_baseline_mb', 'gpu_peak_mb', 'gpu_peak_increase_mb': the same for the GPU (None without GPU).
    """
    res = {}
    pid = os.getpid()
    baseline = _tree_rss_mb(pid)
    cpu_samples = [baseline]
    stop = threading.Event()

    def poll_cpu():
        while not stop.is_set():
            cpu_samples.append(_tree_rss_mb(pid))
            stop.wait(interval)

    t_cpu = threading.Thread(target=poll_cpu)
    t_cpu.daemon = True
    t_cpu.start()

    gpu_tracker = _GpuTracker(interval) if gpu else None

    start = time.perf_counter()
    try:
        yield res
    finally:
        end = time.perf_counter()
        stop.set()
        t_cpu.join()
        cpu_samples.append(_tree_rss_mb(pid))
        peak = max(cpu_samples)
        res['wall_s'] = end - start
        res['cpu_rss_baseline_mb'] = baseline
        res['cpu_rss_peak_mb'] = peak
        res['cpu_rss_peak_increase_mb'] = peak - baseline
        g = gpu_tracker.stop() if gpu_tracker is not None else []
        if g:
            res['gpu_baseline_mb'] = g[0]
            res['gpu_peak_mb'] = max(g)
            res['gpu_peak_increase_mb'] = max(g) - g[0]
        else:
            res['gpu_baseline_mb'] = res['gpu_peak_mb'] = res['gpu_peak_increase_mb'] = None


# ### CPU ##############################################################################################################

def _status_kb(pid, field):
    # Value of a 'Field:   123 kB' line of /proc/<pid>/status in kB, None if unavailable
    try:
        with open('/proc/{}/status'.format(pid)) as f:
            for line in f:
                if line.startswith(field + ':'):
                    return float(line.split()[1])
    except (IOError, OSError, ValueError, IndexError):
        pass
    return None


def _children(pid):
    # Direct children of pid (all threads), via /proc/<pid>/task/<tid>/children
    out = []
    try:
        tids = os.listdir('/proc/{}/task'.format(pid))
    except OSError:
        return out
    for tid in tids:
        try:
            with open('/proc/{}/task/{}/children'.format(pid, tid)) as f:
                out.extend(int(c) for c in f.read().split())
        except (IOError, OSError, ValueError):
            continue
    return out


def _tree_rss_mb(pid):
    # Summed VmRSS of pid and all its descendants in MiB
    total = 0.0
    stack, seen = [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        kb = _status_kb(p, 'VmRSS')
        if kb is not None:
            total += kb
        stack.extend(_children(p))
    return total / 1024.0


# ### GPU ##############################################################################################################

class _GpuTracker(object):
    # Persistent nvidia-smi process printing memory.used (MiB) every interval

    def __init__(self, interval):
        self.samples = []
        self.proc = None
        self.thread = None
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip() or '0'
        cmd = ['nvidia-smi', '-i', gpu, '-lms', str(max(1, int(interval * 1000))),
               '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         universal_newlines=True, bufsize=1)
        except (OSError, ValueError):
            self.proc = None
            return
        first = self.proc.stdout.readline().strip()
        try:
            self.samples.append(float(first))
        except ValueError:
            pass
        self.thread = threading.Thread(target=self._poll)
        self.thread.daemon = True
        self.thread.start()

    def _poll(self):
        for line in self.proc.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                continue

    def stop(self):
        if self.proc is None:
            return []
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)
        return list(self.samples)
