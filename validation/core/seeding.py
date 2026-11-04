"""
Deterministic seeds from config.SEED + a task identity.

stdlib only, Python 3.5-compatible (runs in every method env): no f-strings, no variable annotations.
"""

import hashlib

import config


def derive_seed(*keys):
    """
    Derives a reproducible seed from config.SEED and any number of keys (str, int, float, bool, None).

    The keys are joined to a string (type-tagged, so 1 and '1' differ) and hashed with sha256, so the seed is the
    same in every process, env and Python version (unlike hash()).

    Example:
        derive_seed('tune', 'imstat', 0, 1, 'sample_001')

    Returns:
        int: Seed in [0, 2**31 - 1), usable for numpy, torch, TF, R and sklearn.
    """
    parts = [_key(config.SEED)] + [_key(k) for k in keys]
    digest = hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()
    return int(digest[:16], 16) % (2 ** 31 - 1)


def _key(k):
    if k is None or isinstance(k, (bool, int, float, str)):
        return '{}:{!r}'.format(type(k).__name__, k)
    # numpy scalars -> Python scalars
    if hasattr(k, 'item'):
        return _key(k.item())
    raise TypeError('seed keys must be str, int, float, bool or None, got {!r}'.format(k))
