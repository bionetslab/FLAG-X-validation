"""
Shared preprocessing: a transformation followed by a per-channel scaling, fitted on the training data of a task.

Lightweight numpy-only reimplementation, so it runs in every method env, including the
Python 3.5 envs (fx_deepcytof, fx_acdc): no f-strings, no variable annotations.

Transformations (trafo, trafo_params):
- 'log10_cutoff', {'cutoff': float or {channel: float}}: values <= cutoff are set to log10(cutoff), all others to
  log10(value); a dict gives one cutoff per channel and must cover all channels. Required.
- 'arcsinh', {'cofactor': float}: arcsinh(value / cofactor), same as pytometry normalize_arcsinh. Required.
- 'none', {}

Scalings (scaling, scaling_params), fitted on the pooled transformed training events:
- 'zscore', {}: (x - mean) / std per channel
- 'minmax', {'feature_range': (a, b)}: (x - min) / (max - min) * (b - a) + a per channel, default (0, 1), as sklearn
  MinMaxScaler; test values outside the training range are not clipped
- 'none', {}

A fitted preprocessor is saved as JSON (save / load): constructor arguments plus the fitted scaling parameters as plain
lists, readable in every env.

shared_preprocessor(dataset, arm) builds the input of the experiments (D3): 'shared' = log10 with the dataset's
cut-offs (config.LOG10_CUTOFFS), no scaling; 'native' = untransformed (methods with internal preprocessing).
preprocessor_state(p) gives a fitted preprocessor as a dict for the run JSONs.
"""

import json
import numpy as np

import config


TRAFOS = ('log10_cutoff', 'arcsinh', 'none')
SCALINGS = ('zscore', 'minmax', 'none')


class Preprocessor(object):

    def __init__(self, channels, trafo='none', trafo_params=None, scaling='none', scaling_params=None):
        """
        Args:
            channels (list of str): Channel names in the column order of X, e.g. config.CHANNELS['imstat']. Only used
                to look up per-channel cutoffs and to check the number of columns of X.
            trafo (str): Transformation, one of TRAFOS: 'log10_cutoff', 'arcsinh' or 'none'.
            trafo_params (dict or None): Parameters of the transformation:
                - 'log10_cutoff': {'cutoff': c}, c is a number (same cutoff for all channels, e.g. 100) or a dict
                  {channel name: number} with an entry for every channel (e.g. config.LOG10_CUTOFFS['imstat'])
                - 'arcsinh': {'cofactor': number}, e.g. {'cofactor': 150}
                - 'none': None or {}
            scaling (str): Scaling, one of SCALINGS: 'zscore', 'minmax' or 'none'.
            scaling_params (dict or None): Parameters of the scaling:
                - 'minmax': None or {'feature_range': (a, b)}, the output range of the training data, e.g. (-1, 1);
                  default (0, 1)
                - 'zscore', 'none': None or {}

        Example:
            Preprocessor(CHANNELS['imstat'], 'log10_cutoff', {'cutoff': LOG10_CUTOFFS['imstat']}, 'zscore')
        """
        if trafo not in TRAFOS:
            raise ValueError('trafo must be one of {}, got {!r}'.format(TRAFOS, trafo))
        if scaling not in SCALINGS:
            raise ValueError('scaling must be one of {}, got {!r}'.format(SCALINGS, scaling))
        self.channels = list(channels)
        self.trafo = trafo
        self.trafo_params = dict(trafo_params or {})
        self.scaling = scaling
        self.scaling_params = dict(scaling_params or {})

        # Required transformation parameters
        required = {'log10_cutoff': 'cutoff', 'arcsinh': 'cofactor'}.get(trafo)
        if required is not None and required not in self.trafo_params:
            raise ValueError("trafo {!r} requires trafo_params={{{!r}: ...}}, got {!r}".format(
                trafo, required, self.trafo_params))

        # Per-channel cutoffs in column order (log10_cutoff only)
        self.cutoffs_ = None
        if trafo == 'log10_cutoff':
            cutoff = self.trafo_params['cutoff']
            if isinstance(cutoff, dict):
                missing = [c for c in self.channels if c not in cutoff]
                if missing:
                    raise ValueError('no cutoff for channels {}'.format(missing))
                self.cutoffs_ = np.array([cutoff[c] for c in self.channels], dtype=np.float64)
            else:
                self.cutoffs_ = np.full(len(self.channels), cutoff, dtype=np.float64)

        # Fitted scaling parameters: x_scaled = (x - center_) / scale_
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Fits the scaling on the transformed training events (the transformation itself has no fitted parameters).

        Args:
            X (np.ndarray or list of np.ndarray): Untransformed training data, either one array of shape
                (n_events, n_channels) or a list of such arrays (one per sample, e.g. the curated sample_<NNN>_X.npy
                of all training samples), which are pooled. Columns in the order of channels.

        Returns:
            Preprocessor: self (fitted), so calls can be chained.
        """
        X = self._transform(self._stack(X))
        if self.scaling == 'zscore':
            self.center_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
        elif self.scaling == 'minmax':
            # (x - min) / (max - min) * (b - a) + a  =  (x - center_) / scale_
            # with scale_ = (max - min) / (b - a) and center_ = min - a * scale_
            a, b = self.scaling_params.get('feature_range', (0.0, 1.0))
            x_min, x_max = X.min(axis=0), X.max(axis=0)
            x_range = x_max - x_min
            x_range[x_range == 0] = b - a  # Constant channels -> a (as sklearn)
            self.scale_ = x_range / (b - a)
            self.center_ = x_min - a * self.scale_
        else:
            self.center_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1.0  # Constant channels are only centered
        return self

    def transform(self, X):
        """
        Applies the transformation and the fitted scaling.

        Args:
            X (np.ndarray or list of np.ndarray): Untransformed data, one array of shape (n_events, n_channels) or a
                list of such arrays, which are pooled (row order kept). Columns in the order of channels.

        Returns:
            np.ndarray: float32 array of shape (n_events, n_channels); for a list, the rows of all arrays stacked.
        """
        if self.center_ is None:
            raise RuntimeError('Preprocessor is not fitted')
        X = self._transform(self._stack(X))
        return ((X - self.center_) / self.scale_).astype(np.float32)

    def fit_transform(self, X):
        """fit(X) followed by transform(X); same input and output as transform."""
        return self.fit(X).transform(X)

    def save(self, path):
        """
        Saves the constructor arguments and the fitted scaling parameters (null if not fitted) as JSON.

        Args:
            path (str): Output file, e.g. '<task dir>/preprocessor.json'.
        """
        state = {
            'channels': self.channels,
            'trafo': self.trafo,
            'trafo_params': self.trafo_params,
            'scaling': self.scaling,
            'scaling_params': self.scaling_params,
            'center_': None if self.center_ is None else self.center_.tolist(),
            'scale_': None if self.scale_ is None else self.scale_.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=_to_builtin)

    @classmethod
    def load(cls, path):
        """
        Rebuilds a preprocessor saved with save (via __init__, so the parameters are checked again).

        Args:
            path (str): JSON file written by save.

        Returns:
            Preprocessor: fitted if it was fitted when saved.
        """
        with open(path) as f:
            state = json.load(f)
        p = cls(state['channels'], state['trafo'], state['trafo_params'], state['scaling'], state['scaling_params'])
        if state['center_'] is not None:
            p.center_ = np.array(state['center_'], dtype=np.float64)
            p.scale_ = np.array(state['scale_'], dtype=np.float64)
        return p

    def _stack(self, X):
        # Pool a list of samples into one array; computations in float64
        X = np.vstack(X) if isinstance(X, (list, tuple)) else np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.channels):
            raise ValueError('X must have shape (events, {}), got {}'.format(len(self.channels), X.shape))
        return X.astype(np.float64)

    def _transform(self, X):
        if self.trafo == 'log10_cutoff':
            # log10 only where x > cutoff, so values <= cutoff (incl. negative ones) raise no warning
            out = np.empty_like(X)
            out[:] = np.log10(self.cutoffs_)  # Set each col to its respective cutoff value
            np.log10(X, out=out, where=X > self.cutoffs_)
            return out
        if self.trafo == 'arcsinh':
            return np.arcsinh(X / self.trafo_params['cofactor'])
        return X

    def __repr__(self):
        return 'Preprocessor(trafo={!r}, trafo_params={!r}, scaling={!r}, scaling_params={!r})'.format(
            self.trafo, self.trafo_params, self.scaling, self.scaling_params)


def shared_preprocessor(dataset, arm):
    """
    Unfitted preprocessor of an input arm (D3) for a dataset.

    Args:
        dataset (str): Dataset name (config.DATASETS).
        arm (str): 'shared' (log10 with the cut-offs of config.LOG10_CUTOFFS[dataset], no scaling) or 'native'
            (no transformation; the method preprocesses internally). The runners give it via input_arm(params).

    Returns:
        Preprocessor
    """
    if arm == 'shared':
        return Preprocessor(config.CHANNELS[dataset], 'log10_cutoff', {'cutoff': config.LOG10_CUTOFFS[dataset]})
    if arm == 'native':
        return Preprocessor(config.CHANNELS[dataset], 'none')
    raise ValueError('unknown input arm {!r}'.format(arm))


def preprocessor_state(p):
    """Parameters and fitted scaling of a Preprocessor as a JSON-ready dict (for the run JSONs)."""
    return {'trafo': p.trafo, 'trafo_params': p.trafo_params, 'scaling': p.scaling,
            'scaling_params': p.scaling_params,
            'center_': None if p.center_ is None else p.center_.tolist(),
            'scale_': None if p.scale_ is None else p.scale_.tolist()}


def _to_builtin(obj):
    # JSON fallback for numpy scalars and arrays in the parameters (e.g. a cutoff given as np.float32)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('{!r} is not JSON serializable'.format(obj))
