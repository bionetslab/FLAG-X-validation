"""
Runner: FLAG-X SOM classifier (flagx.gating.SOMClassifier, flagx 0.2.1). Env fx_flagx_cpu (Somoclu, CPU).

- flagx has no seed argument; numpy is seeded anyway. With initialization='pca' the initialization is deterministic,
  but the multithreaded Somoclu training is not: two identical fits give different codebooks (smoke test 2026-11-12,
  max abs diff ~0.03). SOM is therefore a stochastic method in the benchmark (3 model seeds); the seed does not
  control this variation, so single runs are not bit-reproducible.
- A fresh instance is built per fit: SOMClassifier.fit overwrites radius_0 (flagx som_classifier.py L147-151, L203).
- predict returns -1 for events whose best-matching unit has no labelled training events (built-in abstention of the
  method, kept and scored as abstention, D8).
- Threads: Somoclu uses OpenMP (OMP_NUM_THREADS, set in the job scripts).
- Memory: flagx 0.2.1 finds the best-matching units (BMUs) via the full float64 distance matrix events x SOM units
  (som_classifier.py L848-898; the chunked branch L859-866 only applies to <=10x10 SOMs with <=100k events and stacks
  the chunks back into the full matrix). In annotate_som after the training this needs 5 GB per 1M training events
  for a 25x25 SOM (OOM on BLAST110 in the benchmark pilot). _ChunkedBMUSOMClassifier computes the BMUs chunk-wise with
  the same distance function and keeps only the argmin per chunk: identical BMUs, memory independent of the number of
  events. Only the BMU path is changed; the training (Somoclu) is untouched. topographic_error / transform, which use
  the full surface state, are not used by the benchmark and not supported by the subclass.
"""
import numpy as np
from flagx.gating import SOMClassifier

from validation.runners._sklearn_common import fit, input_arm  # noqa: F401

MODULES = ['numpy', 'flagx', 'somoclu', 'sklearn']

BMU_CHUNK_SIZE = 100000  # events per chunk (25x25 SOM: 0.5 GB float64 distances per chunk; as fast as 1M)


class _ChunkedBMUSOMClassifier(SOMClassifier):
    # MOD: memory-efficient BMU computation. Replaces the pair
    #   self._custom_get_bmus(activation_map=self._custom_get_surface_state(data=X))
    # (flagx 0.2.1, flagx/gating/som_classifier.py L848-898, used in annotate_som L395, predict L269,
    # predict_proba L323) by a chunk-wise version: _custom_get_surface_state passes the data through, and
    # _custom_get_bmus computes the distances per chunk with flagx's own _compute_distances (L875-884) and the
    # argmin / divmod of _custom_get_bmus (L895-898). The per-event distances are computed by the same code, so the BMUs
    # are identical to the original.

    def _custom_get_surface_state(self, data):
        return np.ascontiguousarray(data)

    def _custom_get_bmus(self, activation_map):
        data = activation_map  # the data, see _custom_get_surface_state
        codebook_reshaped = self.som_.codebook.reshape(-1, self.som_.codebook.shape[2])
        bmu_indices = np.empty(data.shape[0], dtype=np.int64)
        for start in range(0, data.shape[0], BMU_CHUNK_SIZE):
            stop = min(start + BMU_CHUNK_SIZE, data.shape[0])
            distances = self._compute_distances(data[start:stop], codebook_reshaped)
            bmu_indices[start:stop] = np.argmin(distances, axis=1)
            del distances  # free before the next chunk is allocated (else two chunks are alive at once)
        rows, cols = np.divmod(bmu_indices, self.som_dimensions[1])
        return np.column_stack((rows, cols))


def build(params, seed, dataset, n_jobs):
    np.random.seed(seed)
    params = dict(params)
    if 'som_dimensions' in params:
        params['som_dimensions'] = tuple(params['som_dimensions'])
    return _ChunkedBMUSOMClassifier(verbosity=0, **params)


def predict(model, X, proba=False):
    y_pred = model.predict(X)
    # predict_proba: class distribution of the BMU (columns = model.og_classes_); labels from predict (keeps -1)
    return y_pred, (model.predict_proba(X) if proba else None)
