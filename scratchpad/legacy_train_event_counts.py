"""Env: fx_plot. Events per legacy training sample (y_sample_<NN>_train.npy, read from the .npy header only), from the
legacy np_files of a sibling checkout (the legacy experiment does not record them). The legacy 'random' order uses
sample_00 ... sample_<n-1> (validation/utils/val_utils.py::n_samples_experiment_helper, sample_order_file None).
-> scratchpad/results/legacy_criterion_check/legacy_train_event_counts.csv (dataset, sample_index, n_events)."""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else '/data/bionets/ac07izid/FLAG-X-validation-copy/data/np_files'
DIRS = {'Imstat': 'imstat', 'LT1': 'lymphoma_tube1', 'LT2': 'lymphoma_tube2', 'LT1b': 'lymphoma_tube1_binary',
        'LT2b': 'lymphoma_tube2_binary', 'Flowcyt': 'flowcyt'}
rows = []
for ds, d in DIRS.items():
    trafo = 'log10_cutoff100' if d == 'flowcyt' else 'log10_w_custom_cutoffs'
    p = os.path.join(SRC, d, trafo, 'sample_wise_train')
    i = 0
    while os.path.exists(os.path.join(p, 'y_sample_{:02d}_train.npy'.format(i))):
        y = np.load(os.path.join(p, 'y_sample_{:02d}_train.npy'.format(i)), mmap_mode='r')
        rows.append(dict(dataset=ds, sample_index=i, n_events=len(y)))
        i += 1
out = os.path.join(HERE, 'results', 'legacy_criterion_check', 'legacy_train_event_counts.csv')
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).groupby('dataset')['n_events'].describe().round(0))
