"""
Data curation: raw FCS/CSV files -> one untransformed X (float32, events x channels) and y (int64) .npy file per sample.

Output per dataset in data/curated/<dataset>/:
- sample_<NNN>_X.npy, sample_<NNN>_y.npy: channels in the order of config.CHANNELS[<dataset>], labels as new integers
- meta_samples.csv: new sample name, original filename, number of events, additional sample info from the filename
- meta_labels.csv: new integer label, original label(s), short name, description

Files are loaded with flagx (one FlowDataManager per file). If an FCS file contains a spillover matrix, the sample is
compensated with flagx; no other transformation is applied. Channels are renamed with the per-dataset channel name
mapping below (keys: $PnN); config.CHANNELS decides which channels are kept and their order.

Env: fx_flagx_cpu (envs/flagx_cpu.yml)
"""

import os
import re
import numpy as np
import pandas as pd
from flagx.io import FlowDataManager

from config import DATA_RAW, DATA_CURATED, CHANNELS


OVERWRITE = False  # If False, datasets whose meta_samples.csv already exists are skipped


# ### Imstat ###########################################################################################################

IMSTAT_RAW = os.path.join(DATA_RAW, 'imstat')
IMSTAT_SPLIT_DEVELOPMENT = os.path.join(DATA_RAW, 'imstat_data_split_development.csv')  # Legacy train/test split

# Two channel name variants exist in the raw files ('-PC5.5' vs 'FL4 INT' for the unused 4th fluorescence channel)
IMSTAT_CHANNEL_MAP = {
    'FS PEAK': 'FS_PEAK', 'FS INT': 'FS_INT', 'FS TOF': 'FS_TOF', 'SS INT': 'SS_INT',
    '16-FITC': 'CD16_FITC', '56-PE': 'CD56_PE', '3-ECD': 'CD3_ECD', '-PC5.5': 'FL4_INT', 'FL4 INT': 'FL4_INT',
    '4-PC7': 'CD4_PC7', '19-APC': 'CD19_APC', '14-APC700': 'CD14_APC700', 'FL8 INT': 'FL8_INT',
    '8-PB': 'CD8_PB', '45-CO': 'CD45_CO', 'TIME': 'TIME',
}
IMSTAT_LABEL_KEY = 'population'
IMSTAT_LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}  # Original -> new
IMSTAT_LABEL_NAMES = {  # New -> (short name, description)
    0: ('B', 'B cells'),
    1: ('Th', 'T helper cells'),
    2: ('NK', 'NK cells'),
    3: ('M16', 'atypical CD16+ monocytes'),
    4: ('M', 'other monocytes'),
    5: ('G', 'granulocytes'),
    6: ('X', 'sorted out'),
    7: ('O', 'unclassified'),
}


def curate_imstat():
    dataset = 'imstat'
    filenames = sorted(f for f in os.listdir(IMSTAT_RAW) if f.endswith('.fcs'))
    legacy_split = pd.read_csv(IMSTAT_SPLIT_DEVELOPMENT).set_index('filename')['mode']

    meta = []
    for i, fn in enumerate(filenames):
        df = load_sample(IMSTAT_RAW, fn)
        X, y = to_X_y(df, IMSTAT_CHANNEL_MAP, CHANNELS[dataset], IMSTAT_LABEL_KEY, IMSTAT_LABEL_MAP)
        sample = save_sample(dataset, i, X, y)

        # Filenames: '<date>-<run> <lysis method> ...' or 'ER_<id>_H1_<date>_ED'
        m = re.match(r'^(\d{8}-\d) (\S+) ', fn)
        meta.append({
            'sample': sample,
            'filename_original': fn,
            'n_events': X.shape[0],
            'run_id': m.group(1) if m else None,
            'lysis_method': m.group(2) if m else None,
            'legacy_split': legacy_split[fn[:-len('.fcs')]],
        })

    save_meta(dataset, meta, IMSTAT_LABEL_MAP, IMSTAT_LABEL_NAMES)


# ### Lymphoma (Lt1, Lt2, binary variants Lt1b, Lt2b) ##################################################################

LYMPHOMA_RAW = {
    'lt1': os.path.join(DATA_RAW, 'lymphoma', 'concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels'),
    'lt2': os.path.join(DATA_RAW, 'lymphoma', 'concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels'),
}
LYMPHOMA_RAW['lt1b'] = LYMPHOMA_RAW['lt1']
LYMPHOMA_RAW['lt2b'] = LYMPHOMA_RAW['lt2']

LYMPHOMA_CHANNEL_MAP = {
    'lt1': {
        'FS': 'FS', 'SS': 'SS', 'kappavCD8_FITC': 'kappavCD8_FITC', 'lambdavCD7_PE': 'lambdavCD7_PE',
        'CD23_ECD': 'CD23_ECD', 'CD79bvCD4_PC5.5': 'CD79bvCD4_PC5.5', 'CD5_PC7': 'CD5_PC7', 'CD38_APC': 'CD38_APC',
        'CD19_APC_A700': 'CD19_APC_A700', 'CD20vCD3_APC_A750': 'CD20vCD3_APC_A750', 'FMC7vCD2_PB': 'FMC7vCD2_PB',
        'CD45_KrOr': 'CD45_KrOr', 'TIME': 'TIME',
    },
    'lt2': {
        'FS': 'FS', 'SS': 'SS', 'CD103_FITC': 'CD103_FITC', 'CD43_PE': 'CD43_PE', 'CD25_ECD': 'CD25_ECD',
        'CD10_PC5.5': 'CD10_PC5.5', 'CD200_PC7': 'CD200_PC7', 'CD52_APC': 'CD52_APC',
        'CD11c_APC_A700': 'CD11c_APC_A700', 'CD20_APC_A750': 'CD20_APC_A750', 'IgM_PB': 'IgM_PB',
        'CD19_KrOr': 'CD19_KrOr', 'TIME': 'TIME',
    },
}
LYMPHOMA_CHANNEL_MAP['lt1b'] = LYMPHOMA_CHANNEL_MAP['lt1']
LYMPHOMA_CHANNEL_MAP['lt2b'] = LYMPHOMA_CHANNEL_MAP['lt2']

LYMPHOMA_LABEL_KEY = 'population'
# Original labels: 1 B cells, 9 B cells outside the FS/SS gate (dying B cells), 7 CD45- cells (mostly erythrocytes),
# 10 other cells. Tube 2 contains no label 7.
LYMPHOMA_LABEL_MAP = {
    'lt1': {1: 0, 9: 1, 7: 2, 10: 3},
    'lt2': {1: 0, 9: 1, 10: 2},
    'lt1b': {7: 0, 10: 0, 1: 1, 9: 1},
    'lt2b': {10: 0, 1: 1, 9: 1},
}
LYMPHOMA_LABEL_NAMES = {
    'lt1': {0: ('B', 'B cells'), 1: ('dyB', 'dying B cells'), 2: ('X', 'CD45- cells (mostly erythroid)'),
            3: ('O', 'other cells')},
    'lt2': {0: ('B', 'B cells'), 1: ('dyB', 'dying B cells'), 2: ('O', 'other cells')},
    'lt1b': {0: ('O', 'other cells'), 1: ('B', 'B cells incl. dying B cells')},
    'lt2b': {0: ('O', 'other cells'), 1: ('B', 'B cells incl. dying B cells')},
}


def curate_lymphoma(dataset):
    raw = LYMPHOMA_RAW[dataset]
    filenames = sorted(f for f in os.listdir(raw) if f.endswith('.fcs'))

    meta = []
    for i, fn in enumerate(filenames):
        df = load_sample(raw, fn)
        X, y = to_X_y(
            df, LYMPHOMA_CHANNEL_MAP[dataset], CHANNELS[dataset], LYMPHOMA_LABEL_KEY, LYMPHOMA_LABEL_MAP[dataset]
        )
        sample = save_sample(dataset, i, X, y)

        # Filenames: '<patient id>_<lymphoma subtype>_T<tube>_d13_N<events>k'
        m = re.match(r'^(\d+)_([A-Za-z]+)_T\d_', fn)
        meta.append({
            'sample': sample,
            'filename_original': fn,
            'n_events': X.shape[0],
            'patient_id': m.group(1),
            'subtype': m.group(2),
        })

    save_meta(dataset, meta, LYMPHOMA_LABEL_MAP[dataset], LYMPHOMA_LABEL_NAMES[dataset])


# ### Flowcyt ##########################################################################################################

FLOWCYT_RAW = os.path.join(DATA_RAW, 'flowcyt', 'data_original')

FLOWCYT_CHANNEL_MAP = {
    'FS INT': 'FS_INT', 'SS PEAK': 'SS_PEAK', 'SS INT': 'SS_INT', 'SS TOF': 'SS_TOF',
    'FL1 INT_CD14-FITC': 'CD14_FITC', 'FL2 INT_CD19-PE': 'CD19_PE', 'FL3 INT_CD13-ECD': 'CD13_ECD',
    'FL4 INT_CD33-PC5.5': 'CD33_PC5.5', 'FL5 INT_CD34-PC7': 'CD34_PC7', 'FL6 INT_CD117-APC': 'CD117_APC',
    'FL7 INT_CD7-APC700': 'CD7_APC700', 'FL8 INT_CD16-APC750': 'CD16_APC750', 'FL9 INT_HLA-PB': 'HLA_PB',
    'FL10 INT_CD45-KO': 'CD45_KO', 'TIME': 'TIME',
}
FLOWCYT_LABEL_KEY = 'label'
FLOWCYT_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
FLOWCYT_LABEL_NAMES = {
    0: ('T', 'T cells'),
    1: ('B', 'B cells'),
    2: ('M', 'monocytes'),
    3: ('Ma', 'mast cells'),
    4: ('HSPC', 'hematopoietic stem and progenitor cells'),
    5: ('O', 'other cells'),
}


def curate_flowcyt():
    dataset = 'flowcyt'
    # Filenames: 'Case_<id>.csv', sorted by case id
    filenames = sorted(
        (f for f in os.listdir(FLOWCYT_RAW) if f.endswith('.csv')),
        key=lambda f: int(re.match(r'^Case_(\d+)\.csv$', f).group(1)),
    )

    meta = []
    for i, fn in enumerate(filenames):
        df = load_sample(FLOWCYT_RAW, fn)
        X, y = to_X_y(df, FLOWCYT_CHANNEL_MAP, CHANNELS[dataset], FLOWCYT_LABEL_KEY, FLOWCYT_LABEL_MAP)
        sample = save_sample(dataset, i, X, y)
        meta.append({
            'sample': sample,
            'filename_original': fn,
            'n_events': X.shape[0],
            'case_id': int(re.match(r'^Case_(\d+)\.csv$', fn).group(1)),
        })

    save_meta(dataset, meta, FLOWCYT_LABEL_MAP, FLOWCYT_LABEL_NAMES)


# ### Helpers ##########################################################################################################

def load_sample(raw_dir, fn):
    # Load one FCS/CSV file with flagx; FCS columns are named by $PnN (reindex=False)
    fdm = FlowDataManager(data_file_names=[fn], data_file_path=raw_dir, save_path=raw_dir, verbosity=0)
    fdm.load_data_files_to_anndata(reindex=False)
    adata = fdm.anndata_list_[0]

    # Compensate if the FCS file contains a spillover matrix. flagx only logs compensation errors, so check the log.
    if adata.uns.get('meta', {}).get('spill') is not None:
        fdm.sample_wise_compensation()
        log = fdm.compensation_log_['logs'].iloc[0]
        if log != 'compensation applied successfully':
            raise RuntimeError('Compensation of {} failed: {}'.format(fn, log))

    return adata.to_df()


def to_X_y(df, channel_map, channels, label_key, label_map):
    unmapped = set(df.columns) - set(channel_map) - {label_key}
    if unmapped:
        raise ValueError('Channels without a name mapping: {}'.format(sorted(unmapped)))
    df = df.rename(columns=channel_map)

    X = df[channels].to_numpy(dtype=np.float32)

    y_orig = df[label_key].to_numpy()
    if not np.all(y_orig == np.round(y_orig)):
        raise ValueError('Non-integer labels found')
    y_orig = y_orig.astype(np.int64)
    unknown = set(np.unique(y_orig).tolist()) - set(label_map)
    if unknown:
        raise ValueError('Labels without a mapping: {}'.format(sorted(unknown)))
    y = np.vectorize(label_map.get, otypes=[np.int64])(y_orig)

    return X, y


def save_sample(dataset, i, X, y):
    sample = 'sample_{:03d}'.format(i + 1)
    out_dir = os.path.join(DATA_CURATED, dataset)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, sample + '_X.npy'), X)
    np.save(os.path.join(out_dir, sample + '_y.npy'), y)
    return sample


def save_meta(dataset, meta, label_map, label_names):
    out_dir = os.path.join(DATA_CURATED, dataset)
    meta_labels = pd.DataFrame([
        {
            'label': new,
            'label_original': ';'.join(str(o) for o in sorted(o for o, n in label_map.items() if n == new)),
            'name': name,
            'description': description,
        }
        for new, (name, description) in sorted(label_names.items())
    ])
    meta_labels.to_csv(os.path.join(out_dir, 'meta_labels.csv'), index=False)
    # Written last: its existence marks the dataset as complete
    pd.DataFrame(meta).to_csv(os.path.join(out_dir, 'meta_samples.csv'), index=False)


def is_done(dataset):
    return os.path.exists(os.path.join(DATA_CURATED, dataset, 'meta_samples.csv'))


if __name__ == '__main__':

    curate_functions = {
        'imstat': curate_imstat,
        'lt1': lambda: curate_lymphoma('lt1'),
        'lt2': lambda: curate_lymphoma('lt2'),
        'lt1b': lambda: curate_lymphoma('lt1b'),
        'lt2b': lambda: curate_lymphoma('lt2b'),
        'flowcyt': curate_flowcyt,
    }

    for dataset, curate in curate_functions.items():
        if is_done(dataset) and not OVERWRITE:
            print('{}: already curated, skipped'.format(dataset))
            continue
        print('{}: curating ...'.format(dataset))
        curate()

    print('done')
