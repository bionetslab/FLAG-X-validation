"""
Constants shared by several scripts.

Imported from every env, including the Python 3.5 envs (fx_deepcytof, fx_acdc):
keep this file stdlib-only and Python 3.5-compatible (no f-strings, no variable annotations).
"""

import os

# ### Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(ROOT, 'data', 'raw')
DATA_CURATED = os.path.join(ROOT, 'data', 'curated')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')

# ### Datasets
DATASETS = ['imstat', 'lt1', 'lt2', 'lt1b', 'lt2b', 'flowcyt']

# Channels kept per dataset, in the column order of the curated X matrices (harmonized names, see curate_data.py)
CHANNELS = {
    'imstat': [
        'FS_INT', 'SS_INT', 'CD16_FITC', 'CD56_PE', 'CD3_ECD', 'CD4_PC7', 'CD19_APC', 'CD14_APC700', 'CD8_PB',
        'CD45_CO',
    ],
    'lt1': [
        'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7', 'CD38_APC',
        'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr',
    ],
    'lt2': [
        'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7', 'CD52_APC', 'CD11c_APC_A700',
        'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr',
    ],
    'flowcyt': [
        'FS_INT', 'SS_INT', 'CD14_FITC', 'CD19_PE', 'CD13_ECD', 'CD33_PC5.5', 'CD34_PC7', 'CD117_APC', 'CD7_APC700',
        'CD16_APC750', 'HLA_PB', 'CD45_KO',
    ],
}
CHANNELS['lt1b'] = CHANNELS['lt1']
CHANNELS['lt2b'] = CHANNELS['lt2']
