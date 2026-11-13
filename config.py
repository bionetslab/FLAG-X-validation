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
DATA_SPLITS = os.path.join(ROOT, 'data', 'splits')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')

# ### Randomness
SEED = 42

# ### Datasets (run order of the pipeline: curation, splits, benchmark manifests; plot order: PLOT_DATASETS)
DATASETS = [
    'imstat', 'lt1', 'lt2', 'lt1b', 'lt2b', 'flowcyt', 'blast110_p1', 'blast110_p2', 'blast110_p3', 'blast110_p4',
]

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
    'blast110_p1': [
        'FSC_A', 'SSC_A', 'CD7', 'CD56', 'CD34', 'CD117', 'CD33', 'HLA-DR', 'CD13', 'CD45',
    ],
    'blast110_p2': [
        'FSC_A', 'SSC_A', 'CD15', 'CD22', 'CD34', 'CD117', 'CD19', 'HLA-DR', 'CD13', 'CD45',
    ],
    'blast110_p3': [
        'FSC_A', 'SSC_A', 'CD36', 'CD14', 'CD34', 'CD117', 'CD11b', 'HLA-DR', 'CD13', 'CD45',
    ],
    'blast110_p4': [
        'FSC_A', 'SSC_A', 'CD2', 'CD133', 'CD34', 'CD117', 'CD33', 'HLA-DR', 'CD13', 'CD45',
    ],
}
CHANNELS['lt1b'] = CHANNELS['lt1']
CHANNELS['lt2b'] = CHANNELS['lt2']

# ### Preprocessing
# Lower cutoffs for the log10 transformation (validation.core.preprocessing, trafo 'log10_cutoff'): one value per channel
# (biologically informed, from the paper, legacy/main.py) or one uniform value
LOG10_CUTOFFS = {
    'imstat': {
        'FS_INT': 100000, 'SS_INT': 20000, 'CD16_FITC': 250, 'CD56_PE': 450, 'CD3_ECD': 700, 'CD4_PC7': 1200,
        'CD19_APC': 1700, 'CD14_APC700': 900, 'CD8_PB': 450, 'CD45_CO': 500,
    },
    'lt1': {
        'FS': 100000, 'SS': 20000, 'kappavCD8_FITC': 500, 'lambdavCD7_PE': 400, 'CD23_ECD': 500,
        'CD79bvCD4_PC5.5': 1200, 'CD5_PC7': 300, 'CD38_APC': 700, 'CD19_APC_A700': 150, 'CD20vCD3_APC_A750': 500,
        'FMC7vCD2_PB': 500, 'CD45_KrOr': 1000,
    },
    'lt2': {
        'FS': 100000, 'SS': 20000, 'CD103_FITC': 300, 'CD43_PE': 1500, 'CD25_ECD': 1000, 'CD10_PC5.5': 1000,
        'CD200_PC7': 1000, 'CD52_APC': 150, 'CD11c_APC_A700': 200, 'CD20_APC_A750': 300, 'IgM_PB': 400,
        'CD19_KrOr': 200,
    },
    'flowcyt': 100,
    'blast110_p1': 100, 'blast110_p2': 100, 'blast110_p3': 100, 'blast110_p4': 100,
}
LOG10_CUTOFFS['lt1b'] = LOG10_CUTOFFS['lt1']
LOG10_CUTOFFS['lt2b'] = LOG10_CUTOFFS['lt2']


# ### Plotting
# Order, display names and colors of the datasets and methods in all figures. Only plot scripts use these, so they can be
# changed freely (DATASETS above is the run order: changing it would change the benchmark manifest indices).
PLOT_DATASETS = [
    'imstat', 'flowcyt', 'lt1', 'lt2', 'lt1b', 'lt2b', 'blast110_p1', 'blast110_p2', 'blast110_p3', 'blast110_p4',
]
DATASET_NAMES = {
    'imstat': 'Imstat', 'lt1': 'Lt1', 'lt2': 'Lt2', 'lt1b': 'Lt1b', 'lt2b': 'Lt2b', 'flowcyt': 'Flowcyt',
    'blast110_p1': 'BLAST110 P1', 'blast110_p2': 'BLAST110 P2', 'blast110_p3': 'BLAST110 P3',
    'blast110_p4': 'BLAST110 P4',
}
PLOT_METHODS = [
    'mlp', 'dgcytof', 'deepcytof', 'som', 'gatemeclass', 'gmm_mocking', 'lda_abdelaal',
    'lda_sklearn', 'logreg', 'knn', 'rf', 'hgb'  
]
METHOD_NAMES = {
    'som': 'SOM classifier', 'mlp': 'MLP classifier',
    'dgcytof': 'DGCyTOF', 'deepcytof': 'DeepCyTOF', 'gatemeclass': 'GateMeClass', 'lda_abdelaal': 'LDA-Abdelaal',
    'gmm_mocking': 'GMM-Mocking',
    'rf': 'RF', 'hgb': 'HGB', 'knn': 'kNN', 'logreg': 'LogReg', 'lda_sklearn': 'LDA-sklearn',
}
METHOD_COLORS = {  # Colorblind-friendly, muted: Paul Tol's 'muted' scheme (9) + 3 of his 'light' scheme
    'som': '#88CCEE', 'mlp': '#44AA99', 'dgcytof': '#117733', 'deepcytof': '#999933', 'gatemeclass': '#DDCC77',
    'lda_abdelaal': '#CC6677', 'gmm_mocking': '#882255', 'rf': '#AA4499', 'hgb': '#332288', 'knn': '#EE8866',
    'logreg': '#77AADD', 'lda_sklearn': '#DDDDDD',
}
