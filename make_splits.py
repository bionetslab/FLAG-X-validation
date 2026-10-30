"""
Cross-validation splits: patient-level (stratified) k-fold splits per dataset, saved once and reused by all later steps.

Folds are assigned once per split group on the union of its patients and then mapped to the samples of every dataset in
the group, so a patient is in the same fold in all dataset versions (lt1/lt1b/lt2/lt2b) and panels (BLAST110 tubes).

Output in data/splits/:
- <dataset>.csv: one row per sample; columns sample, patient_id, [stratum], split_0 ... split_<K_OUTER-1> (train/test)
- imstat_inner.csv: inner splits of the tuning dataset, one row per sample (same order as imstat.csv); columns sample,
  split_<i>_inner_<j> for every outer split i and inner split j (train/val, outer test samples are test)
- summary.csv: number of train/test samples per dataset and outer split
- summary_inner.csv: number of train/val/test samples per outer and inner split of the tuning dataset

Env: fx_flagx_cpu (envs/flagx_cpu.yml)
"""

import os
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from config import DATA_CURATED, DATA_SPLITS, DATASETS, SEED


OVERWRITE = False  # If False, existing split files are kept

K_OUTER = 5
K_INNER = 3

# Split group, key: ([datasets, ], patient column in meta_samples.csv, stratum column in meta_samples.csv)
# Each group gets the same split on the patient level, stratify by lymphoma subtype or sample type
SPLIT_GROUPS = {
    'imstat': (['imstat'], 'donor_id', None),
    'flowcyt': (['flowcyt'], 'case_id', None),
    'lymphoma': (['lt1', 'lt1b', 'lt2', 'lt2b'], 'patient_id', 'subtype'),
    'blast110': (['blast110_p1', 'blast110_p2', 'blast110_p3', 'blast110_p4'], 'patient_id', 'sample_type'),
}

# Tuning dataset, gets inner train/val splits for every outer split
INNER_DATASET = 'imstat'


def make_outer_splits(group: str):
    # Get datasets included in group, patient ID key, stratification key
    datasets, patient_key, stratum = SPLIT_GROUPS[group]

    # Union of patients (and their stratum) over all datasets of the group
    meta = {}
    for dataset in datasets:  # Iterate over datasets of the group
        # Load meta_samples.csv
        meta[dataset] = pd.read_csv(os.path.join(DATA_CURATED, dataset, 'meta_samples.csv'), dtype=str)
        # Rename patient_key col to patient_id -> uniform naming
        meta[dataset] = meta[dataset].rename(columns={patient_key: 'patient_id'})
    # Iterate over datasets, combine patient_ID cols and stratum cols into joint df
    patients = pd.concat([m[['patient_id'] + ([stratum] if stratum else [])] for m in meta.values()])
    # Remove duplicate rows -> Unique (patient_id, stratum) value combinations
    patients = patients.drop_duplicates()
    # A patient remains more than once only if its stratum differs between datasets of the group
    assert patients['patient_id'].is_unique, '{}: a patient has different strata across datasets'.format(group)

    # Create pandas series of strata indexed by patient_id
    strata = patients.set_index('patient_id')[stratum] if stratum else None
    # Split patients into (stratified) folds -> Pandas Series, index=patient id, value=fold number
    folds = assign_folds(patients['patient_id'], K_OUTER, SEED, strata)

    for dataset in datasets:
        # Define path to CSV file with split
        path = os.path.join(DATA_SPLITS, dataset + '.csv')
        # Check if split CSV already exists
        if is_done(path):
            print('{}: splits exist, skipped'.format(dataset))
            continue

        # Get meta_samples df (contains patient_id columns)
        m = meta[dataset]
        # folds[m['patient_id']].values = fold number for patients in dataset; m['sample'].values = new filename
        # -> Pandas Series with Index: new filename, Value: fold number
        sample_folds = pd.Series(folds[m['patient_id']].values, index=m['sample'].values)

        # Generate the joint dataframe storing the k train-test splits -> cols: new filename (index), split_i
        roles = fold_roles(sample_folds, ('train', 'test'), 'split')
        # Check if each sample is in test exactly once
        assert ((roles == 'test').sum(axis=1) == 1).all()

        # Merge selected cols of meta_samples with roles df; save
        columns = ['sample', 'patient_id'] + ([stratum] if stratum else [])
        out = pd.concat([m[columns].reset_index(drop=True), roles.reset_index(drop=True)], axis=1)
        out.to_csv(path, index=False)
        print('{}: {} samples, test fold sizes {}'.format(dataset, len(out), (roles == 'test').sum(axis=0).tolist()))


def make_inner_splits():
    # Define path to CSV file with split
    path = os.path.join(DATA_SPLITS, INNER_DATASET + '_inner.csv')
    # Check if split CSV already exists
    if is_done(path):
        print('{}: inner splits exist, skipped'.format(INNER_DATASET))
        return

    # Read in the df with the outer splits, set sample/new filename as index
    outer = pd.read_csv(os.path.join(DATA_SPLITS, INNER_DATASET + '.csv'), dtype=str).set_index('sample')
    # Initialize the df for the inner split
    inner = pd.DataFrame(index=outer.index)
    # Iterate over the outer splits
    for i in range(K_OUTER):
        # Select patient ids of the train samples of the current split
        train = outer.loc[outer['split_{}'.format(i)] == 'train', 'patient_id']
        # Split patients into k_inner folds -> Pandas Series with Index: patient id, Value: fold number
        patient_folds = assign_folds(pd.Series(train.unique()), K_INNER, SEED + i)
        # Expand back to full traiining set (some patients ids appear multiple times), index with sample/new filename
        sample_folds = pd.Series(patient_folds[train].values, index=train.index)
        # Generate the joint dataframe storing the k_inner train-val splits
        roles = fold_roles(sample_folds, ('train', 'val'), 'split_{}_inner'.format(i))
        # Check if each sample is in val exactly once
        assert ((roles == 'val').sum(axis=1) == 1).all()

        # Iterate over columns; add to global df; reindex adds back test samples
        for column in roles:
            inner[column] = roles[column].reindex(inner.index).fillna('test')
        print('{} split {}: {} train samples, val fold sizes {}'.format(
            INNER_DATASET, i, len(roles), (roles == 'val').sum(axis=0).tolist()))

    # Move sample/new filename from index to col; save
    inner.reset_index().to_csv(path, index=False)


def make_summaries():
    # Always rewritten from the saved split files, so the summaries match them
    summary = []
    for dataset in DATASETS:
        splits = pd.read_csv(os.path.join(DATA_SPLITS, dataset + '.csv'), dtype=str)
        for i in range(K_OUTER):
            roles = splits['split_{}'.format(i)]
            summary.append({
                'dataset': dataset, 'split': i, 'n_train': (roles == 'train').sum(), 'n_test': (roles == 'test').sum(),
            })
    pd.DataFrame(summary).to_csv(os.path.join(DATA_SPLITS, 'summary.csv'), index=False)

    summary_inner = []
    inner = pd.read_csv(os.path.join(DATA_SPLITS, INNER_DATASET + '_inner.csv'), dtype=str)
    for i in range(K_OUTER):
        for j in range(K_INNER):
            roles = inner['split_{}_inner_{}'.format(i, j)]
            summary_inner.append({
                'dataset': INNER_DATASET, 'split': i, 'inner': j, 'n_train': (roles == 'train').sum(),
                'n_val': (roles == 'val').sum(), 'n_test': (roles == 'test').sum(),
            })
    pd.DataFrame(summary_inner).to_csv(os.path.join(DATA_SPLITS, 'summary_inner.csv'), index=False)


# ### Helpers ##########################################################################################################

def assign_folds(patients: pd.Series, k: int, seed: int, strata: pd.Series | None = None) -> pd.Series:
    """Returns the test fold (0 ... k-1) per patient, indexed by patient. Stratified if strata are given."""
    patients = patients.sort_values().reset_index(drop=True)  # Sort patients, input order does not matter
    folds = pd.Series(-1, index=patients.values)  # Initialize fold Pandas series with fold assignments
    # Do the k-fold split, scikit-learn splitters return iterators of (train_idx, test_idx)
    if strata is None:
        splits = KFold(n_splits=k, shuffle=True, random_state=seed).split(patients)
    else:
        # Strata with fewer patients than k raise a sklearn warning; they are still
        # assigned, they just do not occur in every fold
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(patients, strata[patients].values)
    # Assign fold numbers
    for fold, (_, test_idx) in enumerate(splits):
        folds.iloc[test_idx] = fold
    return folds


def fold_roles(folds: pd.Series, names: tuple[str, str], prefix: str) -> pd.DataFrame:
    """One column per split (<prefix>_<i>) with names[1] for samples in fold i and names[0] otherwise."""
    k = folds.max() + 1
    # Generate DataFrame from Dict 
    return pd.DataFrame({
        '{}_{}'.format(prefix, i):  # Col name 'prefix_i', e.g. 'split_0'
        folds.map(lambda f: names[1] if f == i else names[0]).values  # names: train, test/val -> current fold is test/val, rest is train
        for i in range(k)  # Iterate over folds numbers
    }, index=folds.index)  # Use new filenames as index


def is_done(path: str) -> bool:
    """Returns False if csv storing the split is not on disk or OVERWRITE True, oterwise returns True"""
    return os.path.exists(path) and not OVERWRITE


if __name__ == '__main__':

    covered = [d for datasets, _, _ in SPLIT_GROUPS.values() for d in datasets]
    assert sorted(covered) == sorted(DATASETS), 'every dataset must be in exactly one split group'
    os.makedirs(DATA_SPLITS, exist_ok=True)

    for group in SPLIT_GROUPS:
        print('{}: making outer splits ...'.format(group))
        make_outer_splits(group)

    print('{}: making inner splits ...'.format(INNER_DATASET))
    make_inner_splits()

    print('writing summaries ...')
    make_summaries()

    print('done')
