
import numpy as np
import pandas as pd
from typing import Union, Tuple, Sequence, Optional

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def set_pandas_print_options():
    # Set pandas print options such that alls columns and rows are displayed
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')


# Todo: maybe refactor this (unknowns should just be excluded or set to others)

def prec_rec_f1_avg(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        exclude_unknowns: bool = True,
        # Whether to also compute the metrics without unknowns (= events for which classification was not possible)
        unknown_label: int = -1,
        verbosity: int = 0,
) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:

    # Compute the precision, recall and F1 for with average micro, macro and weighted
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    prec_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    rec_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    data_prec = [prec_macro, prec_micro, prec_w]
    data_rec = [rec_macro, rec_micro, rec_w]
    data_f1 = [f1_macro, f1_micro, f1_w]

    index = ['prec', 'rec', 'f1']

    columns = ['macro', 'micro', 'weighted']

    data = [data_prec, data_rec, data_f1]

    # Compute the precision, recall and F1 for with average micro, macro and weighted
    if exclude_unknowns:

        # Create y_preds where the unknown samples are excluded
        not_unknowns_bool = np.logical_not(y_pred == unknown_label)

        # Remove events for which there was no prediction possible (i.e. unknowns)
        y_pred_wout_unknowns = y_pred[not_unknowns_bool]
        y_true_wout_unknowns = y_true[not_unknowns_bool]

        # Check the classes of the unclassified events/unknowns
        y_true_unknowns = pd.Series(y_true[np.logical_not(not_unknowns_bool)])
        y_true_unknowns_val_counts = y_true_unknowns.value_counts()

        if verbosity > 0:
            set_pandas_print_options()
            print(f'# ### Label count for unclassifiable events:\n{y_true_unknowns_val_counts}')

        prec_macro_wout_unknowns = precision_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='macro', zero_division=0)
        rec_macro_wout_unknowns = recall_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='macro', zero_division=0)
        f1_macro_wout_unknowns = f1_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='macro', zero_division=0)

        prec_micro_wout_unknowns = precision_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='micro', zero_division=0)
        rec_micro_wout_unknowns = recall_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='micro', zero_division=0)
        f1_micro_wout_unknowns = f1_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='micro', zero_division=0)

        prec_w_wout_unknowns = precision_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='weighted', zero_division=0)
        rec_w_wout_unknowns = recall_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='weighted', zero_division=0)
        f1_w_wout_unknowns = f1_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average='weighted', zero_division=0)

        data_prec_wout_unknowns = [prec_macro_wout_unknowns, prec_micro_wout_unknowns, prec_w_wout_unknowns]
        data_rec_wout_unknowns = [rec_macro_wout_unknowns, rec_micro_wout_unknowns, rec_w_wout_unknowns]
        data_f1_wout_unknowns = [f1_macro_wout_unknowns, f1_micro_wout_unknowns, f1_w_wout_unknowns]

        index = index + ['prec_unkn_excl', 'rec_unkn_excl', 'f1_unkn_excl']

        data = data + [data_prec_wout_unknowns, data_rec_wout_unknowns, data_f1_wout_unknowns]

    res_df = pd.DataFrame(
        data=data,
        index=index,
        columns=columns,
    )

    if verbosity >= 1:
        set_pandas_print_options()
        print(f'# ### Results:\n{res_df}')

    return (res_df, y_true_unknowns_val_counts) if exclude_unknowns else res_df


def prec_rec_f1_class_wise(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        exclude_unknowns: bool = True,
        # Whether to also compute the metrics without unknowns (= events for which classification was not possible)
        unknown_label: int = -1,
        reference_labels: Union[np.ndarray, None] = None,  # All labels that could occur
        verbosity: int = 0
):

    # Compute class-wise precision, recall and F1
    prec_class_wise = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_class_wise = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_class_wise = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Combine the results in a dataframe
    classes = np.union1d(y_true, y_pred)

    res_df_class_wise = pd.DataFrame(
        data=np.vstack((prec_class_wise, rec_class_wise, f1_class_wise)),
        columns=classes.tolist(),
        index=['prec', 'rec', 'f1']
    )

    if exclude_unknowns:
        # Create y_preds where the unknown samples are excluded
        not_unknowns_bool = np.logical_not(y_pred == unknown_label)

        # Remove events for which there was no prediction possible (i.e. unknowns)
        y_pred_wout_unknowns = y_pred[not_unknowns_bool]
        y_true_wout_unknowns = y_true[not_unknowns_bool]

        # Compute class-wise precision, recall and F1
        prec_class_wise_wout_unknowns = precision_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average=None, zero_division=0)
        rec_class_wise_wout_unknowns = recall_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average=None, zero_division=0)
        f1_class_wise_wout_unknowns = f1_score(
            y_true_wout_unknowns, y_pred_wout_unknowns, average=None, zero_division=0)

        classes_wout_unknowns = np.union1d(y_true_wout_unknowns, y_pred_wout_unknowns)

        # Add rows to dataframe, if a class does not occur at all in wout_unknowns, insert Nan
        res_df_class_wise.loc['prec_unkn_excl'] = align_score_arrays(
            arr=classes,
            arr_wout_unknowns=classes_wout_unknowns,
            arr_wout_unknowns_perf=prec_class_wise_wout_unknowns
        )
        res_df_class_wise.loc['rec_unkn_excl'] = align_score_arrays(
            arr=classes,
            arr_wout_unknowns=classes_wout_unknowns,
            arr_wout_unknowns_perf=rec_class_wise_wout_unknowns
        )
        res_df_class_wise.loc['f1_unkn_excl'] = align_score_arrays(
            arr=classes,
            arr_wout_unknowns=classes_wout_unknowns,
            arr_wout_unknowns_perf=f1_class_wise_wout_unknowns
        )

    # Insert nan columns if label from reference not in columns
    if reference_labels is not None:
        # for label in reference_labels:
        #     if label not in res_df_class_wise.columns:
        #         res_df_class_wise[label] = np.nan
        res_df_class_wise = res_df_class_wise.reindex(
            columns=res_df_class_wise.columns.union(reference_labels), fill_value=np.nan
        )

    if verbosity >= 1:
        set_pandas_print_options()
        print(f'# ### Class-wise results:\n{res_df_class_wise}')

    return res_df_class_wise


def confusion_matrix_df(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        verbosity: int = 0
):
    # Compute the confusion matrix
    cf_mtrx = confusion_matrix(y_true, y_pred)
    classes = np.union1d(y_true, y_pred)

    cf_df = pd.DataFrame(
        data=cf_mtrx,
        index=classes,
        columns=classes
    )

    if verbosity >= 1:
        set_pandas_print_options()
        print(f'# ### Confusion matrix:\n{cf_df}')

    return cf_df


def prec_rec_f1_avg_sample_wise(
        y_trues: Sequence[np.ndarray],
        y_preds: Sequence[np.ndarray],
        sample_names: Sequence[str],
        exclude_unknowns: bool = True,
        unknown_label: int = -1,
        verbosity: int = 0,
) -> Tuple[pd.DataFrame, ...]:

    res_dfs = []
    val_counts_y_true_of_unknowns = []
    for y_true, y_pred in zip(y_trues, y_preds):
        res = prec_rec_f1_avg(
            y_true=y_true,
            y_pred=y_pred,
            exclude_unknowns=exclude_unknowns,
            unknown_label=unknown_label,
            verbosity=verbosity
        )

        if exclude_unknowns:
            res_dfs.append(res[0])
            val_counts_y_true_of_unknowns.append(res[1])
        else:
            res_dfs.append(res)

    df_prec, df_rec, df_f1, df_prec_uex, df_rec_uex, df_f1_uex = _res_df_concatenation_helper(
        res_dfs=res_dfs,
        wout_unknowns=exclude_unknowns,
        sample_names=sample_names
    )

    if exclude_unknowns:

        val_counts_y_true_of_unknowns_df = _y_true_unknowns_val_counts_concatenation_helper(
            val_counts=val_counts_y_true_of_unknowns,
            sample_names=sample_names
        )

        out = df_prec, df_rec, df_f1, df_prec_uex, df_rec_uex, df_f1_uex, val_counts_y_true_of_unknowns_df
    else:
        out = df_prec, df_rec, df_f1

    # Returns dfs with columns: macro, micro, weighted; rows: sample_names, std, mean
    return out


def prec_rec_f1_class_sample_wise(
        y_trues: Sequence[np.ndarray],
        y_preds: Sequence[np.ndarray],
        sample_names: Sequence[str],
        exclude_unknowns: bool = True,
        unknown_label: int = -1,
        reference_labels: Union[np.ndarray, None] = None,  # All labels that could occur
        verbosity: int = 0,
):

    res_dfs = []

    for y_true, y_pred in zip(y_trues, y_preds):
        res_dfs.append(
            prec_rec_f1_class_wise(
                y_true=y_true,
                y_pred=y_pred,
                exclude_unknowns=exclude_unknowns,
                unknown_label=unknown_label,
                reference_labels=reference_labels,
                verbosity=verbosity
            )
        )

    out = _res_df_class_wise_concatenation_helper(
        res_dfs=res_dfs,
        wout_unknowns=exclude_unknowns,
        sample_names=sample_names
    )

    if not exclude_unknowns:
        out = out[0:3]

    return out


def confusion_matrix_df_sample_wise(
        y_trues: Sequence[np.ndarray],
        y_preds: Sequence[np.ndarray],
        verbosity: int = 0
):
    cf_dfs = []
    for y_true, y_pred in zip(y_trues, y_preds):
        cf_dfs.append(confusion_matrix_df(y_true, y_pred, verbosity=verbosity))

    return cf_dfs


def align_score_arrays(
        arr: np.ndarray,
        arr_wout_unknowns: np.ndarray,
        arr_wout_unknowns_perf: np.ndarray
):
    """
    Aligns the performance scores of two arrays such that missing labels in the second array
    are replaced with NaN in the aligned score array.

    Parameters:
        arr (array-like): The array with the complete set of labels.
        arr_wout_unknowns (array-like): The array with a subset of labels.
        arr_wout_unknowns_perf (array-like): The performance scores corresponding to labels in `arr_wout_unknowns`.

    Returns:
        np.ndarray: Aligned performance scores for the second array (with NaN for missing labels).
    """
    return np.array([
        arr_wout_unknowns_perf[np.where(arr_wout_unknowns == label)[0][0]] if label in arr_wout_unknowns else np.nan
        for label in arr
    ])


def _res_df_concatenation_helper(
        res_dfs: Sequence[pd.DataFrame],
        wout_unknowns: bool,
        sample_names: Sequence[str],
) -> Tuple[Optional[pd.DataFrame], ...]:
    prec_df = pd.DataFrame(columns=['macro', 'micro', 'weighted'])
    rec_df = pd.DataFrame(columns=['macro', 'micro', 'weighted'])
    f1_df = pd.DataFrame(columns=['macro', 'micro', 'weighted'])
    if wout_unknowns:
        prec_df_wout_unkn = pd.DataFrame(columns=['macro', 'micro', 'weighted'])
        rec_df_wout_unkn = pd.DataFrame(columns=['macro', 'micro', 'weighted'])
        f1_df_wout_unkn = pd.DataFrame(columns=['macro', 'micro', 'weighted'])

    for sample_name, res_df in zip(sample_names, res_dfs):
        prec_df.loc[sample_name] = res_df.loc['prec']
        rec_df.loc[sample_name] = res_df.loc['rec']
        f1_df.loc[sample_name] = res_df.loc['f1']

        if wout_unknowns:
            prec_df_wout_unkn.loc[sample_name] = res_df.loc['prec_unkn_excl']
            rec_df_wout_unkn.loc[sample_name] = res_df.loc['rec_unkn_excl']
            f1_df_wout_unkn.loc[sample_name] = res_df.loc['f1_unkn_excl']

    if wout_unknowns:
        out = [prec_df, rec_df, f1_df, prec_df_wout_unkn, rec_df_wout_unkn, f1_df_wout_unkn]
    else:
        out = [prec_df, rec_df, f1_df, None, None, None]

    for df in out:
        if df is not None:
            std = df.std(axis=0)
            mean = df.mean(axis=0)
            df.loc['std'] = std
            df.loc['mean'] = mean

    return tuple(out)


def _y_true_unknowns_val_counts_concatenation_helper(
        val_counts: Sequence[pd.Series],
        sample_names: Sequence[str],
) -> pd.DataFrame:

    df = pd.concat(val_counts, axis=1, join='outer', ignore_index=True).T.fillna(0)
    df.index = sample_names

    return df


def _res_df_class_wise_concatenation_helper(
        res_dfs: Sequence[pd.DataFrame],
        wout_unknowns: bool,
        sample_names: Sequence[str],
) -> Tuple[pd.DataFrame, ...]:

    # Get rows with respective metric
    res_dfs_prec = [df[df.index == 'prec'] for df in res_dfs]
    res_dfs_rec = [df[df.index == 'rec'] for df in res_dfs]
    res_dfs_f1 = [df[df.index == 'f1'] for df in res_dfs]

    res_df_prec = pd.concat(res_dfs_prec, axis=0, join='outer', ignore_index=True)
    res_df_rec = pd.concat(res_dfs_rec, axis=0, join='outer', ignore_index=True)
    res_df_f1 = pd.concat(res_dfs_f1, axis=0, join='outer', ignore_index=True)
    res_df_prec.index = sample_names
    res_df_rec.index = sample_names
    res_df_f1.index = sample_names

    if wout_unknowns:
        res_dfs_prec_wout_unkn = [df[df.index == 'prec_unkn_excl'] for df in res_dfs]
        res_dfs_rec_wout_unkn = [df[df.index == 'rec_unkn_excl'] for df in res_dfs]
        res_dfs_f1_wout_unkn = [df[df.index == 'f1_unkn_excl'] for df in res_dfs]

        res_df_prec_wout_unkn = pd.concat(res_dfs_prec_wout_unkn, axis=0, join='outer', ignore_index=True)
        res_df_rec_wout_unkn = pd.concat(res_dfs_rec_wout_unkn, axis=0, join='outer', ignore_index=True)
        res_df_f1_wout_unkn = pd.concat(res_dfs_f1_wout_unkn, axis=0, join='outer', ignore_index=True)
        res_df_prec_wout_unkn.index = sample_names
        res_df_rec_wout_unkn.index = sample_names
        res_df_f1_wout_unkn.index = sample_names


        out = [res_df_prec, res_df_rec, res_df_f1, res_df_prec_wout_unkn, res_df_rec_wout_unkn, res_df_f1_wout_unkn]

    else:

        out = [res_df_prec, res_df_rec, res_df_f1, None, None, None]

    for df in out:
        if df is not None:
            std = df.std(axis=0, skipna=True)
            mean = df.mean(axis=0, skipna=True)
            df.loc['std'] = std
            df.loc['mean'] = mean

    return tuple(out)
