
import os
import time
import psutil
import threading
import subprocess
import warnings
import copy
import numpy as np
import pandas as pd
from typing import Union, Tuple, Callable, Dict, List, Literal, Any

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def set_pandas_print_options():
    # Set pandas print options such that alls columns and rows are displayed
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')


def eval_score_with_abstention(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eval_func: Callable[..., Union[float, np.ndarray]],
        eval_func_kwargs: Union[Dict, None] = None,
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        verbosity: int = 1,
) -> Union[float, np.ndarray]:

    # Case1: unknowns_label is None => compute metrics as usual
    # Case2: unknowns_label is not None, others_label is None
    #        - A: No events predicted as unknown => compute metrics as usual
    #        - B: Some events predicted as unknown => exclude unknowns class from any average calculation
    #             - binary: count others as negative predictions
    #             - macro: compute class wise scores, take average withot the unknowns class
    #             - micro: TP, FN are counted globally, unknown=FN automatically
    #             - weighted: no unknowns in y_true => weight 0
    # Case3: unknowns_label is not None, others_label is not None
    #        - A: No events predicted as unknown => compute metrics as usual
    #        - B: Some events predicted as unknown => count unknowns as others

    y_true = y_true.copy()
    y_pred = y_pred.copy()

    if eval_func not in {precision_score, recall_score, f1_score}:
        raise ValueError("eval_func must be one of scikit-learns' precision_score, recall_score, f1_score")

    # Get all labels that occur in the input and prediction output
    labels = np.union1d(y_true, y_pred)

    # Check whether any events were predicted as unknown or unknowns_label is None
    pred_unknowns = np.any(labels == abstention_label)

    if eval_func_kwargs is None:
        eval_func_kwargs = {}

    if 'average' not in eval_func_kwargs:
        eval_func_kwargs['average'] = 'binary'

    if pred_unknowns:  # Case2B, Case3B

        if others_label is None:  # Case2B

            # Determine which kind of average should be computed
            avg = eval_func_kwargs.get('average', None)

            if avg == 'sample':

                raise ValueError("average='sample' is not available in this version.")

            elif avg == 'binary':
                if verbosity >= 1:
                    warnings.warn("For average='binary' unknown events are counted as negative predictions.")

                pos_label = eval_func_kwargs.pop('pos_label', None)
                if pos_label is None:
                    pos_label = 1
                    if verbosity >= 1:
                        warnings.warn("'pos_label' was not passed in eval_func_kwargs. Using default value 1.")

                # Get the actual labels, raise value error if there are more than two
                actual_labels = np.setdiff1d(labels, [abstention_label])
                if len(actual_labels) != 2:
                    raise ValueError(
                        "Target is not binary but average='binary' was requested. "
                        "Please choose another average: ['micro', 'macro', 'weighted']"
                    )

                # Get the negative label
                neg_label = np.setdiff1d(actual_labels, [pos_label]).item()

                # Relabel unknowns as negative predictions
                y_pred[y_pred == abstention_label] = neg_label

                # Compute eval metric
                out = eval_func(y_true, y_pred, pos_label=pos_label, **eval_func_kwargs)

            elif avg == 'macro':
                # Calculate class-wise metric for all classes, including the unknowns
                eval_func_kwargs.pop('average', None)

                if 'labels' in eval_func_kwargs:
                    eval_func_kwargs.pop('labels')
                    warnings.warn("'labels' in eval_func_kwargs was overridden by internal label handling.")

                class_wise_scores = eval_func(y_true, y_pred, labels=labels, average=None, **eval_func_kwargs)

                # Remove the score corresponding to the unknowns
                class_wise_scores = class_wise_scores[labels != abstention_label]

                # Compute mean
                out = class_wise_scores.mean()

            else:
                # Calculate average as usual:
                # - 'micro' (TP, FN are counted globally, unknown=FN)
                # - 'weighted' (no unknowns in y_true => weight 0)

                out = eval_func(y_true, y_pred, **eval_func_kwargs)

        else:  # Case3B
            # Relabel unknowns to others
            y_pred[y_pred == abstention_label] = others_label

            out = eval_func(y_true, y_pred, **eval_func_kwargs)

    else:  # Case1, Case2A, Case3A
        out = eval_func(y_true, y_pred, **eval_func_kwargs)

    return out


def prec_rec_f1_avg(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        pos_label: Union[int, None] = None,
        verbosity: int = 1,
) -> pd.DataFrame:

    score_names = ['precision', 'recall', 'f1-score']
    score_fcts = [precision_score, recall_score, f1_score]

    averages = ['macro', 'micro', 'weighted']

    if pos_label is not None:
        averages += ['binary']

    res_df = pd.DataFrame(index=score_names, columns=averages)

    for sn, sf in zip(score_names, score_fcts):
        for avg in averages:

            kwargs = {'average': avg}
            if pos_label is not None:
                kwargs['pos_label'] = pos_label

            res_df.loc[sn, avg] = eval_score_with_abstention(
                y_true=y_true,
                y_pred=y_pred,
                eval_func=sf,
                eval_func_kwargs=kwargs,
                abstention_label=abstention_label,
                others_label=others_label,
                verbosity=verbosity,
            )

    if verbosity >= 2:
        set_pandas_print_options()
        print(f'# ### Performance scores:\n{res_df}\n')

    return res_df


def prec_rec_f1_class_wise(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        verbosity: int = 1,
) -> pd.DataFrame:

    score_names = ['precision', 'recall', 'f1-score']
    score_fcts = [precision_score, recall_score, f1_score]

    classes = np.union1d(y_true, y_pred)

    if others_label is not None:
        classes = classes[classes != abstention_label]

    res_df = pd.DataFrame(index=score_names, columns=classes)

    for sn, sf in zip(score_names, score_fcts):
        res_df.loc[sn, :] = eval_score_with_abstention(
            y_true=y_true,
            y_pred=y_pred,
            eval_func=sf,
            eval_func_kwargs={'average': None},
            abstention_label=abstention_label,
            others_label=others_label,
            verbosity=verbosity,
        )

    if verbosity >= 2:
        set_pandas_print_options()
        print(f'# ### Performance scores:\n{res_df}\n')

    return res_df


def prec_rec_f1_avg_sample_wise(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        pos_label: Union[int, None] = None,
        verbosity: int = 1,
) -> Tuple[pd.DataFrame, ...]:

    score_names = ['precision', 'recall', 'f1-score']
    score_fcts = [precision_score, recall_score, f1_score]

    averages = ['macro', 'micro', 'weighted']

    if pos_label is not None:
        averages += ['binary']

    res_dfs = []

    for sn, sf in zip(score_names, score_fcts):

        res_df = pd.DataFrame(columns=averages)

        for i, (yt, yp) in enumerate(zip(y_trues, y_preds)):

            # Compute macro, micro and weighted average score
            scores = []
            for avg in averages:

                kwargs = {'average': avg}
                if pos_label is not None:
                    kwargs['pos_label'] = pos_label

                scores.append(
                    eval_score_with_abstention(
                        y_true=yt,
                        y_pred=yp,
                        eval_func=sf,
                        eval_func_kwargs=kwargs,
                        abstention_label=abstention_label,
                        others_label=others_label,
                        verbosity=verbosity,
                    )
                )
            res_df.loc[i, :] = scores

        m = res_df.mean(axis=0)
        std = res_df.std(axis=0)
        res_df.loc['mean', :] = m
        res_df.loc['std', :] = std

        res_dfs.append(res_df.copy())

    if verbosity >= 2:
        set_pandas_print_options()
        print(f'# ### Precision:\n{res_dfs[0]}\n')
        print(f'# ### Recall:\n{res_dfs[1]}\n')
        print(f'# ### F1:\n{res_dfs[2]}\n')

    return tuple(res_dfs)


def prec_rec_f1_class_sample_wise(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        verbosity: int = 1,
) -> Tuple[pd.DataFrame, ...]:

    score_names = ['precision', 'recall', 'f1-score']
    score_fcts = [precision_score, recall_score, f1_score]

    res_dfs = []

    for sn, sf in zip(score_names, score_fcts):

        res_df_list = []

        for i, (yt, yp) in enumerate(zip(y_trues, y_preds)):

            classes = np.union1d(yt, yp)

            if others_label is not None:
                classes = classes[classes != abstention_label]

            res_df = pd.DataFrame(columns=classes)

            res_df.loc[i, :] = eval_score_with_abstention(
                y_true=yt,
                y_pred=yp,
                eval_func=sf,
                eval_func_kwargs={'average': None},
                abstention_label=abstention_label,
                others_label=others_label,
                verbosity=verbosity,
            )

            res_df_list.append(res_df.copy())

        # Concatenate
        res_df_list_concat = pd.concat(res_df_list, axis=0, join='outer')
        m = res_df_list_concat.mean(axis=0)
        std = res_df_list_concat.std(axis=0)
        res_df_list_concat.loc['mean', :] = m
        res_df_list_concat.loc['std', :] = std

        res_dfs.append(res_df_list_concat)

    if verbosity >= 2:
        set_pandas_print_options()
        print(f'# ### Precision:\n{res_dfs[0]}\n')
        print(f'# ### Recall:\n{res_dfs[1]}\n')
        print(f'# ### F1:\n{res_dfs[2]}\n')

    return tuple(res_dfs)


def abstention_counts(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        abstention_label: Union[int, None] = None,
        verbosity: int = 1,
) -> pd.Series:

    abst_counts = pd.Series(y_true[y_pred == abstention_label]).value_counts()

    abst_counts.name = 'abstention_counts'

    if verbosity >= 2:
        print(f'# ### Abstention counts:\n{abst_counts}\n')

    return abst_counts


def abstention_counts_sample_wise(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
        abstention_label: Union[int, None] = None,
        verbosity: int = 1,
) -> pd.DataFrame:

    abstention_counts_list = []

    for yt, yp in zip(y_trues, y_preds):
        abstention_counts_list.append(
            abstention_counts(y_true=yt, y_pred=yp, abstention_label=abstention_label, verbosity=0)
        )

    res_df = pd.DataFrame(abstention_counts_list)
    res_df.index = list(range(len(y_trues)))

    m = res_df.mean(axis=0)
    std = res_df.std(axis=0)
    total = res_df.sum(axis=0)
    res_df.loc['total', :] = total
    res_df.loc['mean', :] = m
    res_df.loc['std', :] = std

    if verbosity >= 2:
        print(f'# ### Abstention counts:\n{res_df}\n')

    return res_df


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
        print(f'# ### Confusion matrix:\n{cf_df}\n')

    return cf_df


def confusion_matrix_df_sample_wise(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
):
    cf_dfs = []
    for y_true, y_pred in zip(y_trues, y_preds):
        cf_dfs.append(confusion_matrix_df(y_true, y_pred, verbosity=0))

    return cf_dfs


def eval_wrapper(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        pos_label: Union[int, None] = None,
        verbosity: int = 1,
) -> Tuple[pd.DataFrame, ...]:

    res_df_avg = prec_rec_f1_avg(
        y_true=y_true, y_pred=y_pred, abstention_label=abstention_label, others_label=others_label, pos_label=pos_label,
        verbosity=verbosity
    )

    res_df_cw = prec_rec_f1_class_wise(
        y_true=y_true, y_pred=y_pred, abstention_label=abstention_label, others_label=others_label, verbosity=verbosity
    )

    cf_mat = confusion_matrix_df(y_true=y_true, y_pred=y_pred, verbosity=verbosity)

    out = (res_df_avg, res_df_cw, cf_mat)

    if abstention_label is not None:
        abst_counts = abstention_counts(
            y_true=y_true, y_pred=y_pred, abstention_label=abstention_label, verbosity=verbosity
        )

        out += (abst_counts, )

    return out


def eval_wrapper_sample_wise(
        y_trues: List[np.ndarray],
        y_preds: List[np.ndarray],
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        pos_label: Union[int, None] = None,
        verbosity: int = 1,
) -> Tuple[Union[pd.DataFrame, List[pd.DataFrame]], ...]:

    res_df_avg_prec, res_df_avg_rec, res_df_avg_f1 = prec_rec_f1_avg_sample_wise(
        y_trues=y_trues, y_preds=y_preds, abstention_label=abstention_label, others_label=others_label,
        pos_label=pos_label, verbosity=verbosity
    )

    res_df_cw_prec, res_df_cw_rec, res_df_cw_f1 = prec_rec_f1_class_sample_wise(
        y_trues=y_trues, y_preds=y_preds, abstention_label=abstention_label, others_label=others_label,
        verbosity=verbosity
    )

    cf_mats = confusion_matrix_df_sample_wise(y_trues=y_trues, y_preds=y_preds)

    out = (res_df_avg_prec, res_df_avg_rec, res_df_avg_f1, res_df_cw_prec, res_df_cw_rec, res_df_cw_f1, cf_mats)

    if abstention_label is not None:
        abst_counts = abstention_counts_sample_wise(
            y_trues=y_trues, y_preds=y_preds, abstention_label=abstention_label, verbosity=verbosity
        )

        out += (abst_counts, )

    return out


def get_time_str(seconds: float) -> Tuple[str, int, int, float]:

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    time_str = f'{h}h {m}m {s}s'

    return time_str, h, m, s

def get_error_dataframe(
        failure_combinations: List[str],
        failure_points: List[str],
        error_types: List[str],
        error_messages: List[str],
) -> pd.DataFrame:
    error_df = pd.DataFrame()
    error_df['setting'] = failure_combinations
    error_df['failure_point'] = failure_points
    error_df['error_type'] = error_types
    error_df['error_message'] = error_messages
    if not error_df.empty:
        error_df['error_message'] = error_df['error_message'].str.replace("\n", " ", regex=True)

    return error_df


def n_samples_experiment_helper(
        classifier: Any,
        n_samples: List[int],
        n_events: List[Union[int, Literal['all']]],
        data_p: Union[str, None] = None,
        save_p: Union[str, None] = None,
        abstention_label: Union[int, None] = None,
        others_label: Union[int, None] = None,
        pos_label: Union[int, None] = None,
        sample_order_file: Union[str, None] = None,
        seed: Union[int, None] = None,
):

    if data_p is None:
        data_p = os.getcwd()

    if save_p is None:
        save_p = os.getcwd()

    if seed is not None:
        np.random.seed(seed)

    # Load the test data
    x_test = np.load(os.path.join(data_p, 'x_test.npy'))
    y_test = np.load(os.path.join(data_p, 'y_test.npy'))

    # Load the sample-wise test data
    samples_p_test = os.path.join(data_p, 'sample_wise_test')
    n_samples_test = len([f for f in os.listdir(samples_p_test) if f.startswith('x_')])
    sample_names_test = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples_test)]
    samples_x_test_filenames = [f'x_{sn}.npy' for sn in sample_names_test]
    samples_x_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_x_test_filenames]

    samples_y_test_filenames = [f'y_{sn}.npy' for sn in sample_names_test]
    samples_y_test = [np.load(os.path.join(samples_p_test, f)) for f in samples_y_test_filenames]

    # Get list of index tuples to iterate in desired order
    iter_list = _get_expanding_iterator_list(n=len(n_events), m=len(n_samples))

    # Init dfs to track performance, prec, rec, f1, micro, macro, weighted, binary (if available)
    dummy_df = pd.DataFrame(np.nan, index=n_events, columns=n_samples)
    n_modes = 3 if pos_label is None else 4
    res_dfs = [dummy_df.copy() for _ in range(n_modes * 3)]
    metrics = ['prec', ] * n_modes + ['rec', ] * n_modes + ['f1'] * n_modes
    modes = ['micro', 'macro', 'weighted'] * 3 if pos_label is None else ['micro', 'macro', 'weighted', 'binary'] * 3

    for i, j in iter_list:  # i = n_ds_fractions, j = n_samples

        # Deepcopy classifier before training etc in each loop
        clf = copy.deepcopy(classifier)

        # ### Define path for saving results
        current_save_p = os.path.join(
            save_p,
            'detailed_res',
            f'nevents_{n_events[i]}_nsamples_{n_samples[j]}'
        )
        os.makedirs(current_save_p, exist_ok=True)

        # ### Load the train data
        data_p_load = os.path.join(data_p, 'sample_wise_train')

        if sample_order_file is None:

            sample_names_train = [f'sample_{str(i).zfill(2)}_train' for i in range(n_samples[j])]

        else:

            # Load the filenames in a fixed order from a .txt file
            with open(sample_order_file, 'r') as file:
                og_filenames = [line.strip() for line in file if line.strip()]

            # Shorten the list of filenames to the desired length
            og_filenames = og_filenames[:n_samples[j]]

            # Load the df that stores the mapping from old (.fcs) to new (.npy) filenames
            og_to_new_fns_df_train = pd.read_csv(
                os.path.join(data_p_load, 'sample_names_mapping_train.csv'), index_col=0
            )

            # Get the corresponding new filenames
            sample_names_train = [
                og_to_new_fns_df_train.loc[og_to_new_fns_df_train['og_sample_name'] == og_sn, 'new_sample_name'].iloc[0]
                for og_sn in og_filenames
            ]
            sample_names_train = [sn[:-4] for sn in sample_names_train]

        # Load the samples
        x_trains = [np.load(os.path.join(data_p_load, f'x_{sn}.npy')) for sn in sample_names_train]
        y_trains = [np.load(os.path.join(data_p_load, f'y_{sn}.npy')) for sn in sample_names_train]

        # Downsample
        if n_events[i] != 'all':
            keep_bools = []
            for y in y_trains:

                ds_keep_bool = get_downsampling_bool(
                    y=y, target_num_events=n_events[i], stratified=True
                )
                keep_bools.append(ds_keep_bool)

            x_trains = [x[kb, :] for x, kb in zip(x_trains, keep_bools)]
            y_trains = [y[kb] for y, kb in zip(y_trains, keep_bools)]

        # Concatenate and shuffle rows
        x_train = np.concatenate(x_trains, axis=0)
        y_train = np.concatenate(y_trains, axis=0)

        # Shuffle the train data row-wise
        shuffle_permutation = np.random.permutation(x_train.shape[0])
        x_train = x_train[shuffle_permutation, :]
        y_train = y_train[shuffle_permutation]

        # ### Fit the classifier
        print('# ### Starting fit ...')
        st_fit = time.time()
        clf.fit(X=x_train, y=y_train)
        et_fit = time.time()
        fit_time_sek = et_fit - st_fit
        fit_time_str, h, m, s = get_time_str(seconds=fit_time_sek)
        print(f'# ### Fit finished, time: {fit_time_sek} s = {fit_time_str}\n')

        fit_time_df = pd.DataFrame(
            data=[[fit_time_sek, h, m, s]], index=['fit_time'], columns=['total s', 'h', 'm', 's']
        )
        fit_time_df.to_csv(os.path.join(current_save_p, 'fit_time_df.csv'))

        clf.save(filepath=current_save_p)

        # ### Predict
        print('# ### Starting prediction ...')
        st_pred = time.time()
        y_pred = clf.predict(X=x_test)
        et_pred = time.time()
        pred_time_sek = et_pred - st_pred
        pred_time_str, h, m, s = get_time_str(seconds=pred_time_sek)
        print(f'# ### Prediction finished, time: {pred_time_sek} s = {pred_time_str}\n')

        pred_time_df = pd.DataFrame(
            data=[[pred_time_sek, h, m, s]], index=['pred_time'], columns=['total s', 'h', 'm', 's']
        )
        pred_time_df.to_csv(os.path.join(current_save_p, 'pred_time_df.csv'))

        np.save(os.path.join(current_save_p, 'y_pred.npy'), y_pred)

        print('# ### Starting sample-wise prediction ...')
        samples_y_pred = []
        samples_pred_times = []
        for x, sn in zip(samples_x_test, sample_names_test):
            st = time.time()
            samples_y_pred.append(clf.predict(X=x))
            et = time.time()
            samples_pred_times.append(et - st)

        samples_pred_times_df = pd.DataFrame(index=sample_names_test, columns=['pred_time'])
        samples_pred_times_df['pred_time'] = samples_pred_times
        m = samples_pred_times_df['pred_time'].mean(axis=0)
        std = samples_pred_times_df['pred_time'].std(axis=0)
        samples_pred_times_df.loc['mean'] = m
        samples_pred_times_df.loc['std'] = std
        samples_pred_times_df.to_csv(os.path.join(current_save_p, 'samples_pred_times_df.csv'))

        print(f'# ### Sample-wise prediction finished, avg time per sample: {m}\n')

        os.makedirs(os.path.join(current_save_p, 'samples_y_pred'), exist_ok=True)
        for y, sn in zip(samples_y_pred, sample_names_test):
            np.save(os.path.join(current_save_p, 'samples_y_pred', f'y_pred_{sn}.npy'), y)

        # ### Evaluate
        # Compute evaluation metrics for samples concatenated to one
        out = eval_wrapper(
            y_true=y_test,
            y_pred=y_pred,
            abstention_label=abstention_label,
            others_label=others_label,
            pos_label=pos_label,
            verbosity=1
        )

        out[0].to_csv(os.path.join(current_save_p, 'res_df_avg.csv'))
        out[1].to_csv(os.path.join(current_save_p, 'res_df_cw.csv'))
        out[2].to_csv(os.path.join(current_save_p, 'cf_mat.csv'))
        if abstention_label is not None:
            out[3].to_csv(os.path.join(current_save_p, 'abst_counts.csv'))

        # Compute sample-wise evaluation metrics
        out_sw = eval_wrapper_sample_wise(
            y_trues=samples_y_test,
            y_preds=samples_y_pred,
            abstention_label=abstention_label,
            others_label=others_label,
            pos_label=pos_label,
            verbosity=2,
        )

        out_sw[0].to_csv(os.path.join(current_save_p, 'res_df_sw_avg_prec.csv'))
        out_sw[1].to_csv(os.path.join(current_save_p, 'res_df_sw_avg_rec.csv'))
        out_sw[2].to_csv(os.path.join(current_save_p, 'res_df_sw_avg_f1.csv'))

        out_sw[3].to_csv(os.path.join(current_save_p, 'res_df_sw_cw_prec.csv'))
        out_sw[4].to_csv(os.path.join(current_save_p, 'res_df_sw_cw_rec.csv'))
        out_sw[5].to_csv(os.path.join(current_save_p, 'res_df_sw_cw_f1.csv'))

        if abstention_label is not None:
            out_sw[7].to_csv(os.path.join(current_save_p, 'abst_counts.csv'))

        os.makedirs(os.path.join(current_save_p, 'confusion_matrices_sw'), exist_ok=True)
        for cf_df, sn in zip(out_sw[6], sample_names_test):
            cf_df.to_csv(os.path.join(current_save_p, 'confusion_matrices_sw', f'cf_mat_{sn}.csv'))

        for res_df, metric, mode in zip(res_dfs, metrics, modes):

            if metric == 'prec':
                out_df = out_sw[0]
            elif metric == 'rec':
                out_df = out_sw[1]
            else:  # f1
                out_df = out_sw[2]

            res_df.loc[n_events[i], n_samples[j]] = out_df.loc['mean', mode]

            res_df.to_csv(os.path.join(save_p, f'res_df_{metric}_{mode}.csv'))


def _get_expanding_iterator_list(n: int, m: int) -> List[Tuple[int, int]]:
    max_dim = max(n, m)
    seen = set()
    out = []
    for size in range(1, max_dim + 1):
        for i in range(size):
            for j in range(size):
                if i < n and j < m and (i, j) not in seen:
                    out.append((i, j))
                    seen.add((i, j))
    return out


def get_downsampling_bool(y: np.ndarray, target_num_events: int, stratified: bool = False) -> np.ndarray:

    num_events = y.shape[0]

    keep_mask = np.zeros_like(y, dtype=bool)

    if target_num_events >= num_events:
        keep_mask[:] = True

    elif stratified:

        unique_labels, counts = np.unique(y, return_counts=True)
        selected_indices = []

        for label, count in zip(unique_labels, counts):

            # Get the number of events of this class to keep
            target_num_events_class = int(round(target_num_events * (count / num_events)))
            target_num_events_class = min(target_num_events_class, count)

            # Get the indices where y == class
            class_indices = np.where(y == label)[0]

            # Randomly draw from the indices and append to list
            if target_num_events_class > 0:
                selected = np.random.choice(class_indices, target_num_events_class, replace=False)
                selected_indices.extend(selected)

        # Adjust the number of samples to the exact desired number (account for rounding errors)
        if len(selected_indices) > target_num_events:
            selected_indices = np.random.choice(selected_indices, target_num_events, replace=False)
        elif len(selected_indices) < target_num_events:
            remaining_unselected_events = np.setdiff1d(np.arange(num_events), selected_indices)
            additional_events = np.random.choice(
                remaining_unselected_events, target_num_events - len(selected_indices), replace=False
            )
            selected_indices.extend(additional_events)

        keep_mask[selected_indices] = True

    else:

        selected_indices = np.random.choice(np.arange(num_events), target_num_events, replace=False)

        keep_mask[selected_indices] = True

    return keep_mask


def get_cpu_memory_mb(process: psutil.Process) -> float:
    total_mem = 0
    try:
        with process.oneshot():
            children = process.children(recursive=True)
            all_procs = [process] + children
            for proc in all_procs:

                try:
                    if proc.is_running():
                        total_mem += proc.memory_info().rss

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    except Exception as e:
        print(f'CPU memory tracking failed with error:\n{e}')

    total_mem /= 1024 ** 2

    return total_mem


def track_memory_cpu(interval=0.1):
    """
    Tracks total memory (RSS) of the current process + children.
    Returns a list of memory samples (in MB).
    """

    process = psutil.Process(os.getpid())
    memory_samples = [get_cpu_memory_mb(process=process)]
    stop_event = threading.Event()

    # Initial sample
    def poll():
        while not stop_event.is_set():
            mem = get_cpu_memory_mb(process=process)
            memory_samples.append(mem)
            stop_event.wait(interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()

    return memory_samples, stop_event, thread


def track_memory_gpu(interval=0.1):
    """
    Tracks GPU 0 memory usage over time in a background thread.
    Returns (samples_list, stop_event, thread).
    """
    memory_samples = []
    stop_event = threading.Event()

    interval_ms = max(1, int(interval * 1000))

    # Start a persistent nvidia-smi process
    try:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i", "0",  # Pin 1st GPU
                f"-lms", str(interval_ms),  # Sampling interval in ms
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
    except TypeError:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i", "0",
                f"-lms", str(interval_ms),
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,  # instead of text=True
            bufsize=1
        )

    # Get first sample
    first = 0
    if proc.stdout is not None:
        try:
            first_line = proc.stdout.readline().strip()

            if first_line:
                first = int(first_line)

        except Exception as e:
            pass

    memory_samples.append(first)

    def poll():
        try:
            for line in proc.stdout:
                try:
                    mem = int(line.strip())
                except ValueError:
                    continue

                memory_samples.append(mem)

                if stop_event.is_set():
                    break
        finally:
            # Clean up process when stopping
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                pass

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()

    return memory_samples, stop_event, thread


def scalability_wrapper(
        function: Callable,
        function_params: Union[Dict[str, Any], None]= None,
        track_gpu: bool = False,
        tracking_interval: float = 0.1,
        res_dir: Union[str, None] = None,
        res_filename: Union[str, None] = None,
) -> Tuple[pd.DataFrame, Any]:

    # Start memory tracking
    memory_samples_cpu, stop_event_cpu, tracker_thread_cpu = track_memory_cpu(interval=tracking_interval)
    if track_gpu:
        memory_samples_gpu, stop_event_gpu, tracker_thread_gpu = track_memory_gpu(interval=tracking_interval)

    wall_start = time.perf_counter()

    try:

        if function_params is not None:
            function_output = function(**function_params)
        else:
            function_output = function()

    finally:

        wall_end = time.perf_counter()

        # Stop memory tracker
        stop_event_cpu.set()
        tracker_thread_cpu.join()

        if track_gpu:
            stop_event_gpu.set()
            tracker_thread_gpu.join()

    # Analyze results
    wall_time = wall_end - wall_start

    memory_peak_cpu = max(memory_samples_cpu)
    memory_average_cpu = sum(memory_samples_cpu) / len(memory_samples_cpu)

    if track_gpu:
        memory_peak_gpu = max(memory_samples_gpu)
        memory_average_gpu = sum(memory_samples_gpu) / len(memory_samples_gpu)
    else:
        memory_peak_gpu = None
        memory_average_gpu = None

    res = {
        'wall_time': wall_time,
        'mem_peak_cpu': memory_peak_cpu,
        'mem_avg_cpu': memory_average_cpu,
        'samples_cpu': len(memory_samples_cpu),
        'mem_peak_gpu': memory_peak_gpu,
        'mem_avg_gpu': memory_average_gpu,
        'samples_gpu': len(memory_samples_gpu) if track_gpu else None,
    }
    res_df = pd.DataFrame([res])

    if res_dir is not None:
        if res_filename is None:
            res_filename = 'scalability_results.csv'
        res_df.to_csv(os.path.join(res_dir, res_filename))

    return res_df, function_output




