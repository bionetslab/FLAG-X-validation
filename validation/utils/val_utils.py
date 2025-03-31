
import warnings
import numpy as np
import pandas as pd
from typing import Union, Tuple, Sequence, Optional, Callable, Dict, List

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

    # todo: add binary option (if pos label given, also compute binary)

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


