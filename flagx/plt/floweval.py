
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import warnings

from typing import Union, Tuple, List, Literal, Sequence
from .flowsom import SomClassifier
from .flowdata import FlowDataManager
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier


# Todo: https://scikit-learn.org/dev/model_selection.html


def get_eval_logger(
        filename: str = 'evaluation.log',
        filepath: Union[str, None] = None,
) -> logging.Logger:

    if filepath is None:
        filepath = os.getcwd()

    # Convert the full path to absolute path for comparison consistency
    full_path = os.path.abspath(os.path.join(filepath, filename))

    # Create a logger specific to evaluation
    eval_logger = logging.getLogger("evaluation")
    eval_logger.setLevel(logging.INFO)

    # Ensure this logger doesn't propagate messages to the root logger
    # Todo: leave in ???
    eval_logger.propagate = False

    # Set a formatter for the evaluation file
    # formatter = logging.Formatter('%(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    # Check if the logger already has handlers to avoid duplicates
    if not eval_logger.hasHandlers():
        # Initial setup: add both file and console handlers
        file_handler = logging.FileHandler(full_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        eval_logger.addHandler(console_handler)

    else:
        # Flag to track if there is an existing file handler with the same log filename
        file_handler_exists = False
        for handler in eval_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                # Check if the existing file handler's path matches the new path
                if os.path.abspath(handler.baseFilename) == full_path:
                    file_handler_exists = True
                else:
                    # Remove the old file handler if the filename is different
                    eval_logger.removeHandler(handler)
                    handler.close()

                    # Add a new file handler with the updated filename
                    file_handler = logging.FileHandler(full_path, mode="w")
                    file_handler.setLevel(logging.INFO)
                    file_handler.setFormatter(formatter)
                    eval_logger.addHandler(file_handler)
                break

        # If no matching file handler exists, add one with the specified filename
        if not file_handler_exists:
            file_handler = logging.FileHandler(full_path, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_classification_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: np.ndarray,
        class_priors: np.ndarray,
        compare_to_dummy_classifier: bool = False,
        additional_dummy_classifier_kwargs: Union[dict, None] = None,
        plot: bool = False,
        additional_plot_kwargs: Union[dict, None] = None,
        log_filename: str = 'eval_classification.log',
        filepath: Union[str, None] = None,
) -> None:
    if filepath is None:
        filepath = os.getcwd()

    # ### Get logger
    eval_logger = get_eval_logger(filename=log_filename, filepath=filepath)
    eval_logger.info('# ###### Classification performance metrics ###### #\n')

    # ### Check labels in test and train set, if necessary subset class_priors to labels present in test set
    test_classes = np.union1d(y_true, y_pred)
    class_priors = _check_classes(
        classes=classes, class_priors=class_priors, test_classes=test_classes, logger=eval_logger)

    if compare_to_dummy_classifier:
        if additional_dummy_classifier_kwargs is None:
            additional_dummy_classifier_kwargs = dict()
        x_train, y_train = additional_dummy_classifier_kwargs.pop('train_data', (np.array([[]]), np.array([])))
        if x_train.size == 0 or y_train.size == 0:
            compare_to_dummy_classifier = False
            eval_logger.warning('WARNING: # ### No train_data was passed, '
                                'proceeding without comparing to dummy classifier\n')
        else:
            strategy = additional_dummy_classifier_kwargs.pop('strategy', 'stratified')
            # ## Note on 'strategy':
            # 'most_frequent': predict: most frequent value in y_train, proba: one-hot encoded dist
            # 'prior': predict: most frequent value in y_train, proba: empirical class distribution
            # 'stratified': predict/proba: sample according empirical class distribution
            # 'uniform': predict: uniform at random across classes
            # 'constant': predict: specified constant value
            dummy_clf = DummyClassifier(strategy=strategy, **additional_dummy_classifier_kwargs)
            dummy_clf.fit(X=x_train, y=y_train)
            y_pred_dummy = dummy_clf.predict(np.zeros((y_true.shape[0], x_train.shape[1])))
            eval_logger.info(f"# ### Comparing to dummy classifier with prediction strategy: '{strategy}'\n")

    # ### Accuracy
    if compare_to_dummy_classifier:
        eval_logger.info('# ### SOM classifier ### #')
    overall_accuracy = accuracy_score(y_true, y_pred)
    eval_logger.info(f'# ### Accuracy: {overall_accuracy}')
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    eval_logger.info(f'# ### Balanced accuracy: {balanced_accuracy}, '
                     f'(adjusted such that random performance ~= 0, perfect classification ~= 1)\n')
    if compare_to_dummy_classifier:
        eval_logger.info('# ### Dummy classifier ### #')
        overall_accuracy_dummy = accuracy_score(y_true, y_pred_dummy)
        eval_logger.info(f'# ### Accuracy: {overall_accuracy_dummy}')
        balanced_accuracy_dummy = balanced_accuracy_score(y_true, y_pred_dummy, adjusted=True)
        eval_logger.info(f'# ### Balanced accuracy: {balanced_accuracy_dummy}\n')

    # ### Class-wise precision, recall, F1-score
    if compare_to_dummy_classifier:
        eval_logger.info('# ### SOM classifier ### #')
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    perc_rec_f1 = pd.DataFrame(columns=test_classes)
    perc_rec_f1.loc['precision'] = precision
    perc_rec_f1.loc['recall'] = recall
    perc_rec_f1.loc['F1'] = f1
    eval_logger.info(f'# ### Class-wise precision, recall, F1:\n{perc_rec_f1.to_string()}\n')
    if compare_to_dummy_classifier:
        eval_logger.info('# ### Dummy classifier ### #')
        perc_rec_f1 = pd.DataFrame(columns=test_classes)
        perc_rec_f1.loc['precision'] = precision_score(y_true, y_pred_dummy, average=None)
        perc_rec_f1.loc['recall'] = recall_score(y_true, y_pred_dummy, average=None)
        perc_rec_f1.loc['F1'] = f1_score(y_true, y_pred_dummy, average=None)
        eval_logger.info(f'# ### Class-wise precision, recall, F1:\n{perc_rec_f1.to_string()}\n')

    # ### Average F1-scores
    if compare_to_dummy_classifier:
        eval_logger.info('# ### SOM classifier ### #')
    f1_micro = f1_score(y_true, y_pred, average='micro')  # total TP, FN => F1 (global)
    f1_macro = f1_score(y_true, y_pred, average='macro')  # class-wise TP, FN => class-wise F1 => mean
    f1_weighted = f1_score(y_true, y_pred, average='weighted')  # class-wise TP, FN => class-wise F1
    # => weighted average, weight_i ~= number of samples with y_true = i / number of samples in y_true
    # => classes with mre samples are given a higher weight
    eval_logger.info(
        f'# ### Average F1-scores:\n# micro: {f1_micro}\n# macro: {f1_macro}\n# weighted: {f1_weighted}\n')
    if compare_to_dummy_classifier:
        eval_logger.info('# ### Dummy classifier ### #')
        eval_logger.info(
            f'# ### Average F1-scores:\n'
            f'# micro: {f1_score(y_true, y_pred_dummy, average="micro")}\n'
            f'# macro: {f1_score(y_true, y_pred_dummy, average="macro")}\n'
            f'# weighted: {f1_score(y_true, y_pred_dummy, average="weighted")}\n')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    if compare_to_dummy_classifier:
        eval_logger.info('# ### SOM classifier ### #')
    eval_logger.info(f'# ### Confusion matrix:\n{cm}\n')
    if compare_to_dummy_classifier:
        eval_logger.info('# ### Dummy classifier ### #')
        eval_logger.info(f'# ### Confusion matrix:\n{confusion_matrix(y_true, y_pred_dummy)}\n')

    if plot:
        if additional_plot_kwargs is None:
            additional_plot_kwargs = dict()

        save = additional_plot_kwargs.pop('save', False)
        filename = additional_plot_kwargs.pop('filename', 'confusion_matrix.png')
        show = additional_plot_kwargs.pop('show', False)

        cm_disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_true, y_pred=y_pred, labels=test_classes, **additional_plot_kwargs)

        if save:
            plt.savefig(os.path.join(filepath, filename))
            plt.close('all')

        if show:
            plt.show()

    '''
    # AUC-ROC for each class (one-vs-rest)
    auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')

    # Precision-Recall Curve & Average Precision Score for each class
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

    '''


def _check_classes(
        classes: np.ndarray,
        class_priors: np.ndarray,
        test_classes: np.ndarray,
        logger: logging.Logger,
):
    idx = np.argsort(classes)
    classes = classes[idx]
    class_priors = class_priors[idx]

    if not np.all(np.isin(classes, test_classes)) and np.all(np.isin(test_classes, classes)):
        logger.warning('WARNING: # ### The classes in the test set do not coincide with the classes in the train set, '
                       'evaluation metrics will only be calculated on the classes present in the test set')
        only_in_classes = np.setdiff1d(classes, test_classes)
        only_in_test_classes = np.setdiff1d(test_classes, classes)
        if only_in_classes.size != 0:
            logger.info(f'# ### Classes {only_in_classes} appear only in the train and not in the test set')
        if only_in_test_classes.size != 0:
            logger.info(f'# ### Classes {only_in_test_classes} appear only in the test and not in the train set')

        class_priors = class_priors[np.isin(classes, test_classes)]

    return class_priors


def evaluate_som_results(
        som_classifier: SomClassifier,
        x_train: Union[np.ndarray, None] = None,
        x_test: Union[np.ndarray, None] = None,
        report_matrices: bool = False,
        log_filename: str = 'eval_som.log',
        filepath: Union[str, None] = None,
):
    if filepath is None:
        filepath = os.getcwd()

    # ### Get logger
    eval_logger = get_eval_logger(filename=log_filename, filepath=filepath)
    eval_logger.info('# ###### SOM performance metrics ###### #\n')

    only_in_input = som_classifier.unpredictable_classes()
    if only_in_input.size != 0:
        eval_logger.info(f'# ### The SOM classifier fails to predict the classes {only_in_input}\n')

    if report_matrices:
        if x_train is not None:
            eval_logger.info(
                f'# ### Activation frequencies (train):\n{som_classifier.get_activation_frequencies(x=x_train)}')
        if x_test is not None:
            eval_logger.info(
                f'# ### Activation frequencies (test):\n{som_classifier.get_activation_frequencies(x=x_test)}')

    if report_matrices:
        eval_logger.info(
            f'# ### Unit impurity (entropy):\n{som_classifier.calculate_unit_impurity(impurity_measure="entropy")}\n')
        eval_logger.info(
            f'# ### Unit impurity (gini):\n{som_classifier.calculate_unit_impurity(impurity_measure="gini")}\n')
    eval_logger.info(
        f'# ### Mean impurity (entropy): {som_classifier.calculate_mean_impurity(impurity_measure="entropy")}')
    eval_logger.info(
        f'# ### Mean impurity (gini): {som_classifier.calculate_mean_impurity(impurity_measure="gini")}\n')

    if x_train is not None:
        eval_logger.info(f'# ### Quantization error (train): {som_classifier.calculate_quantization_error(x=x_train)}')
    if x_test is not None:
        eval_logger.info(f'# ### Quantization error (test): {som_classifier.calculate_quantization_error(x=x_test)}')

    if x_train is not None:
        eval_logger.info(f'# ### Topographical error (train): {som_classifier.calculate_topographic_error(x=x_train)}')
    if x_test is not None:
        eval_logger.info(f'# ### Topographical error (test): {som_classifier.calculate_topographic_error(x=x_test)}')


def generate_som_plots_x(
        som_classifier: SomClassifier,
        x: Union[np.ndarray, None] = None,
        filename_prefix: Union[str, None] = None,
        filepath: Union[str, None] = None,
):
    if filepath is None:
        filepath = os.getcwd()
    if filename_prefix is None:
        filename_prefix = ''

    # Plot activation frequencies
    freq = som_classifier.get_activation_frequencies(x=x)
    sns.heatmap(freq, cmap='Blues', annot=True, annot_kws={'size': 6})
    plt.savefig(os.path.join(filepath, f'{filename_prefix}som_activation_frequencies.png'))
    plt.close('all')


def generate_som_plots_general(
        som_classifier: SomClassifier,
        filename_prefix: Union[str, None] = None,
        filepath: Union[str, None] = None,
):
    if filepath is None:
        filepath = os.getcwd()
    if filename_prefix is None:
        filename_prefix = ''

    # Plot classes of the units in the trained SOM
    som_grid_classes = np.vectorize(lambda v: som_classifier.new_to_og_classes_dict_.get(v, v))(
        som_classifier.som_unit_labels_)
    sns.heatmap(som_grid_classes, cmap='tab10', cbar=False, annot=True)
    plt.title('Classes')
    plt.savefig(os.path.join(filepath, f'{filename_prefix}som_classes.png'))
    plt.close('all')

    # Plot class distribution in som units
    plot_som_pies(
        class_counts_per_unit=som_classifier.class_counts_per_unit_, som_dimensions=som_classifier.som_dimensions,
        label_mapping=som_classifier.new_to_og_classes_dict_
    )
    plt.savefig(os.path.join(filepath, f'{filename_prefix}som_class_distributions.png'))
    plt.close('all')

    # Plot impurities
    entropies = som_classifier.calculate_unit_impurity(impurity_measure='entropy')
    entropies = np.round(entropies, decimals=3)
    sns.heatmap(entropies, cmap='Reds', annot=True, annot_kws={'size': 6})
    plt.savefig(os.path.join(filepath, f'{filename_prefix}som_entropies.png'))
    plt.close('all')

    ginies = som_classifier.calculate_unit_impurity(impurity_measure='gini')
    ginies = np.round(ginies, decimals=3)
    sns.heatmap(ginies, cmap='Reds', annot=True, annot_kws={'size': 6})
    plt.savefig(os.path.join(filepath, f'{filename_prefix}som_ginies.png'))
    plt.close('all')


def plot_som_pies(
        class_counts_per_unit: np.ndarray,
        som_dimensions: Tuple[int, int],
        label_mapping: Union[dict, None] = None,
        title: Union[str, None] = None,
        axs: Union[plt.Axes, None] = None,
        figsize: Tuple[int, int] = (9, 9),
        dpi: int = 100,
):

    # Create a new figure and axes if not provided
    if axs is None:
        fig, axs = plt.subplots(
            som_dimensions[0], som_dimensions[1],
            figsize=figsize, subplot_kw={'aspect': 'equal'}, dpi=dpi,
        )

    else:
        fig = plt.gcf()  # Get the current figure if axes are provided

    # Ensure axes is a 2D array for easy iteration
    if axs.ndim == 1:
        axs = axs.reshape(som_dimensions)

    # Calculate fractions and iterate over axes and plot the corresponding pie chart
    som_grid_fractions = np.zeros_like(class_counts_per_unit)
    n_events_per_unit = class_counts_per_unit.sum(axis=2, keepdims=True)
    nonzero_bool = (n_events_per_unit != 0).squeeze(axis=2)
    som_grid_fractions[nonzero_bool, :] = class_counts_per_unit[nonzero_bool, :] / n_events_per_unit[nonzero_bool, :]
    # som_grid_fractions = class_counts_per_unit / class_counts_per_unit.sum(axis=2, keepdims=True)

    for i in range(som_dimensions[0]):
        for j in range(som_dimensions[1]):
            class_fracs = som_grid_fractions[i, j, :]

            ax = axs[i, j]

            if np.all(class_fracs == 0):
                ax.set_facecolor('black')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            else:
                # ax.text(0.5, 0.5, f'{i, j}')  # just for checking

                # Plot the pie chart on the current axis
                patches, _ = ax.pie(class_fracs)
                ax.set_xticks([])
                ax.set_yticks([])

    # Add labels to the columns
    for col, ax in enumerate(axs[0]):
        fig.text(ax.get_position().x0 + ax.get_position().width / 2, 0.92, f'{col}', ha='center',
                 va='bottom')

    # Add labels to the rows
    for row, ax in enumerate(axs[:, 0]):
        fig.text(0.05, ax.get_position().y0 + ax.get_position().height / 2, f'{row}', ha='right', va='center')

    # Create a legend for the figure using patches from one of the plots
    if label_mapping is not None:
        labels = [label_mapping[key] for key in range(som_grid_fractions.shape[2])]
    else:
        labels = np.array(list(range(som_grid_fractions.shape[2])))

    fig.legend(handles=patches, labels=labels, loc='center right', ncol=1)

    if title is not None:
        fig.suptitle(title)

    return fig, axs


def plot_1d_metric_over_time(
        metric_train: np.ndarray,
        n_epochs: int,
        tracking_interval: int,
        metric_val: Union[np.ndarray, None] = None,
        title: Union[str, None] = None,
        xlabel: Union[str, None] = 'Epoch',
        ylabel: Union[str, None] = 'Metric',
        ax: Union[plt.Axes, None] = None,
        filename: Union[str, None] = None,
        filepath: Union[str, None] = None,
):

    # epochs = range(2, (metric_train.shape[0] + 1) * 2, 2)
    epochs = list(range(tracking_interval, n_epochs + 1, tracking_interval))
    if n_epochs % tracking_interval > 1:
        epochs += [n_epochs, ]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(epochs, metric_train, label='Train', color='blue')

    if metric_val is not None:
        ax.plot(epochs, metric_val, label='Val', color='orange')

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # ax.set_xticks(epochs)
    # ax.set_xticklabels(epochs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.legend()
    # ax.grid(True)

    if filename is not None:
        if filepath is None:
            filepath = os.getcwd()
        plt.savefig(os.path.join(filepath, filename))


def plot_nd_metric_over_time(
        metrics: np.ndarray,
        n_epochs: int,
        tracking_interval: int,
        title: Union[str, None] = None,
        xlabel: Union[str, None] = 'Epoch',
        ylabel: Union[str, None] = 'Metric',
        metric_labels: Union[List[str], None] = None,
        colors: Union[List[str], None] = None,
        ax: Union[plt.Axes, None] = None,
        filename: Union[str, None] = None,
        filepath: Union[str, None] = None,
):
    epochs = list(range(tracking_interval, n_epochs + 1, tracking_interval))
    if n_epochs % tracking_interval > 1:
        epochs += [n_epochs, ]

    n_metrics, _ = metrics.shape
    # epochs = range(2, (n_epochs + 1) * 2, 2)

    if colors is None:
        colors = plt.cm.tab10.colors

    if ax is None:
        fig, ax = plt.subplots()

    # Loop through each metric and plot
    for i in range(n_metrics):
        color = colors[i % len(colors)]
        label_train = f'{metric_labels[i]}' if metric_labels and i < len(metric_labels) else f'{i+1}'
        ax.plot(epochs, metrics[i], label=label_train, color=color, linestyle='-')

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # ax.set_xticks(epochs)
    # ax.set_xticklabels(epochs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.legend()
    # ax.grid(True)

    # Save the plot if filename is provided
    if filename is not None:
        if filepath is None:
            filepath = os.getcwd()
        plt.savefig(os.path.join(filepath, filename))


def evaluate_sample_wise(
        som_c: SomClassifier,
        flow_manager: FlowDataManager,
        mode: Literal['test', 'val'] = 'test',
        create_data_loader_kwargs: Union[dict, None] = None,
        log_filename: str = 'eval_sample_wise.log',
        filepath: Union[str, None] = None,
        save_res_df: bool = False,
        res_df_filename: str = 'res_df.csv',
):
    assert som_c.is_fitted_, 'SomClassifier must be fitted before evaluation'
    if mode == 'test':
        data_list = flow_manager.test_data
        assert data_list != [], "'perform_data_split()' must have been called on FlowDataManager"

    else:
        data_list = flow_manager.val_data
        assert data_list != [], "'perform_data_split()' must have been called on FlowDataManager, " \
                                "with the fraction of the data allocated to the validation set greater than 0"

    if create_data_loader_kwargs is None:
        create_data_loader_kwargs = dict()

    if filepath is None:
        filepath = os.getcwd()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')

    # ### Get logger
    eval_logger = get_eval_logger(filename=log_filename, filepath=filepath)

    res_df = pd.DataFrame(
        columns=[
            '# events',
            'Accuracy',
            'Precision macro', 'Precision micro', 'Precision weighted',
            'Recall macro', 'Recall micro', 'Recall weighted',
            'F1 macro', 'F1 micro', 'F1 weighted',
        ]
    )
    for cl in som_c.og_classes_:
        res_df[f'# {cl}'] = None
    for cl in som_c.og_classes_:
        res_df[f'Prec {cl}'] = None
    for cl in som_c.og_classes_:
        res_df[f'Rec {cl}'] = None
    for cl in som_c.og_classes_:
        res_df[f'F1 {cl}'] = None

    for i, d in enumerate(data_list):

        if isinstance(d, sc.AnnData):
            try:
                sample_name = d.uns['filename']
            except KeyError:
                sample_name = f'Sample_{i}'
        else:
            sample_name = d

        eval_logger.info('# ###########################################################################################'
                         '###### #')
        eval_logger.info(f'# ###### {sample_name}, Number: {i} ###### #')

        dummy_data_list = [d, ]
        dummy_data_loader = flow_manager.create_data_loader_worker(
            data_list=dummy_data_list,
            save_path=flow_manager.save_path,
            batch_size=-1,
            shuffle=False,
            **create_data_loader_kwargs
        )

        x_val_test, y_val_test = next(iter(dummy_data_loader))

        y_pred = som_c.predict(X=x_val_test)

        test_classes = np.union1d(y_val_test, y_pred)

        # ### Examine classes and their distribution in the sample
        true_classes = np.unique(y_val_test)
        pred_classes = np.unique(y_pred)
        eval_logger.info(f'# ### Classes in val-/test-set: {true_classes}')
        eval_logger.info(f'# ### Predicted classes: {pred_classes}')
        label_series = pd.Series(y_val_test)
        class_counts = label_series.value_counts()
        class_fracs = label_series.value_counts(normalize=True)
        eval_logger.info(f'# ### Absolute counts of the classes:\n'
                         f'{pd.DataFrame([class_counts.values], columns=class_counts.index, index=["Count"])}')
        eval_logger.info(f'# ### Relative frequencies of the classes:\n'
                         f'{pd.DataFrame([class_fracs.values], columns=class_fracs.index, index=["Freq"])}')

        # ### Number of events in sample
        n_events = x_val_test.shape[0]
        eval_logger.info(f'# ### Number of events: {n_events}\n')

        # ### Accuracy
        overall_accuracy = accuracy_score(y_val_test, y_pred)
        eval_logger.info(f'# ### Accuracy: {overall_accuracy}\n')

        # ### Class-wise precision, recall, F1-score
        precision = [None] * len(som_c.og_classes_)
        recall = [None] * len(som_c.og_classes_)
        f1 = [None] * len(som_c.og_classes_)
        # Calculate metrics for all present classes
        dummy_precision = precision_score(y_val_test, y_pred, labels=test_classes, average=None)
        dummy_recall = recall_score(y_val_test, y_pred, labels=test_classes, average=None)
        dummy_f1 = f1_score(y_val_test, y_pred, labels=test_classes, average=None)
        # Assign computed values to respective indices in the result lists
        for idx, cl in enumerate(test_classes):
            # Get index of the class in the array with the og classes
            class_index = som_c.og_classes_.tolist().index(cl)
            precision[class_index] = dummy_precision[idx]
            recall[class_index] = dummy_recall[idx]
            f1[class_index] = dummy_f1[idx]

        # precision = precision_score(y_val_test, y_pred, average=None)
        # recall = recall_score(y_val_test, y_pred, average=None)
        # f1 = f1_score(y_val_test, y_pred, average=None)
        perc_rec_f1 = pd.DataFrame(columns=som_c.og_classes_)
        perc_rec_f1.loc['precision'] = precision
        perc_rec_f1.loc['recall'] = recall
        perc_rec_f1.loc['F1'] = f1
        eval_logger.info(f'# ### Class-wise precision, recall, F1:\n{perc_rec_f1.to_string()}\n')

        # ### Average Precision and Recall scores
        prec_micro = precision_score(y_val_test, y_pred, average='micro')
        prec_macro = precision_score(y_val_test, y_pred, average='macro')
        prec_weighted = precision_score(y_val_test, y_pred, average='weighted')
        rec_micro = recall_score(y_val_test, y_pred, average='micro')
        rec_macro = recall_score(y_val_test, y_pred, average='macro')
        rec_weighted = recall_score(y_val_test, y_pred, average='weighted')
        eval_logger.info(
            f'# ### Average Precision:\n# micro: {prec_micro}\n# macro: {prec_macro}\n# weighted: {prec_weighted}\n')
        eval_logger.info(
            f'# ### Average Recall:\n# micro: {rec_micro}\n# macro: {rec_macro}\n# weighted: {rec_weighted}\n')

        # ### Average F1-scores
        f1_micro = f1_score(y_val_test, y_pred, average='micro')  # total TP, FN => F1 (global)
        f1_macro = f1_score(y_val_test, y_pred, average='macro')  # class-wise TP, FN => class-wise F1 => mean
        f1_weighted = f1_score(y_val_test, y_pred, average='weighted')  # class-wise TP, FN => class-wise F1
        # => weighted average, weight_i ~= number of samples with y_true = i / number of samples in y_true
        # => classes with mre samples are given a higher weight
        eval_logger.info(
            f'# ### Average F1-scores:\n# micro: {f1_micro}\n# macro: {f1_macro}\n# weighted: {f1_weighted}\n')

        # Confusion Matrix (dims: y_true x y_pred, => Rows with many entries ~ hard to classify classes)
        cm = confusion_matrix(y_val_test, y_pred, labels=som_c.og_classes_)

        eval_logger.info(f'# ### Confusion matrix:\n{cm}\n')

        res_df.loc[sample_name] = [
            n_events,
            overall_accuracy,
            prec_macro, prec_micro, prec_weighted,
            rec_macro, rec_micro, rec_weighted,
            f1_macro, f1_micro, f1_weighted
        ] + class_counts.sort_index().sort_index().reindex(som_c.og_classes_, fill_value=0).to_list() \
                                  + precision + recall + f1

    if save_res_df:
        res_df.to_csv(os.path.join(filepath, res_df_filename))

    return res_df


def plot_sample_wise_evaluation(
        res_df: pd.DataFrame,
        dpi: int = 300,
        jitter: Union[float, bool] = 0.3,
        dot_size: Union[float, None] = 10,
        color: str = 'skyblue',
        fontsize: float = 8.0,
        xlabel: Union[str, None] = None,
        ylabel: Union[str, None] = None,
        ax: Union[plt.Axes, None] = None,
        **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    long_sample_id = (sum(len(str(sample_id)) >= 4 for sample_id in res_df.index.to_list()) >= 1)

    if long_sample_id:
        # Create mapping from sample ids to integers
        sample_id_df = pd.DataFrame(
            data={'id': list(range(res_df.shape[0]))},
            index=res_df.index.copy(),
        )

    df_melted = res_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
    # => df with columns: 'index' (sample ids), 'Metric' (Accuracy, F1_micro, F1_macro), 'Value' (value of metric)

    ax = sns.stripplot(
        data=df_melted, x='Metric', y='Value', jitter=jitter, size=dot_size, color=color, ax=ax, **kwargs)

    categories = df_melted['Metric'].unique()
    collections = ax.collections

    if len(categories) != len(collections):
        raise ValueError("Mismatch between categories and collections. Check plot setup.")

    for category, collection in zip(categories, collections):

        # Filter DataFrame for this category
        category_data = df_melted[df_melted['Metric'] == category]

        # Retrieve offsets for this collection
        offsets = collection.get_offsets()
        x_coords = offsets[:, 0]
        y_coords = offsets[:, 1]

        # Annotate points
        for (x, y), (_, row) in zip(zip(x_coords, y_coords), category_data.iterrows()):
            ax.text(
                x=x,
                y=y,
                s=str(row['index']) if not long_sample_id else sample_id_df.loc[row['index'], 'id'],
                # Annotate with sample index
                fontsize=fontsize,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('')


def plot_param_lineplot(
        res_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        xlog10: bool = False,
        xlog10plusone: bool = False,
        custom_x_ticks: Union[Sequence[float], Literal['auto'], None] = None,
        x_label: Union[str, None] = None,
        y_label: Union[str, None] = None,
        markersize: float = 6.0,
        x_axis_grid: bool = False,
        abline_param_values: bool = False,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
):
    assert not xlog10 or not xlog10plusone, \
        "Choose only one transformation for the x-axis either 'xlog10' or 'xlog10plusone'"
    if ax is None:
        _, ax = plt.subplots(dpi=dpi)

    x = res_df[x_col].to_numpy()
    x_og = x.copy()
    y = res_df[y_col].to_numpy()

    if xlog10:
        x = np.log10(x)
    if xlog10plusone:
        x = np.log10(1 + x)

    ax.plot(x, y, marker='o', markersize=markersize)

    if custom_x_ticks is not None:
        if custom_x_ticks == 'auto':
            x_low = x_og.min()
            x_high = x_og.max()
            xt = [x_low, ]
            current_value = x_low
            while True:
                # Determine the current step size based on magnitude
                step = 10 ** (len(str(current_value)) - 1)

                # Find the next value based on the step
                next_value = ((current_value // step) + 1) * step

                # Stop if the next value exceeds x_high
                if next_value >= x_high:
                    break

                xt.append(next_value)
                current_value = next_value

            xt.append(x_high)

            def is_power_of_ten(n):
                while n % 10 == 0:
                    n //= 10
                return n == 1

            xt_labels = [tick if is_power_of_ten(n=tick) else '' for tick in xt]
            xt_labels[0] = x_low
            # xt_labels[-1] = x_high

        else:
            xt = custom_x_ticks
            xt_labels = custom_x_ticks

        if xlog10:
            xt = np.log10(xt)
        elif xlog10plusone:
            xt = np.log10(1 + xt)

        ax.set_xticks(xt, labels=xt_labels)

    ax.xaxis.grid(x_axis_grid)
    ax.yaxis.grid(False)

    if abline_param_values:
        offset = 0.05

        for i, value in enumerate(x):
            ax.axvline(x=value, color='grey', linestyle='-', linewidth=0.8, zorder=1)

            y_position = y.max() * 0.9 + (i % 2) * offset
            ax.text(
                value, y_position, f"{x_og[i]:.2f}",
                ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontsize=10, color='grey'
            )

    if x_label is not None:
        ax.set_xlabel(x_label)
    else:
        if xlog10:
            ax.set_xlabel(f'log10({x_col})')
        elif xlog10plusone:
            ax.set_xlabel(f'log10(1 + {x_col})')
        else:
            ax.set_xlabel(x_col)

    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(y_col)


def plot_param_stripplot(
        res_df: pd.DataFrame,
        id_var: str,
        val_var: str,
        val_name: Union[str, None] = None,
        jitter: Union[bool, float] = True,
        xlabel: str = '',
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
):
    val_name = val_name if val_name is not None else 'Mean Test Score'

    try:
        res_df[id_var] = res_df[id_var].astype(str)
    except TypeError:
        print("'id_var' values could not be turned into str")
        return

    # val_name = 'Score'
    df_melted = res_df.reset_index().melt(
        id_vars=['index', id_var], value_vars=val_var, value_name=val_name)

    if ax is None:
        _, ax = plt.subplots(dpi=dpi)

    ax = sns.stripplot(
        data=df_melted, x=id_var, y=val_name, jitter=jitter,
        size=11, color='orange', edgecolor='auto', linewidth=1.0, ax=ax)
    ax.set_xlabel(xlabel)

    categories = df_melted[id_var].unique()
    collections = ax.collections

    if len(categories) != len(collections):
        raise ValueError("Mismatch between categories and collections. Check plot setup.")

    for category, collection in zip(categories, collections):

        # Filter DataFrame for this category
        category_data = df_melted[df_melted[id_var] == category]

        # Retrieve offsets for this collection
        offsets = collection.get_offsets()
        x_coords = offsets[:, 0]
        y_coords = offsets[:, 1]

        # Annotate points
        for (x, y), (_, row) in zip(zip(x_coords, y_coords), category_data.iterrows()):
            ax.text(
                x=x,
                y=y,
                s=str(row['index']),
                # Annotate with sample index
                fontsize=8,
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )


def plot_support_hists(
        som_c: SomClassifier,
        n_bins: int = 100,
        verbosity: int = 0,
        dpi: int = 100,
        plot_class_wise: bool = False,
        n_bins_class_wise: int = 20,
        plot_act_freq: bool = False,
        save_p: Union[str, None] = None,
        ax: Union[plt.Axes, None] = None,
):
    support = som_c.class_counts_per_unit_.sum(axis=2)
    if verbosity >= 1:
        print(f'# ### Total support (n events): {support.sum()}')
        print(f'# ### Max support: {support.max()}')
        print(f'# ### Min support: {support.min()}')
        print(f'# ### Class of unit with smallest support in training set: '
              f'{som_c.som_unit_labels_[np.unravel_index(np.argmin(support), support.shape)]}')

    if ax is None:
        _, ax = plt.subplots(dpi=dpi)

    ax.hist(support.flatten(), bins=n_bins, color='lightblue', edgecolor='grey')

    if save_p is not None:
        plt.savefig(os.path.join(save_p, 'support_hist.png'))
        plt.close('all')

    if plot_class_wise:
        # Plot the for each class the size of the support (events for which the unit is bmu)
        # of the units that predict this class
        # => Are there classes with BMUs that predict them that are bmu only for very few cells
        colormap = plt.get_cmap('tab10', len(som_c.og_classes_))
        for i, c in enumerate(som_c.classes_):
            c_bool = (som_c.som_unit_labels_ == c)
            c_supp_vals = support[c_bool]
            fig, ax_dummy = plt.subplots(dpi=dpi)
            ax_dummy.hist(
                c_supp_vals, color=colormap(i), alpha=0.6, edgecolor='grey', label=f'{int(c)}', bins=n_bins_class_wise)
            ax_dummy.set_title(
                f'Class: {som_c.new_to_og_classes_dict_[c]}, '
                f'Total units: {c_bool.sum()}, '
                f'total support: {int(c_supp_vals.sum())}')
            if save_p is not None:
                plt.savefig(os.path.join(save_p, f'support_hist_c{som_c.new_to_og_classes_dict_[c]}.png'))
                plt.close('all')

    if plot_act_freq:
        freq = support / support.sum()
        freq_sci = np.vectorize(lambda x: f"{x:.2e}")(freq)

        fig, ax_dummy = plt.subplots(dpi=dpi)
        sns.heatmap(freq, cmap='Blues', annot=freq_sci, annot_kws={'size': 2}, fmt='', ax=ax_dummy)
        if save_p is not None:
            plt.savefig(os.path.join(save_p, 'act_freq_heatmap.png'))
            plt.close('all')

        fig, ax_dummy = plt.subplots(dpi=dpi)
        ax_dummy.hist(freq.flatten(), color='green', alpha=0.6, edgecolor='grey', bins=n_bins)
        if save_p is not None:
            plt.savefig(os.path.join(save_p, 'act_freq_hist.png'))
            plt.close('all')

        if verbosity >= 1:
            print(f'Activation frequencies sorted: {np.sort(freq.flatten()).tolist()}')
            print(f'Activation frequency is zero {(freq == 0).sum()} times')


def plot_support_hist_w_class_perc(
        som_c: SomClassifier,
        class_label: Union[int, None] = None,
        n_bins: int = 20,
        plot_percentages: bool = False,
        fontsize: Union[float, None] = None,
        plot_title: bool = False,
        ax: Union[plt.Axes, None] = None,
        dpi: int = 100,
):
    # ### Get class count per unit and number of classes from SOM classifier
    cc_per_unit = som_c.class_counts_per_unit_.copy()  # somdim0 x somdim1 x n_classes
    n_classes = som_c.og_classes_.shape[0]

    # ### Calculate support of each unit
    support = cc_per_unit.sum(axis=2).flatten()  # shape: n_units

    # Flatten the class count per unit array
    cc_per_unit_flat = cc_per_unit.reshape(-1, n_classes)  # shape: n_units x n_classes
    # Note: flatten, reshape works row by row
    # => entries in 0-dimension correspond to same unit in 'support' and 'cc_per_unit_flat'

    # ### If passed, subset 'support' and 'cc_per_unit_flat' to units associated with 'class_label'
    if class_label is not None:
        new_class_label = None
        for key, value in som_c.new_to_og_classes_dict_.items():
            if value == class_label:
                new_class_label = key
        if new_class_label is not None:
            c_bool = (som_c.som_unit_labels_ == new_class_label).flatten()
            support = support[c_bool]
            cc_per_unit_flat = cc_per_unit_flat[c_bool, :]
        else:
            warnings.warn(
                "'class_label' is not a class that the SomClassifier can predict, "
                "proceeding without subsetting SOM units", UserWarning)

    # Define histogram (bin values and bins edges)
    bin_vals, bins = np.histogram(support, bins=n_bins)

    # ### For each bin calculate the label fraction across all events that are in the support of units of that bin
    # Initialize counts for each label in the bins, shape: n_bins x n_classes
    bin_wise_label_count = np.zeros((n_bins, n_classes))

    # Calculate counts for each bin and label
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        # Create bool for which units the support value is in the respective bin
        if i == len(bins) - 2:  # Last bin, included upper bound
            in_bin = (low <= support) & (support <= high)
        else:
            in_bin = (low <= support) & (support < high)
        # in_bin = (support >= low) & (support < high)
        if in_bin.any():
            # Sum label counts across units that are in the bin
            bin_wise_label_count[i, :] = cc_per_unit_flat[in_bin].sum(axis=0)

    # Calculate the label fractions in each bin, shape: n_bins x n_classes
    relative_bin_wise_label_count = np.zeros_like(bin_wise_label_count)
    row_sums = bin_wise_label_count.sum(axis=1)
    non_zero_rows = row_sums != 0
    relative_bin_wise_label_count[non_zero_rows] = (
            bin_wise_label_count[non_zero_rows] / row_sums[non_zero_rows, None]
    )

    # ### Plotting
    # Define centers of bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Define colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    # Define bottom and top of the respective bars (one barplot per class),
    bar_edges = np.zeros((n_bins, n_classes + 1))
    for i in range(n_classes):
        bar_edges[:, i + 1] = bar_edges[:, i].copy() + relative_bin_wise_label_count[:, i] * bin_vals

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)

    # Plot the stacked bar histogram
    for i in range(n_classes):
        ax.bar(
            bin_centers,
            bar_edges[:, i + 1] - bar_edges[:, i],  # Bar height starts from bottom
            width=np.diff(bins),
            bottom=bar_edges[:, i],
            color=colors[i],
            edgecolor='black',
            align='center',
            label=f'{int(som_c.new_to_og_classes_dict_[i])}'
        )

    if plot_percentages:
        for i in range(n_bins):
            percentages_str = '%: '
            for j in range(n_classes):
                frac = relative_bin_wise_label_count[i, j]
                if frac > 0:
                    percentages_str += f'{int(som_c.new_to_og_classes_dict_[j])}: {np.round(frac * 100, 2)}'
                    percentages_str += ', ' if j <= n_classes - 2 else ''
            ax.text(
                x=bin_centers[i], y=ax.get_ylim()[1] * 0.05, s=percentages_str, rotation=90, ha='center',
                fontsize=fontsize
            )

    # Add legend and labels
    # ax.set_title('Distribution of support sizes across SOM units')
    ax.set_xlabel('Support, all units' if class_label is None else f'Support, Class {class_label} units')
    ax.set_ylabel('# SOM units')
    ax.legend(title='Labels')

    if plot_title:
        ax.set_title(
            f'Class: {class_label}, '
            f'Total units: {support.shape[0]}, '
            f'Total support: {int(support.sum())}')


def plot_param_heatmap(
        res_df: pd.DataFrame,
        param_row: str,
        param_col: str,
        performance_score: str = 'mean_test_score',
        other_params: Union[dict, None] = None,
        dpi: int = 100,
        ax: Union[plt.Axes, None] = None,
):
    if ax is None:
        _, ax = plt.subplots(dpi=dpi)

    # Problem, param pairs are not unique, two options:
    # 1.) Aggregate to remove duplicates
    #     agg_res_df = res_df.groupby([param_row, param_col])[performance_score].mean().reset_index()
    # 2.) Subset dataframe ~= fix all other parameters to specified value

    res_df = res_df.copy()

    if other_params is None:
        other_params = {}

    title_str = ''
    for key, val in other_params.items():
        res_df = res_df[res_df[key] == str(val)]
        title_str += f'{key}: {val} '

    # Pivot the data
    heatmap_data = res_df.pivot(index=param_row, columns=param_col, values=performance_score)

    # Plot the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".5f", ax=ax)
    ax.set_xlabel(param_col)
    ax.set_ylabel(param_row)
    ax.set_title(title_str)


