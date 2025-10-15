
import os

PLOT_DIR = './results/plots'

DATASET_TO_DIR = {
    'Imstat': 'imstat',
    'LT1': 'lymphoma_tube1', 'LT1b': 'lymphoma_tube1_binary',
    'LT2': 'lymphoma_tube2', 'LT2b': 'lymphoma_tube2_binary',
    'Flowcyt': 'flowcyt'
}

METHOD_TO_DIR = {
    'GateMeClass': 'gatemeclass',
    'DGCyTOF': 'dgcytof',
    'FCNN': 'fcnn',
    'SOM-Classifier': 'som',
    'SOM-Clf.': 'som'
}

DATASET_TO_NUM_SAMPLES = {
    'Imstat': 75,
    'LT1': 58, 'LT1b': 58,
    'LT2': 58, 'LT2b': 58,
    'Flowcyt': 18,
}


def fig1_gating_performance():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from validation.plt import plot_performance_score_box_plot, plot_cell_pop_size_pred_vs_gt, annotate_mosaic


    ####################################################################################################################
    dataset_name_psize = 'Imstat'
    dataset_names_perf = ['Flowcyt', 'Imstat', 'LT1', 'LT1b', 'LT2', 'LT2b']

    method_names = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']


    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    ####################################################################################################################

    conversion_mapping_y_label = {'f1': 'F1', 'prec': 'Precision', 'rec': 'Recall'}

    lm_imstat = {
        '-1': 'NA',
        '1': 'B',  # B cell
        '2': 'Th',  # T helper
        '3': 'NK',  # NK cell
        '4': 'M16', # (CD16+)',  # atypical CD16+ Monocytes
        '5': 'M',  # Other Monocytes
        '6': 'G',  # Granulocytes
        '7': 'X',  # Sorted out
        '8': 'O'  # Unclassified
    }

    # Define label order for the legend
    label_order = ['NA', 'B', 'Th', 'NK', 'M', 'M16', 'G', 'X', 'O']

    # Initialize the mosaic
    figsize = (6.5, 8)  # (8, 10)
    ratios = [3, 3, 4]  # [1/4, 1/4, 1/2]
    fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EE
        """,
        gridspec_kw={'height_ratios': ratios}
    )

    # --- Plot the population percentages
    # Load the plot data
    base_path_y_true = './data/np_files'
    base_path_y_pred = './results/gating_performance'

    # Load the sample-wise data (ground truth and prediction)
    dataset_dir_psize = DATASET_TO_DIR[dataset_name_psize]
    y_true_path = os.path.join(base_path_y_true, dataset_dir_psize, 'log10_w_custom_cutoffs', 'sample_wise_test')

    n_samples = len([f for f in os.listdir(y_true_path) if f.startswith('y_')])
    y_true_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
    y_trues = [np.load(os.path.join(y_true_path, f)).astype(int) for f in y_true_filenames]

    # Define color mapping
    color_mapping = dict(
        (str(k), c)
        for k, c in zip([-1] + list(range(1, 9)), sns.color_palette("Accent", len(lm_imstat)))
    )

    for key, method in zip(['A', 'C', 'D', 'B'], method_names):

        # Load the predicted labels
        data_trafo = 'log10_w_custom_cutoffs' if dataset_name_psize != 'Flowcyt' else 'log10_cutoff100'
        if method == 'GateMeClass':
            data_trafo = 'arcsinh_cofactor150'

        method_dir = METHOD_TO_DIR[method]

        y_pred_path = os.path.join(base_path_y_pred, method_dir, dataset_dir_psize, data_trafo, 'samples_y_pred')
        y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]

        y_preds = []
        success_idx = []
        for i, fn in enumerate(y_pred_filenames):
            try:
                y_preds.append(np.load(os.path.join(y_pred_path, fn)).astype(int))
                success_idx.append(i)
            except FileNotFoundError:
                print(f'# No y_pred found for: {method}, {dataset_name_psize}, {data_trafo}, {fn}')

        y_trues_plot = [y_trues[i] for i in success_idx]
        y_preds_plot = y_preds

        plot_cell_pop_size_pred_vs_gt(
            y_trues=y_trues_plot,
            y_preds=y_preds_plot,
            percentage=True,
            palette=color_mapping,
            title=method,
            point_size=11.0,
            show_r2=True,
            show_pearson=True,
            ax=axd[key],
        )

    # --- Plot the performance scores
    # Load the results dataframes
    base_path = './results/gating_performance'
    res_dfs = []
    for dataset in dataset_names_perf:
        res_dfs_sub = []
        for method in method_names:

            data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            if method == 'GateMeClass':
                data_trafo = 'arcsinh_cofactor150'

            res_df_path = os.path.join(
                base_path,
                METHOD_TO_DIR[method],
                DATASET_TO_DIR[dataset],
                data_trafo,
                f'res_df_sw_avg_{performance_score}.csv'
            )

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')
            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# No results found for dataset: '{dataset}', method: '{method}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    plot_performance_score_box_plot(
        sample_wise_res_dfs=res_dfs,
        method_names=method_names,
        dataset_names=dataset_names_perf,
        score_mode=performance_score_mode,
        y_label=conversion_mapping_y_label[performance_score] + ' Score',
        sns_boxplot_kwargs=None,
        plot_points=True,
        point_kwargs=None,
        boxplot_alpha=0.9,
        ax=axd['E'],
    )

    # Manually adjust axis labels
    ax_label_fontsize = 12

    for key in ['A', 'B', 'C', 'D']:
        ax = axd[key]
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel('Pred. Population Size (%)', fontsize=ax_label_fontsize)
        ax.tick_params(labelsize=ax_label_fontsize - 2)

        # Manually change the legend labels
        handles, labels = axd[key].get_legend_handles_labels()
        labels = [lm_imstat[label] for label in labels]

        # Reorder handles and labels (sort based on predefined order)
        ordered = sorted(zip(handles, labels), key=lambda x: label_order.index(x[1]))
        handles, labels = zip(*ordered)

        axd[key].legend(handles, labels, ncol=2, markerscale=1.5, loc='lower right', fontsize=8)

    ax_e = axd['E']
    ax_e.set_xlabel(None)
    ax_e.set_ylabel(ax_e.get_ylabel(), fontsize=ax_label_fontsize)
    ax_e.tick_params(axis='y', labelsize=ax_label_fontsize - 2)
    ax_e.tick_params(axis='x', labelsize=ax_label_fontsize)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(os.path.join(PLOT_DIR, 'fig1_gating_performance.png'), dpi=fig.dpi)


def fig5_num_samples():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patheffects as pe

    from matplotlib.lines import Line2D
    from validation.plt import annotate_mosaic

    # Configuration ####################################################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    dataset_names = ['Flowcyt', 'LT1', 'LT2', 'Imstat', 'LT1b', 'LT2b']

    method_names = ['FCNN', 'SOM-Classifier']

    n_events = ['all', ]

    plot_all_samples_score = True

    ####################################################################################################################

    all_records = []
    for dataset in dataset_names:

        num_samples = DATASET_TO_NUM_SAMPLES[dataset]
        dataset_dir = DATASET_TO_DIR[dataset]

        for method in method_names:
            for n in n_events:
                for i in range(1, num_samples + 1):

                    method_dir = METHOD_TO_DIR[method]
                    data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
                    if method == 'GateMeClass':
                        data_trafo = 'arcsinh_cofactor150'

                    if i < num_samples:
                        file_path = os.path.join(
                            './results/num_samples_num_events',
                            method_dir,
                            dataset_dir,
                            data_trafo,
                            'random',
                            'detailed_res',
                            f'nevents_{n}_nsamples_{i}',
                            f'res_df_sw_avg_{performance_score}.csv'
                        )
                    else:  # Load previously computed scores for all samples and all events
                        file_path = os.path.join(
                            './results/gating_performance',
                            method_dir,
                            dataset_dir,
                            data_trafo,
                            f'res_df_sw_avg_{performance_score}.csv'
                        )

                    try:
                        df = pd.read_csv(file_path, index_col=0)
                        df = df.drop(index=['mean', 'std'], errors='ignore')
                        for val in df[performance_score_mode]:
                            all_records.append({
                                'dataset': dataset,
                                'method': method,
                                'order': 'random',
                                'n_events': n,
                                'n_samples': i,
                                'score': val
                            })

                    except FileNotFoundError as e:
                        # print(f"# Missing: {file_path}")
                        continue

    # Create DataFrame
    res_df = pd.DataFrame(all_records)

    # Subset to numbers of samples to be plotted
    imstat_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['Imstat']) &
            (res_df['dataset'] == 'Imstat')
    )
    lt_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['LT1']) &
            (res_df['dataset'].isin(['LT1', 'LT2', 'LT1b', 'LT2b']))
    )
    flowcyt_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['Flowcyt']) &
            (res_df['dataset'] == 'Flowcyt')
    )
    keep_bool = (
            (res_df['n_samples'] == 1) |
            (res_df['n_samples'] % 5 == 0) |
            imstat_full | lt_full | flowcyt_full
    )
    plot_df = res_df[keep_bool].copy()

    print(plot_df)

    # --- Plot performance comparison for methods
    # Define a palette
    mn = ['dummy0', 'dummy1', 'FCNN', 'SOM-Classifier']
    palette = dict(zip(mn, sns.color_palette('Set2', len(mn))))

    fig = plt.figure(figsize=(8, 5), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        DEF
        LLL
        """,
        gridspec_kw={'height_ratios': [1, 1, 0.2]}
    )

    plot_labels = list('ABCDEF')

    for dataset, plot_label in zip(dataset_names, plot_labels):

        # Subset to the dataset
        plot_df_sub = plot_df.loc[
            (plot_df['dataset'] == dataset)
            & (plot_df['n_events'] == 'all')
            & (plot_df['order'] == 'random')
        ].copy()

        ax = axd[plot_label]

        sns.lineplot(
            data=plot_df_sub,
            x='n_samples',
            y='score',
            hue='method',
            errorbar=('ci', 95),
            n_boot=1000,
            seed=42,
            err_style='band',
            marker='o',
            markersize=4,
            palette=palette,
            ax=ax,
        )

        if plot_all_samples_score:
            for method in method_names:

                # Get the score for all samples
                num_samples = DATASET_TO_NUM_SAMPLES[dataset]
                score_df = plot_df_sub[
                    (plot_df_sub['method'] == method) &
                    (plot_df_sub['n_samples'] == num_samples)
                ]
                score = score_df['score'].mean()

                color = palette.get(method, 'grey')
                ax.axhline(
                    y=score,
                    linestyle='--',
                    linewidth=1,
                    color=color,
                    alpha=0.8,
                )

                x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
                y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                y_pos = score + y_offset  # just above the line

                ax.text(
                    x=x_pos,
                    y=y_pos,
                    s=f'{score:.3f}',
                    color=color,
                    va='bottom',
                    ha='right',
                    fontsize=8,
                    alpha=0.95,
                    clip_on=True,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
                )

            # Define dummy legend entry for all sample performance
            all_samples_legend = Line2D([], [], linestyle='--', color='grey', linewidth=1, label='All Samples')
            handles, labels = ax.get_legend_handles_labels()
            if 'All Samples' not in labels:
                handles.append(all_samples_legend)
                labels.append('All Samples')
            ax.legend(handles=handles, labels=labels, fontsize=9)

        else:
            ax.legend()

        # Set title and axis labels
        ax.set_title(dataset)
        ax.set_xlabel('No. of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')

        # Set min and max number of samples as x ticks
        x_min, x_max = plot_df_sub['n_samples'].min(), plot_df_sub['n_samples'].max()
        current_ticks = ax.get_xticks()
        current_ticks = [tick for tick in current_ticks if x_min <= tick <= x_max]
        new_ticks = [x_min] + current_ticks + [x_max]
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in new_ticks])

    # Remove legend in subplots and add in separate panel
    handles, labels = axd['A'].get_legend_handles_labels()
    for ax in axd.values():
        if ax.get_legend():
            ax.get_legend().remove()
    axd['L'].axis('off')
    axd['L'].legend(
        handles=handles,
        labels=labels,
        loc='center',
        ncol=len(labels),
        fontsize=12,
        frameon=True,
    )

    # Adjust font sizes
    ax_label_fontsize = 12
    for label in plot_labels:
        ax = axd[label]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16, excluded=['L', ])

    plt.savefig(os.path.join(PLOT_DIR, 'fig4_num_samples.png'), dpi=fig.dpi, bbox_inches='tight')
    plt.close('all')


def fig6_precision_recall():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.ticker import FormatStrFormatter
    from validation.utils import eval_score_with_abstention
    from validation.plt import annotate_mosaic
    from sklearn.metrics import precision_score, recall_score

    # ### Set flags and important variables here #######################################################################
    datasets = ['LT1b', 'LT2b']
    methods = ['FCNN', 'SOM-Classifier']

    thresholds = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    ]

    generate_plot_df = False
    ####################################################################################################################

    if generate_plot_df:

        long_data = []

        for dataset in datasets:

            # Load the sample-wise test data
            data_p = os.path.join(
                './data/np_files', DATASET_TO_DIR[dataset], 'log10_w_custom_cutoffs', 'sample_wise_test'
            )
            n_samples = len([f for f in os.listdir(data_p) if f.startswith('y_')])
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_y_test = [np.load(os.path.join(data_p, f'y_{sn}.npy')) for sn in sample_names]

            for method in methods:

                # Load the probabilistic predictions
                pred_p = os.path.join(
                    './results/probabilistic_predictions',
                    METHOD_TO_DIR[method] + ('_20' if method == 'SOM-Classifier' else ''),
                    DATASET_TO_DIR[dataset],
                    'log10_w_custom_cutoffs/samples_y_proba'
                )
                samples_y_proba = [np.load(os.path.join(pred_p, f'y_proba_{sn}.npy')) for sn in sample_names]

                # Get the predicted probability for class 1
                samples_y_proba = [y[:, 1] for y in samples_y_proba]

                # Get predictions for each threshold and each sample
                for threshold in thresholds:

                    # Get prediction for current threshold
                    samples_y_preds = [(y_prob >= threshold).astype(int) for y_prob in samples_y_proba]

                    # Compute binary precision and recall for each sample
                    precisions_binary = []
                    recalls_binary = []
                    for i, (yt, yp) in enumerate(zip(samples_y_test, samples_y_preds)):

                        precisions_binary.append(
                            eval_score_with_abstention(
                                y_true=yt,
                                y_pred=yp,
                                eval_func=precision_score,
                                eval_func_kwargs={'pos_label': 1, 'average': 'binary', 'zero_division': np.nan},
                                abstention_label=-1 if method == 'SOM-Classifier' else None,
                                others_label=None,
                                verbosity=2,
                            )
                        )

                        recalls_binary.append(
                            eval_score_with_abstention(
                                y_true=yt,
                                y_pred=yp,
                                eval_func=recall_score,
                                eval_func_kwargs={'pos_label': 1, 'average': 'binary', 'zero_division': np.nan},
                                abstention_label=-1 if method == 'SOM-Classifier' else None,
                                others_label=None,
                                verbosity=2,
                            )
                        )

                    precisions_binary_ex_nan = [x for x in precisions_binary if not np.isnan(x)]
                    recalls_binary_ex_nan = [x for x in recalls_binary if not np.isnan(x)]
                    long_data.append({
                        'Dataset': dataset,
                        'Method': method,
                        'Threshold': threshold,
                        'Precision': sum(precisions_binary_ex_nan) / len(precisions_binary_ex_nan),
                        'Recall': sum(recalls_binary_ex_nan) / len(recalls_binary_ex_nan),
                    })

        plot_df = pd.DataFrame(long_data)
        plot_df.to_csv('./results/probabilistic_predictions/plot_df.csv')

    else:
        plot_df = pd.read_csv('./results/probabilistic_predictions/plot_df.csv', index_col=0)

    plot_df_long = plot_df.melt(
        id_vars=['Dataset', 'Method', 'Threshold'],
        value_vars=['Precision', 'Recall'],
        var_name='Metric',
        value_name='Score'
    )

    # ### Plotting
    fig = plt.figure(figsize=(8, 4), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        LL
        """,
        gridspec_kw={'height_ratios': [1, 0.1]},
    )

    # Define the palette
    mn = ['dummy0', 'dummy1', 'FCNN', 'SOM-Classifier']
    palette = dict(zip(mn, sns.color_palette('Set2', len(mn))))

    # Define marker and line styles
    marker_styles = {
        'Precision': 'o',
        'Recall': '^'
    }

    line_styles = {
        'Precision': (4, 2),
        'Recall': (1, 0)
    }

    for dataset, plot_label in zip(datasets, ['A', 'B']):

        # Subset to dataset and exclude extreme values
        df_sub = plot_df_long.loc[
            (plot_df_long['Dataset'] == dataset) &
            (plot_df_long['Threshold'] != 0.01) &
            (plot_df_long['Threshold'] != 0.99)
        ].copy()

        ax = axd[plot_label]

        sns.lineplot(
            data=df_sub,
            x='Threshold',
            y='Score',
            hue='Method',
            style='Metric',
            dashes=line_styles,
            markers=marker_styles,
            markersize=5,
            markeredgecolor='black',
            markeredgewidth=0.5,
            linewidth=1.5,
            palette=palette,
            ax=ax,
        )

        ax.set_title(dataset)
        ax.grid(True, alpha=0.6)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Remove legend in subplots and add in separate panel
    handles, labels = axd['A'].get_legend_handles_labels()
    for ax in axd.values():
        if ax.get_legend():
            ax.get_legend().remove()
    axd['L'].axis('off')
    axd['L'].legend(
        handles=handles,
        labels=labels,
        loc='center',
        ncol=2,
        fontsize=10,
        frameon=True,
    )

    # Adjust font sizes
    ax_label_fontsize = 12
    for key in ['A', 'B']:
        ax = axd[key]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16, excluded=['L', ])

    plt.savefig(os.path.join(PLOT_DIR, 'fig6_precision_recall.png'), dpi=fig.dpi)
    plt.close('all')


def fig1s_population_sizes():
    import os
    import random
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from itertools import chain
    from validation.plt import plot_cell_pop_size_pred_vs_gt, annotate_mosaic

    random.seed(43)

    datasets = ['Flowcyt', 'LT1', 'LT1b', 'LT2', 'LT2b']

    methods = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    # Define mappings from integer to letter labels
    lm_lt1 = {
        '1': 'B',  # B cells
        '9': 'dyB',  # Dying B
        '7': 'X',  # Erythroid (CD45-)
        '10': 'O'  # Others
    }
    lm_lt2 = lm_lt1

    lm_lt1b = {
        '1': 'B',
        '0': 'O'
    }
    lm_lt2b = lm_lt1b

    lm_flowcyt = {
        '0': 'T',  # T lymphocyte
        '1': 'B',  # B lymphocyte
        '2': 'M',  # Monocyte
        '3': 'Ma',  # Mast cell
        '4': 'HSPC',  # Hematopoietic stem and progenitor cell
        '5': 'O'  # Others
    }

    label_display_order = ['HSPC', 'M', 'Ma', 'T', 'B', 'dyB', 'X', 'O']

    # Build global colormap
    dataset_name_to_label_mapping = {
        'LT1': lm_lt1, 'LT1b': lm_lt1b, 'LT2': lm_lt2, 'LT2b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }
    all_letter_labels = set(chain.from_iterable(m.values() for m in dataset_name_to_label_mapping.values()))
    all_letter_labels = list(sorted(all_letter_labels))
    random.shuffle(all_letter_labels)

    # global_palette = sns.color_palette("hls", len(all_letter_labels))
    global_palette = sns.color_palette('Set1')
    global_color_mapping = dict(zip(all_letter_labels, global_palette))

    # Initialize the mosaic
    figsize = (9.5, 10)
    mosaic_str = '''
        ABCDE
        FGHIJ
        KLMNO
        PQRST
        UVWXY
    '''

    fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(mosaic_str)

    legend_subplots = list('AFKPU')
    plot_subplots = list('BCDEGHIJLMNOQRSTVWXY')
    legend_reference_subplots = list('EJOTY')

    count = 0
    for dataset in datasets:

        for method in methods:

            trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            if method == 'GateMeClass':
                trafo = 'arcsinh_cofactor150'

            # Load the sample-wise data (ground truth and prediction, skip missing files)
            y_true_path = os.path.join('./data/np_files', DATASET_TO_DIR[dataset], trafo, 'sample_wise_test')
            n_samples = len([f for f in os.listdir(y_true_path) if f.startswith('y_')])
            y_true_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]

            y_pred_path = os.path.join(
                './results/gating_performance', METHOD_TO_DIR[method], DATASET_TO_DIR[dataset], trafo, 'samples_y_pred'
            )
            y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]

            y_trues = []
            y_preds = []
            for yt_fn, yp_fn in zip(y_true_filenames, y_pred_filenames):
                try:
                    y_true = np.load(os.path.join(y_true_path, yt_fn)).astype(int)
                    y_pred = np.load(os.path.join(y_pred_path, yp_fn)).astype(int)
                    y_trues.append(y_true)
                    y_preds.append(y_pred)
                except FileNotFoundError:
                    print(f'# No y_pred found for: {method}, {dataset}, {trafo}, {yp_fn}')

            # Define color mapping
            cell_type_labels = np.unique(np.concatenate(y_trues)).tolist()
            label_mapping = dataset_name_to_label_mapping[dataset]
            color_mapping = {
                str(int_label): global_color_mapping[label_mapping[str(int_label)]]
                for int_label in cell_type_labels
            }

            ax = axd[plot_subplots[count]]

            plot_cell_pop_size_pred_vs_gt(
                y_trues=y_trues,
                y_preds=y_preds,
                percentage=True,
                palette=color_mapping,
                title=method,
                point_size=22.0,
                # show_r2=True,
                # show_pearson=True,
                ax=ax,
            )

            # ax.set_title(f'{method}, {dataset_name}')
            ax.set_xlabel('Pop. Size (%)')
            ax.set_ylabel('Pred. Pop. Size (%)')
            ax.grid(False)
            ax.get_legend().remove()

            _, x_max = ax.get_xlim()
            x_upper = math.ceil(x_max / 10) * 10
            x_upper = min(x_upper, 100)
            x_middle = math.ceil(x_upper / (2 * 10)) * 10
            x_ticks = [0, x_middle, x_upper]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(x) for x in x_ticks])

            _, y_max = ax.get_ylim()
            y_upper = math.ceil(y_max / 10) * 10
            y_upper = min(y_upper, 100)
            y_middle = math.ceil(y_upper / (2 * 10)) * 10
            y_ticks = [0, y_middle, y_upper]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(y) for y in y_ticks])

            count += 1

    # Build legends
    for legend_subplot_key, legend_reference_key, dataset_name in zip(
            legend_subplots, legend_reference_subplots, datasets
    ):

        # Get labels and handles from reference subplot, convert to letter labels, reorder
        label_mapping = dataset_name_to_label_mapping[dataset_name]
        handles, labels = axd[legend_reference_key].get_legend_handles_labels()
        labels = [label_mapping[label] for label in labels]

        label_to_handle = dict(zip(labels, handles))
        ordered_labels = [l for l in label_display_order if l in label_to_handle]
        ordered_handles = [label_to_handle[l] for l in ordered_labels]

        axd[legend_subplot_key].legend(
            ordered_handles,
            ordered_labels,
            frameon=False,
            ncol=2 if len(handles) >= 6 else 1,
            loc='center',
            markerscale=1.75,
        )
        axd[legend_subplot_key].axis('off')
        axd[legend_subplot_key].set_title(f'Dataset: {dataset_name}')

    annotate_mosaic(fig=fig, axd=axd, fontsize=14)
    plt.savefig(os.path.join(PLOT_DIR, 'fig1s_population_sizes'), dpi=fig.dpi)
    plt.close('all')


def fig2s_gating_performance_class_wise():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from validation.plt import plot_performance_score_box_plot_cw, annotate_mosaic

    ####################################################################################################################
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT1b', 'LT2', 'LT2b']

    methods = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    performance_score = 'f1'  # f1, prec, rec

    ####################################################################################################################

    conversion_mapping_y_label = {'f1': 'F1', 'prec': 'Precision', 'rec': 'Recall'}

    # Define mappings from integer to letter labels
    lm_imstat = {
        '1': 'B',  # B cell
        '2': 'Th',  # T helper
        '3': 'NK',  # NK cell
        '4': 'M16',  # (CD16+)',  # atypical CD16+ Monocytes
        '5': 'M',  # Other Monocytes
        '6': 'G',  # Granulocytes
        '7': 'X',  # Sorted out
        '8': 'O'  # Unclassified
    }

    lm_lt1 = {
        '1': 'B',  # B cells
        '9': 'dyB',  # Dying B
        '7': 'X',  # Erythroid (CD45-)
        '10': 'O'  # Others
    }
    lm_lt2 = lm_lt1

    lm_lt1b = {
        '1': 'B',
        '0': 'O'
    }
    lm_lt2b = lm_lt1b

    lm_flowcyt = {
        '0': 'T',  # T lymphocyte
        '1': 'B',  # B lymphocyte
        '2': 'M',  # Monocyte
        '3': 'Ma',  # Mast cell
        '4': 'HSPC',  # Hematopoietic stem and progenitor cell
        '5': 'O'  # Others
    }

    label_mappings = {
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1b': lm_lt1b, 'LT2': lm_lt2, 'LT2b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }

    label_display_order = ['HSPC', 'M', 'M16', 'Ma', 'T', 'Th', 'NK', 'G', 'B', 'dyB', 'X', 'O']

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        LL
        """,
        gridspec_kw={'height_ratios': [1, 1, 1, 0.2]},
    )

    # Define a palette
    palette = dict(zip(methods, sns.color_palette('Set2', len(methods))))

    # ### Plot the performance scores

    res_dfs = []
    for dataset in datasets:
        res_dfs_sub = []
        for method in methods:

            trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            if method == 'GateMeClass':
                trafo = 'arcsinh_cofactor150'

            res_df_path = os.path.join(
                './results/gating_performance',
                METHOD_TO_DIR[method],
                DATASET_TO_DIR[dataset],
                trafo,
                f'res_df_sw_cw_{performance_score}.csv'
            )

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')

                # Change the column names to letter labels
                label_mapping = label_mappings[dataset]
                res_df = res_df.rename(columns=label_mapping)

            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for '{dataset}', '{method}', '{trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    for res_df, dataset, label in zip(res_dfs, datasets, ['A', 'B', 'C', 'D', 'E', 'F']):

        plot_performance_score_box_plot_cw(
            sample_wise_res_dfs=res_df,
            method_names=methods,
            y_label=conversion_mapping_y_label[performance_score] + ' Score',
            title=dataset,
            palette=palette,
            label_order=label_display_order,
            sns_boxplot_kwargs=None,
            plot_points=True,
            point_kwargs=None,
            boxplot_alpha=0.9,
            ax=axd[label],
        )

        legend = axd[label].get_legend()
        legend.set_title(None)

    # Remove legend in subplots and add in separate panel
    handles, labels = axd['B'].get_legend_handles_labels()
    for ax in axd.values():
        if ax.get_legend():
            ax.get_legend().remove()
    axd['L'].axis('off')
    axd['L'].legend(
        handles=handles,
        labels=labels,
        loc='center',
        ncol=4,
        frameon=True,
        fontsize=11,
        # title='Gating Method'
        # handlelength=2, handleheight=1.5
    )

    annotate_mosaic(fig=fig, axd=axd, fontsize=16, excluded=['L', ])

    plt.savefig(os.path.join(PLOT_DIR, 'fig2s_gating_performance_class_wise.png'), dpi=fig.dpi)
    plt.close('all')


def fig3s_num_samples_num_events():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patheffects as pe

    from matplotlib.lines import Line2D
    from validation.plt import annotate_mosaic

    np.random.seed(42)

    # Configuration ####################################################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    dataset_names = ['Imstat', 'LT1', 'LT2', 'LT1b', 'LT2b', 'Flowcyt']

    method_names = ['FCNN', 'SOM-Clf.']

    n_events = [5000, 10000, 20000, 50000, 'all']

    plot_all_samples_score = True
    ####################################################################################################################

    # --- Load data
    all_records = []
    for dataset in dataset_names:

        num_samples = DATASET_TO_NUM_SAMPLES[dataset]
        dataset_dir = DATASET_TO_DIR[dataset]

        for method in method_names:
            for n in n_events:
                for i in range(1, num_samples + 1):

                    method_dir = METHOD_TO_DIR[method]
                    data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
                    if method == 'GateMeClass':
                        data_trafo = 'arcsinh_cofactor150'

                    if i == num_samples and n == 'all':
                        file_path = os.path.join(
                            './results/gating_performance',
                            method_dir,
                            dataset_dir,
                            data_trafo,
                            f'res_df_sw_avg_{performance_score}.csv'
                        )
                    else:
                        file_path = os.path.join(
                            './results/num_samples_num_events',
                            method_dir,
                            dataset_dir,
                            data_trafo,
                            'random',
                            'detailed_res',
                            f'nevents_{n}_nsamples_{i}',
                            f'res_df_sw_avg_{performance_score}.csv'
                        )

                    try:
                        df = pd.read_csv(file_path, index_col=0)
                        df = df.drop(index=['mean', 'std'], errors='ignore')
                        for val in df[performance_score_mode]:
                            all_records.append({
                                'dataset': dataset,
                                'method': method,
                                'order': 'random',
                                'n_events': n,
                                'n_samples': i,
                                'score': val
                            })

                    except FileNotFoundError as e:
                        # print(f"# Missing: {file_path}")
                        continue

    # Create DataFrame
    res_df = pd.DataFrame(all_records)
    res_df['n_samples'] = res_df['n_samples'].astype(int)

    # Subset to numbers of samples to be plotted
    imstat_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['Imstat']) &
            (res_df['dataset'] == 'Imstat')
    )
    lt_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['LT1']) &
            (res_df['dataset'].isin(['LT1', 'LT2', 'LT1b', 'LT2b']))
    )
    flowcyt_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['Flowcyt']) &
            (res_df['dataset'] == 'Flowcyt')
    )
    keep_bool = (
            (res_df['n_samples'] == 1) |
            (res_df['n_samples'] % 5 == 0) |
            imstat_full | lt_full | flowcyt_full
    )
    plot_df = res_df[keep_bool].copy()

    # --- Plot performance comparison for random vs ordered and num events
    plot_combinations = [
        ('Flowcyt', 'SOM-Clf.'), ('LT1', 'SOM-Clf.'), ('LT2', 'SOM-Clf.'),
        ('Imstat', 'SOM-Clf.'), ('LT1b', 'SOM-Clf.'), ('LT2b', 'SOM-Clf.'),
        ('Flowcyt', 'FCNN'), ('LT1', 'FCNN'), ('LT2', 'FCNN'),
        ('Imstat', 'FCNN'), ('LT1b', 'FCNN'), ('LT2b', 'FCNN'),
    ]

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
    layout_str = '''
            ABC
            DEF
            ZZZ
            GHI
            JKL
        '''
    axd = fig.subplot_mosaic(layout_str, gridspec_kw={'height_ratios': [1, 1, 0.2, 1, 1]})

    plot_labels = list('ABCDEFGHIJKL')

    for comb, plot_label in zip(plot_combinations, plot_labels):

        dataset = comb[0]
        method = comb[1]

        df_sub = plot_df.loc[
            (plot_df['dataset'] == dataset)
            & (plot_df['method'] == method)
            ].copy()

        ax = axd[plot_label]

        sns.lineplot(
            data=df_sub,
            x='n_samples',
            y='score',
            hue='n_events',
            errorbar=None,  # ('ci', 95),
            # n_boot=1000,
            # seed=42,
            # err_style='band',
            marker='o',
            markersize=3,
            palette='magma',
            ax=ax,
        )

        if plot_all_samples_score:
            # Get the score for all samples
            df_all_data_score = df_sub[
                (df_sub['n_samples'] == DATASET_TO_NUM_SAMPLES[dataset]) &
                (df_sub['n_events'] == 'all')
                ]

            score = df_all_data_score['score'].mean()

            color = 'dimgray'

            ax.axhline(
                y=score,
                linestyle='--',
                linewidth=1,
                color=color,
                alpha=0.8,
                # label=f'{method_name} (all samples)',
            )

            x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            y_pos = score - y_offset
            va = 'top'

            # Draw text
            ax.text(
                x=x_pos,
                y=y_pos,
                s=f'{score:.3f}',
                color=color,
                va=va,
                ha='right',
                fontsize=8,
                alpha=0.95,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
            )

            # Define dummy legend entry for all sample performance
            all_samples_legend = Line2D([], [], linestyle='--', color=color, linewidth=1, label='All Samples')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(all_samples_legend)
            labels.append('All Samples')
            ax.legend(handles=handles, labels=labels)

            bootstrap_ci = True
            if bootstrap_ci:
                n_boot = 10000
                scores = df_all_data_score['score'].to_numpy()
                boot_means = np.random.choice(scores, size=(n_boot, len(scores)), replace=True).mean(axis=1)
                ci_lower, ci_upper = np.percentile(boot_means, [5, 95])# [2.5, 97.5])
                # std_scores = scores.std()
                # ci_lower = scores.mean() - std_scores
                # ci_upper = scores.mean() + std_scores

                ax.axhspan(
                    ci_lower,
                    ci_upper,
                    xmin=0, xmax=1,  # spans full width of axes (0% to 100%)
                    facecolor=color,
                    alpha=0.2,
                    zorder=0
                )

                ax.axhline(
                    y=ci_lower,
                    linestyle='-',
                    linewidth=1,
                    color=color,
                    alpha=0.3,
                    zorder=0
                )

                ax.axhline(
                    y=ci_upper,
                    linestyle='-',
                    linewidth=1,
                    color=color,
                    alpha=0.3,
                    zorder=0
                )

                n_samples = sorted(list(set(df_sub['n_samples'])))
                n_samples.remove(1)
                n_events = list(set(df_sub['n_events']))
                n_events.remove('all')
                n_events = sorted(n_events) + ['all', ]

                first_valid = None
                for ns in n_samples:
                    for ne in n_events:
                        ns_ne_scores = df_sub.loc[
                            (df_sub['n_samples'] == ns) & (df_sub['n_events'] == ne),
                            'score'
                        ]

                        ns_ne_score = ns_ne_scores.mean()

                        is_valid = (ns_ne_score >= ci_lower) # & (ns_ne_std <= ci_upper)

                        if is_valid and first_valid is None:
                            first_valid = (ns, ne, ns_ne_score)
                            break
                    if first_valid is not None:
                        break

                ns_val, ne_val, score_val = first_valid

                # Plot a red "X" at the data point
                ax.scatter(
                    ns_val,
                    score_val,
                    facecolors='none',
                    edgecolors='red',
                    marker='o',
                    linewidths=1.5,
                    s=50,
                    zorder=5
                )

                ax.annotate(
                    f'No. Samples: {ns_val},\nNo. Events: {ne_val}',
                    xy=(ns_val, score_val),
                    xytext=(0.4, 0.15),
                    textcoords='axes fraction',
                    xycoords='data',
                    ha='left', va='center',
                    color='black',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='lightgrey'),
                    arrowprops=dict(
                        arrowstyle='-',
                        color='red',
                        lw=0.5,
                        shrinkB=3,
                    )
                )

        else:
            ax.legend()

        ax.set_title(f'{dataset} | {method}')
        ax.set_xlabel('No. of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')
        ax.legend(fontsize=8)  # title='Number of Events')

        # Set min and max number of samples as x ticks
        x_min, x_max = df_sub['n_samples'].min(), df_sub['n_samples'].max()
        current_ticks = ax.get_xticks()
        current_ticks = [tick for tick in current_ticks if x_min <= tick <= x_max]
        new_ticks = [x_min] + current_ticks + [x_max]
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in new_ticks])

    # Remove legend in subplots and add in separate panel
    handles, labels = axd['A'].get_legend_handles_labels()
    for ax in axd.values():
        if ax.get_legend():
            ax.get_legend().remove()
    axd['Z'].axis('off')
    axd['Z'].legend(
        handles=handles,
        labels=labels,
        loc='center',
        ncol=5,
        fontsize=12,
        frameon=True,
    )

    # Adjust font sizes
    ax_label_fontsize = 12
    for key in plot_labels:
        ax = axd[key]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16, excluded=['Z', ])

    plt.savefig(os.path.join(PLOT_DIR, 'fig3s_num_samples_num_events.png'), dpi=fig.dpi)
    plt.close('all')


def fig4s_sample_order():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patheffects as pe

    from matplotlib.lines import Line2D

    from validation.plt import annotate_mosaic

    # Configuration ####################################################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    datasets = ['LT1', 'LT2', 'LT1b', 'LT2b']
    max_n_samples = 20
    max_n_trails = 30

    methods = ['FCNN', 'SOM-Classifier']

    ####################################################################################################################

    # Load the results for ordered samples
    data_ordered = []
    for dataset in datasets:
        for method in methods:

            file_path_ordered = os.path.join(
                './results/n_samples_n_events',
                METHOD_TO_DIR[method],
                DATASET_TO_DIR[dataset],
                'log10_w_custom_cutoffs',
                'ordered',
                f'res_df_{performance_score}_{performance_score_mode}.csv'
            )

            df_ordered = pd.read_csv(file_path_ordered, index_col=0)

            for i in range(1, max_n_samples + 1):

                score = df_ordered.loc['all', str(i)]

                data_ordered.append({
                    'dataset': dataset,
                    'method': method,
                    'order': 'ordered',
                    'n_samples': i,
                    'score': score
                })

    # Load the results for randomly ordered samples
    data_random = []
    for dataset in datasets:
        for method in methods:

            file_path_random = os.path.join(
                './results/random_sample_order_trials',
                METHOD_TO_DIR[method],
                DATASET_TO_DIR[dataset],
                'log10_w_custom_cutoffs',
                f'res_df_{performance_score}_{performance_score_mode}.csv'
            )

            df_random = pd.read_csv(file_path_random, index_col=0)

            for n in df_random.index[: max_n_trails]:
                for i in df_random.columns[: max_n_samples]:
                    score = df_random.loc[n, i]
                    data_random.append({
                        'dataset': dataset,
                        'method': method,
                        'order': 'random',
                        'trial_no': n,
                        'n_samples': i,
                        'score': score
                    })

    # Load the results for all samples
    full_dataset_performances = dict()
    for dataset in datasets:
        for method in methods:
            file_path = os.path.join(
                './results/pred_eval',
                METHOD_TO_DIR[method],
                DATASET_TO_DIR[dataset],
                'log10_w_custom_cutoffs',
                f'res_df_sw_avg_{performance_score}.csv'
            )

            df = pd.read_csv(file_path, index_col=0)
            score = df.loc['mean', performance_score_mode]

            if dataset in full_dataset_performances:
                full_dataset_performances[dataset][method] = score
            else:
                full_dataset_performances[dataset] = {method: score}

    # Create DataFrames
    df_ordered = pd.DataFrame(data_ordered)
    df_ordered['n_samples'] = df_ordered['n_samples'].astype(int)

    df_random = pd.DataFrame(data_random)
    df_random['n_samples'] = df_random['n_samples'].astype(int)

    # --- Plot performance comparison for methods
    # Define a palette
    mn = ['dummy0', 'dummy1', 'FCNN', 'SOM-Classifier']
    palette = dict(zip(mn, sns.color_palette('Set2', len(mn))))

    fig = plt.figure(figsize=(8, 7), constrained_layout=True, dpi=300)
    mosaic = '''
        AB
        CD
        LL
    '''
    axd = fig.subplot_mosaic(mosaic, gridspec_kw={'height_ratios': [1, 1, 0.1]})

    for dataset, plot_label in zip(datasets, list('ABCD')):

        ax = axd[plot_label]

        # Plot the average performance across random order trials
        df_random_sub = df_random.loc[(df_random['dataset'] == dataset)].copy()
        sns.lineplot(
            data=df_random_sub,
            x='n_samples',
            y='score',
            hue='method',
            errorbar=('ci', 95),
            n_boot=1000,
            seed=42,
            err_style='band',
            marker='o',
            markersize=4,
            palette=palette,
            ax=ax,
        )

        # Plot the performance for ordered samples
        df_ordered_sub = df_ordered.loc[(df_ordered['dataset'] == dataset)].copy()
        sns.lineplot(
            data=df_ordered_sub,
            x='n_samples',
            y='score',
            hue='method',
            marker='^',
            markersize=4,
            palette=palette,
            ax=ax,
        )

        # Set title and axis labels
        ax.set_title(dataset)
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')

        # Set min and max number of samples as x ticks
        x_min, x_max = 1, max_n_samples
        current_ticks = ax.get_xticks()
        current_ticks = [tick for tick in current_ticks if x_min <= tick <= x_max]
        new_ticks = [x_min] + current_ticks + [x_max]
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in new_ticks])

        # Plot the performance with the full dataset
        for method in methods:

            score = full_dataset_performances[dataset][method]
            color = palette.get(method, 'grey')

            ax.axhline(
                y=score,
                linestyle='--',
                linewidth=1,
                color=color,
                alpha=0.8,
            )

            x_pos = ax.get_xlim()[1] * 0.98
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            y_pos = score - y_offset
            va = 'top'

            # Draw text
            ax.text(
                x=x_pos,
                y=y_pos,
                s=f'{score:.3f}',
                color=color,
                va=va,
                ha='right',
                fontsize=8,
                alpha=0.95,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
            )

    # Define all legend components
    method_handles = [
        Line2D([0], [0], color=palette[method], lw=2, label=method)
        for method in methods
    ]

    order_handles = [
        Line2D([0], [0], marker='o', color='grey', linestyle='None', label='Random'),
        Line2D([0], [0], marker='^', color='grey', linestyle='None', label='Ordered')
    ]

    all_samples_handle = [
        Line2D([0], [0], linestyle='--', color='grey', linewidth=1, label='All Samples')
    ]

    # Combine all handles
    all_handles = method_handles + order_handles + all_samples_handle

    # Create single unified legend
    ncol = 3
    axd['L'].legend(
        handles=_to_column_major(handles=all_handles, ncol=ncol),
        loc='center',
        ncol=ncol,
        fontsize=12,
        frameon=True
    )

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(os.path.join(PLOT_DIR, 'fig4s_sample_order.png'), dpi=fig.dpi)
    plt.close('all')


def fig5s_minority_count():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.patheffects as pe

    from matplotlib.ticker import LogLocator
    from scipy.stats import spearmanr

    from validation.utils.val_utils import get_downsampling_bool, _get_expanding_iterator_list
    from validation.plt import annotate_mosaic

    # ### Set flags and important variables here #######################################################################
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT2', 'LT1b', 'LT2b']
    method_names = ['FCNN', 'SOM-Classifier']

    dataset_to_n_samples = {
        'Flowcyt': list(range(1,6)) + list(range(10, 16, 5)) + [18, ],
        'Imstat': list(range(1,21)) + list(range(25, 76, 5)),
        'LT1': list(range(1,21)) + list(range(25, 56, 5)) + [58, ],
        'LT2': list(range(1,21)) + list(range(25, 56, 5)) + [58, ],
        'LT1b': list(range(1,21)) + list(range(25, 56, 5)) + [58, ],
        'LT2b': list(range(1,21)) + list(range(25, 56, 5)) + [58, ],
    }
    n_events = [5000, 10000, 20000, 50000, 'all']

    plot_num_events = False

    generate_res_df = True

    ####################################################################################################################

    # Generate or load the results dataframe
    if generate_res_df:

        # ### Generate the minority count dataframe
        res_dfs_minority_count = []
        for dataset in datasets:

            # Set random seed
            np.random.seed(42)

            # Set data path
            trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            data_p = os.path.join('./data/np_files', DATASET_TO_DIR[dataset], trafo, 'sample_wise_train')

            # Get number of samples for which to compute population size information
            n_samples = dataset_to_n_samples[dataset]

            # Generate iter list (same as for n samples experiment)
            iter_list = _get_expanding_iterator_list(n=len(n_events), m=len(n_samples))

            n_samples_list = []
            n_events_list = []
            median_minority_count_list = []
            total_minority_count_list = []
            mean_minority_count_list = []
            min_minority_count_list = []

            for i, j in iter_list:

                # Load the label vectors of the train samples
                sample_names_train = [f'sample_{str(i).zfill(2)}_train' for i in range(n_samples[j])]
                y_trains = [np.load(os.path.join(data_p, f'y_{sn}.npy')) for sn in sample_names_train]

                # Downsample
                if n_events[i] != 'all':
                    keep_bools = []
                    for y in y_trains:
                        ds_keep_bool = get_downsampling_bool(
                            y=y, target_num_events=n_events[i], stratified=True
                        )
                        keep_bools.append(ds_keep_bool)

                    y_trains = [y[kb] for y, kb in zip(y_trains, keep_bools)]

                # Concatenate and shuffle row-wise (just for consistency with n samples experiment)
                y_train = np.concatenate(y_trains, axis=0)
                shuffle_permutation = np.random.permutation(y_train.shape[0])
                y_train = y_train[shuffle_permutation]

                # Get the minority class size for each sample
                minority_counts = []
                for labels in y_trains:
                    unique, counts = np.unique(labels, return_counts=True)
                    minority_counts.append(counts.min())

                # Compute median, total, mean, and min across samples
                median_minority_count = np.median(minority_counts)
                total_minority_count = sum(minority_counts)
                mean_minority_count = sum(minority_counts) / len(minority_counts)
                min_minority_count = min(minority_counts)

                median_minority_count_list.append(median_minority_count)
                total_minority_count_list.append(total_minority_count)
                mean_minority_count_list.append(mean_minority_count)
                min_minority_count_list.append(min_minority_count)
                n_samples_list.append(n_samples[j])
                n_events_list.append(n_events[i])

            res_dfs_minority_count_dataset = pd.DataFrame(
                {
                    'n_samples': n_samples_list,
                    'n_events': n_events_list,
                    'median_minority_count': median_minority_count_list,
                    'total_minority_count': total_minority_count_list,
                    'mean_minority_count': mean_minority_count_list,
                    'min_minority_count': min_minority_count_list,
                    'dataset': [dataset, ] * len(median_minority_count_list)
                }
            )

            res_dfs_minority_count.append(res_dfs_minority_count_dataset)

            res_df_minority_count_dataset_wide = res_dfs_minority_count_dataset.pivot(
                index='n_events',
                columns='n_samples',
                values='median_minority_count'
            )

            print(f'# ### Minority class count {dataset}:\n{res_df_minority_count_dataset_wide}')

            # res_df_minority_count_dataset_wide.to_csv(
            #     os.path.join(PLOT_DIR, f'minority_count_{dataset.replace(' ', '')}.csv')
            # )

        res_df_minority_count = pd.concat(res_dfs_minority_count, axis=0, ignore_index=True)
        # res_df_minority_count.to_csv(os.path.join(PLOT_DIR, 'minority_count.csv'))

        # ### Generate the performance dataframe
        all_records = []
        for dataset in datasets:
            for method in method_names:
                for n in n_events:
                    max_n = DATASET_TO_NUM_SAMPLES[dataset]
                    for i in range(1, max_n + 1):

                        data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'

                        if i == max_n and n == 'all':  # Load previously computed scores for (all samples, all events)
                            file_path = os.path.join(
                                './results/gating_performance',
                                METHOD_TO_DIR[method],
                                DATASET_TO_DIR[dataset],
                                data_trafo,
                                'res_df_sw_avg_f1.csv'
                            )
                        else:
                            file_path = os.path.join(
                                './results/num_samples_num_events',
                                METHOD_TO_DIR[method],
                                DATASET_TO_DIR[dataset],
                                data_trafo,
                                'random',
                                'detailed_res',
                                f'nevents_{n}_nsamples_{i}',
                                'res_df_sw_avg_f1.csv'
                            )

                        try:
                            df = pd.read_csv(file_path, index_col=0)
                            df = df.drop(index=['mean', 'std'], errors='ignore')
                            for val in df['macro']:
                                all_records.append({
                                    'dataset': dataset,
                                    'method': method,
                                    'n_events': n,
                                    'n_samples': i,
                                    'score': val
                                })

                        except FileNotFoundError as e:
                            # print(f"# Missing: {file_path}")
                            continue

        # Concatenate and aggregate
        res_df_performance = pd.DataFrame(all_records)
        res_df_performance['n_samples'] = res_df_performance['n_samples'].astype(int)
        res_df_performance = (
            res_df_performance
            .groupby(['dataset', 'method', 'n_events', 'n_samples'], as_index=False)
            .agg({'score': 'mean'})
        )
        # res_df_performance.to_csv(os.path.join(PLOT_DIR, f'performance.csv'))

        # Merge dataframes
        res_df = pd.merge(res_df_performance, res_df_minority_count, on=['n_samples', 'n_events', 'dataset'])
        res_df.to_csv(os.path.join(PLOT_DIR, 'res_df_minority_count.csv'))

    else:
        res_df = pd.read_csv(os.path.join(PLOT_DIR, 'res_df_minority_count.csv'), index_col=0)
        n_events_col = [int(n) if n != 'all' else n for n in res_df['n_events']]
        res_df['n_events'] = n_events_col


    # Load performance scores for all samples
    all_records = []
    for dataset in datasets:
        for method in method_names:

            data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'

            file_path = os.path.join(
                './results/gating_performance',
                METHOD_TO_DIR[method],
                DATASET_TO_DIR[dataset],
                data_trafo,
                f'res_df_sw_avg_f1.csv'
            )

            df = pd.read_csv(file_path, index_col=0)

            all_records.append({
                'dataset': dataset,
                'method': method,
                'score': df.loc['mean', 'macro']
            })

    # Create DataFrame
    res_df_all_data_performance = pd.DataFrame(all_records)

    # Subset to numbers of samples to be plotted
    imstat_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['Imstat']) &
            (res_df['dataset'] == 'Imstat')
    )
    lt_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['LT1']) &
            (res_df['dataset'].isin(['LT1', 'LT2', 'LT1b', 'LT2b']))
    )
    flowcyt_full = (
            (res_df['n_samples'] == DATASET_TO_NUM_SAMPLES['Flowcyt']) &
            (res_df['dataset'] == 'Flowcyt')
    )
    keep_bool = (
            (res_df['n_samples'] == 1) |
            (res_df['n_samples'] % 5 == 0) |
            imstat_full | lt_full | flowcyt_full
    )
    plot_df = res_df[keep_bool].copy()

    # palette = sns.color_palette('crest', as_cmap=True)
    palette = sns.color_palette('RdBu', as_cmap=True)

    mc_modes = ['total', 'mean', 'median', 'min']
    mc_mode_to_ax_label = {
        'total': 'Total minority count',
        'mean': 'Mean minority count',
        'median': 'Median minority count',
        'min': 'Min. minority count',
    }

    for mc_mode in mc_modes:

        fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
        axd = fig.subplot_mosaic(
            '''
            XYZ
            ABC
            DEF
            GHI
            JKL
            ''',
            gridspec_kw={'height_ratios': [0.5, 1, 1, 1, 1]}
        )

        dataset_method_tuples = [(ds, mth) for mth in method_names for ds in datasets]
        for subplot_key, (dataset, method) in zip(list('ABCDEFGHIJKL'), dataset_method_tuples):

            # Subset the dataframe
            keep_bool_dataset = (plot_df['dataset'] == dataset)
            keep_bool_method = (plot_df['method'] == method)
            keep_bool = keep_bool_dataset & keep_bool_method
            plot_df_sub = plot_df[keep_bool].copy()

            ax = axd[subplot_key]

            sns.scatterplot(
                plot_df_sub,
                x=f'{mc_mode}_minority_count',
                y='score',
                hue='n_samples',
                style='n_events' if plot_num_events else None,
                legend=True,
                palette=palette,
                ax=ax,
            )

            # Format x axis
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))

            # Add all data performance hline
            row_bool = (
                    (res_df_all_data_performance['dataset'] == dataset)
                    & (res_df_all_data_performance['method'] == method)
            )
            score = res_df_all_data_performance.loc[row_bool, 'score'].iloc[0]

            color = 'black'
            ax.axhline(
                y=score,
                linestyle='--',
                linewidth=1,
                color=color,
                alpha=0.8,
            )

            x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            y_pos = score - y_offset
            va = 'top'
            ax.text(
                x=x_pos,
                y=y_pos,
                s=f'{score:.3f}',
                color=color,
                va=va,
                ha='right',
                fontsize=8,
                alpha=0.95,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=1.0, foreground='white')]
            )

            # Print spearman correlation
            rho, p = spearmanr(plot_df_sub[f'{mc_mode}_minority_count'].to_numpy(), plot_df_sub['score'].to_numpy())
            print(f'# rho={rho:.4f}, p={p:.4f}')
            ax.text(
                0.95, 0.05, fr'$r_S = {np.round(rho, 4)}$',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

            ax.set_title(f'{dataset} | {method}')

        # Add colorbar
        ax = axd['X']
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cbar = mpl.colorbar.ColorbarBase(
            ax,
            cmap=palette,
            norm=norm,
            orientation='horizontal',
        )
        cbar.set_ticks([])
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['1 Sample', 'All\nSamples'])
        cbar.set_label('Number of Samples', fontsize=10, labelpad=5)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.label.set_horizontalalignment('center')

        if plot_num_events:
            # Add a legend
            ax = axd['Z']
            handles, labels = axd['A'].get_legend_handles_labels()

            style_handles = []
            style_labels = []
            for h, l in zip(handles, labels):
                if l in [str(i) for i in n_events]:
                    style_handles.append(h)
                    style_labels.append(l)

            ax.axis('off')
            ax.legend(
                style_handles,
                style_labels,
                title='Number of events',
                ncol=2,
                frameon=True,
            )
        else:
            axd['Z'].axis('off')

        # Add table
        table_data = [
            ['Dataset', 'All Samples'],
            ['Flowcyt', DATASET_TO_NUM_SAMPLES['Flowcyt']],
            ['Imstat', DATASET_TO_NUM_SAMPLES['Imstat']],
            ['LT1, LT2', DATASET_TO_NUM_SAMPLES['LT1']],
        ]

        ax = axd['Y']
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')

        # Format
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.2)

        ncols = len(table_data[0])
        for col in range(ncols):
            cell = table[(0, col)]
            cell.set_facecolor('lightgreen')
            cell.set_text_props(weight='bold')

        ax.axis('off')


        for key, ax in axd.items():
            if not key in {'X', 'Y', 'Z'}:
                ax.legend_.remove()

                ax.set_ylabel('Macro F1')
                ax.set_xlabel(mc_mode_to_ax_label[mc_mode])

        annotate_mosaic(fig, axd, fontsize=None, excluded=['X', 'Y', 'Z'])

        fig.savefig(os.path.join(PLOT_DIR, f'fig5s_minority_count_{mc_mode}.png'), dpi=fig.dpi)


def fig6s_gating_performance_local():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    from validation.plt import plot_performance_score_box_plot, plot_performance_score_box_plot_cw, annotate_mosaic

    ####################################################################################################################
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT1b', 'LT2', 'LT2b']

    methods = ['FCNN', 'SOM-Classifier']

    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary
    ####################################################################################################################

    conversion_mapping_y_label = {'f1': 'F1', 'prec': 'Precision', 'rec': 'Recall'}

    # Define mappings from integer to letter labels
    lm_imstat = {
        '1': 'B',  # B cell
        '2': 'Th',  # T helper
        '3': 'NK',  # NK cell
        '4': 'M16',  # (CD16+)',  # atypical CD16+ Monocytes
        '5': 'M',  # Other Monocytes
        '6': 'G',  # Granulocytes
        '7': 'X',  # Sorted out
        '8': 'O'  # Unclassified
    }

    lm_lt1 = {
        '1': 'B',  # B cells
        '9': 'dyB',  # Dying B
        '7': 'X',  # Erythroid (CD45-)
        '10': 'O'  # Others
    }
    lm_lt2 = lm_lt1

    lm_lt1b = {
        '1': 'B',
        '0': 'O'
    }
    lm_lt2b = lm_lt1b

    lm_flowcyt = {
        '0': 'T',  # T lymphocyte
        '1': 'B',  # B lymphocyte
        '2': 'M',  # Monocyte
        '3': 'Ma',  # Mast cell
        '4': 'HSPC',  # Hematopoietic stem and progenitor cell
        '5': 'O'  # Others
    }

    label_mappings = {
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1b': lm_lt1b, 'LT2': lm_lt2, 'LT2b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }

    label_display_order = ['HSPC', 'M', 'M16', 'Ma', 'T', 'Th', 'NK', 'G', 'B', 'dyB', 'X', 'O']

    # Define a palette
    palette = dict(zip(['dummy0', 'dummy1'] + methods, sns.color_palette('Set2', len(methods) + 2)))

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AAAL
        BBCC
        DDEE
        FFGG
        """,
        # gridspec_kw={'height_ratios': [1/4, 1/4, 1/2]}
    )

    # --- Plot performance scores
    # Load the results dataframes
    base_path = './results/local_training'

    res_dfs = []
    for dataset in datasets:
        res_dfs_sub = []
        for method in methods:

            data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            res_df_path = os.path.join(
                base_path,
                METHOD_TO_DIR[method] + ('_20' if method == 'SOM-Classifier' else ''),
                DATASET_TO_DIR[dataset],
                data_trafo,
                f'res_df_sw_avg_{performance_score}.csv'
            )

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')
            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for '{dataset}', '{method}', '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    plot_performance_score_box_plot(
        sample_wise_res_dfs=res_dfs,
        method_names=methods,
        dataset_names=datasets,
        score_mode=performance_score_mode,
        y_label=conversion_mapping_y_label[performance_score] + ' Score',
        title='All Datasets | Macro',
        palette=palette,
        sns_boxplot_kwargs=None,
        plot_points=True,
        point_kwargs=None,
        boxplot_alpha=0.9,
        ax=axd['A'],
    )

    # --- Plot class-wise performance scores
    res_dfs = []
    for dataset in datasets:
        res_dfs_sub = []
        for method in methods:

            data_trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
            res_df_path = os.path.join(
                base_path,
                METHOD_TO_DIR[method] + ('_20' if method == 'SOM-Classifier' else ''),
                DATASET_TO_DIR[dataset],
                data_trafo,
                f'res_df_sw_cw_{performance_score}.csv'
            )

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')

                # Change the column names to letter labels
                label_mapping = label_mappings[dataset]
                res_df = res_df.rename(columns=label_mapping)

            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for '{dataset}', '{method}', '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    for rdf, dsn, label in zip(res_dfs, datasets, ['B', 'C', 'D', 'E', 'F', 'G']):

        plot_performance_score_box_plot_cw(
            sample_wise_res_dfs=rdf,
            method_names=methods,
            y_label=conversion_mapping_y_label[performance_score] + ' Score',
            title=dsn + ' | Class-wise',
            palette=palette,
            label_order=label_display_order,
            sns_boxplot_kwargs=None,
            plot_points=True,
            point_kwargs=None,
            boxplot_alpha=0.9,
            ax=axd[label],
        )

    # Plot the legend separately
    handles, labels = axd['A'].get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if isinstance(h, Patch)]
    handles, labels = zip(*filtered) if filtered else ([], [])
    axd['L'].axis('off')
    axd['L'].legend(handles, labels, loc='center', frameon=False, ncol=1)

    for key, ax in axd.items():
        if key != 'L':
            ax.set_xlabel(None)
            ax.get_legend().remove()

    annotate_mosaic(fig=fig, axd=axd, fontsize=16, excluded=['L'])

    plt.savefig(os.path.join(PLOT_DIR, 'fig6s_gating_performance_local.png'), dpi=fig.dpi)


def fig7s_dataset_size():

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from validation.plt import plot_sample_sizes, annotate_mosaic


    ####################################################################################################################
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT2']

    ####################################################################################################################

    y_trains = []
    y_tests = []
    for dataset in datasets:

        # Load the sample-wise data
        trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
        base_path = os.path.join('./data/np_files', DATASET_TO_DIR[dataset], trafo)

        y_train_dir = os.path.join(base_path, 'sample_wise_train')
        num_y_trains = len([f for f in os.listdir(y_train_dir) if f.startswith('y_')])
        y_train_filenames = [f'y_sample_{str(i).zfill(2)}_train.npy' for i in range(num_y_trains)]
        y_trains.append([np.load(os.path.join(y_train_dir, f)).astype(int) for f in y_train_filenames])

        y_test_dir = os.path.join(base_path, 'sample_wise_test')
        num_y_test = len([f for f in os.listdir(y_test_dir) if f.startswith('y_')])
        y_test_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(num_y_test)]
        y_tests.append([np.load(os.path.join(y_test_dir, f)).astype(int) for f in y_test_filenames])

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 11), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        """
    )

    for dataset, ytr, yte, labels in zip(datasets, y_trains, y_tests, ['AB', 'CD', 'EF', 'GH']):

        plot_sample_sizes(
            ys=ytr, title=f'{dataset} Train', abline_mean=True, abline_std=True, print_total=True, ax=axd[labels[0]]
        )
        plot_sample_sizes(
            ys=yte, title=f'{dataset} Test', abline_mean=True, abline_std=True, print_total=True, ax=axd[labels[1]]
        )

        axd[labels[0]].legend(loc='lower right')
        axd[labels[1]].legend(loc='lower right')

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig(os.path.join(PLOT_DIR, 'fig7s_dataset_size.png'), dpi=fig.dpi)


def fig8s_class_balance():

    import os
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from itertools import chain
    from validation.plt import plot_class_balance, annotate_mosaic

    random.seed(42)

    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT2', 'LT1b', 'LT2b']

    # Define mappings from integer to letter labels
    lm_imstat = {
        '1': 'B',  # B cell
        '2': 'Th',  # T helper
        '3': 'NK',  # NK cell
        '4': 'M16',  # (CD16+)',  # atypical CD16+ Monocytes
        '5': 'M',  # Other Monocytes
        '6': 'G',  # Granulocytes
        '7': 'X',  # Sorted out
        '8': 'O'  # Unclassified
    }

    lm_lt1 = {
        '1': 'B',  # B cells
        '9': 'dyB',  # Dying B
        '7': 'X',  # Erythroid (CD45-)
        '10': 'O'  # Others
    }
    lm_lt2 = lm_lt1

    lm_lt1b = {
        '1': 'B',
        '0': 'O'
    }
    lm_lt2b = lm_lt1b

    lm_flowcyt = {
        '0': 'T',  # T lymphocyte
        '1': 'B',  # B lymphocyte
        '2': 'M',  # Monocyte
        '3': 'Ma',  # Mast cell
        '4': 'HSPC',  # Hematopoietic stem and progenitor cell
        '5': 'O'  # Others
    }

    label_mappings = {
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1b': lm_lt1b, 'LT2': lm_lt2, 'LT2b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }

    label_display_order = ['HSPC', 'M', 'M16', 'Ma', 'T', 'Th', 'NK', 'G', 'B', 'dyB', 'X', 'O']

    all_letter_labels = set(chain.from_iterable(m.values() for m in label_mappings.values()))
    all_letter_labels = list(sorted(all_letter_labels))
    random.shuffle(all_letter_labels)

    # global_palette = sns.color_palette("hls", len(all_letter_labels))
    global_palette = sns.color_palette('Set3')
    global_color_mapping = dict(zip(all_letter_labels, global_palette))

    y_trains = []
    y_tests = []
    for dataset in datasets:
        # Load the sample-wise data
        trafo = 'log10_w_custom_cutoffs' if dataset != 'Flowcyt' else 'log10_cutoff100'
        base_path = os.path.join('./data/np_files', DATASET_TO_DIR[dataset], trafo)

        y_train_dir = os.path.join(base_path, 'sample_wise_train')
        num_y_trains = len([f for f in os.listdir(y_train_dir) if f.startswith('y_')])
        y_train_filenames = [f'y_sample_{str(i).zfill(2)}_train.npy' for i in range(num_y_trains)]
        y_trains.append([np.load(os.path.join(y_train_dir, f)).astype(int) for f in y_train_filenames])

        y_test_dir = os.path.join(base_path, 'sample_wise_test')
        num_y_test = len([f for f in os.listdir(y_test_dir) if f.startswith('y_')])
        y_test_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(num_y_test)]
        y_tests.append([np.load(os.path.join(y_test_dir, f)).astype(int) for f in y_test_filenames])

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 11), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AABB
        CCDD
        EEFF
        GGHH
        IJKL
        """
    )

    for dataset, ytr, yte, labels in zip(datasets, y_trains, y_tests, ['AB', 'CD', 'EF', 'GH', 'IJ', 'KL']):

        # Remap to letter labels
        label_map = label_mappings[dataset]

        ytr_remapped = [np.array([label_map[str(l)] for l in arr]) for arr in ytr]
        yte_remapped = [np.array([label_map[str(l)] for l in arr]) for arr in yte]

        plot_class_balance(
            ys=ytr_remapped,
            title=f'{dataset} Train',
            palette=global_color_mapping,
            label_order=label_display_order,
            ax=axd[labels[0]]
        )
        plot_class_balance(
            ys=yte_remapped,
            title=f'{dataset} Test',
            palette=global_color_mapping,
            label_order=label_display_order,
            ax=axd[labels[1]]
        )

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig(os.path.join(PLOT_DIR, 'fig8s_class_balance.png'), dpi=fig.dpi)


def _to_column_major(handles, ncol):
    """Reorder handles from row-major to column-major layout."""
    import numpy as np
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    # pad handles so we can reshape cleanly
    padded = handles + [None] * (nrow * ncol - n)
    arr = np.array(padded).reshape(nrow, ncol)
    # flatten column-major (Fortran order)
    reordered = arr.T.flatten()
    # drop padding
    return [h for h in reordered if h is not None]


def fig3_iterative_refinement():

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.patches import Polygon
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    from flagx.io import FlowDataManager
    from validation.plt import annotate_mosaic


    fdm = FlowDataManager(
        data_file_names=['annotated_train_data.fcs', ],  # ['annotated_train_data_downsampled_100000_events.fcs', ],
        data_file_path='./results/pipeline_workflow/Imstat/output'
    )
    fdm.load_data_files_to_anndata()
    adata = fdm.anndata_list_[0]

    print(adata)
    print(adata.var_names)

    fig = plt.figure(figsize=(8, 8), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        '''
        AB
        CD
        ''',
        # gridspec_kw={'height_ratios': [1, 1]}
    )

    # --- Panel A: Ground truth gating in UMAP
    ax = axd['A']

    population = adata[:, 'population'].X.flatten()
    margin = 2**20 * 0.05
    min_val = margin
    max_val = 2**20 - margin

    original_min = 1
    original_max = 8
    original_range = original_max - original_min
    original_population = ((population - min_val) / (max_val - min_val) * original_range + original_min).astype(int)

    mask_nk_cells = (original_population == 3)

    x_umap = adata[:, 'umap_1'].X.flatten()
    y_umap = adata[:, 'umap_2'].X.flatten()

    ax.scatter(
        x=x_umap[mask_nk_cells],
        y=y_umap[mask_nk_cells],
        c='#4B9B69',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=2,
        label='Ground Truth NK Cells'
    )

    ax.scatter(
        x=x_umap[~mask_nk_cells],
        y=y_umap[~mask_nk_cells],
        c='lightgrey',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=1,
        label='Others'
    )

    start_positions = [(750000, 650000), (660000, 330000), (960000, 420000)]
    end_positions = [(660000, 550000), (600000, 450000), (830000, 420000)]

    for start_position, end_position in zip(start_positions, end_positions):
        ax.annotate(
            '',  # no text
            xy=end_position,
            xytext=start_position,
            arrowprops=dict(
                arrowstyle='-|>,head_length=1,head_width=0.5',
                color='crimson',
                lw=4
            )
        )

    find_arrow_pos = False
    if find_arrow_pos:

        xticks = np.arange(0, 1000000 + 1, 100000)
        yticks = np.arange(0, 1000000 + 1, 100000)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(6)

        ax.grid(True, which='major', linestyle='-', color='black', alpha=0.9)

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    ax.legend(markerscale=5)

    # --- Panel B: Ground truth gating in trained SOM
    ax = axd['B']

    x_som = adata[:, 'som_1'].X.flatten()
    y_som = adata[:, 'som_2'].X.flatten()

    ax.scatter(
        x=x_som[mask_nk_cells],
        y=y_som[mask_nk_cells],
        c='#4B9B69',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=2,
        label='Ground Truth NK Cells'
    )

    ax.scatter(
        x=x_som[~mask_nk_cells],
        y=y_som[~mask_nk_cells],
        c='lightgrey',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=1,
        label='Others'
    )

    start_positions = [(750000, 880000), (930000, 550000), (750000, 150000)]
    end_positions = [(860000, 810000), (810000, 630000), (610000, 150000)]

    for start_position, end_position in zip(start_positions, end_positions):
        ax.annotate(
            '',
            xy=end_position,
            xytext=start_position,
            arrowprops=dict(
                arrowstyle='-|>,head_length=1,head_width=0.5',
                color='crimson',
                lw=4
            )
        )

    # Plot gate
    poly_points = np.array([
        [470000, 275000],
        [430000, 275000],
        [430000, 390000],
        [390000, 390000],
        [390000, 450000],
        [450000, 510000],
        [575000, 510000],
        [660000, 415000],
        [620000, 375000],
        [620000, 320000],
        [470000, 320000],
    ])
    poly = Polygon(
        poly_points,
        closed=True,
        facecolor='none',
        edgecolor='crimson',
        linewidth=1
    )
    ax.add_patch(poly)

    find_arrow_pos = False
    if find_arrow_pos:

        xticks = np.arange(0, 1000000 + 1, 50000)
        yticks = np.arange(0, 1000000 + 1, 50000)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(3)

        ax.grid(True, which='major', linestyle='-', color='black', alpha=0.9)

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlabel('SOM Grid 1')
    ax.set_ylabel('SOM Grid 2')

    ax.legend(markerscale=5, loc='upper left')

    # --- Panel C: Gating to core NK population in SOM
    ax = axd['C']

    points = adata[:, ['som_1', 'som_2']].X

    shapely_poly = ShapelyPolygon(poly_points)
    mask_core_nk_population = np.array([
        shapely_poly.contains(Point(p)) or shapely_poly.touches(Point(p)) for p in points
    ])

    ax.scatter(
        x=x_som[mask_core_nk_population],
        y=y_som[mask_core_nk_population],
        c='#B94B4B',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=2,
        label='Core NK Cells'
    )

    ax.scatter(
        x=x_som[~mask_core_nk_population],
        y=y_som[~mask_core_nk_population],
        c='lightgrey',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=1,
        label='Others'
    )

    # Plot gate
    poly = Polygon(
        poly_points,
        closed=True,
        facecolor='none',
        edgecolor='crimson',
        linewidth=1
    )
    ax.add_patch(poly)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('SOM Grid 1')
    ax.set_ylabel('SOM Grid 2')

    ax.legend(markerscale=5, loc='lower left')

    # --- Panel D: Core NK population in UMAP
    ax = axd['D']

    ax.scatter(
        x=x_umap[mask_core_nk_population],
        y=y_umap[mask_core_nk_population],
        c='#B94B4B',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=2,
        label='Core NK Cells'
    )

    ax.scatter(
        x=x_umap[~mask_core_nk_population],
        y=y_umap[~mask_core_nk_population],
        c='lightgrey',
        s=2,
        linewidths=0.1,
        edgecolors='darkgrey',
        alpha=0.9,
        zorder=1,
        label='Others'
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    ax.legend(markerscale=5, loc='lower left')


    annotate_mosaic(fig=fig, axd=axd, fontsize=16, excluded=None)

    fig.savefig(os.path.join(PLOT_DIR, 'fig3_iterative_refinement.png'), dpi=fig.dpi)



if __name__ == '__main__':

    os.makedirs(PLOT_DIR, exist_ok=True)

    import matplotlib

    matplotlib.use('Agg')

    # fig1_gating_performance()

    # fig3_iterative_refinement()

    # fig5_num_samples()

    # fig6_precision_recall()

    # fig1s_population_sizes()

    # fig2s_gating_performance_class_wise()

    # fig3s_num_samples_num_events()

    # fig4s_sample_order()  # todo

    # fig5s_minority_count()

    # fig6s_gating_performance_local()

    # fig7s_dataset_size()

    # fig8s_class_balance()

    print('done')



