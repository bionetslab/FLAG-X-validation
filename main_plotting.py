
import os

PLOT_DIR = './results/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

DATASET_TO_DIR = {
    'Imstat': 'imstat',
    'LT1': 'lymphoma_tube1', 'LT1b': 'lymphoma_tube1_binary',
    'LT2': 'lymphoma_tube2', 'LT2b': 'lymphoma_tube2_binary',
    'Flowcyt': 'flowcyt'
}

METHOD_TO_DIR = {
    'GateMeClass': 'gatemeclass_no_abstention',
    'DGCyTOF': 'dgcytof',
    'FCNN': 'fcnn',
    'SOM-Classifier': 'som_classifier'
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

        axd[key].legend(handles, labels, ncol=2, markerscale=1.5, loc='lower right')

    ax_e = axd['E']
    ax_e.set_xlabel(None)
    ax_e.set_ylabel(ax_e.get_ylabel(), fontsize=ax_label_fontsize)
    ax_e.tick_params(axis='y', labelsize=ax_label_fontsize - 2)
    ax_e.tick_params(axis='x', labelsize=ax_label_fontsize)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(os.path.join(PLOT_DIR, 'fig1_gating_performance.png'), dpi=fig.dpi)


def fig4_num_samples():

    # Todo: remove legends and place in separate panel above

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

    n_events = [100, 1000, 5000, 10000, 20000, 50000, 'all']

    dataset_names = ['Flowcyt', 'LT1', 'LT2', 'Imstat', 'LT1b', 'LT2b']

    method_names = ['FCNN', 'SOM-Classifier']

    plot_all_samples_score = True

    ####################################################################################################################

    all_records = []
    for dataset in dataset_names:

        # max_n = dsn_to_maxn[ds]
        # ds_dir = conversion_mapping_datasets[ds]

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
        """,
        gridspec_kw=None
    )

    plot_labels = list('ABCDEF')

    for dataset, plot_label in zip(dataset_names, plot_labels):

        # Subset to the dataset and all events (no per sample downsampling)
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

    # Adjust font sizes
    ax_label_fontsize = 12
    for label in plot_labels:
        ax = axd[label]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig(os.path.join(PLOT_DIR, 'fig4_num_samples.png'), dpi=fig.dpi)
    plt.close('all')


# Todo: ...
def main_precision_and_recall_plot_manuscript():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.ticker import FormatStrFormatter
    from validation.utils import prec_rec_f1_avg_sample_wise
    from validation.plt import annotate_mosaic

    # ### Set flags and important variables here #######################################################################
    data_sets = ['LT1 b', 'LT2 b']
    trafo = 'log10_channelwisecutoff'  # 'arcsinh_cofactor150', 'log10_channelwisecutoff'

    methods = ['FCNN', 'SOM-Classifier']

    thresholds = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
    ]

    generate_plot_df = False

    plot_dir = os.path.join(os.getcwd(), 'results/plots')

    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {'LT1 b': 'lymphoma_tube1_binary', 'LT2 b': 'lymphoma_tube2_binary',}

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {'FCNN': 'softmax', 'SOM-Classifier': 'som'}

    if generate_plot_df:

        pos_label = 1

        long_data = []

        for data_set in data_sets:

            # Load the sample-wise test data
            samples_p = os.path.join(
                os.getcwd(), 'data/np_files', conversion_mapping_datasets[data_set], trafo, 'sample_wise_test'
            )
            n_samples = len([f for f in os.listdir(samples_p) if f.startswith('y_')])
            sample_names = [f'sample_{str(i).zfill(2)}_test' for i in range(n_samples)]
            samples_y_test = [np.load(os.path.join(samples_p, f'y_{sn}.npy')) for sn in sample_names]

            for m in methods:

                # Load the probabilistic predictions
                pred_p = os.path.join(
                    os.getcwd(),
                    'results/probabilistic_pred',
                    conversion_mapping_methods[m],
                    conversion_mapping_datasets[data_set],
                    trafo,
                    'samples_y_proba'
                )
                samples_y_proba = [np.load(os.path.join(pred_p, f'y_proba_{sn}.npy')) for sn in sample_names]

                # Get the predicted probability for class 1
                samples_y_proba = [y[:, 1] for y in samples_y_proba]

                # Get predictions for each threshold and each sample
                for threshold in thresholds:

                    # Get prediction for current threshold
                    samples_y_preds = [(y_prob >= threshold).astype(int) for y_prob in samples_y_proba]

                    print('# ### Calculating evaluation metrics:', data_set, m, threshold)

                    res_df_avg_prec, res_df_avg_rec, res_df_avg_f1 = prec_rec_f1_avg_sample_wise(
                        y_trues=samples_y_test,
                        y_preds=samples_y_preds,
                        abstention_label=None,
                        others_label=None,
                        pos_label=pos_label,
                        verbosity=0,
                    )

                    long_data.append({
                        'Dataset': data_set,
                        'Method': m,
                        'Threshold': threshold,
                        'Precision': res_df_avg_prec.loc['mean', 'binary'],
                        'Recall': res_df_avg_rec.loc['mean', 'binary']
                    })

        plot_df = pd.DataFrame(long_data)

        plot_df.to_csv(os.path.join(os.getcwd(), 'results/probabilistic_pred/plot_df.csv'))

    else:

        plot_df = pd.read_csv(os.path.join(os.getcwd(), 'results/probabilistic_pred/plot_df.csv'), index_col=0)


    plot_df_long = plot_df.melt(
        id_vars=['Dataset', 'Method', 'Threshold'],
        value_vars=['Precision', 'Recall'],
        var_name='Metric',
        value_name='Score'
    )

    # ### Plotting
    # Define a palette
    mn = ['dummy0', 'dummy1', 'FCNN', 'SOM-Classifier']
    palette = dict(zip(mn, sns.color_palette('Set2', len(mn))))

    fig = plt.figure(figsize=(8, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        """,
        gridspec_kw=None
    )

    for ds, plot_label in zip(data_sets, ['A', 'B']):

        # Subset to dataset
        # df_sub = plot_df_long.loc[plot_df_long['Dataset'] == ds].copy()
        df_sub = plot_df_long.loc[
            (plot_df_long['Dataset'] == ds) &
            (plot_df_long['Threshold'] != 0.01) &
            (plot_df_long['Threshold'] != 0.99)
        ].copy()

        ax = axd[plot_label]

        marker_styles = {
            'Precision': 'o',
            'Recall': '^'
        }

        line_styles = {
            'Precision': (4, 2),
            'Recall': (1, 0)
        }

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

        ax.set_title(ds)
        ax.grid(True, alpha=0.6)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if ds == 'LT2 b':
            ax.legend(loc='lower center')

    # Adjust font sizes
    ax_label_fontsize = 12
    for key in ['A', 'B']:
        ax = axd[key]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'recall_precision_manuscript.png'),
        dpi=fig.dpi
    )
    plt.close('all')


def main_population_size_plot_supplement():
    import os
    import random
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from itertools import chain
    from validation.plt import plot_cell_pop_size_pred_vs_gt, annotate_mosaic

    random.seed(43)
    ####################################################################################################################
    dataset_names = ['Flowcyt', 'LT1', 'LT1 b', 'LT2', 'LT2 b']

    method_names = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    data_trafo = 'log10_channelwisecutoff'

    plot_dir = os.path.join(os.getcwd(), 'results/plots')

    base_path_y_true = os.path.join(os.getcwd(), 'data/np_files')
    base_path_y_pred = os.path.join(os.getcwd(), 'results/pred_eval')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GateMeClass': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier',
        'SOM-Classifier': 'som_classifier'
    }

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
        'LT1': lm_lt1, 'LT1 b': lm_lt1b, 'LT2': lm_lt2, 'LT2 b': lm_lt2b, 'Flowcyt': lm_flowcyt
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
        P.QRS
        T.UVW
    '''

    fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(mosaic_str)

    legend_subplots = list('AFKPT')
    plot_subplots = list('BCDEGHIJLMNOQRSUVW')
    legend_reference_subplots = list('EJOSW')

    count = 0
    for dataset_name in dataset_names:

        dataset_dir = conversion_mapping_datasets[dataset_name]

        # Set data trafo to be used
        is_flowcyt = (dataset_dir == 'flowcyt')
        base_trafo = 'log10_cutoff100' if is_flowcyt and data_trafo == 'log10_channelwisecutoff' else 'log10_channelwisecutoff'

        # Load the sample-wise data (ground truth and prediction)
        y_true_path = os.path.join(base_path_y_true, dataset_dir, base_trafo, 'sample_wise_test')

        n_samples = len([f for f in os.listdir(y_true_path) if f.startswith('y_')])
        y_true_filenames = [f'y_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]
        y_trues = [np.load(os.path.join(y_true_path, f)).astype(int) for f in y_true_filenames]

        # Define color mapping

        cell_type_labels = np.unique(np.concatenate(y_trues)).tolist()
        label_mapping = dataset_name_to_label_mapping[dataset_name]
        color_mapping = {
            str(int_label): global_color_mapping[label_mapping[str(int_label)]]
            for int_label in cell_type_labels
        }

        for method in method_names:

            method_dir = conversion_mapping_methods[method]

            # No results for gatemeclass and LT2, LT2 b -> continue
            if method_dir == 'gatemeclass_no_abstention' and dataset_name in {'LT2', 'LT2 b'}:
                continue

            # Always show results for arcsinh for gatemeclass
            if method_dir == 'gatemeclass_no_abstention' and base_trafo in {'log10_channelwisecutoff', 'log10_cutoff100'}:
                data_trafo_load = 'arcsinh_cofactor150'
            else:
                data_trafo_load = base_trafo


            y_pred_path = os.path.join(base_path_y_pred, method_dir, dataset_dir, data_trafo_load, 'samples_y_pred')
            y_pred_filenames = [f'y_pred_sample_{str(i).zfill(2)}_test.npy' for i in range(n_samples)]

            # Load the predictions, skip missing files
            y_preds = []
            y_trues_plot = []
            for yt, f in zip(y_trues, y_pred_filenames):
                try:
                    yp = np.load(os.path.join(y_pred_path, f)).astype(int)
                    y_preds.append(yp)
                    y_trues_plot.append(yt)
                except FileNotFoundError:
                    print(f'# ### No y_pred found for: {method}, {dataset_name}, {data_trafo_load}, {f}')

            ax = axd[plot_subplots[count]]

            plot_cell_pop_size_pred_vs_gt(
                y_trues=y_trues_plot,
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
            legend_subplots, legend_reference_subplots, dataset_names
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
    plt.savefig('./results/plots/population_sizes_supplement.png', dpi=fig.dpi)
    plt.close('all')


def main_performance_score_plot_supplement():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from validation.plt import plot_performance_score_box_plot_cw, annotate_mosaic


    ####################################################################################################################
    dataset_names = ['Flowcyt', 'Imstat', 'LT1', 'LT1 b', 'LT2', 'LT2 b']

    method_names = ['GateMeClass', 'DGCyTOF', 'FCNN', 'SOM-Classifier']

    data_trafo = 'log10_channelwisecutoff'  # log10_channelwisecutoff, arcsinh_cofactor150, log10_cutoff100

    performance_score = 'f1'  # f1, prec, rec

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'GateMeClass': 'gatemeclass_no_abstention', 'GMC wa': 'gatemeclass_w_abstention',
        'DGCyTOF': 'dgcytof', 'FCNN': 'softmax_classifier',
        'SOM-Classifier': 'som_classifier'
    }

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
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1 b': lm_lt1b, 'LT2': lm_lt2, 'LT2 b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }

    label_display_order = ['HSPC', 'M', 'M16', 'Ma', 'T', 'Th', 'NK', 'G', 'B', 'dyB', 'X', 'O']

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        CD
        EF
        """
    )

    # Define a palette
    palette = dict(zip(method_names, sns.color_palette("Set2", len(method_names))))

    # ### Plot the performance scores
    # Load the results dataframes
    base_path = os.path.join(os.getcwd(), 'results/pred_eval')

    res_dfs = []
    for dataset_name in dataset_names:
        res_dfs_sub = []
        for method in method_names:

            ds = conversion_mapping_datasets[dataset_name]
            m = conversion_mapping_methods[method]

            # Always show results for arcsinh for gatemeclass
            if m == 'gatemeclass_no_abstention' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'arcsinh_cofactor150'
            elif ds == 'flowcyt' and data_trafo == 'log10_channelwisecutoff':
                data_trafo_load = 'log10_cutoff100'
            else:
                data_trafo_load = data_trafo

            res_df_path = os.path.join(base_path, m, ds, data_trafo_load, f'res_df_sw_cw_{performance_score}.csv')

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')

                # Change the column names to letter labels
                label_mapping = label_mappings[dataset_name]
                res_df = res_df.rename(columns=label_mapping)

            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    for rdf, dsn, label in zip(res_dfs, dataset_names, ['A', 'B', 'C', 'D', 'E', 'F']):

        plot_performance_score_box_plot_cw(
            sample_wise_res_dfs=rdf,
            method_names=method_names,
            y_label=conversion_mapping_y_label[performance_score] + ' Score',
            title=dsn,
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

    '''# ### Manually adjust axis labels
    ax_label_fontsize = 12

    for key in ['A', 'B']:
        ax = axd[key]
        ax.set_xlabel(None, fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)

        ax.tick_params(axis='x', labelsize=ax_label_fontsize)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)'''

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig(f'./results/plots/performance_supplement.png', dpi=fig.dpi)
    plt.close('all')


def main_performance_plot_local_supplement():

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    from validation.plt import plot_performance_score_box_plot, plot_performance_score_box_plot_cw, annotate_mosaic

    ####################################################################################################################
    dataset_names = ['Flowcyt', 'Imstat', 'LT1', 'LT1 b', 'LT2', 'LT2 b']

    method_names = ['FCNN', 'SOM-Classifier']

    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    # Convert method names to corresponding dir names
    conversion_mapping_methods = {
        'FCNN': 'softmax',
        'SOM-Classifier': 'som'
    }

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
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1 b': lm_lt1b, 'LT2': lm_lt2, 'LT2 b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }

    label_display_order = ['HSPC', 'M', 'M16', 'Ma', 'T', 'Th', 'NK', 'G', 'B', 'dyB', 'X', 'O']

    # Define a palette
    methods = ['dummy0', 'dummy1'] + method_names
    palette = dict(zip(methods, sns.color_palette("Set2", len(methods))))

    # Initialize the mosaic
    fig = plt.figure(figsize=(8, 10), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AAAB
        CCDD
        EEFF
        GGHH
        """,
        # gridspec_kw={'height_ratios': [1/4, 1/4, 1/2]}
    )

    # ### Plot the performance scores
    # Load the results dataframes
    base_path = os.path.join(os.getcwd(), 'results/local_training')

    res_dfs = []
    for ds in dataset_names:
        res_dfs_sub = []
        for m in method_names:

            ds_dir = conversion_mapping_datasets[ds]
            method_dir = conversion_mapping_methods[m]
            data_trafo = 'log10_channelwisecutoff' if ds != 'Flowcyt' else 'log10_cutoff100'

            res_df_path = os.path.join(
                base_path, method_dir, ds_dir, data_trafo, f'res_df_sw_avg_{performance_score}.csv'
            )

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')
            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    plot_performance_score_box_plot(
        sample_wise_res_dfs=res_dfs,
        method_names=method_names,
        dataset_names=dataset_names,
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

    res_dfs = []
    for dataset_name in dataset_names:
        res_dfs_sub = []
        for method in method_names:

            ds = conversion_mapping_datasets[dataset_name]
            m = conversion_mapping_methods[method]

            data_trafo = 'log10_channelwisecutoff' if dataset_name != 'Flowcyt' else 'log10_cutoff100'

            res_df_path = os.path.join(base_path, m, ds, data_trafo, f'res_df_sw_cw_{performance_score}.csv')

            try:
                res_df = pd.read_csv(res_df_path, index_col=0)
                res_df = res_df.drop(index=['mean', 'std'], errors='ignore')

                # Change the column names to letter labels
                label_mapping = label_mappings[dataset_name]
                res_df = res_df.rename(columns=label_mapping)

            except FileNotFoundError:
                res_df = pd.DataFrame()
                print(f"# ### No results found for dataset: '{ds}', method: '{m}', data trafo: '{data_trafo}'")

            res_dfs_sub.append(res_df)
        res_dfs.append(res_dfs_sub)

    for rdf, dsn, label in zip(res_dfs, dataset_names, ['C', 'D', 'E', 'F', 'G', 'H']):

        plot_performance_score_box_plot_cw(
            sample_wise_res_dfs=rdf,
            method_names=method_names,
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

    # Extract legend handles and labels from axd['A']
    handles, labels = axd['A'].get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if isinstance(h, Patch)]
    handles, labels = zip(*filtered) if filtered else ([], [])

    # Plot the legend separately in panel 'B'
    axd['B'].axis('off')
    axd['B'].legend(handles, labels, loc='center', frameon=False, ncol=1)

    for key, ax in axd.items():
        if key != 'B':
            ax.set_xlabel(None)
            ax.get_legend().remove()

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig('./results/plots/performance_local_supplement.png', dpi=fig.dpi)


def main_n_samples_plot_supplement():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.lines import Line2D
    from validation.plt import annotate_mosaic

    # Configuration ####################################################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    n_events = [100, 1000, 5000, 10000, 20000, 50000, 'all']

    dataset_names = ['Imstat', 'LT1', 'LT2', 'LT1 b', 'LT2 b', 'Flowcyt']

    method_names = ['FCNN', 'SOM-Clf.']

    n_events_plot = [5000, 10000, 20000, 50000, 'all']

    dataset_to_max_n = {'Imstat': 75, 'LT1 b': 73, 'LT2 b': 73, 'LT1': 73, 'LT2': 73, 'Flowcyt': 22}

    plot_all_samples_score = True

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    os.makedirs(plot_dir, exist_ok=True)

    ####################################################################################################################

    # Directory mappings
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    conversion_mapping_methods = {'FCNN': 'softmax', 'SOM-Clf.': 'som'}

    all_records = []

    for ds in dataset_names:
        max_n = dataset_to_max_n[ds]
        ds_dir = conversion_mapping_datasets[ds]
        data_trafo = 'log10_channelwisecutoff' if ds != 'Flowcyt' else 'log10_cutoff100'

        for method in method_names:
            for n in n_events:
                for i in range(1, max_n + 1):

                    method_dir = conversion_mapping_methods[method]

                    if i == max_n and n == 'all':    # Load previously computed scores for (all samples, all events)
                        file_path = os.path.join(
                            './results/pred_eval',
                            method_dir + '_classifier',
                            ds_dir,
                            data_trafo,
                            f'res_df_sw_avg_{performance_score}.csv'
                        )
                    else:
                        file_path = os.path.join(
                            './results/n_samples_n_events',
                            method_dir,
                            ds_dir,
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
                                'dataset': ds,
                                'method': method,
                                'n_events': n,
                                'n_samples': i,
                                'score': val
                            })

                    except FileNotFoundError as e:
                        # print(f"# Missing: {file_path}")
                        continue

    # Create DataFrame
    df_all = pd.DataFrame(all_records)
    df_all['n_samples'] = df_all['n_samples'].astype(int)

    # Subset dataframe
    keep_bool_n_events = df_all['n_events'].isin(n_events_plot)
    keep_bool_n_samples = (
            (df_all['n_samples'] % 5 == 0) |
            (df_all['n_samples'] == 1) |
            (df_all['n_samples'] >= 70) |
            ((df_all['n_samples'] == 22) & (df_all['dataset'] == 'Flowcyt'))
    )
    keep_bool = np.logical_and(keep_bool_n_events, keep_bool_n_samples)
    df_all = df_all[keep_bool]

    print(df_all)

    # ### Plot performance comparison for random vs ordered and num events
    plot_combinations = [
        ('Flowcyt', 'SOM-Clf.'), ('LT1', 'SOM-Clf.'), ('LT2', 'SOM-Clf.'),
        ('Imstat', 'SOM-Clf.'), ('LT1 b', 'SOM-Clf.'), ('LT2 b', 'SOM-Clf.'),
        ('Flowcyt', 'FCNN'), ('LT1', 'FCNN'), ('LT2', 'FCNN'),
        ('Imstat', 'FCNN'), ('LT1 b', 'FCNN'), ('LT2 b', 'FCNN'),
    ]

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
    layout_str = '''
            ABC
            DEF
            GHI
            JKL
        '''
    axd = fig.subplot_mosaic(layout_str)

    plot_labels = [c for c in layout_str if c.isalpha()]

    for comb, plot_label in zip(plot_combinations, plot_labels):
        dataset = comb[0]
        method = comb[1]

        df_sub = df_all.loc[
            (df_all['dataset'] == dataset) &
            (df_all['method'] == method)
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
                (df_sub['n_samples'] == dataset_to_max_n[dataset]) &
                (df_sub['n_events'] == 'all')
            ]

            score = df_all_data_score['score'].mean()

            color = 'grey'

            ax.axhline(
                y=score,
                linestyle='--',
                linewidth=1,
                color=color,
                alpha=0.8,
                # label=f'{method_name} (all samples)',
            )

            x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            y_pos = score + y_offset  # just above the line
            va = 'bottom'

            if y_pos > ax.get_ylim()[1]:  # would be clipped at the top
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
                clip_on=False
            )

            # Define dummy legend entry for all sample performance
            all_samples_legend = Line2D([], [], linestyle='--', color='grey', linewidth=1, label='All Samples')
            handles, labels = ax.get_legend_handles_labels()
            if 'All Samples' not in labels:
                handles.append(all_samples_legend)
                labels.append('All Samples')
            ax.legend(handles=handles, labels=labels, title='Method')

        else:
            ax.legend(title='Method')

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

    # Adjust font sizes
    ax_label_fontsize = 12
    for key in plot_labels:
        ax = axd[key]
        ax.set_title(ax.get_title(), fontsize=ax_label_fontsize + 2)
        ax.set_xlabel(ax.get_xlabel(), fontsize=ax_label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=ax_label_fontsize)
        ax.tick_params(axis='x', labelsize=ax_label_fontsize - 2)
        ax.tick_params(axis='y', labelsize=ax_label_fontsize - 2)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'n_samples_supplement.png'),
        dpi=fig.dpi
    )
    plt.close('all')


def main_n_samples_ordered_plot_supplement():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.lines import Line2D

    from validation.plt import annotate_mosaic

    # Configuration ####################################################################################################
    performance_score = 'f1'  # f1, prec, rec
    performance_score_mode = 'macro'  # macro, micro, weighted, binary

    dataset_names = ['LT1', 'LT1 b']
    max_n_samples = 20
    max_n_trails = 30

    method_names = ['FCNN', 'SOM-Classifier']

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    os.makedirs(plot_dir, exist_ok=True)

    ####################################################################################################################

    # Directory mappings
    conversion_mapping_datasets = {'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary'}

    conversion_mapping_methods = {'FCNN': 'softmax', 'SOM-Classifier': 'som'}

    data_ordered = []
    for ds in dataset_names:
        ds_dir = conversion_mapping_datasets[ds]

        for method in method_names:
            method_dir = conversion_mapping_methods[method]

            file_path_ordered = os.path.join(
                './results/n_samples_n_events',
                method_dir,
                ds_dir,
                'log10_channelwisecutoff',
                'ordered',
                f'res_df_{performance_score}_{performance_score_mode}.csv'
            )

            df_ordered = pd.read_csv(file_path_ordered, index_col=0)

            for i in range(1, max_n_samples + 1):

                score = df_ordered.loc['all', str(i)]

                data_ordered.append({
                    'dataset': ds,
                    'method': method,
                    'order': 'ordered',
                    'n_samples': i,
                    'score': score
                })

    data_random = []
    for ds in dataset_names:
        ds_dir = conversion_mapping_datasets[ds]

        for method in method_names:
            method_dir = conversion_mapping_methods[method]

            file_path_random = os.path.join(
                './results/random_sample_order_trials',
                method_dir,
                ds_dir,
                'log10_channelwisecutoff',
                f'res_df_{performance_score}_{performance_score_mode}.csv'
            )

            try:
                df_random = pd.read_csv(file_path_random, index_col=0)
            except FileNotFoundError:
                continue

            for n in df_random.index[: max_n_trails]:
                for i in df_random.columns[: max_n_samples]:
                    score = df_random.loc[n, i]
                    data_random.append({
                        'dataset': ds,
                        'method': method,
                        'order': 'random',
                        'trial_no': n,
                        'n_samples': i,
                        'score': score
                    })

    # Load the results for all samples
    dict_all_data = dict()
    for ds in dataset_names:
        ds_dir = conversion_mapping_datasets[ds]
        for method in method_names:
            method_dir = conversion_mapping_methods[method]
            file_path = os.path.join(
                './results/pred_eval',
                method_dir + '_classifier',
                ds_dir,
                'log10_channelwisecutoff',
                f'res_df_sw_avg_{performance_score}.csv'
            )

            df = pd.read_csv(file_path, index_col=0)
            score = df.loc['mean', performance_score_mode]

            if ds in dict_all_data:
                dict_all_data[ds][method] = score
            else:
                dict_all_data[ds] = {method: score}


    # Create DataFrames
    df_ordered = pd.DataFrame(data_ordered)
    df_ordered['n_samples'] = df_ordered['n_samples'].astype(int)

    df_random = pd.DataFrame(data_random)
    df_random['n_samples'] = df_random['n_samples'].astype(int)

    # ### Plot performance comparison for methods
    # Define a palette
    mn = ['dummy0', 'dummy1', 'FCNN', 'SOM-Classifier']
    palette = dict(zip(mn, sns.color_palette('Set2', len(mn))))

    fig = plt.figure(figsize=(8, 3), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        AB
        """,
        gridspec_kw=None
    )

    for ds, plot_label in zip(dataset_names, ['A', 'B']):

        ax = axd[plot_label]

        df_random_sub = df_random.loc[(df_random['dataset'] == ds)].copy()

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


        df_ordered_sub = df_ordered.loc[(df_ordered['dataset'] == ds)].copy()

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
        ax.set_title(ds)
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(f'{performance_score_mode.capitalize()} {performance_score.capitalize()} Score')

        # Set min and max number of samples as x ticks
        x_min, x_max = 1, max_n_samples
        current_ticks = ax.get_xticks()
        current_ticks = [tick for tick in current_ticks if x_min <= tick <= x_max]
        new_ticks = [x_min] + current_ticks + [x_max]
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in new_ticks])

        # Plot the all samples scores
        for method in method_names:

            score = dict_all_data[ds][method]
            color = palette.get(method, 'grey')

            ax.axhline(
                y=score,
                linestyle='--',
                linewidth=1,
                color=color,
                alpha=0.8,
                # label=f'{method_name} (all samples)',
            )

            # x_pos = ax.get_xlim()[1] * 0.98  # slightly inside right edge
            # y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            # y_pos = score + y_offset  # just above the line

            # Draw text
            # ax.text(
            #     x=x_pos,
            #     y=y_pos,
            #     s=f'{score:.3f}',
            #     color=color,
            #     va='bottom',
            #     ha='right',
            #     fontsize=8,
            #     alpha=1.0,
            #       clip_on=True
            # )

        # Define all legend components
        method_handles = [
            Line2D([0], [0], color=palette[method], lw=2, label=method)
            for method in method_names
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

        legend_loc = 'center right' if ds == 'LT1' else 'lower right'

        # Create single unified legend
        ax.legend(handles=all_handles, title='Method & Order', loc=legend_loc)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'n_samples_random_vs_ordered_supplement.png'),
        dpi=300
    )
    plt.close('all')


def main_dataset_size_plot_supplement():

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from validation.plt import plot_sample_sizes, annotate_mosaic


    ####################################################################################################################
    dataset_names = ['Flowcyt', 'Imstat', 'LT1', 'LT2']

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    y_trains = []
    y_tests = []
    for ds in dataset_names:

        # Load the sample-wise data
        base_path = os.path.join(os.getcwd(), 'data/np_files', conversion_mapping_datasets[ds], 'arcsinh_cofactor150')

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

    for ds, ytr, yte, labels in zip(dataset_names, y_trains, y_tests, ['AB', 'CD', 'EF', 'GH']):

        plot_sample_sizes(
            ys=ytr, title=f'{ds} Train', abline_mean=True, abline_std=True, print_total=True, ax=axd[labels[0]]
        )
        plot_sample_sizes(
            ys=yte, title=f'{ds} Test', abline_mean=True, abline_std=True, print_total=True, ax=axd[labels[1]]
        )

        axd[labels[0]].legend(loc='lower right')
        axd[labels[1]].legend(loc='lower right')

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig('./results/plots/dataset_sizes_supplement.png', dpi=fig.dpi)


def main_dataset_balance_plot_supplement():

    import os
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from itertools import chain
    from validation.plt import plot_class_balance, annotate_mosaic

    random.seed(24)
    ####################################################################################################################
    dataset_names = ['Flowcyt', 'Imstat', 'LT1', 'LT2', 'LT1 b', 'LT2 b']

    plot_dir = os.path.join(os.getcwd(), 'results/plots')
    ####################################################################################################################

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    conversion_mapping_datasets = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

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
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1 b': lm_lt1b, 'LT2': lm_lt2, 'LT2 b': lm_lt2b, 'Flowcyt': lm_flowcyt
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
    for ds in dataset_names:
        # Load the sample-wise data
        base_path = os.path.join(os.getcwd(), 'data/np_files', conversion_mapping_datasets[ds],
                                 'arcsinh_cofactor150')

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

    for ds, ytr, yte, labels in zip(dataset_names, y_trains, y_tests, ['AB', 'CD', 'EF', 'GH', 'IJ', 'KL']):

        # Remap to letter labels
        label_map = label_mappings[ds]

        ytr_remapped = [np.array([label_map[str(l)] for l in arr]) for arr in ytr]
        yte_remapped = [np.array([label_map[str(l)] for l in arr]) for arr in yte]

        plot_class_balance(
            ys=ytr_remapped,
            title=f'{ds} Train',
            palette=global_color_mapping,
            label_order=label_display_order,
            ax=axd[labels[0]]
        )
        plot_class_balance(
            ys=yte_remapped,
            title=f'{ds} Test',
            palette=global_color_mapping,
            label_order=label_display_order,
            ax=axd[labels[1]]
        )

    # ### Manually adjust axis labels
    # for key in ['F', 'G', 'H', 'J', 'K', 'L']:
    #     ax = axd[key]
    #     ax.set_ylabel(None)

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)

    plt.savefig('./results/plots/dataset_balances_supplement.png', dpi=fig.dpi)


def main_pipeline_workflow():
    import os
    import random
    import readfcs
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')
    random.seed(42)

    from seaborn import scatterplot
    from flagx import GatingPipeline

    train = True

    # ###### Initial training ###### #
    # ### Set parameters
    dataset = 'LT2'  # Imstat, LT1, LT2

    if dataset == 'Imstat':
        channels = ['FS INT', 'SS INT', '16-FITC', '56-PE', '3-ECD', '4-PC7', '19-APC', '14-APC700', '8-PB', '45-CO']
        label_key = 'population'
        cutoff_dict = {
            'FS INT': 100000, 'SS INT': 20000, '16-FITC': 250, '56-PE': 450, '3-ECD': 700,
            '4-PC7': 1200,
            '19-APC': 1700,
            '14-APC700': 900,
            '8-PB': 450, '45-CO': 500
        }

    elif dataset == 'LT1':
        channels = [
            'FS', 'SS', 'kappavCD8_FITC', 'lambdavCD7_PE', 'CD23_ECD', 'CD79bvCD4_PC5.5', 'CD5_PC7',
            'CD38_APC', 'CD19_APC_A700', 'CD20vCD3_APC_A750', 'FMC7vCD2_PB', 'CD45_KrOr'
        ]
        label_key = 'population'
        cutoff_dict = {
            'FS': 100000, 'SS': 20000, 'kappavCD8_FITC': 500, 'lambdavCD7_PE': 400, 'CD23_ECD': 500,
            'CD79bvCD4_PC5.5': 1200, 'CD5_PC7': 300, 'CD38_APC': 700,
            'CD19_APC_A700': 150,
            'CD20vCD3_APC_A750': 500, 'FMC7vCD2_PB': 500, 'CD45_KrOr': 1000
        }

    else:  # LT2
        channels = [
            'FS', 'SS', 'CD103_FITC', 'CD43_PE', 'CD25_ECD', 'CD10_PC5.5', 'CD200_PC7',
            'CD52_APC', 'CD11c_APC_A700', 'CD20_APC_A750', 'IgM_PB', 'CD19_KrOr'
        ]
        label_key = 'population'
        cutoff_dict = {
            'FS': 100000, 'SS': 20000, 'CD103_FITC': 300, 'CD43_PE': 1500, 'CD25_ECD': 1000,
            'CD10_PC5.5': 1000, 'CD200_PC7': 1000, 'CD52_APC': 150, 'CD11c_APC_A700': 200,
            'CD20_APC_A750': 300, 'IgM_PB': 400, 'CD19_KrOr': 200,
        }

    dataset_to_dir = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma/concatenated_labled_fcs_format_21Blood4Bcell_T1_Labels',
        'LT2': 'lymphoma/concatenated_labled_fcs_format_22Blood4Bcell_T2_Labels',
    }

    # ## Define train and test files
    if dataset == 'Imstat':
        train_data_fns = [
            '20150312-1 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019511 001.fcs',
            'ER_000000_H1_150305_ED.fcs',
            '20150320-1 IOTest Test 16-56-3-4-19-14-8-45 00019651 001.fcs',
            '20150317-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019558 001.fcs',
            '20150318-3 VersaLyse VersaLyseFix 16-56-3-4-19-14-8-45 00019593 001.fcs',
            'ER_000050_H1_150311_ED.fcs',
            '20150312-2 IOTest Test 16-56-3-4-19-14-8-45 00019495 001.fcs',
            '20150312-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019496 001.fcs',
            '20150320-1 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019654 001.fcs',
            'ER_000018_H1_150306_ED.fcs',
            '20150320-3 IOTest Test 16-56-3-4-19-14-8-45 00019661 001.fcs',
            '20150319-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019617 001.fcs',
            '20150320-2 VersaLyse VersaLyse 16-56-3-4-19-14-8-45 00019658 001.fcs',
            'ER_000025_H1_150309_ED.fcs',
            '20150318-3 IOTest Test 16-56-3-4-19-14-8-45 00019588 001.fcs',
            '20150318-1 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019572 001.fcs',
            '20150317-2 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019557 001.fcs',
            'ER_000006_H1_150305_ED.fcs',
            '20150312-1 IOTest Test 16-56-3-4-19-14-8-45 00019508 001.fcs',
            '20150319-3 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019637 001.fcs',
        ]
        test_data_fns = [
            'ER_000057_H1_150311_ED.fcs',
            '20150318-2 Quick Quick 16-56-3-4-19-14-8-45 00019581 001.fcs',
            '20150318-1 VersaLyseFix VersaLyseFix 16-56-3-4-19-14-8-45 00019574 001.fcs',
            '20150320-2 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019657 001.fcs',
            '20150318-1 IOTest Test 16-56-3-4-19-14-8-45 00019571 001.fcs',
            'ER_000001_H1_150305_ED.fcs',
            '20150320-3 Quick Quick 16-56-3-4-19-14-8-45 00019660 001.fcs',
            '20150312-2 VersaLyse VersaLyse 16-56-3-4-19-14-8-45 00019500 001.fcs',
            '20150318-3 Facslysing FacsLysing 16-56-3-4-19-14-8-45 00019592 001.fcs',
            '20150319-3 IOTest Test 16-56-3-4-19-14-8-45 00019619 001.fcs',
        ]

    elif dataset == 'LT1':
        train_data_fns = [
            '4790_NB_T1_d13_N48k.fcs',
            '1910_CLL_T1_d13_N100k.fcs',
            '3280_DLBCL_T1_d13_N73k.fcs',
            '1560_MCL_T1_d13_N100k.fcs',
            '0290_FL_T1_d13_N100k.fcs',
            '3630_HCL_T1_d13_N100k.fcs',
            '0550_LPL_T1_d13_N100k.fcs',
            '4010_MZL_T1_d13_N100k.fcs',
            '1350_MBL_T1_d13_N100k.fcs',
            '3110_BL_T1_d13_N100k.fcs',
            '3180_UC_T1_d13_N100k.fcs',
            '6110_NB_T1_d13_N74k.fcs',
            '0480_CLL_T1_d13_N100k.fcs',
            '1090_DLBCL_T1_d13_N100k.fcs',
            '1500_MCL_T1_d13_N100k.fcs',
            '1030_FL_T1_d13_N100k.fcs',
            '3090_HCL_T1_d13_N100k.fcs',
            '0760_LPL_T1_d13_N100k.fcs',
            '2490_MZL_T1_d13_N83k.fcs',
            '3420_MBL_T1_d13_N100k.fcs',
        ]
        test_data_fns = [
            '5630_NB_T1_d13_N100k.fcs',
            '0910_CLL_T1_d13_N100k.fcs',
            '0220_DLBCL_T1_d13_N100k.fcs',
            '1990_FL_T1_d13_N100k.fcs',
            '2740_HCL_T1_d13_N40k.fcs',
            '2100_LPL_T1_d13_N100k.fcs',
            '3730_MZL_T1_d13_N100k.fcs',
            '2990_MBL_T1_d13_N100k.fcs',
            '3670_UC_T1_d13_N100k.fcs',
            '4490_NB_T1_d13_N100k.fcs',
        ]
    else:
        train_data_fns = [
            '4790_NB_T2_d13_N100k.fcs',
            '1910_CLL_T2_d13_N100k.fcs',
            '3280_DLBCL_T2_d13_N65k.fcs',
            '1560_MCL_T2_d13_N100k.fcs',
            '0290_FL_T2_d13_N100k.fcs',
            '3630_HCL_T2_d13_N100k.fcs',
            '0550_LPL_T2_d13_N100k.fcs',
            '4010_MZL_T2_d13_N100k.fcs',
            '1350_MBL_T2_d13_N100k.fcs',
            '3110_BL_T2_d13_N100k.fcs',
            '3180_UC_T2_d13_N100k.fcs',
            '6110_NB_T2_d13_N79k.fcs',
            '0480_CLL_T2_d13_N100k.fcs',
            '1090_DLBCL_T2_d13_N100k.fcs',
            '1500_MCL_T2_d13_N100k.fcs',
            '1030_FL_T2_d13_N100k.fcs',
            '3090_HCL_T2_d13_N100k.fcs',
            '1740_LPL_T2_d13_N100k.fcs',
            '2490_MZL_T2_d13_N100k.fcs',
            '3420_MBL_T2_d13_N100k.fcs',
        ]
        test_data_fns = [
            '5630_NB_T2_d13_N100k.fcs',
            '0910_CLL_T2_d13_N100k.fcs',
            '0220_DLBCL_T2_d13_N100k.fcs',
            '1990_FL_T2_d13_N100k.fcs',
            '2740_HCL_T2_d13_N68k.fcs',
            '2100_LPL_T2_d13_N100k.fcs',
            '3730_MZL_T2_d13_N100k.fcs',
            '2990_MBL_T2_d13_N100k.fcs',
            '3670_UC_T2_d13_N100k.fcs',
            '4490_NB_T2_d13_N100k.fcs',
        ]

    # ### Randomized train and test file selection
    # data_fns = sorted(os.listdir(data_dir))
    # random.shuffle(data_fns)

    # train_data_fns = data_fns[0:75]
    # train_data_fns = data_fns[0:20]
    # test_data_fns_unfiltered = data_fns[75:]

    # if data_subdir != 'imstat':
        # Select test files from each condition
    #     conditions = {'NB', 'CLL', 'DLBCL', 'MCL', 'FL'}  # 'HCL', 'LPL', 'MZL', 'MBL', 'BL', 'UC'}
    #     conditions_seen = set()
    #     test_data_fns = []
    #     for fn in test_data_fns_unfiltered:
    #         fn_parts = fn.split('_')
    #         condition = fn_parts[1]

    #        if condition in conditions and condition not in conditions_seen:
    #             test_data_fns.append(fn)
    #             conditions_seen.add(condition)

    #         if conditions == conditions_seen:
    #             break
    # else:
    #     test_data_fns = test_data_fns_unfiltered[0:5]

    data_subdir = dataset_to_dir[dataset]

    save_path = os.path.join('results/pipeline_workflow', dataset)
    os.makedirs(save_path, exist_ok=True)

    data_dir = os.path.join('./data/raw', data_subdir)

    with open(os.path.join(save_path, 'train_samples.txt'), 'w') as f:
        for line in train_data_fns:
            f.write(line + '\n')

    with open(os.path.join(save_path, 'test_samples.txt'), 'w') as f:
        for line in test_data_fns:
            f.write(line + '\n')

    preprocessing_kwargs = {'flavour': 'log10_w_custom_cutoffs', 'flavour_kwargs': {'cutoffs': cutoff_dict}}

    save_path_som = os.path.join(save_path, 'som')
    os.makedirs(save_path_som, exist_ok=True)

    save_path_fcnn = os.path.join(save_path, 'fcnn')
    os.makedirs(save_path_fcnn, exist_ok=True)

    if train:
        # ### Train the SOM-classifier gating pipeline
        som_kwargs = {
            'som_topology': 'planar',
            'som_grid_type': 'rectangular',
            'som_dimensions': (25, 25),
            'neighborhood': 'gaussian',
            'gaussian_neighborhood_sigma': 0.25,
            'initialization': 'pca',
            'n_epochs': 1000,
            'radius_0': -0.25,
            'radius_n': 0.01,
            'radius_cooling': 'linear',
            'learning_rate_0': 0.5,
            'learning_rate_n': 0.05,
            'learning_rate_decay': 'exponential',
            'verbosity': 2
        }

        gp_som = GatingPipeline(
            train_data_file_path=data_dir,
            train_data_file_names=train_data_fns,
            train_data_file_type='fcs',
            save_path=save_path_som,
            channels=channels,
            label_key=label_key,
            channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
            relabel_data_kwargs=None,
            preprocessing_kwargs=preprocessing_kwargs,
            gating_method='som',
            gating_method_kwargs=som_kwargs,
            verbosity=2,
        )

        gp_som.train()

        gp_som.save(filename='trained_pipeline_som.pkl')

        # ### Train the FCNN-softmax-classifier gating pipeline
        fcnn_kwargs = {'layer_sizes': (128, 64, 32), 'n_epochs': 20, 'device': 'cuda', 'verbosity': 2}

        gp_fcnn = GatingPipeline(
            train_data_file_path=data_dir,
            train_data_file_names=train_data_fns,
            train_data_file_type='fcs',
            save_path=save_path_fcnn,
            channels=channels,
            label_key=label_key,
            channel_names_alignment_kwargs={'reference_channel_names': 0},  # Use 1st file as reference
            relabel_data_kwargs=None,
            preprocessing_kwargs=preprocessing_kwargs,
            gating_method='fcnn',
            gating_method_kwargs=fcnn_kwargs,
            verbosity=2,
        )

        gp_fcnn.train()

        gp_fcnn.save(filename='trained_pipeline_fcnn.pkl')

        del gp_som, gp_fcnn

    # ###### Inference with new data ###### #
    # ### Set parameters
    output_dir = os.path.join(save_path, 'output')
    output_dir_test_samples = os.path.join(output_dir, 'test_samples')
    os.makedirs(output_dir_test_samples, exist_ok=True)

    dim_red_methods = ('som', 'pca', 'umap', 'tsne')
    dim_red_method_kwargs = (None, None, {'n_jobs': 12}, {'n_jobs': 12})

    # ### Inference with the SOM pipeline
    gp_som = GatingPipeline.load(filename='trained_pipeline_som.pkl', filepath=save_path_som)

    gp_som.verbosity = 2

    # Train data
    gp_som.inference(
        data_file_path=data_dir,
        data_file_names=train_data_fns,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_sample_wise=False,
        save_path=output_dir,
        save_filenames='annotated_train_data.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=False,
        fcs_metadata_dicts=None,
    )

    # Test data individual samples
    for fn in test_data_fns:
        print(f'annotated_{fn}')
        gp_som.inference(
            data_file_path=data_dir,
            data_file_names=[fn, ],
            gate=True,
            dim_red_methods=dim_red_methods,
            dim_red_method_kwargs=dim_red_method_kwargs,
            save_sample_wise=False,
            save_path=output_dir_test_samples,
            save_filenames=f'annotated_{fn}',  # Todo: fix input format (expects list if samplewise== True)
            val_range=(0.0, 2 ** 20),
            keep_unscaled=False,
            fcs_metadata_dicts=None,
        )

    # Test data samples concatenated
    gp_som.inference(
        data_file_path=data_dir,
        data_file_names=test_data_fns,
        gate=True,
        dim_red_methods=dim_red_methods,
        dim_red_method_kwargs=dim_red_method_kwargs,
        save_sample_wise=False,
        save_path=output_dir,
        save_filenames='annotated_test_data.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=False,
        fcs_metadata_dicts=None,
    )

    # ### Inference with the FCNN pipeline
    gp_fcnn = GatingPipeline.load(filename='trained_pipeline_fcnn.pkl', filepath=save_path_fcnn)

    # Train data
    gp_fcnn.inference(
        data_file_path=output_dir,
        data_file_names=['annotated_train_data.fcs', ],
        gate=True,
        dim_red_methods=None,
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=output_dir,
        save_filenames='annotated_train_data.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=False,
        fcs_metadata_dicts=None,
    )

    # Test data individual samples
    for fn in test_data_fns:
        gp_fcnn.inference(
            data_file_path=output_dir_test_samples,
            data_file_names=[f'annotated_{fn}', ],
            gate=True,
            dim_red_methods=None,
            dim_red_method_kwargs=None,
            save_sample_wise=False,
            save_path=output_dir_test_samples,
            save_filenames=f'annotated_{fn}',
            val_range=(0.0, 2 ** 20),
            keep_unscaled=False,
            fcs_metadata_dicts=None,
        )

    # Test data samples concatenated
    gp_fcnn.inference(
        data_file_path=output_dir,
        data_file_names=['annotated_test_data.fcs', ],
        gate=True,
        dim_red_methods=None,
        dim_red_method_kwargs=None,
        save_sample_wise=False,
        save_path=output_dir,
        save_filenames='annotated_test_data.fcs',
        val_range=(0.0, 2 ** 20),
        keep_unscaled=False,
        fcs_metadata_dicts=None,
    )

    del gp_som, gp_fcnn

    # ###### Output validation ###### #
    annotated_test_data = readfcs.read(os.path.join(output_dir, 'annotated_test_data.fcs'))
    print("# ### Annotated test data:\n", annotated_test_data)
    df = annotated_test_data.to_df()
    print("# Channels:\n", df.columns)

    for drm in dim_red_methods:
        fig, ax = plt.subplots(dpi=300)
        scatterplot(data=df, x=f'{drm}_1', y=f'{drm}_2', s=1, hue='sample_id', palette='deep', ax=ax)
        plt.legend(title='Sample ID', markerscale=4)
        plt.savefig(os.path.join(output_dir, f'sample_id_dimred_{drm}.png'), dpi=300)
        plt.close('all')
        for gm in ['som', 'fcnn']:
            fig, ax = plt.subplots(dpi=300)
            scatterplot(data=df, x=f'{drm}_1', y=f'{drm}_2', s=1, hue=f'prediction_{gm}', palette='deep', ax=ax)
            plt.legend(title='Pred', markerscale=4)
            plt.savefig(os.path.join(output_dir, f'gating_{gm}_dimred_{drm}.png'), dpi=300)
            plt.close('all')


def main_pipeline_output_downsampling():

    import os
    import numpy as np

    from flagx.io import FlowDataManager, export_to_fcs
    from validation.utils.val_utils import get_downsampling_bool

    np.random.seed(42)

    # ### Set parameters
    datasets = ['Imstat', 'LT1', 'LT2']
    data_files = ['annotated_train_data.fcs', 'annotated_test_data.fcs']
    label_key = 'population'
    sample_id_key = 'sample_id'

    target_num_events = 100000

    double_stratified = False
    # Stratify w.r.t. num events per sample and cell types, if False fixed  num events per sample

    for dataset in datasets:

        results_path = os.path.join(os.getcwd(), 'results/pipeline_workflow', dataset, 'output')

        for data_file in data_files:

            # Instantiate a datamanager
            fdm = FlowDataManager(
                data_file_names=[data_file, ],
                data_file_type=None,
                data_file_path=results_path,
                save_path=results_path,
                verbosity=2,
            )

            # Load data file to anndata
            fdm.load_data_files_to_anndata()

            adata = fdm.anndata_list_[0]

            # Extract the labels
            col_index = adata.var_names.get_loc(label_key)
            labels = adata.X[:, col_index]

            # Extract the sample ids
            col_index = adata.var_names.get_loc(sample_id_key)
            sample_ids = adata.X[:, col_index]

            if double_stratified:  # ### Double stratified downsampling

                # Get the downsampling bool where stratification w.r.t. num events per sample is used
                ds_bool_sample_based = get_downsampling_bool(
                    y=sample_ids,
                    target_num_events=target_num_events,
                    stratified=True
                )

                # Get the event count per sample
                # (used as target num events for population size-based, sample-wise stratified downsampling)
                sample_ids_downsampled = sample_ids[ds_bool_sample_based]
                sample_ids_unique, counts = np.unique(sample_ids_downsampled, return_counts=True)

                # Downsample per sample, stratify w.r.t. population sizes
                ds_bool = np.zeros_like(sample_ids).astype(bool)
                for sample_id, count in zip(sample_ids_unique, counts):

                    sample_id_bool = (sample_ids == sample_id)

                    labels_current_sample = labels[sample_id_bool]

                    ds_bool_current_sample = get_downsampling_bool(
                        y=labels_current_sample,
                        target_num_events=count,
                        stratified=True
                    )

                    ds_bool[sample_id_bool] = ds_bool_current_sample
            else:  # ### Same num events per sample, stratify w.r.t. population sizes

                # Define num events per sample such that target_num_events is reached
                unique_sample_ids = np.unique(sample_ids)
                num_samples = unique_sample_ids.shape[0]
                base = target_num_events // num_samples
                remainder = target_num_events % num_samples
                events_per_sample = np.full(num_samples, base, dtype=int)
                events_per_sample[:remainder] += 1

                # Downsample per sample, stratify w.r.t. population sizes
                ds_bool = np.zeros_like(sample_ids).astype(bool)
                for sample_id, num_events in zip(unique_sample_ids, events_per_sample):
                    sample_bool = (sample_ids == sample_id)
                    labels_current_sample = labels[sample_bool]

                    ds_bool_current_sample = get_downsampling_bool(
                        y=labels_current_sample,
                        target_num_events=num_events,
                        stratified=True
                    )
                    ds_bool[sample_bool] = ds_bool_current_sample

            # Apply downsampling
            adata_downsampled = adata[ds_bool, :].copy()

            print(f'# ### Num events before: {adata.n_obs}, after: {adata_downsampled.n_obs}')
            print(adata_downsampled)

            export_to_fcs(
                data_list=[adata_downsampled, ],
                save_path=results_path,
                save_filenames=data_file[:-4] + f'_downsampled_{target_num_events}_events.fcs',
            )


def main_plot_minority_count():

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from validation.utils.val_utils import get_downsampling_bool, _get_expanding_iterator_list
    from validation.plt import annotate_mosaic

    # ### Set flags and important variables here #######################################################################
    datasets = ['Flowcyt', 'LT1', 'LT2', 'Imstat', 'LT1 b', 'LT2 b']

    plot_dir = os.path.join(os.getcwd(), 'results/plots')

    # Create dir to save plots into
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    dataset_to_datasetdir = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    dataset_to_n_samples = {
        'Flowcyt': list(range(1,6)) + list(range(10, 22, 5)) + [22, ],
        'Imstat': list(range(1,21)) + list(range(25, 76, 5)),
        'LT1': list(range(1,21)) + list(range(25, 71, 5)) + [73, ],
        'LT2': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
        'LT1 b': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
        'LT2 b': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
    }

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
        'Imstat': lm_imstat, 'LT1': lm_lt1, 'LT1 b': lm_lt1b, 'LT2': lm_lt2, 'LT2 b': lm_lt2b, 'Flowcyt': lm_flowcyt
    }

    dataset_to_minority_class = {
        'Flowcyt': 3,
        'Imstat': 4,
        'LT1': 9,
        'LT2': 9,
        'LT1 b': 1,
        'LT2 b': 1,
    }

    n_events = [100, 1000, 5000, 10000, 20000, 50000, 'all']

    n_events_plot = [100, 1000, 5000, 10000, 20000, 50000, 'all']

    generate_plot_dfs = False

    ####################################################################################################################

    if generate_plot_dfs:
        res_dfs = []
        res_dfs_wide = []
        for dataset in datasets:

            # Set random seed
            np.random.seed(42)

            # Set data path
            trafo = 'log10_channelwisecutoff' if dataset != 'Flowcyt' else 'log10_cutoff100'
            data_p = os.path.join(
                os.getcwd(), 'data/np_files', dataset_to_datasetdir[dataset], trafo, 'sample_wise_train'
            )

            # Get number of samples for which to compute population size information
            n_samples = dataset_to_n_samples[dataset]

            # Generate iter list (same as for n samples experiment)
            iter_list = _get_expanding_iterator_list(n=len(n_events), m=len(n_samples))

            n_samples_list = []
            n_events_list = []
            minority_count_list = []
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

                # Concatenate and shuffle rows
                y_train = np.concatenate(y_trains, axis=0)

                # Shuffle row-wise (just for consistency with n samples experiment)
                shuffle_permutation = np.random.permutation(y_train.shape[0])
                y_train = y_train[shuffle_permutation]

                # Get count for minority class
                minority_class_count = (y_train == dataset_to_minority_class[dataset]).sum()

                minority_count_list.append(minority_class_count)
                n_samples_list.append(n_samples[j])
                n_events_list.append(n_events[i])

            res_df = pd.DataFrame(
                {
                    'n_samples': n_samples_list,
                    'n_events': n_events_list,
                    'minority_count': minority_count_list,
                    'dataset': [dataset] * len(minority_count_list)
                }
            )

            res_dfs.append(res_df)

            res_df_wide = res_df.pivot(index='n_events', columns='n_samples', values='minority_count')

            print(f'# ### Minority class count {dataset}:\n{res_df_wide}')

            res_dfs_wide.append(res_df_wide)

            # res_df_wide.to_csv(os.path.join(plot_dir, f'minority_class_{dataset.replace(' ', '')}.csv'))

        res_df_concat = pd.concat(res_dfs, axis=0, ignore_index=True)
        res_df_concat.to_csv(os.path.join(plot_dir, f'minority_class.csv'))

    else:

        # paths = [os.path.join(plot_dir, f'minority_class_{dataset.replace(' ', '')}.csv') for dataset in datasets]
        # res_dfs_wide = [pd.read_csv(path, index_col=0) for path in paths]

        # Load df
        res_df_concat = pd.read_csv(os.path.join(plot_dir, f'minority_class.csv'), index_col=0)
        n_events_col = [int(n) if n != 'all' else n for n in res_df_concat['n_events']]
        res_df_concat['n_events'] = n_events_col

    # Subset the dataframe w.r.t. n_events, n_samples
    keep_bool_n_events = res_df_concat['n_events'].isin(n_events_plot)
    keep_bool_n_samples = (
            (res_df_concat['n_samples'] % 5 == 0) |
            (res_df_concat['n_samples'] == 1) |
            (res_df_concat['n_samples'] >= 70) |
            ((res_df_concat['n_samples'] == 22) & (res_df_concat['dataset'] == 'Flowcyt'))
    )
    keep_bool = np.logical_and(keep_bool_n_events, keep_bool_n_samples)
    res_df_concat = res_df_concat[keep_bool]

    # ### Load the performance df
    method_names = ['FCNN', 'SOM-Classifier']

    dataset_to_max_n = {'Imstat': 75, 'LT1': 73, 'LT2': 73, 'LT1 b': 73, 'LT2 b': 73, 'Flowcyt': 22}

    # Directory mappings
    method_to_dir = {'FCNN': 'softmax', 'SOM-Classifier': 'som'}

    all_records = []

    for dataset in datasets:

        max_n = dataset_to_max_n[dataset]
        ds_dir = dataset_to_datasetdir[dataset]
        data_trafo = 'log10_channelwisecutoff' if dataset != 'Flowcyt' else 'log10_cutoff100'

        for method in method_names:
            for n in n_events:
                for i in range(1, max_n + 1):

                    method_dir = method_to_dir[method]

                    if i == max_n and n == 'all':  # Load previously computed scores for (all samples, all events)
                        file_path = os.path.join(
                            './results/pred_eval',
                            method_dir + '_classifier',
                            ds_dir,
                            data_trafo,
                            'res_df_sw_avg_f1.csv'
                        )
                    else:
                        file_path = os.path.join(
                            './results/n_samples_n_events',
                            method_dir,
                            ds_dir,
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

    # Create DataFrame
    res_df_performance = pd.DataFrame(all_records)
    res_df_performance['n_samples'] = res_df_performance['n_samples'].astype(int)

    # Subset the dataframe w.r.t. n_events, n_samples
    keep_bool_n_events_perf = res_df_performance['n_events'].isin(n_events_plot)
    keep_bool_n_samples_perf = (
            (res_df_performance['n_samples'] % 5 == 0) |
            (res_df_performance['n_samples'] == 1) |
            (res_df_performance['n_samples'] >= 70) |
            ((res_df_performance['n_samples'] == 22) & (res_df_performance['dataset'] == 'Flowcyt'))
    )
    keep_bool_perf = np.logical_and(keep_bool_n_events_perf, keep_bool_n_samples_perf)
    res_df_performance = res_df_performance[keep_bool_perf]

    res_df_performance = (
        res_df_performance
        .groupby(['dataset', 'method', 'n_events', 'n_samples'], as_index=False)
        .agg({'score': 'mean'})
    )

    # Merge with minority counts dataframe
    res_df_performance_som = res_df_performance[res_df_performance['method'] == 'SOM-Classifier']
    res_df_performance_fcnn = res_df_performance[res_df_performance['method'] == 'FCNN']

    res_df_performance_som_joint = pd.merge(
        res_df_concat, res_df_performance_som, on=['n_events', 'n_samples', 'dataset']
    )
    res_df_performance_fcnn_joint = pd.merge(
        res_df_concat, res_df_performance_fcnn, on=['n_events', 'n_samples', 'dataset']
    )

    res_df = pd.concat([res_df_performance_som_joint, res_df_performance_fcnn_joint], axis=0).reset_index(drop=True)


    # ### Plot 1: Per dataset lineplot: x=n_samples, y=n_minority_events
    fig = plt.figure(figsize=(8, 8), constrained_layout=True, dpi=300)
    axd = fig.subplot_mosaic(
        """
        ABC
        DEF
        """
    )

    for dataset, key in zip(datasets, list('ABCDEF')):

        ax = axd[key]

        # Subset the dataframe
        keep_bool_dataset = (res_df_concat['dataset'] == dataset)
        res_df_concat_sub = res_df_concat[keep_bool_dataset]

        sns.lineplot(
            data=res_df_concat_sub,
            x='n_samples',
            y='minority_count',
            hue='n_events',
            errorbar=None,
            marker='o',
            markersize=3,
            palette='magma',
            ax=ax,
        )

        # Set title
        label_mapping = label_mappings[dataset]
        minority_class = dataset_to_minority_class[dataset]
        minority_class_label = label_mapping[str(minority_class)]
        ax.set_title(f'{dataset} | {minority_class_label} Count')

    fig.savefig(os.path.join(plot_dir, 'minority_count_per_dataset.png'), dpi=fig.dpi)
    plt.close(fig)


    # ### Plot 2: Per dataset lineplot: x=n_minority_events, y=macro_f1
    res_df['minority_count_plus_one'] = res_df['minority_count'] + 1

    print(res_df)

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
    layout_str = '''
                ABC
                DEF
                GHI
                JKL
            '''
    axd = fig.subplot_mosaic(layout_str)

    plot_labels = [c for c in layout_str if c.isalpha()]

    count = 0
    for method in method_names:
        for dataset in datasets:

            # Subset df to dataset and method

            plot_df = res_df[(res_df['method'] == method) & (res_df['dataset'] == dataset)]

            ax = axd[plot_labels[count]]

            sns.lineplot(
                plot_df,
                x='minority_count_plus_one',
                y='score',
                hue='n_events',
                errorbar=None,  # 'sd',
                # err_style='bars',
                marker='o',
                markersize=3,
                palette='magma',
                legend=True,
                ax=ax,
            )

            ax.set_title(f'{dataset} | {method}')
            ax.set_xlabel('Minority Count + 1')
            ax.set_ylabel('Macro F1 Score')

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                fontsize=6,
                title=None,
                loc='lower right',
                handlelength=1.5,
            )

            ax.set_xscale('log', base=10)

            from matplotlib.ticker import LogLocator
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=range(1, 10), numticks=100))

            count += 1

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'minority_count_vs_f1_score.png'),
        dpi=fig.dpi
    )
    plt.close('all')

    # ### Plot 3: Per dataset lineplot: x=n_minority_events, y=macro_f1

    log10 = True
    interm_ticks = False

    tick_fs = 8
    ax_fs = 8

    fig = plt.figure(figsize=(8, 9), constrained_layout=True, dpi=300)
    layout_str = '''
                    ABC
                    DEF
                    GHI
                    JKL
                '''
    axd = fig.subplot_mosaic(layout_str)

    plot_labels = [c for c in layout_str if c.isalpha()]

    count = 0
    for method in method_names:
        for dataset in datasets:
            # Subset df to dataset and method

            plot_df = res_df[(res_df['method'] == method) & (res_df['dataset'] == dataset)].copy()

            if log10:
                plot_df['minority_count_raw'] = plot_df['minority_count'].copy()
                plot_df['minority_count'] = np.log10(plot_df['minority_count'] + 1)

            mc = plot_df['minority_count'].to_numpy()
            plot_df['minority_count_scaled'] = - (mc - mc.min()) / (mc.max() - mc.min())
            sc = plot_df['score'].to_numpy()
            plot_df['score_scaled'] = (sc - sc.min()) / (sc.max() - sc.min())

            ax = axd[plot_labels[count]]

            sns.lineplot(
                plot_df,
                x='n_samples',
                y='minority_count_scaled',
                hue='n_events',
                errorbar=None,  # 'sd',
                # err_style='bars',
                marker='o',
                markersize=3,
                palette='magma',
                legend=True,
                ax=ax,
            )

            sns.lineplot(
                plot_df,
                x='n_samples',
                y='score_scaled',
                hue='n_events',
                errorbar=None,  # 'sd',
                # err_style='bars',
                marker='o',
                markersize=3,
                palette='magma',
                legend=True,
                ax=ax,
            )

            ax.set_title(f'{dataset} | {method}')
            ax.set_xlabel('No. of Training Samples', fontsize=ax_fs)
            ax.set_ylabel('Minority Count | Macro F1', fontsize=ax_fs)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicates while preserving order
            ax.legend(
                by_label.values(),
                by_label.keys(),
                fontsize=6,
                title=None,
                loc='center right',
                handlelength=1.5,  # shrink marker size a bit
            )

            # Set min and max number of samples as x ticks
            x_min, x_max = plot_df['n_samples'].min(), plot_df['n_samples'].max()
            current_ticks = ax.get_xticks()
            current_ticks = [tick for tick in current_ticks if x_min <= tick <= x_max]
            new_ticks = [x_min] + current_ticks + [x_max]
            ax.set_xticks(new_ticks)
            ax.set_xticklabels([str(int(tick)) for tick in new_ticks], fontsize=tick_fs)

            # Set x ticks
            y_ticks_pos_label = np.round(np.linspace(sc.min(), sc.max(), 4), 2)
            y_ticks_pos_position = ((y_ticks_pos_label - sc.min()) / (sc.max() - sc.min())).tolist()

            if not log10:
                # Linear case (unchanged logic)
                y_ticks_neg_label = np.linspace(mc.min(), mc.max(), 4, dtype=int)
                y_ticks_neg_position = (-(y_ticks_neg_label - mc.min()) / (mc.max() - mc.min())).tolist()

                zero_label = f'{y_ticks_neg_label[0]} | {y_ticks_pos_label[0]}'
                y_ticks_labels = (
                        [str(l) for l in y_ticks_neg_label[1:]] + [zero_label] + [str(l) for l in y_ticks_pos_label[1:]]
                )
                y_ticks_positions = y_ticks_neg_position[1:] + [0.0] + y_ticks_pos_position[1:]

            else:
                # ---------- LOG case for the negative side ----------
                minority_raw = plot_df['minority_count_raw'].to_numpy()
                min_raw = int(np.nanmin(minority_raw))
                max_raw = int(np.nanmax(minority_raw))

                log_ticks_raw = []
                decade_exponents = []

                if max_raw >= 10:
                    # Build candidate ticks
                    max_decade = int(np.floor(np.log10(max_raw)))
                    if interm_ticks:
                        # Intermediate ticks (10,20,...,90, 100,200,...), but we'll filter by min_raw next
                        for d in range(1, max_decade + 1):  # start at 10^1
                            base = 10 ** d
                            for m_ in range(1, 10):
                                v = m_ * base
                                if v <= max_raw:
                                    log_ticks_raw.append(v)
                        decade_exponents = list(range(1, max_decade + 1))
                    else:
                        decade_exponents = list(range(1, max_decade + 1))
                        log_ticks_raw = [10 ** d for d in decade_exponents]

                    # *** KEY: keep only ticks that are within the plotted data range ***
                    # (values below min_raw map above the 0 line after min-max scaling)
                    log_ticks_raw = [v for v in log_ticks_raw if v >= min_raw]

                # Positions in transformed space (log10(x+1)), then min-max scale and flip negative
                log_ticks_vals = np.log10(np.array(log_ticks_raw, dtype=float) + 1.0) if len(
                    log_ticks_raw) else np.array([])

                denom = (mc.max() - mc.min()) if (mc.max() > mc.min()) else 1.0
                y_ticks_neg_position = (-(log_ticks_vals - mc.min()) / denom).tolist() if len(log_ticks_vals) else []

                # Labels: 10^d only for decade ticks that survived filtering
                decade_raw_values = {10 ** d for d in decade_exponents}
                y_ticks_neg_labels = [
                    (r"$10^{%d}$" % int(np.log10(v)) if v in decade_raw_values else "")
                    for v in log_ticks_raw
                ]

                # Combine ticks
                zero_label = f'{min_raw} | {y_ticks_pos_label[0]}'
                y_ticks_labels = y_ticks_neg_labels + [zero_label] + [str(l) for l in y_ticks_pos_label[1:]]
                y_ticks_positions = y_ticks_neg_position + [0.0] + y_ticks_pos_position[1:]

            # Apply to the axis
            ax.set_yticks(y_ticks_positions, labels=y_ticks_labels, fontsize=tick_fs)

            ax.axhline(y=-0.001, color='green', linewidth=1.0)
            ax.axhline(y=0.001, color='blue', linewidth=1.0)

            ax.grid()

            count += 1

    annotate_mosaic(fig=fig, axd=axd, fontsize=16)
    plt.savefig(
        os.path.join(plot_dir, f'minority_count_and_f1_score.png'),
        dpi=fig.dpi
    )
    plt.close('all')


def main_minority_count_figure():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.patheffects as pe

    from matplotlib.ticker import LogLocator

    from validation.utils.val_utils import get_downsampling_bool, _get_expanding_iterator_list
    from validation.plt import annotate_mosaic

    # ### Set flags and important variables here #######################################################################
    datasets = ['Flowcyt', 'Imstat', 'LT1', 'LT2', 'LT1 b', 'LT2 b']
    method_names = ['FCNN', 'SOM-Classifier']

    plot_dir = os.path.join(os.getcwd(), 'results/minority_count')
    os.makedirs(plot_dir, exist_ok=True)

    # Convert dataset names to corresponding dir names
    dataset_to_datasetdir = {
        'Imstat': 'imstat',
        'LT1': 'lymphoma_tube1', 'LT1 b': 'lymphoma_tube1_binary',
        'LT2': 'lymphoma_tube2', 'LT2 b': 'lymphoma_tube2_binary',
        'Flowcyt': 'flowcyt'
    }

    dataset_to_n_samples = {
        'Flowcyt': list(range(1, 6)) + list(range(10, 22, 5)) + [22, ],
        'Imstat': list(range(1, 21)) + list(range(25, 76, 5)),
        'LT1': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
        'LT2': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
        'LT1 b': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
        'LT2 b': list(range(1, 21)) + list(range(25, 71, 5)) + [73, ],
    }
    n_events = [5000, 10000, 20000, 50000, 'all']  # [100, 1000, 5000, 10000, 20000, 50000, 'all']

    plot_num_events = False

    generate_res_df = False

    ####################################################################################################################

    # Generate or load the results dataframe
    if generate_res_df:

        # ### Generate the minority count dataframe
        res_dfs_minority_count = []
        for dataset in datasets:

            # Set random seed
            np.random.seed(42)

            # Set data path
            trafo = 'log10_channelwisecutoff' if dataset != 'Flowcyt' else 'log10_cutoff100'
            data_p = os.path.join(
                os.getcwd(), 'data/np_files', dataset_to_datasetdir[dataset], trafo, 'sample_wise_train'
            )

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

                # Get the median minority class size across samples
                minority_counts = []
                for labels in y_trains:
                    unique, counts = np.unique(labels, return_counts=True)
                    minority_counts.append(counts.min())

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

            res_df_minority_count_dataset_wide.to_csv(
                os.path.join(plot_dir, f'minority_count_{dataset.replace(' ', '')}.csv')
            )

        res_df_minority_count = pd.concat(res_dfs_minority_count, axis=0, ignore_index=True)
        res_df_minority_count.to_csv(os.path.join(plot_dir, f'minority_count.csv'))

        # ### Generate the performance dataframe
        dataset_to_max_n = {'Imstat': 75, 'LT1': 73, 'LT2': 73, 'LT1 b': 73, 'LT2 b': 73, 'Flowcyt': 22}
        method_to_dir = {'FCNN': 'softmax', 'SOM-Classifier': 'som'}
        all_records = []
        for dataset in datasets:

            max_n = dataset_to_max_n[dataset]
            ds_dir = dataset_to_datasetdir[dataset]
            data_trafo = 'log10_channelwisecutoff' if dataset != 'Flowcyt' else 'log10_cutoff100'

            for method in method_names:
                for n in n_events:
                    for i in range(1, max_n + 1):

                        method_dir = method_to_dir[method]

                        if i == max_n and n == 'all':  # Load previously computed scores for (all samples, all events)
                            file_path = os.path.join(
                                './results/pred_eval',
                                method_dir + '_classifier',
                                ds_dir,
                                data_trafo,
                                'res_df_sw_avg_f1.csv'
                            )
                        else:
                            file_path = os.path.join(
                                './results/n_samples_n_events',
                                method_dir,
                                ds_dir,
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

        res_df_performance.to_csv(os.path.join(plot_dir, f'performance.csv'))

        # Merge dataframes
        res_df = pd.merge(res_df_performance, res_df_minority_count, on=['n_samples', 'n_events', 'dataset'])

        res_df.to_csv(os.path.join(plot_dir, 'res_df.csv'))

    else:
        res_df = pd.read_csv(os.path.join(plot_dir, 'res_df.csv'), index_col=0)
        n_events_col = [int(n) if n != 'all' else n for n in res_df['n_events']]
        res_df['n_events'] = n_events_col


    # Load performance scores for all samples
    all_records = []
    for ds in datasets:

        ds_dir = dataset_to_datasetdir[ds]
        data_trafo = 'log10_channelwisecutoff' if ds != 'Flowcyt' else 'log10_cutoff100'

        for method in method_names:

            method_dir = 'som' if method == 'SOM-Classifier' else 'softmax'

            file_path = os.path.join(
                './results/pred_eval',
                method_dir + '_classifier',
                ds_dir,
                data_trafo,
                f'res_df_sw_avg_f1.csv'
            )

            df = pd.read_csv(file_path, index_col=0)

            all_records.append({
                'dataset': ds,
                'method': method,
                'score': df.loc['mean', 'macro']
            })

    # Create DataFrame
    res_df_all_data_performance = pd.DataFrame(all_records)

    # Subset dataframe to values to be plotted
    keep_bool_n_events = res_df['n_events'].isin(n_events)
    keep_bool_n_samples = (
            (res_df['n_samples'] % 5 == 0) |
            (res_df['n_samples'] == 1) |
            (res_df['n_samples'] >= 70) |
            ((res_df['n_samples'] == 22) & (res_df['dataset'] == 'Flowcyt'))
    )
    keep_bool = np.logical_and(keep_bool_n_events, keep_bool_n_samples)
    res_df = res_df[keep_bool].copy()

    print(res_df)

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
            keep_bool_dataset = (res_df['dataset'] == dataset)
            keep_bool_method = (res_df['method'] == method)
            keep_bool = keep_bool_dataset & keep_bool_method
            res_df_plot = res_df[keep_bool].copy()

            ax = axd[subplot_key]

            sns.scatterplot(
                res_df_plot,
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
        cbar.set_ticklabels(['1 sample', 'all\nsamples'])
        cbar.set_label('Number of samples', fontsize=10, labelpad=5)
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
            ['Flowcyt', 22],
            ['Imstat', 75],
            ['LT1, LT2', 73]
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

        fig.savefig(os.path.join(plot_dir, f'minority_count_{mc_mode}.png'), dpi=fig.dpi)







