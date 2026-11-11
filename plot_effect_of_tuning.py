"""
Effect of hyperparameter tuning (PLAN.md step 6): box plots of the paired differences best - default config, one box
per method, before they are averaged into the effect-of-tuning tables of tune_hyperparameters_eval.py (Part 4).

    effect_outer_box.pdf    effect at the outer level (Imstat benchmark: per-split best): delta per inner fold,
                            all 5 outer splits x 3 inner folds pooled (15 points per method)
    effect_overall_box.pdf  effect at the overall level (consensus best, other datasets): delta per outer split
                            (5 points per method)

Partial results as in the tables: units where best or default failed (NaN) are left out; the tick label then gives
the number of valid units. Each point is one unit; the box shows median and quartiles, whiskers 1.5 IQR.

Env: fx_plot (envs/plot.yml)

Inputs: results/tune_hyperparameters_eval/effect_of_tuning/effect_units_{outer,overall}.csv
Outputs: figures/tune_hyperparameters_eval/effect_{outer,overall}_box.pdf
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS, FIGURES


OVERWRITE = False  # If False, existing figures are skipped

RES_DIR = os.path.join(RESULTS, 'tune_hyperparameters_eval', 'effect_of_tuning')
FIG_DIR = os.path.join(FIGURES, 'tune_hyperparameters_eval')

# x-axis order: FLAG-X, published methods, tabular baselines
METHOD_NAMES = {
    'som': 'FLAG-X SOM', 'mlp': 'FLAG-X MLP',
    'dgcytof': 'DGCyTOF', 'deepcytof': 'DeepCyTOF', 'gatemeclass': 'GateMeClass', 'lda_abdelaal': 'LDA-Abdelaal',
    'gmm_mocking': 'GMM-Mocking',
    'rf': 'RF', 'hgb': 'HGB', 'knn': 'kNN', 'logreg': 'LogReg', 'lda_sklearn': 'LDA-sklearn',
}

BOX_COLOR = '#4C72B0'
POINT_COLOR = '#333333'
JITTER = 0.12  # Half-width of the horizontal point jitter (deterministic, seed 0)


def plot_box(units, title, path):
    """
    One box per method of the paired deltas (NaN units left out), points overlaid, zero line.

    Args:
        units (pd.DataFrame): effect_units_*.csv: one row per method x unit with column delta.
        title (str): Axes title.
        path (str): Output PDF.
    """
    methods = [m for m in METHOD_NAMES if m in set(units['method'])]
    data, labels = [], []
    for m in methods:
        d = units.loc[units['method'] == m, 'delta']
        data.append(d.dropna().values)
        n_valid, n = d.notna().sum(), len(d)
        labels.append(METHOD_NAMES[m] + ('' if n_valid == n else '\n({}/{})'.format(n_valid, n)))

    fig, ax = plt.subplots(figsize=(9, 3.6), layout='constrained')
    ax.axhline(0, color='0.5', linewidth=1, linestyle='--', zorder=0)
    ax.boxplot(data, positions=range(len(data)), widths=0.55, showfliers=False, patch_artist=True,
               boxprops={'facecolor': BOX_COLOR, 'alpha': 0.35, 'edgecolor': BOX_COLOR},
               medianprops={'color': BOX_COLOR, 'linewidth': 2},
               whiskerprops={'color': BOX_COLOR}, capprops={'color': BOX_COLOR})
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + rng.uniform(-JITTER, JITTER, len(d)), d, s=12, color=POINT_COLOR, alpha=0.8, zorder=3,
                   linewidths=0)
    ax.set_xticks(range(len(data)), labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
    ax.set_ylabel('Δ macro F1 (best − default)')
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', color='0.9', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_effect_outer():
    path = os.path.join(FIG_DIR, 'effect_outer_box.pdf')
    if os.path.exists(path) and not OVERWRITE:
        return
    units = pd.read_csv(os.path.join(RES_DIR, 'effect_units_outer.csv'))
    plot_box(units, 'Effect of tuning, outer level: per-split best vs. default, Δ per inner fold (5 splits × 3 folds)',
             path)


def plot_effect_overall():
    path = os.path.join(FIG_DIR, 'effect_overall_box.pdf')
    if os.path.exists(path) and not OVERWRITE:
        return
    units = pd.read_csv(os.path.join(RES_DIR, 'effect_units_overall.csv'))
    plot_box(units, 'Effect of tuning, overall level: consensus best vs. default, Δ per outer split (5 splits)', path)


if __name__ == '__main__':
    plot_effect_outer()
    plot_effect_overall()
