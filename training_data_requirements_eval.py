"""
Evaluation of the training data requirements (PLAN.md step 8, working_notes/training_data_requirements_eval_plan.md):
score curves over n_samples x n_events, the minimal training configuration per method x dataset (paired
non-inferiority to the full-data baseline of step 7), its variability over trials / folds, the sensitivity to the
criterion, and a cross-fitted selection-bias check. NOT IMPLEMENTED YET (skeleton; the plan is under review).

Env: fx_plot (pandas).

Usage:
    python training_data_requirements_eval.py [--in DIR] [--baseline DIR] [--out DIR]
        --in: step-8 output root (default results/training_data_requirements)
        --baseline: step-7 eval output (default results/benchmark_eval; reads scores/<dataset>.csv)
        --out: output directory (default results/training_data_requirements_eval)

Aggregation (see the working note, section Theory):
    unit = test sample (each tested once over the outer folds); per cell: score per test sample averaged over the
    trials -> pooled over the folds -> paired difference d to the step-7 baseline of the same sample (seed mean) ->
    mean d with a bootstrap CI (resampling test samples; donors for Imstat). The ('all', 'all') cell is not in the
    step-8 grid, so the baseline always comes from step 7.

Criteria (CRITERIA): minimal cell = the smallest cell (ORDER: greedy or joint) whose lower CI bound of mean d is
above -delta; plus the legacy criterion (mean score within the 95 % CI of the baseline) for comparison. A cell is
eligible only if all folds and trials have scores.

Outputs (<out>): atoms.csv, scores/<dataset>.csv, sample_scores.csv, curves.csv, minimal.csv, crossfit.csv,
minimal_configs.json (read by step 9). Columns: see the working note.
Each part is skipped if its outputs exist (OVERWRITE = False); the parts after it then read them.
"""

import argparse
import os

import config

IN_DIR = os.path.join(config.RESULTS, 'training_data_requirements')
BASELINE_DIR = os.path.join(config.RESULTS, 'benchmark_eval')
OUT_DIR = os.path.join(config.RESULTS, 'training_data_requirements_eval')
OVERWRITE = False

# TODO (open decisions, working note): margins, primary criterion, grid order, bootstrap settings
PRIMARY_DELTA = 0.02  # Non-inferiority margin (macro F1; F1 of label 1 for binary datasets), proposed
CRITERIA = [('noninf', 0.01), ('noninf', 0.02), ('noninf', 0.05), ('legacy_ci', None)]
ORDERS = ['greedy', 'joint']  # greedy: fewest samples, then fewest events; joint: fewest total training events
N_BOOT = 10000
CI_LEVEL = 0.95
BOOT_UNIT = {'imstat': 'donor'}  # Resampling unit per dataset ('sample' unless listed)


# ---------------------------------------------------------------------------------------------------------------
# Part 1: atom table (status, resources, n_train_events)
# ---------------------------------------------------------------------------------------------------------------

def atom_table(manifest, in_dir):
    """One row per manifest atom: identity, status (incl. skipped / missing), phase, fit time / memory."""
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------
# Part 2: score tables (per atom x test sample, per-class scores as in benchmark_eval.py)
# ---------------------------------------------------------------------------------------------------------------

def score_table(dataset, manifest, atoms, in_dir):
    """One row per atom x test sample of the dataset: counts -> prec / rec / f1 per class."""
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------
# Part 3: per-sample trial means, paired with the step-7 baseline
# ---------------------------------------------------------------------------------------------------------------

def baseline_scores(dataset, baseline_dir):
    """Step-7 score per method x test sample (seed mean) from <baseline_dir>/scores/<dataset>.csv."""
    raise NotImplementedError


def sample_scores(dataset, scores, baseline):
    """Per method x cell x test sample: trial mean (NaN if any trial is missing), baseline, d = cell - baseline."""
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------
# Part 4: curves (pooled over folds) with bootstrap CIs; per-fold / per-trial means
# ---------------------------------------------------------------------------------------------------------------

def bootstrap_ci(d, groups, seed):
    """Percentile CI of mean(d), resampling groups (test samples or donors) with replacement."""
    raise NotImplementedError


def curve_table(dataset, samples):
    """Per method x cell: mean score, mean d, CI of mean d, n; per-fold and per-trial means."""
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------
# Part 5: minimal configurations (pooled, per trial, per fold; criteria x orders)
# ---------------------------------------------------------------------------------------------------------------

def passes(row, criterion, delta):
    """True if a cell meets the criterion (noninf: CI lower bound > -delta; legacy_ci: inside the baseline CI)."""
    raise NotImplementedError


def minimal_cell(curve, criterion, delta, order):
    """The smallest eligible cell that passes (greedy or joint order), or None."""
    raise NotImplementedError


def minimal_table(curves):
    """Per method x dataset x criterion x order: pooled minimal cell, plus the per-trial and per-fold minima."""
    raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------
# Part 6: cross-fitted selection (select on folds != k, evaluate on fold k)
# ---------------------------------------------------------------------------------------------------------------

def crossfit_table(samples):
    """Per method x dataset x criterion x fold: selected cell, its score on fold k, in-sample comparison."""
    raise NotImplementedError


def main(in_dir, baseline_dir, out_dir):
    raise NotImplementedError('skeleton: see working_notes/training_data_requirements_eval_plan.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--in', dest='in_dir', default=IN_DIR, help='step-8 output root (default: %(default)s)')
    parser.add_argument('--baseline', default=BASELINE_DIR, help='step-7 eval output (default: %(default)s)')
    parser.add_argument('--out', default=OUT_DIR, help='output directory (default: %(default)s)')
    args = parser.parse_args()
    main(args.in_dir, args.baseline, args.out)
