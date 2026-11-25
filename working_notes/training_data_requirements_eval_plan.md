# Training data requirements: evaluation plan (PLAN.md step 8)

Status: decided 2026-11-25, implementation in progress. Script: `training_data_requirements_eval.py`.
Experiment design: `working_notes/training_data_requirements_plan.md`. Reviewer comment: R4.5
(`../flagx_revision/REVISION_PLAN.md`), also R2.19 / R2.21 and the test set reuse note (REVISION_PLAN.md:687).

## Goal

From the step-8 atoms (`results/training_data_requirements/`) and the full-data baseline of step 7
(`results/benchmark_eval/scores/`): the score curves over n_samples × n_events, the minimal training configuration
per method × dataset under a fixed criterion, its variability over trials / folds, a sensitivity analysis over
criteria, and a selection-bias check (cross-fitting). The minimal configurations are read by step 9.

## Decisions (2026-11-25)

- **Margin δ** (2026-11-18): primary δ = `config.DELTA` = 0.02 macro F1 (F1 of label 1 for binary datasets), a stated
  convention; no data-driven δ (the seed variability of step 7 rests on 3 seeds only). Testing "Δ = 0" was rejected:
  it accepts noisy cells early and rejects precise ones (the margin is then the CI width, i.e. depends on the test
  size).
- **Criteria computed:** non-inferiority over a δ sweep of 21 log-spaced values 0.001–0.1 (R10 series: 0.001, 0.00125,
  0.0016, 0.002, …, 0.01, …, 0.02, …, 0.05, …, 0.1), plus the legacy criterion (mean score ≥ lower bound of the
  baseline's bootstrap CI) for comparison only. The δ sensitivity figure is drawn from `minimal.csv` (prototype
  `scratchpad/plot_delta_sensitivity.py`).
- **Order:** greedy (fewest samples, then fewest events per sample) is primary: it keeps the number of annotated
  samples (the clinical labelling cost) small, as in legacy Table 1. Joint (fewest total training events) is reported
  as secondary.
- **Stability rule** (a cell passes only if every larger n_samples at the same n_events also passes; ineligible larger
  cells are ignored): minimal cells are computed **with and without** it; the primary is chosen after seeing the
  results. Costs nothing extra (same pass table).
- **Bootstrap:** N_BOOT = 10000, percentile 95 % two-sided CI (one-sided α = 0.025). Resampling unit: donors for Imstat
  (125 samples / 26 donors, up to 5 per donor), test samples elsewhere (one sample per patient). The CI is conditional
  on the trial-averaged models; the trial variance is reported as the spread of the per-trial minima (no two-level
  bootstrap).
- **Cut-off cells** (timeout / oom / skipped / out of budget / failed / missing): never eligible; a cell counts only if
  all its lanes (folds × trials) are in budget and `ok`.
- **Step 9 config** (pooled or cross-fitted per fold): still open, decided after comparing them;
  `minimal_configs.json` holds both.
- **Compute budget:** the cutoff rule is applied as stated (below); no check of its assumption and no automated
  outlier detection. Outlier cutoffs are checked and resubmitted manually.

## Compute budget (post hoc, decided 2026-11-24)

Before any score analysis, the eval applies the run's cutoff rule again over all atoms, since lanes run in parallel and
atoms that finished `ok` before a smaller atom of another lane timed out are not rewritten:

- Cutoff per method × dataset: x = the smallest `n_train_events` of any `timeout` / `oom` atom (any fold / trial).
- Every atom with `n_train_events` ≥ x is outside the budget, whatever its status (`in_budget`, `excluded_by` in
  `atoms.csv`). A cell then counts only if all folds and trials are in budget and have scores.
- No slowdown factors: the CPU arrays run on Gold 6326 only; the GPU array is mixed, but each cell runs in 15 lanes, so
  its cutoff already reflects the slower GPU (`working_notes/training_data_requirements_plan.md`, "Hardware").
- Predict-phase timeouts (BLAST110 test folds) depend on the test size, not the training size; they still count
  (conservative), and `phase` shows them.

## Method

**Cells across lanes.** A cell is a grid point (n_samples, n_events) of the dataset's grid. The manifest drops
duplicates per lane (n_samples ≥ the split's training samples = `all`; an event cap ≥ the largest selected sample =
`all` events), so a cell can be missing in some lanes although it exists in others. Per lane, a cell is therefore
resolved to the atom with the same training data: n_samples ≥ the split's training samples → `all`; a missing
(n_samples, n_events) → (n_samples, `all`); (`all`, `all`) → the step-7 model (score = baseline, d = 0, its training
size = all events of the split). `curves.csv` counts these aliases (`n_alias`, `n_baseline_alias`). The data are
identical; only the join shuffle (and for the step-7 alias the seeds and cores) differ.

**Scores.** Per test sample: macro F1 (NaN-mean of the per-class F1, as in step 7) or the F1 of label 1 for binary
datasets. Per cell × test sample: the trial mean over the trials with a defined score (NaN if a trial did not predict
the sample); baseline = the step-7 seed mean (over the seeds that predicted it, as in `plot_benchmark.py`);
d = trial mean − baseline. Samples with an undefined d (score or baseline NaN) are left out of the means (`n_test` vs
`n_test_total`).

**Scopes.** A scope is the part of the data a statistic is computed on. `curves.csv` and `minimal.csv` are single
long tables with a `scope` column (14 rows per cell with 3 trials and 5 folds; filter `scope == 'pooled'` for the main
result):

| scope | data | used for |
|---|---|---|
| `pooled` | all folds, trial means | main result: the curves and the minimal cell; step 9 |
| `trial_<t>` | all folds, trial t only | minimal cell per trial: variability over sample orders (R4.5) |
| `fold_<k>` | fold k only, trial means | minimal cell per fold (spread); evaluating the cross-fitted cell on fold k |
| `excl_<k>` | all folds except k, trial means | selecting the cross-fitted cell for fold k |

A cell is eligible in a scope if all its lanes in that scope are in budget and `ok`. Example: one failed atom in fold 3
/ trial 2 makes the cell ineligible in `pooled`, `trial_2`, `fold_3` and every `excl_<k>` with k ≠ 3, but not in the
other scopes.

**Bootstrap.** Per dataset and scope one fixed matrix of resampled units (donors / samples, seed
`derive_seed('tdr_eval', dataset, scope)`), shared by all methods and cells (common random numbers: differences
between cells are not resampling noise). The bootstrap mean of a cell = sum of d over the drawn units / number of drawn
samples with a defined d (ratio estimator, so donors with several samples weigh by their sample count). Percentile CI
2.5 / 97.5 of mean d; the same resamples give the baseline CI for the legacy criterion.

**Minimal cell** per method × dataset × criterion × order × stable × scope: among the eligible cells that pass,
greedy = smallest (n_samples, n_events) in lexicographic order (`all` = largest); joint = smallest mean total training
events over the scope's lanes (ties: greedy order). Passing: `noninf`: ci_low > −δ; `legacy_ci`: mean score ≥ the
baseline's ci_low. `stable`: in addition every eligible cell with larger n_samples and the same n_events passes.

**Cross-fitting** per fold k: minimal cell selected in scope `excl_k`, evaluated in scope `fold_k` (its mean d and
score); compared with the pooled minimal cell evaluated in `fold_k` (in-sample). The row `fold = all` pools the five
evaluations (n-weighted). Caveat (statistics.md): only approximately nested, the models of the other folds saw fold k's
patients in training.

## Outputs (`results/training_data_requirements_eval/`)

- `atoms.csv`: one row per manifest atom: identity, `n_train_events` (manifest), status (ok / failed / timeout / oom /
  skipped / started / missing; `skipped` also for atoms without a JSON that a cutoff covers), phase, error (last
  line), skipped_by, n_test_done, fit time / peak memory, prediction time (sum), host, cpu_model, gpu, `in_budget`,
  `excluded_by` (the smallest timeout / oom atom of the method × dataset if out of budget).
- `train_class_counts/<dataset>.csv` (2026-11-25, for the legacy minority count figure): one row per dataset × split ×
  trial × cell × training sample, for the cells that at least one method completed (`ok`): `n_<l>` per class (events
  of class l from this sample in the training set, after downsampling; from the atom JSON). No method column: the
  data seeds do not contain the method, so all methods train on the same data; the eval stops if two methods that
  completed a cell report different counts. Per method: join with its `ok` atoms in `atoms.csv`. Everything else
  (per-sample minority counts and their total / mean / median / min, class counts of the whole training set) is
  derived from this table, e.g. in the figure script. At most ~320k rows (one method's atoms × training samples).
- `scores/<dataset>.csv`: one row per predicted atom × test sample (only predicted samples, to keep the size down):
  identity, n_cells, abstention, prediction time, per class `n_` / `npred_` / `prec_` / `rec_` / `f1_` (as
  `benchmark_eval.py`).
- `sample_scores.csv`: one row per method × dataset × cell × test sample: fold, the per-trial scores, the trial mean,
  the baseline, d.
- `curves.csv` (long): one row per method × dataset × cell × scope: n_lanes, n_alias, n_baseline_alias, n_ok, eligible,
  n_train_events (mean / min / max over the lanes), n_test / n_test_total, mean score, mean baseline, mean d, sd d,
  ci_low / ci_high of mean d, base_ci_low / base_ci_high.
- `minimal.csv`: one row per method × dataset × criterion (`noninf` with δ, `legacy_ci`) × order × stable × scope:
  found, the minimal cell (n_samples, n_events, mean n_train_events), its mean d, CI, mean score.
- `crossfit.csv`: one row per method × dataset × criterion × order × stable × fold (0..k-1 and `all`): the cross-fitted
  cell and its fold-k mean d / score / n, the pooled cell and its fold-k mean d / score.
- `minimal_configs.json`: per method × dataset, primary criterion (noninf δ = 0.02, greedy), with and without the
  stability rule: the pooled cell and the cross-fitted cell per fold. Read by step 9.

## Back-check on the legacy results (2026-11-18)

Data: `legacy/results/num_samples_num_events/{fcnn,som}/<dataset>/.../random/detailed_res/nevents_<e>_nsamples_<s>/res_df_sw_avg_f1.csv`
(macro F1 per test sample; one split, one sample order). Baseline: `legacy/results/gating_performance/` (full data, as
in `legacy/main_plotting.py::fig5s_num_samples_num_events`). Same test samples in both (50 / 40 / 12 rows); paired by
row. Legacy rule: greedy (samples first, then events), n_samples ≥ 2. On the legacy figure grid (n_samples multiples
of 5 + full, events 5k–all), `legacy_ci` (mean ≥ lower bound of the baseline bootstrap CI) reproduces Table 1.

| grid | dataset | method | base | CI hw | legacy_ci | Δ ≥ −0.01 | Δ ≥ −0.02 | Δ ≥ −0.05 | noninf 0.01 | noninf 0.02 | noninf 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | Imstat | MLP | 0.956 | 0.011 | 10×20k | 10×20k | 5×10k | 5×5k | 10×50k | 10×20k | 5×5k |
| legacy | Imstat | SOM | 0.945 | 0.013 | 5×5k | 5×5k | 5×5k | 5×5k | 5×10k | 5×5k | 5×5k |
| legacy | LT1 | MLP | 0.857 | 0.020 | 15×all | 15×all | 15×all | 15×50k | 15×all | 15×all | 15×all |
| legacy | LT1 | SOM | 0.702 | 0.012 | 5×5k | 5×20k | 5×5k | 5×5k | 10×all | 5×20k | 5×5k |
| legacy | LT2 | MLP | 0.848 | 0.034 | 15×all | 30×all | 25×50k | 10×all | 58×all | 30×all | 15×all |
| legacy | LT2 | SOM | 0.632 | 0.018 | 5×5k | 5×5k | 5×5k | 5×5k | 5×10k | 5×5k | 5×5k |
| legacy | LT1b | MLP | 0.971 | 0.020 | 5×5k | 5×5k | 5×5k | 5×5k | 5×10k | 5×5k | 5×5k |
| legacy | LT1b | SOM | 0.957 | 0.024 | 5×5k | 5×5k | 5×5k | 5×5k | 5×50k | 5×5k | 5×5k |
| legacy | LT2b | MLP | 0.956 | 0.026 | 5×5k | 5×5k | 5×5k | 5×5k | 5×50k | 5×5k | 5×5k |
| legacy | LT2b | SOM | 0.944 | 0.028 | 5×5k | 5×5k | 5×5k | 5×5k | 5×5k | 5×5k | 5×5k |
| legacy | Flowcyt | MLP | 0.865 | 0.066 | 5×50k | 5×all | 5×all | 5×50k | 10×all | 10×50k | 5×all |
| legacy | Flowcyt | SOM | 0.798 | 0.043 | 5×20k | 5×20k | 5×20k | 5×20k | 5×all | 5×20k | 5×20k |
| full | Imstat | MLP | | | 3×50k | 4×50k | 3×10k | 2×10k | 7×all | 3×all | 2×20k |
| full | Imstat | SOM | | | 2×10k | 2×10k | 2×5k | 2×5k | 2×10k | 2×10k | 2×5k |
| full | LT1 | MLP | | | 7×all | 8×all | 7×all | 6×all | 13×50k | 8×all | 6×all |
| full | LT1 | SOM | | | 3×20k | 3×all | 3×20k | 2×5k | 7×50k | 3×all | 2×all |
| full | LT2 | MLP | | | 6×all | 30×all | 13×all | 6×all | 58×all | 30×all | 6×all |
| full | LT2 | SOM | | | 2×5k | 3×5k | 2×5k | 2×5k | 3×5k | 3×5k | 2×5k |
| full | LT1b | MLP | | | 2×10k | 3×50k | 2×20k | 2×5k | 3×50k | 3×50k | 2×10k |
| full | LT1b | SOM | | | 3×5k | 3×5k | 3×5k | 2×all | 5×50k | 4×10k | 3×5k |
| full | LT2b | MLP | | | 2×20k | 3×all | 3×all | 2×1k | 5×50k | 3×all | 3×1k |
| full | LT2b | SOM | | | 2×10k | 2×50k | 2×10k | 2×5k | 3×50k | 3×20k | 2×10k |
| full | Flowcyt | MLP | | | 2×all | 5×all | 4×all | 2×all | 10×all | 10×50k | 3×all |
| full | Flowcyt | SOM | | | 3×50k | 5×20k | 4×50k | 3×50k | 5×all | 5×20k | 3×50k |

`Δ ≥ −x`: mean paired difference to the baseline (point estimate, no CI); `noninf x`: lower bound of the 95 % paired
bootstrap CI of the mean difference > −x. `full` = all legacy cells (n_samples 2–20 in steps of 1, then steps of 5;
events 100 – all).

Findings:
- **Leniency is real where the baseline CI is wide.** LT2 MLP (CI half-width 0.034): `legacy_ci` accepts 15×all at
  Δ = −0.028; Flowcyt MLP (0.066): 5×50k at Δ = −0.030. `noninf 0.02` gives 30×all and 10×50k there and agrees with
  `legacy_ci` on the datasets with narrow CIs.
- **Pairing is much more precise:** median half-width of the paired CI (cells with Δ > −0.03) is 0.005–0.017 vs.
  0.011–0.066 for the baseline CI (Flowcyt MLP 0.017 vs. 0.066). A non-inferiority test on the paired difference is
  therefore not more lenient for small test sets, and more powerful than the legacy rule.
- **The legacy "5 samples" is mostly the grid floor** (first cell ≥ 2 of the figure grid is 5). On the full grid most
  minima are 2–4 samples (LT1 / LT2 MLP 6–30).
- **Single order → noisy, non-monotone curves.** The first passing cell is often followed by failing larger cells
  (same n_events): e.g. `noninf 0.02` LT2 MLP 30×all fails again at 40, 45, 50, 55; LT1b SOM 4×10k fails again at
  11 of 24 larger n; also for `legacy_ci` (LT2 MLP 6×all: 10 of 22). The trial × fold mean of step 8 smooths this;
  in addition a stability rule ("passes and all larger n_samples at the same n_events pass") is worth considering.
- δ = 0.01 is too strict for a single-order curve (LT2 MLP needs all 58 samples); δ = 0.05 adds little over the point
  estimate. δ = 0.02 as primary looks reasonable.

Scripts, tables and figure (git-ignored `scratchpad/`, env `fx_plot`): `scratchpad/legacy_criterion_check.py` →
`scratchpad/results/legacy_criterion_check/legacy_deltas.csv` (Δ, SD, paired CI per method × dataset × cell) and
`legacy_minimal.csv`; `scratchpad/plot_legacy_criterion.py` →
`scratchpad/figures/legacy_criterion_check/legacy_criterion.png` (Δ over n_samples per events cap with the
paired CI band, −δ, the legacy threshold, and both selected cells; the CI bar at the circle is the tested quantity).
`scratchpad/plot_delta_sensitivity.py` → `scratchpad/figures/legacy_criterion_check/delta_sensitivity.png` and
`scratchpad/results/legacy_criterion_check/delta_sensitivity.csv` (minimal cell over δ). On the legacy data the
greedy minimal sample number falls steeply up to δ ≈ 0.02 and is flat afterwards for most datasets (LT2 / LT1 MLP:
slow decline up to δ ≈ 0.05–0.09); the events per sample of the minimal cell jump around (greedy selection, one
order), so a joint rule (fewest total events) may give a smoother second axis.
Joint rule added (2026-11-18; total events per cell = sum over the first n samples of min(events, cap), counts from
the legacy np_files via `scratchpad/legacy_train_event_counts.py` →
`scratchpad/results/legacy_criterion_check/legacy_train_event_counts.csv`): the minimal total events fall smoothly with
δ (monotone by construction), but the joint rule trades samples for events: it picks many samples with few events
(SOM: 20–60 samples × 100–1k events at δ ≥ 0.02), and the sample number of the minimal cell jumps. Greedy keeps the
number of annotated samples (the clinical labelling cost) small; joint minimizes compute / data volume. They answer
different questions, so the rule should follow the claim (open decision above).

## Theory

@[statistics.md](statistics.md), covers:
- aggregation across folds,
- the criterion, 
- the paired bootstrap CI
- the test–CI duality (α = 0.025), 
- equivalence vs. non-inferiority, 
- and the references with DOIs and passages.
