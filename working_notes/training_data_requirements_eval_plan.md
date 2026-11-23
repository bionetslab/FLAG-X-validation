# Training data requirements: evaluation plan (PLAN.md step 8)

Status: draft for review (2026-11-18). Script: `training_data_requirements_eval.py` (skeleton only, not implemented).
Experiment design: `working_notes/training_data_requirements_plan.md`. Reviewer comment: R4.5
(`../flagx_revision/REVISION_PLAN.md`), also R2.19 / R2.21 and the test set reuse note (REVISION_PLAN.md:687).

## Goal

From the step-8 atoms (`results/training_data_requirements/`) and the full-data baseline of step 7
(`results/benchmark_eval/scores/`): the score curves over n_samples × n_events, the minimal training configuration
per method × dataset under a fixed criterion, its variability over trials / folds, a sensitivity analysis over
criteria, and a selection-bias check (cross-fitting). The minimal configurations are read by step 9.

## Planned outputs (`results/training_data_requirements_eval/`)

- `atoms.csv`: one row per manifest atom: status (ok / failed / timeout / oom / skipped / missing), phase, fit time /
  peak memory, n_train_events, skipped_by.
- `scores/<dataset>.csv`: one row per atom × test sample (per-class counts → f1 / prec / rec, macro F1 or F1 of
  label 1 for binary datasets), as `benchmark_eval.py`.
- `sample_scores.csv`: per method × dataset × cell × test sample: score averaged over the trials (NaN if any trial
  has no score), plus the step-7 baseline (seed mean) and the paired difference d = cell − baseline.
- `curves.csv`: per method × dataset × cell: mean score and mean d over all test samples (pooled over folds),
  bootstrap CI of mean d (donor-level resampling for Imstat), n, and the per-fold / per-trial means.
- `minimal.csv`: per method × dataset × criterion: minimal cell, selected on all folds (pooled); also per trial and
  per fold (spread), and joint minimum (fewest total training events) vs. greedy (samples first, then events).
- `crossfit.csv`: per method × dataset × criterion × fold k: cell selected on the folds ≠ k, its score on fold k,
  and the pooled selection for comparison (in-sample vs. cross-fitted score at the minimal config).
- `minimal_configs.json`: the minimal cell per method × dataset under the primary criterion (pooled; and the
  per-fold cross-fitted cells), read by step 9.

## Open decisions (to settle before implementation)

- Margin δ (decided 2026-11-18): default δ = 0.02 macro F1, kept as a stated convention; no data-driven δ (the seed
  variability of step 7 rests on 3 seeds only, too few). Instead the dependence on δ is shown explicitly: a δ
  sensitivity figure (minimal samples, events per sample and total training events of the minimal cell over δ = 0.001–0.1, per
  method × dataset; prototype `scratchpad/plot_delta_sensitivity.py`). Testing "Δ = 0" instead was rejected: it
  accepts noisy cells early and rejects precise ones (the margin is then the CI width, i.e. depends on the test size).
- Order of the grid for "minimal" (greedy samples-first as in legacy, or joint by total events; both reported, one
  primary).
- Step 9 uses the pooled or the cross-fitted per-fold config (decide after comparing them).
- Treatment of cells cut off (timeout / oom / skipped): not eligible as minimal; a cell counts only if all folds
  and trials have scores.
- Bootstrap: N_BOOT, percentile CI, resampling unit (test sample; donor for Imstat).
- Stability rule: the minimal cell must also hold for all larger n_samples at the same n_events (see the back-check).

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
