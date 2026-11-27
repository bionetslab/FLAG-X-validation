# Sample order robustness (PLAN.md step 9)

Scripts: `sample_order_robustness.py` (implemented 2026-11-25), `sample_order_robustness_eval.py` (2026-11-27). SLURM: `slurm_scripts/sor_<array>.sh <scheme>`.

## Question
Step 8 gives one minimal cell (n_min training samples × ne_min events per sample) per method × dataset, selected on the mean over 3 trials (sample orders). How much does that result depend on the sample order? The same runs answer two questions:
1. **Share of passing trials:** the share of single trials that meet the step-8 primary criterion at the minimal cell (paired non-inferiority to the full-data model, δ = `config.DELTA` = 0.02).
2. **Spread of the per-trial minimum:** for each trial, the smallest n_samples (at ne_min) that passes.

Caveat for the text: a single trial's d is noisier than the 3-trial mean that selected n_min, so the share is expected to be below 1 even without any order effect.

## Design (decided 2026-11-25, user)
- Mirrors `training_data_requirements.py` (step 8) on a smaller grid and imports its machinery: one child process per atom, per-atom time limit, cutoff / dominance, lanes, resubmission via `missing`, `status`.
- **Atom** = method × dataset × outer split × trial × n_samples, with n_samples = every integer 1 … n_min (values ≥ the split's training samples are dropped) and n_events = ne_min.
- **Lane** = method × dataset × split × trial, ordered split → dataset → method → trial; atoms small → large.
- **Minimal cell:** the pooled cell of the step-8 eval's `minimal_configs.json` (primary criterion: noninf, δ = 0.02, greedy); `STABLE` selects the variant with or without the stability rule (TODO: decide after the real step-8 eval). Method × dataset without a minimal cell or with n_min = 'all' are skipped.
- **Trials:** `N_TRIALS = 30` (TODO: final number; legacy Fig S6 used 100 random orders on LT only). Numbered from 0 with their own seeds. The results stand on their own: the step-8 trials are **not** reused (this replaces the earlier plan to continue the trial numbering).
- **Two arms**, one output root each (`results/sample_order_robustness/<scheme>/`):
  - `random`: a random permutation of the outer training samples per trial (as in step 8, new seeds).
  - `stratified`: round-robin over strata. The samples are shuffled within each stratum. Each round takes the next sample of every stratum that still has samples left, in a stratum order that is drawn anew each round (with n not a multiple of the number of strata, different strata are left out). Which sample comes from a stratum is random. Strata (`STRATA`, a `meta_samples.csv` column): LT1 / LT2 / LT1b / LT2b `subtype` (11 subtypes; BL has 2 samples, NB 9), Imstat `lysis` (5 protocols × 25; the donor is the CV group), BLAST110 `sample_type` (AML_Dx / AML_FU / NBM). Flowcyt has no strata (healthy bone marrow), so it runs only in the random arm.
- **Methods / arrays / params / time limits:** as in step 8 (RF, HGB, MLP, SOM; D6 params frozen into the manifest; 1 h per atom). Test samples are predicted in full; counts + resources, no predictions.
- **Seeds** (key `('order_robust', scheme)`): order `derive_seed(*key, dataset, split, trial)`; event order per sample `derive_seed(*key, dataset, split, trial, sample)` (`event_order`, nested over n_samples); join `derive_seed(*key, dataset, split, trial, n_samples, n_events, 'join')`; model `derive_seed(*key, method, dataset, split, trial)`. The data seeds do not contain the method.

## Size (dummy minimal cells, 2026-11-25)
Lanes: random 6,000 (sklearn_cpu 3,000, flagx_cpu 1,500, flagx_gpu 1,500), stratified 5,400 (2,700 / 1,350 / 1,350); all below `MaxArraySize` = 10,000. Atoms (dummy cells): random 56,250, stratified 54,750. The real counts depend on the step-8 minimal cells.

TODO before the run: estimate the lane time from `pred_s` in the step-8 `atoms.csv` (the full test-set prediction dominates, e.g. BLAST110) and set the SLURM `--time` / the lane split accordingly.

## Changes to `training_data_requirements.py` (step 8 unchanged)
- `run_atom(..., order=None, seed_key=('train_req',), order_scheme=None)`: training sample order and seed prefix as arguments; `order_scheme` written to the JSON if given.
- `run_child` / `run_lane(..., atom_cmd=None)`: the child command (incl. `--out`) of the calling script; `run_lane(..., overwrite=None)`.
- Checked: manifest, counts and atom JSONs (except times / metadata) of a step-8 smoke lane are identical to the version before the change.

## Eval (`sample_order_robustness_eval.py`, implemented 2026-11-27)
Env `fx_plot`; reuses the step-8 eval (`training_data_requirements_eval.py`): atom table with the post-hoc budget, scores, paired d to the step-7 baseline, bootstrap curves, `minimal_cell`. Output `results/sample_order_robustness_eval/`.
- **Trial criterion:** scope `trial_<t>` = trial t over all 5 folds (one order per fold): mean d with a percentile CI from the bootstrap over patients (donors for Imstat), on the resamples of the step-8 pooled / `trial_<t>` scopes (same seed), shared by all trials, n and schemes. Pass = eligible (all 5 fold lanes ok and in budget) and `ci_low > -δ`; legacy criterion as well. Scope `pooled` = mean over all trials (sanity check against step 8).
- **Decisions (user, 2026-11-27):**
  - a trial with a failed / timed-out / out-of-budget fold lane counts as **not passing** (share over all trials; `n_failed` reported);
  - the full step-8 δ sweep (0.001–0.1) plus the legacy criterion; the summary at `config.DELTA`;
  - per-trial minimum without and with the stability rule (every eligible larger tested n also passes; only n ≤ n_min are tested; not found = above n_min); the summary follows the manifest's `stable` column (the run script's `STABLE`);
  - **no comparison of the arms in the eval** (aggregation per arm only); the arms are compared in the figures.
- **Shares:** counts and shares only, no CI (user, 2026-11-27: aggregate first; a binomial CI, e.g. Clopper–Pearson, can be computed later from `n_pass` / `n_trials` and `n_min_le_*` / `n_trials`). Conditional on the test set (the trials share it); a share below 1 at n_min is expected because a single trial is noisier than the 3-trial mean that selected n_min.
- **Outputs:** `atoms.csv`, `scores/<scheme>/<dataset>.csv`, `sample_scores.csv`, `trial_curves.csv` (pooled + `trial_<t>`), `trial_minimal.csv` (per trial × criterion × δ × stable), `share.csv` (per n: `share_pass` and the cumulative `share_min_le_{naive,stable}` with the counts), `summary.csv` (per scheme × method × dataset at n_min: `n_pass`, share, `n_failed`, quantiles of the per-trial minimum as order statistics (inf = not found), pooled mean d, step-8 context: pooled mean d / ci_low and passing step-8 trials at the cell).
- **Run-script changes for the eval:** the manifest keeps n_samples = the split's training samples (`>` instead of `>=`; larger n_min is resolved to that atom in the eval) and records the step-8 variant in a `stable` column.

Next: `plot_sample_order_robustness.py` (arm comparison in figures).
