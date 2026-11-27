# Benchmark plan

Status: steps 0–6 done (2026-11-11); step 7: final run done (1373 / 1400 atoms ok, 2026-11-27), eval tables and figures on the final results (2026-11-27); step 8 (training data requirements): `training_data_requirements.py` implemented and smoke-tested (2026-11-17), run pending (open TODOs); eval `training_data_requirements_eval.py` implemented and tested on a synthetic fixture (2026-11-25). Built from `NOTES.md`.

This file holds the rules, the open todos and a one-line summary of each decision. What was done, with outputs, checks, findings and the reasoning behind each decision, is logged in `DONE.md` (append only). Small decisions (k, seeds, grids, ...) are made when we reach the step.

---

## 1. General rules

**Reproducibility**
- All randomness is seeded. Seeds are derived deterministically from one global seed plus the task identity (dataset, fold, method, repetition), so any single task can be rerun in isolation with identical results.
- Splits are generated once, saved, and are the single source of truth. Every later step (tuning, benchmark, follow-up experiments) reads them; no step creates its own splits.
- Downsampling happens on the fly inside a task, with the task's seed. Downsampled data is never stored.
- Every task records the software versions of its env, the git commit, and the wall-clock time and peak memory (CPU, and GPU where used) of training and of the inference on each test sample next to its outputs (step 5).

**No leakage**
- Test samples are never used for tuning, fitting, or for fitting any data-dependent preprocessing (scalers, cutoffs, channel selection). Fixed transformations with constant parameters (e.g. arcsinh with a fixed cofactor) are allowed.
- Tuning uses only the inner (train/val) splits of the outer training folds.

**Structure**
- One script per step. Parameters are constants at the top of the script, and the docstring names the env to run it in.
- Steps communicate only via files on disk. Each step reads the outputs of the previous one and writes into its own directory. Intermediate results are always saved.
- Every step that runs many tasks enumerates them as a list (e.g. method × dataset × fold). A task is addressed by its index, so the same script runs sequentially (loop over all indices) or as a SLURM array job (`SLURM_ARRAY_TASK_ID`).
- Tasks are idempotent and resumable: a task skips if its finished outputs exist, and writes atomically (temp file → rename) so a killed job leaves no half-written results.
- Prediction and scoring are separate. Each task stores, per test sample, the raw evaluation counts (confusion matrix incl. abstentions, per-class TP/FP/FN/TN, class counts) and, where feasible, the predicted labels and probabilities (step 5). Metrics and aggregation are computed afterwards from these (step 10).

**Environments**
- One isolated conda env per method (or method group) and per analysis step (data curation, evaluation, plotting). Env files live in `envs/`, each with a header comment stating its purpose and tested versions.
- R-based methods install `r-base` and all R packages in their own env. The system R is never used.
- The shared helpers (`validation/core/`, step 5: small modular functions for data and split loading, downsampling, joining samples, preprocessing, seeding, evaluation counts, resource tracking; no orchestration) depend only on numpy (+ the standard library) and must run in every method env, including the Python 3.5 envs (`fx_deepcytof`, `fx_acdc`): no f-strings, no `typing.Self`, no walrus operator, etc.

**Methods**
- Every method is used through its wrapper in `validation/gating/` with the common sklearn-like interface (see `CLAUDE.md`). Methods are not reimplemented; copied code is cited with URL, commit and line numbers.
- Method-specific details stay in the wrapper and in a small per-method runner. The shared helpers know nothing about individual methods.

**Extensibility**
- Adding a dataset = adding one `curate_<dataset>` function (raw path, channel and label maps as constants above it) plus its entries in `config.DATASETS` and `config.CHANNELS`. Everything downstream picks it up from the curated files.
- Adding a method = adding a wrapper, an env file, and a runner (`validation/runners/<method>.py`) that builds the wrapper with its hyperparameters, plus its grid / array entry in the experiment scripts. Nothing else changes.

---

## 2. Steps (chronological)

### Steps 0–3 -- DONE
Scaffolding, data curation (10 datasets), splits (`data/splits/`, K_OUTER = 5, K_INNER = 3), shared preprocessing (`Preprocessor`). Details: `DONE.md`.

### Step 4: Method integration -- DONE
Wrappers, envs and runners (`validation/runners/`) for FLAG-X SOM / MLP, DGCyTOF, GateMeClass, LDA-Abdelaal, DeepCyTOF and the tabular baselines (RF, HGB, LogReg, kNN, LDA-sklearn, GMM-Mocking); smoke-tested on Imstat; all envs created and checked on Woody / TinyGPU. Details: `DONE.md`.

### Step 5: Shared helpers (`validation/core/`) -- DONE
Done: data loading, downsampling, joining, seeding, evaluation counts, resource tracking, output / run helpers, scores (`DONE.md`). Rules: each helper does one thing, is stateless (seeds are passed in), numpy + stdlib only, Python 3.5-compatible, and knows nothing about individual methods. **No orchestration:** experiment scripts compose the helpers and write their own code where none fits. Helpers are added when a second script needs them.
- [x] Score helpers (D8 rule): `validation/core/scores.py` (2026-11-10).

**What every experiment stores** (convention, using the helpers above):
- per test sample: the evaluation counts. Any metric (abstention-aware or not, per class or averaged) is computed later from them, without the per-cell predictions (step 10);
- optional per-cell outputs (flag per experiment): predicted labels and, where the method has `predict_proba`, predicted probabilities. On where feasible (main benchmark); off where storage would blow up (e.g. training-data requirements) and in the tuning;
- wall-clock time and peak CPU / GPU memory for training (`fit`, incl. the preprocessing fit) and for the inference on each test sample separately (`predict` / `predict_proba` + preprocessing transform); the scalability experiment (step 9) reuses these;
- run metadata: parameters, seed, env versions, git commit, hardware.
- Downsampling is applied to the training samples; test samples are predicted in full unless an experiment states otherwise.

### Step 6: Hyperparameter tuning (Imstat) -- DONE
Done: `tune_hyperparameters.py`, atom format, smoke test, grid review (295 configs, 4425 atoms), frozen manifests (committed), SLURM scripts and resources (Woody 4 cores, TinyGPU 1 GPU: RTX 2080 Ti / RTX 3080 / V100; `DONE.md`, `working_notes/tuning_plan.md`, `slurm_scripts/README.md`); envs and Imstat data on the cluster; all 4425 atoms run (4157 ok, 268 failed; 2026-11-10).
- [x] `tune_hyperparameters_eval.py` (2026-11-10): aggregation chain sample → donor → inner → outer → overall, mean / SD / n per level, NaN for failed atoms propagates (`working_notes/tuning_plan.md` section 8, `DONE.md`).
- [x] Best configs (D6, 2026-11-10): argmax mean at the outer level (per outer split, Imstat) and the overall level (consensus); incomplete configs excluded; ties → default, then lowest config_id → `results/tune_hyperparameters_eval/best_configs/params_{outer,overall}_<method>.json` (read by step 7).
- [x] Effect of tuning (2026-11-11): best vs. default per method, Δ per unit one level below the selection level (overall: per outer split; outer: per inner fold), partial results used; grid flatness; box plots (`plot_effect_of_tuning.py`; `DONE.md`).

### Step 7: Main benchmark
- [x] `benchmark.py` (2026-11-11): atoms = method × dataset × outer split × model seed (1400 in 6 arrays, `slurm_scripts/bench_<array>.sh`), counts + predicted labels per test sample, optional training downsampling per (method, dataset) (`TRAIN_DOWNSAMPLING`, off) for a second run after OOM / timeout (`working_notes/benchmark_plan.md`, `DONE.md`).
- [x] Run setup (2026-11-11): Woody arrays with 32 cores (248 GB); per-atom progress (`phase`, fit resources and counts written after the fit and after each test sample); commands `pilot` and `cap` (`DONE.md`).
- [x] Pilot (2026-11-12): 116 / 120 ok; SOM × BLAST110 OOM in flagx's unit annotation (full events × units distance matrix), fixed in the runner by a chunk-wise BMU computation with identical results; SOM training is not deterministic (already 3 seeds) (`DONE.md`).
- [x] Caps (2026-11-12): none needed; with the SOM fix every method fits on every dataset at full size, `TRAIN_DOWNSAMPLING` stays off.
- [x] Final run (2026-11-16 to 2026-11-27): 1373 / 1400 atoms ok; 27 failures (15 DGCyTOF in the prediction: 13 OOM, 2 Python errors; 12 GateMeClass BLAST110 OOM in the fit), all reported as results, no caps / reruns; no time-outs (`DONE.md`).
- [x] `benchmark_eval.py` tables (2026-11-11): fit table `atoms.csv` (status, failure type and phase, fit resources), wide per-dataset score tables `scores/<dataset>.csv` (one row per atom × test sample, per-class n / prec / rec / F1, prediction time / memory, n_cells), `failures.csv` (`DONE.md`).
- [x] Aggregation (2026-11-27, as implemented in `benchmark_eval.py` / `plot_benchmark.py`): sample score = macro F1 (multi-class) or F1 of label 1 (binary), averaged over the seeds that predicted the sample; per dataset over all test samples (folds pooled); fold × seed means and fold / seed SD; paired per-sample differences between methods with bootstrap CI and equivalence / non-inferiority at `config.DELTA`; no aggregation over datasets. Run on the final results (`DONE.md`).
- [x] `plot_benchmark.py` (2026-11-13): box plot per dataset × method of the sample-wise scores (seed mean; macro F1, binary datasets F1 of label 1), marker = test fold; global plot order, names and colors of datasets / methods in `config.py` (section Plotting) (`DONE.md`).
- Tasks: method × dataset × outer fold × model seed. Each task trains on the training samples and predicts every test sample.
- Model seeds (R4.4, R4.m3 seed variability): 3 seeds per fold for stochastic methods (FLAG-X MLP/SOM, DGCyTOF, DeepCyTOF, GateMeClass, RF, HGB, GMM-Mocking; initialization, internal sampling / validation splits, training order and on-the-fly downsampling), 1 for deterministic ones (LogReg, kNN, LDA-sklearn, LDA-Abdelaal; 2026-11-11). Seeds are derived from `config.SEED` + task identity. Report the mean and SD over seeds. If BLAST110 is too expensive, reduce its seeds or downsample harder and state this. The splits stay unchanged (1 CV repetition); tuning uses a single seed (stated as a limitation).
- **[D8, decided] Abstention:** the task is full assignment of every cell.
  - Where abstention is only an option of a method, it is switched off: LDA `rejection_threshold=0` (default), GateMeClass `reject_option=False` (default), DeepCyTOF `rejection_threshold=0` (the 0.4 rejection exists only in the code, not in the paper; it also filters the cells used for the MMD calibration). The FLAG-X MLP never abstains.
  - Where it is built into the method (DGCyTOF's calibration step; FLAG-X SOM: cells whose best-matching unit has no labelled training events), the method is run as published and its abstentions are scored as below.
- Hyperparameters (D6): Imstat tasks use the per-fold parameters of their outer split (`best_configs/params_outer_<method>.json`, `['splits'][str(split)]['params']`); all other datasets use the consensus parameters (`params_overall_<method>.json`, `['params']`).
- Run sequentially or as a SLURM array job; GPU only where it helps (FLAG-X MLP, DGCyTOF).

### Step 8: Training data requirements
How the score, fit / predict time and memory depend on the number of training samples × events per sample, under a fixed small budget (4 CPUs or 1 GPU, 1 h per atom). Determines the reduced / minimal training sets used in step 9. Design: `working_notes/training_data_requirements_plan.md`.
- Script `training_data_requirements.py` (all Py3.5-compatible like `benchmark.py`); methods constant at the top: RF, HGB, MLP, SOM (decided 2026-11-24, `DONE.md`).
- Atom = method × dataset × outer split × trial × n_samples × n_events. Trial = training sample order + downsampling seed (model seed tied to the trial); the first n samples of a per-trial permutation of the outer training samples (nested, same for all methods); events downsampled per sample on the fly (first n of a fixed stratified per-sample event order, so nested over n_events); test samples predicted in full; counts + resources per atom, no predictions; params as in step 7 (D6).
- Execution: one SLURM array task per *lane* (method × dataset × fold × trial; arrays `sklearn_cpu`, `flagx_cpu`, `flagx_gpu`, order fold first). The lane worker runs its atoms small → large, each as a child process with a 1 h timeout; a timeout / OOM is the method's cutoff: every atom with at least as many total training events (`n_train_events`) of the same method × dataset, in any fold / trial, is skipped without running (re-applied post hoc in the eval). An atom is only started if it can finish within the lane's job time minus a buffer; unfinished lanes are resubmitted and resume.
- [x] Design (working note) and implementation: `training_data_requirements.py`, `slurm_scripts/tdr_<array>.sh`, helpers `outer_splits` / `best_config` moved to `validation/core/` (2026-11-17, `DONE.md`). Grid without duplicates of `'all'` and without (`'all'`, `'all'`) (= main benchmark): 600 lanes, 59,524 atoms (grid of 2026-11-18: n_samples ~×1.5 steps per dataset, n_events ×3 steps for all datasets, informed by the legacy results).
- [ ] Run (after the TODOs below).
- [x] Method selection (2026-11-24): RF, HGB, MLP (top performers of step 7) + SOM (needed for Workflow 1; `DONE.md`).
- [x] Cutoff rule and hardware (2026-11-24): cutoff by total training events (online and re-checked post hoc in the eval); no calibration run, no slowdown factors (`DONE.md`).
- [x] Eval (2026-11-25, `DONE.md`): `training_data_requirements_eval.py` (post-hoc budget, per-sample trial means paired with step 7, bootstrap curves per scope, minimal cells, cross-fitting, `minimal_configs.json`; training class counts per sample for the minority count figure); primary criterion paired non-inferiority δ = 0.02, greedy order; plan `working_notes/training_data_requirements_eval_plan.md`. Tested on a synthetic fixture only.
- [x] Figures (2026-11-25, `DONE.md`): `plot_training_data_requirements.py` (Δ figures as the focus, minority count figures, legacy-style score figures kept), developed on dummy results; formatting in one final pass.
- [ ] Run the eval on the real step-8 results; then choose the stability rule (with / without) and the step-9 config (pooled or cross-fitted per fold).

### Step 9: Follow-up experiments
Reuse the same splits, curated data and runners; only the task list changes. Candidates from the legacy results (`legacy/results/`), to be confirmed (the reduced/minimal training sets come from step 8):
- local training: FLAG-X SOM and MLP trained on the reduced/minimal training sets in the local (CPU, limited resources) setting of Table 1,
- scalability (runtime and memory vs data size),
- robustness to sample order: `sample_order_robustness.py` (implemented 2026-11-25, `DONE.md`; plan `working_notes/sample_order_robustness_plan.md`). 30 new trials per fold with 1 … n_min training samples at the events per sample of the step-8 minimal cell (n_min, ne_min), mirroring the step-8 setup; two arms with own output roots: `random` (random permutations) and `stratified` (round-robin over strata: LT subtype, Imstat lysis, BLAST110 sample_type; Flowcyt only in `random`). New trials from 0 with own seeds; the step-8 trials are not reused (decided 2026-11-25; replaces "continue the trial numbering"). Criterion: the primary criterion of the step-8 eval (paired non-inferiority, δ = 0.02, greedy; legacy criterion only for comparison). Gives the share of passing trials at the minimal cell and the per-trial minimal n_samples. Legacy ran these trials only on LT1 / LT2 (+ binary).
  - [ ] TODOs in the script: `STABLE` (after the step-8 eval decision), `N_TRIALS` (30 for now), lane time estimate from the step-8 `pred_s` before the run (SLURM `--time` / lane split).
  - [x] Eval `sample_order_robustness_eval.py` (2026-11-27, `DONE.md`): share of passing trials (counts only; CI later if needed), per-trial minimal n_samples, share by n; failed trials count as not passing; δ sweep; both stability variants; no arm comparison (figures). Tested on dummy results.
  - [ ] Run (after the step-8 eval); figure script `plot_sample_order_robustness.py` (arm comparison),
- probabilistic predictions / abstention behaviour,
- channel leave-out,
- class weighting ablation: methods with a class-weight option (FLAG-X MLP, RF, HGB, LogReg `class_weight='balanced'`; LDA-sklearn uniform `priors`) with and without class weighting; not part of the tuning grids (all tuned without),
- preprocessing ablation (decided, see D3): rerun a subset (e.g. a few methods × datasets) with the alternative transformations (uniform cut-off 100, R4.m1; arcsinh) and normalizations, on the reduced/minimal training sets from the training data requirements experiment.

### Step 10: Evaluation
- Separate env. Reads the stored per-sample counts (step 5; per-cell labels / probabilities where stored, e.g. for calibration and abstention analyses), computes sample-wise scores (precision/recall/F1 etc.; port from `legacy/validation/utils/val_utils.py`), writes score tables. The scores implemented in the wrappers (`score`) are not used.
- Scoring rule (D8): an abstained cell (-1) counts as a false negative for its true class and as a false positive for no class. The abstention rate is reported per sample alongside the scores.
- Main comparison on the hard labels only. Predicted probabilities (where stored) are used only for secondary analyses on the methods that provide them (e.g. calibration, abstention behaviour), since not all methods have them (GateMeClass) and they differ in meaning (e.g. DGCyTOF's softmax is from before its reassignment step).
- Aggregation across folds, datasets and methods; statistical comparison (to be decided at this step).

### Step 11: Figures and tables
- Separate env. Reads only score tables, writes to `figures/`.

### Step 12: Documentation
- README "Scripts" section: order of scripts, env per script, how to run sequentially and on SLURM.

---

## 3. Decisions

One line per decision; reasoning and sources in `DONE.md`.

| ID | Decision | Status |
|---|---|---|
| D1 | Datasets | **decided:** Imstat (also used for tuning), Lt1, Lt2, Lt1b, Lt2b, Flowcyt, BLAST110 (p1–p4) |
| D2 | Storage of curated data | **decided:** per-sample `.npy` (X, y) + `meta_samples.csv`, `meta_labels.csv` |
| D3 | Preprocessing ownership | **decided:** the same shared input for all; native input only as a tuning arm for DeepCyTOF (LDA-Abdelaal: shared input only, its internal transformations are not flow-appropriate; no sklearn baseline has internal preprocessing), or for stability (none so far); post hoc ablation |
| D3a | Shared transformation | **decided:** log10 with cut-offs as in the paper (per-channel for Imstat/LT, 100 for Flowcyt/BLAST110); alternatives in the ablation |
| D3b | Shared normalization | **decided:** none; z-score / min-max in the ablation |
| D4 | Prior-knowledge methods (ACDC, Bayesian trees) | **decided:** dropped; both need an expert marker table and cannot derive it from labelled data |
| D5 | Config format | **decided:** small shared `config.py` at root, per-script constants otherwise, no registries |
| D6 | Use of tuned parameters | **decided:** both. Per-fold best for Imstat (nested CV); one consensus set (best mean inner score over all outer splits) for all other datasets |
| D7 | Compensation | **decided:** with flagx if the FCS file has a spillover matrix (only BLAST110) |
| D8 | Abstention | **decided:** switched off where it is an option (LDA, GateMeClass, DeepCyTOF); built-in abstention (DGCyTOF, FLAG-X SOM) kept; abstained cells count as FN for their true class, abstention rate reported (step 10) |
