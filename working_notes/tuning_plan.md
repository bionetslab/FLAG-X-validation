# Tuning plan: step 6 (hyperparameter tuning on Imstat) + step 5 helpers

> Status: implemented (2026-11-04); smoke-tested for every method; timing run see the end. Deviations from the reviewed draft are marked **[impl]**.

## Context
PLAN.md step 6 tunes every benchmark method on the Imstat inner splits (`data/splits/imstat_inner.csv`, 5 outer × 3 inner) and selects per-fold and consensus parameters (D6). Reviewers criticised that only the SOM was tuned (R2.15, R3.m2, R4.3). Tabular sklearn baselines and the GMM classifier of Mocking et al. are added (R4.3).

The step 5 helpers (`validation/core/`) and the step 4 runners are written now, since tuning is their first user. The helpers are small and stateless, so the main benchmark (step 7) and the training-data-requirements experiment (step 8) can reuse them.

Scope of this step:
- Implement everything up to and including the atomic results.
- Aggregation, scores, selection and plots come later and get no helpers now. The atom format is designed to make them easy.

## 1. Decisions taken
- **Input (D3):** all methods get the shared input (log10 with cut-offs, no scaling). Methods with internal preprocessing tune it as a grid parameter (LDA-Abdelaal, DeepCyTOF); if it is selected, they get the raw curated data. No sklearn method has internal preprocessing, so all of them use the shared input only (no scaling arm).
- **Training samples:** all events (`TRAIN_FRAC = 1.0`). **[impl]** Raised from 10 % (timing run) via 20 % to 1.0: the benchmark trains on all available data, so the size-dependent parameters are tuned at (close to) the training size they are used with (`DONE.md`, step 6).
- **Val samples:** class-stratified fraction per sample (`VAL_FRAC`, 1.0 = no downsampling).
- **Seeds:** the data seed depends only on (split, inner, sample), so all methods and configs see identical events. The model seed is `derive_seed('tune', method, split, inner)`, the same for all configs of a method.
- **Grids:** mid-sized. All values of a categorical parameter are included; continuous ones get at least low / mid / high. Each grid contains the default / published config. The user reviews the grids (section 6).
- **LDA:** both versions. `lda_abdelaal` (the R wrapper `validation/gating/lda.py`) and `lda_sklearn`.
- **Boosting:** `HistGradientBoostingClassifier`.
- **GMM (Mocking et al.):** one GMM per class, prediction = argmax of the per-class log-likelihood. All GMM hyperparameters are shared across classes.
- **Execution:** one SLURM array per (env, cpu/gpu) combination. One array task = one atom = (method, config, outer split, inner split), fit + predict. CPU arrays run on Woody, GPU arrays on TinyGPU (frontend `tinyx`).

## 2. New method code

### `validation/gating/gmm_mocking.py`: `GMMMockingClassifier`
- Source: `https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py` L5–27, cited in comments.
- Usage by Mocking et al.:
  - `02-GMMclf.ipynb`: `RandomizedSearchCV` / `GridSearchCV` over `n_components_class0` 1–20 and `n_components_class1` 1–10, with `covariance_type='full'`.
  - Data: BLAST110 at 5K events per sample.
- Follows the wrapper contract:
  - `BaseEstimator, ClassifierMixin`;
  - `_process_class_labels`, so `predict` returns the original labels;
  - `score` = macro F1;
  - `save` / `load` via pickle.
- `__init__`: `n_components=1, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', random_state=None`, i.e. all `GaussianMixture` parameters, shared by the per-class GMMs.
- `# MOD:` changes to the upstream code:
  - multiclass: L27's rule `(ll1 > ll0)` becomes an argmax over classes, which is identical for 2 classes (ties go to the lower class, as upstream);
  - `random_state` is actually passed on (upstream L11 sets it to `None`);
  - `fit` returns `self`;
  - one shared `n_components` instead of `n_components_class0/1`.
- `predict_proba`: softmax of the per-class log-likelihoods, i.e. posteriors under uniform class priors (the quantity the decision rule compares). This should be flagged in the docstring.
- **[impl]** The input is cast to float64. With float32 input (the Preprocessor output), GaussianMixture fails with 'ill-defined empirical covariance' on BLAST110 (many identical values at the log10 cut-off); the upstream class fails the same way. With float64, predictions are identical to the upstream class (same `random_state`) for 1/3/5 components, full/diag covariance (BLAST110 p1, 20k training events).
- Env: `fx_sklearn`.

### sklearn baselines (no wrappers; built directly in the runners)
The sklearn classes already fulfil the contract (original labels, `predict_proba`, picklable):
- `rf` (RandomForestClassifier)
- `hgb` (HistGradientBoostingClassifier)
- `logreg` (LogisticRegression)
- `knn` (KNeighborsClassifier)
- `lda_sklearn` (LinearDiscriminantAnalysis)

### `envs/sklearn.yml` → `fx_sklearn`
- Lightweight, no flagx / torch / somoclu.
- Contents: python 3.12, numpy, scikit-learn (conda-forge, strict priority, versions pinned after creation, header comment as in the other env files).

### `envs/flagx_gpu.yml` → `fx_flagx_gpu`
- The GPU env announced in CLAUDE.md.
- Contents: flagx 0.2.1 (bioconda) + CUDA torch, matching the TinyGPU drivers.
- Used for MLP and DGCyTOF.

## 3. Shared helpers: `validation/core/`
All modules are numpy + stdlib only and Python 3.5-compatible.

| Module | Functions | Notes / reuse |
|---|---|---|
| `data.py` | `load_sample(dataset, sample) -> (X, y)`; `load_meta_samples(dataset)`, `load_meta_labels(dataset)` → list of dicts (csv module); `load_split(dataset, column, inner=False) -> {'train': [...], 'val': [...], 'test': [...]}` | paths from `config`. The benchmark uses `split_<i>`, the tuning `split_<i>_inner_<j>` |
| `sampling.py` | `downsample_indices(y, n=None, frac=None, stratified=False, seed=None) -> sorted idx`; `join_samples(Xs, ys, sample_ids, shuffle=True, seed=None) -> X, y, sample_ids` | stratified = port of legacy `get_downsampling_bool` (`legacy/validation/utils/val_utils.py:750`), but with `np.random.RandomState(seed)`. If `n` is at least the sample size, or `frac >= 1`, all events are kept. The data-requirements experiment uses the same function with `n` |
| `seeding.py` | `derive_seed(*keys) -> int` | sha256 of `config.SEED` + keys → `[0, 2**31-1)`; independent of process and Python version |
| `evaluation.py` | `confusion_counts(y_true, y_pred, labels) -> (K, K+1)` (last column = abstention -1); `class_counts(cm) -> dict` (per class: tp, fp, fn, tn, n_true, n_pred, n_abstained) | D8: an abstained cell = FN for its true class, FP for none. `labels` come from `meta_labels.csv`, so every sample has the same shape. No score functions now; they come with the aggregation |
| `resources.py` | `track_resources(gpu=False, interval=0.05)`: a context manager that yields a dict filled on exit (`wall_s`, `cpu_rss_baseline_mb`, `cpu_rss_peak_mb`, `cpu_rss_peak_increase_mb`, `gpu_peak_mb`) | port of `scalability_wrapper` / `track_memory_cpu/gpu` (`val_utils.py:822–928`). A polling thread reads VmRSS from `/proc` for the process and its children; GPU via `nvidia-smi`; `time.perf_counter` |
| `runs.py` | `save_json(path, obj)`, `save_csv(path, header, rows)` (atomic: tmp + `os.replace`); `run_metadata(modules)` (Python and module `__version__`s, conda env, git commit + dirty flag, host, CPU model / threads, GPU names, SLURM job / array ids); `n_threads()` (`SLURM_CPUS_PER_TASK`, else `os.cpu_count()`) | the benchmark (step 7) reuses them |

`Preprocessor` (`validation/core/preprocessing.py`) is reused unchanged.

## 4. Runners: `validation/runners/<method>.py`
All runners share one interface:
- `build(params, seed, dataset, n_jobs)`
- `fit(model, X, y, sample_ids)`
- `predict(model, X) -> (y_pred, proba or None)`
- `INPUT_ARMS`: the param key and its values for the shared / native input, empty if the method has none.

The experiment script imports only the requested runner (`importlib`), so each env loads only its own method.

| Runner | Env | Notes |
|---|---|---|
| `som` | `fx_flagx_cpu` | flagx `SOMClassifier`, a fresh instance per fit (`fit` overwrites `radius_0`) |
| `mlp` | `fx_flagx_gpu` | flagx `MLPClassifier`. Torch is seeded here (flagx has no seed). `class_weight='balanced'` becomes `loss_params={'weight': tensor}` computed from the training labels |
| `dgcytof` | `fx_flagx_gpu` | `random_state=seed`; built-in abstention kept (D8) |
| `gatemeclass` | `fx_gatemeclass` | `marker_names=CHANNELS[dataset]`, `gmc_reject_option=False`, `gmc_seed=seed` |
| `lda_abdelaal` | `fx_lda` | `rejection_threshold=0`; `transformation` = input arm |
| `deepcytof` | `fx_deepcytof` (Py3.5) | `denoise=False`, `rejection_threshold=0`; `sample_ids` passed; one `predict_proba` call per sample, then argmax; `K.clear_session()`; `n_threads` |
| `rf`, `hgb`, `logreg`, `knn`, `lda_sklearn`, `gmm_mocking` | `fx_sklearn` | `random_state=seed` and `n_jobs` where available |

The number of threads comes from `n_threads()` (also exported as `OMP_NUM_THREADS` in the job scripts).

## 5. `tune_hyperparameters.py` (root; Py3.5-compatible, runs in every env)

**Constants**
- `ARRAYS`: array name → (env, device, methods):

  | Array | Env | Device | Methods |
  |---|---|---|---|
  | `sklearn_cpu` | `fx_sklearn` | cpu | rf, hgb, logreg, knn, lda_sklearn, gmm_mocking |
  | `flagx_cpu` | `fx_flagx_cpu` | cpu | som |
  | `flagx_gpu` | `fx_flagx_gpu` | gpu | mlp, dgcytof |
  | `gatemeclass_cpu` | `fx_gatemeclass` | cpu | gatemeclass |
  | `lda_cpu` | `fx_lda` | cpu | lda_abdelaal |
  | `deepcytof_cpu` | `fx_deepcytof` | cpu | deepcytof |

- `GRIDS` (section 6), `TRAIN_FRAC = 1.0`, `VAL_FRAC = 1.0`, `OVERWRITE = False`, `DATASET = 'imstat'`.

**Commands**
- `python tune_hyperparameters.py manifest <array>`: expands the grids and writes:
  - `results/tune_hyperparameters/configs/<method>.csv` (`config_id`, grid position, `is_default`, params JSON);
  - `results/tune_hyperparameters/manifests/<array>.csv` (`index, atom_id, method, config_id, split, inner`). It prints the number of atoms, which gives the `--array` range.
  - `config_id` = a short hash of the canonical params JSON. It stays stable when the grid is edited, so finished atoms are kept.
- `python tune_hyperparameters.py run <array> [<index>|all]`: the index is taken from `SLURM_ARRAY_TASK_ID` if not given. The atom is looked up in the frozen manifest. No offset is needed: `MaxArraySize` is 10000 on Woody and TinyGPU, and the largest array has 3405 atoms.

**One atom**
1. Skip if the atom's JSON exists (unless `OVERWRITE`); otherwise write it with `status: started`.
2. Load the train / val samples.
3. Downsample them (train `TRAIN_FRAC`, val `VAL_FRAC`, both stratified).
4. Join and shuffle the training samples.
5. Choose the input arm: shared → `Preprocessor('log10_cutoff', LOG10_CUTOFFS)`; native → `Preprocessor('none')` and the method's own preprocessing on.
6. Track the preprocessing fit + `build` + `fit`.
7. Per val sample: track transform + `predict`, then `confusion_counts`.
8. Write the atom files atomically.

**Atom files**
- Location: `results/tune_hyperparameters/atoms/<method>/<config_id>/split_<i>_inner_<j>.json` (+ `_counts.csv`).
- `…_counts.csv` (long and tidy, self-identifying, so all atoms concatenate directly):
  - identifiers: `method, config_id, split, inner, sample, n_events_sample, n_events_used`;
  - class counts: `label, n_true, n_pred, n_abstained, tp, fp, fn, tn`;
  - confusion row: `pred_<label>…, pred_abstained`;
  - prediction performance only: time and memory are only in the JSON. **[impl]** The per-sample predict time / memory columns were removed again (2026-11-05).
- `….json`:
  - **identity:** `atom_id`, method, array, env, device, `config_id`, params;
  - **data:** dataset, split, inner; train and val sample lists; labels (`meta_labels`); `TRAIN_FRAC` / `VAL_FRAC`, stratified; `n_train_events` and `n_train_events_per_class` (all classes of the dataset, 0 if absent);
  - **seeds:** data seeds (rule), model seed;
  - **preprocessing:** the fitted Preprocessor state, input arm;
  - **outcome:** status `ok` / `failed` (+ traceback);
  - **resources (time and memory, per atom):**
    - `resources.fit`: preprocessing fit + `build` + `fit`;
    - `resources.predict[<val sample>]`: transform + `predict` of that sample;
    - each entry is the `track_resources` dict (`wall_s`, `cpu_rss_baseline_mb`, `cpu_rss_peak_mb`, `cpu_rss_peak_increase_mb`, `gpu_peak_mb`; `null` without a GPU);
  - `run_metadata`, start / end time.
- **Failures and timeouts. Rule: an atom that does not finish puts its config out.**
  - **Start marker:** the first thing an atom does is write its JSON with `status: started`. At the end it overwrites it with `ok` or `failed`.
  - **Crash:** a Python exception is caught and written as `status: failed` + traceback, with no counts. This is expected for GateMeClass "V" and LDA-Abdelaal native `log`; the result counts, nothing is worked around.
  - **Timeout, OOM kill, node failure:** Python cannot write anything, so the JSON stays at `started`. After all jobs are done, `started` simply means "did not finish". If the reason is wanted, it is in the job log or in `sacct`. No signal trap, no extra code.
  - **Timeout limit:** every array runs with `--time=24:00:00`, the cluster maximum, so a timeout always means "longer than 24 h".
  - **Config out:** a config with a non-`ok` atom in outer split i is not eligible for the per-fold selection of split i. It is also not eligible for the consensus, which needs all 15 atoms.
  - **Resubmission:** atoms with any existing JSON (`ok`, `failed`, `started`) are skipped, so a resubmission runs only atoms that never started. This also avoids duplicates while jobs are still running. To rerun an atom deliberately, delete its JSON.
- Per-cell predictions / probabilities: off in tuning.

**SLURM scripts: `slurm_scripts/tune_<array>.sh`, one per (env, cluster) array**
- Scripts: `tune_sklearn_cpu.sh`, `tune_flagx_cpu.sh`, `tune_gatemeclass_cpu.sh`, `tune_lda_cpu.sh`, `tune_deepcytof_cpu.sh` (Woody) and `tune_flagx_gpu.sh` (TinyGPU).
- Each script:
  - has the resources as `#SBATCH` lines: `--time=24:00:00`, `--export=NONE`, `--cpus-per-task`, `--mem`, and `--gres=gpu:1` for the GPU array;
  - runs `unset SLURM_EXPORT_ENV`, loads the module and activates its conda env, sets `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`;
  - runs `python tune_hyperparameters.py run <array>`.
- The array range is not hard-coded. It is passed at submission from the manifest size.
- Usage:
  1. `python tune_hyperparameters.py manifest <array>` prints N.
  2. `sbatch --array=0-$((N-1))%<throttle> slurm_scripts/tune_<array>.sh`, or `sbatch.tinygpu …` for `flagx_gpu`.
  3. To resume, resubmit the same command.
- Woody: `--mem` ≤ 7.75 GB per core. TinyGPU: GPU type (rtx3080 / a100) to be chosen.
- `MaxArraySize` = 10000 on both Woody and TinyGPU (checked by the user, 2026-11-04). Every array fits into one submission.
- Logs go to `results/tune_hyperparameters/logs/<array>/%A_%a.out`. Repo, data and envs live on the cluster (`$WORK`); paths come from `config.ROOT`.
- `slurm_scripts/README.md`: the submit / resume commands per array.

**Later (not implemented now):** an aggregation script reads manifests + atoms. It computes macro F1 per val sample (D8 rule) → inner means → per-fold and consensus selection (D6), and reports failures / missing atoms and plots.

## 6. Grids (atoms = configs × 15)

The grids are defined in `GRIDS` of `tune_hyperparameters.py`; this section holds their justification (values, fixed parameters, defaults, sources). Inline comments in the script only point to legacy code. Principle (grid review 2026-11-05): each grid is defined around the method's default / published configuration, which is always part of the grid.

**FLAG-X SOM** (28 configs, 420 atoms; revised in the grid review 2026-11-05)
- Tuned: `som_dimensions` {15², 20², 25², 30²} × `n_epochs` {50, 100, 250, 500, 750, 1000, 2000}.
- Fixed at the values selected in the original tuning (legacy/main.py L928-943): planar, rectangular, gaussian with `gaussian_neighborhood_sigma=0.1`, `initialization='pca'`, exponential radius cooling and learning-rate decay, `learning_rate_0=0.1`, `learning_rate_n=0.001`, `radius_0=-0.5` (negative = fraction of the map size, flagx), `radius_n=0.1`.
- Default: the paper config (25², 1000 epochs, legacy schedule), inside the grid.
- Rationale (legacy results, `legacy/results/som_parameter_influence_study/`, `som_parameter_tuning/.../full_grid/`):
  - One-at-a-time study (10 % data, 3-fold CV), range of mean macro F1 per parameter: map size 0.096 (largest), `radius_n` 0.069 (only for values ≥ 2; flat for ≤ 1, 0.1 best), σ 0.033, `n_epochs` 0.027 (noise, no trend from 10 to 15,000), `radius_0` 0.022, initialization / topology 0.013–0.015; neighborhood, learning rates, grid type, decay and cooling ≤ 0.004 (cross-validation SD ~0.002).
  - Full grid (243 configs): share of score variance map size 0.19, `radius_0` 0.10 (the collapse of −0.25 on maps ≥ 20², F1 ≈ 0.86 with the fixed settings), σ 0.02, learning rates ≤ 0.01; −0.5 best at 25².
  - Epoch calibration (all training events, tuned config): flat from 10 to 6000 epochs (0.9716–0.9749).
  - So map size is the most influential parameter; the fixed parameters had a negligible effect or are fixed at their optimum. Epochs are tuned not because they were influential but because the earlier calibration used a different setting (event-level split, other training size) and they drive the runtime (local training, scalability).
- Caveats (to state in the revision): the legacy validation was at the event level (optimistic); one-at-a-time variation misses interactions; σ had a moderate effect (0.033); the fixed values were selected on the old Imstat training set, which may overlap with today's outer test folds, so the Imstat per-fold scores are slightly optimistic (accepted: Imstat results are read with caution anyway; other datasets are not affected).
- Suggested wording: "Based on the sensitivity analysis and grid search of the original study, map size had the largest effect on performance, whereas learning rates, decay functions, grid type and neighborhood function changed macro F1 by less than 0.005. We therefore fixed these at the previously selected values and tuned map size and the number of epochs. The number of epochs was re-tuned because the earlier calibration used a different training set size, and it determines training time."

**FLAG-X MLP** (36 configs, 540 atoms; revised in the grid review 2026-11-05)
- Tuned: `layer_sizes` {(64, 32, 16), (128, 64, 32), (256, 128, 64)} (small / medium / large, depth fixed at 3) × `lr` {1e-4, 1e-3, 1e-2} × `batch_size` {128, 1024} × schedule:
  - 20 epochs, no early stopping, `validation_fraction=0.0` (all training events);
  - early stopping on 10 % of the training events (`validation_fraction=0.1`, `n_iter_no_change=5`, flagx `tol` 1e-4), `n_epochs=1000` as a safety cap; flagx restores the weights of the best epoch and does not refit on all events.
- Not tuned: no explicit regularization (Adam default, no weight decay; flagx's MLP has no dropout); ReLU, Adam, early-stopping patience / tolerance fixed by flagx; no class weighting (separate ablation, PLAN.md step 8).
- Default: the paper config ((128, 64, 32), lr 1e-3, batch 128, 20 epochs on all events). The flagx version used for the paper had no validation split; with flagx 0.2.1 `validation_fraction=0.0` reproduces it (flagx 0.2.1 splits off `validation_fraction` = 0.1 by default even without early stopping, checked).
- Training course per atom in the JSON (`fit_info`: epochs trained, early stopped, best epoch, loss curves).

**DGCyTOF** (54 configs, 810 atoms; revised in the grid review 2026-11-05)
- Tuned: `layer_sizes` {(64, 32, 16), (128, 64, 32), (256, 128, 64)} (small / medium / large, as for the MLP) × `n_epochs` {10, 20, 50} (no early stopping in DGCyTOF) × `lr` {1e-4, 1e-3, 1e-2} × `batch_size` {128, 1024} (as for the MLP).
- `lr`: hard-coded `lr=0.001` in the official package (`DGCyTOF_Package/DGCyTOF/__init__.py` L118); exposed in the wrapper as a marked addition, default unchanged. The authors themselves searched lr {0.01, 0.001, 0.0001, 0.00001} × batch size {128, 256, 512} (100 epochs) in their notebooks (cells 20–21, commented out, no results reported): [CyTOF1.ipynb](https://github.com/lijcheng12/DGCyTOF/blob/6ff36677e8b341e8f0c7154e13d740062861cbb6/Code_Study/DGCyTOF/CyTOF1/CyTOF1.ipynb), [CyTOF2.ipynb](https://github.com/lijcheng12/DGCyTOF/blob/6ff36677e8b341e8f0c7154e13d740062861cbb6/Code_Study/DGCyTOF/CyTOF2/CyTOF2.ipynb).
- Fixed: `val_size=0.2`, the share of the training events held out for DGCyTOF's validation / calibration step (its results set the confidence threshold of the built-in abstention). **Not tuned by the authors:** their code uses `test_size=0.20` (stratified, cell 9 of both notebooks); the paper says "half-half" (Methods, Step 1); no other value appears anywhere.
- Default: the wrapper defaults = package / notebook settings ((128, 64, 32), 20 epochs, lr 1e-3, batch 128, `val_size` 0.2).
- Other findings (notebooks @ `6ff3667`): epochs 20 (CyTOF1) and 50 (CyTOF2); the saved models they load are named `40-60_50epochs_trained_model.pt` / `40-60_100epochs_trained_model.pt`, i.e. trained with other settings than the notebooks show. The paper's particle-swarm optimisation of layers / regularization / dropout is not in the published code.
- Cost: the calibration step dominates (~48 min per atom on CPU in the timing run) and may not use the GPU (unchecked); whether DGCyTOF runs on TinyGPU or Woody is decided with the array resources at the end.

**GateMeClass** (18 configs, 270 atoms; reviewed 2026-11-05, unchanged)
- Grid around the method's defaults: `gmc_gmm_parameterization` {E, V} (all values) × `gmc_k` {10, 20, 40} × `gmc_sampling` {0.05, 0.1, 0.2}; fixed `gmc_sampling_imp_vars=0.05`.
- Default: E, k 20, sampling 0.1, `sampling_imp_vars` 0.05 = the function defaults of `GateMeClass_train` / `GateMeClass_annotate` (repo @ `a93e020`, [R/GateMeClass.R](https://github.com/simo1c/GateMeClass/blob/a93e0202336f7048324c7c8b9b60ee965104f4d5/R/GateMeClass.R)); k 20 and sampling 0.1 are also used in every tutorial example ([tutorial/readme.md](https://github.com/simo1c/GateMeClass/blob/a93e0202336f7048324c7c8b9b60ee965104f4d5/tutorial/readme.md) L73-76, L170-172, L236-238, L267-268).
- How the method works (code @ `a93e020`):
  1. `GateMeClass_train`: per marker a 1-D Gaussian mixture (mclust) splits the reference cells into −/+ (or −/m/+), giving each cell a +/− signature; for every pair of cell types a decision tree (rpart via caret) finds the marker that best separates them → marker table (each cell type as a few signs, e.g. CD3+CD4+).
  2. `GateMeClass_annotate`: a random subset of the test sample (`sampling`) is taken; the per-marker mixtures are fitted again on that subset (on the test data itself); cells whose signature matches a marker-table entry get that label; all other cells (unmatched ones and those outside the subset) are labelled by a k-NN classifier trained on the matched cells.
- Parameters:
  - `gmc_gmm_parameterization` (train + annotate): variance model of the per-marker mixtures, E = equal, V = variable variance per component. V also switches on ranked set sampling (RSS: the mixture is fitted on a ranked subsample, the rest is predicted); this is the only other effect of E/V. Paper: "The choice of the variance parameter strictly depends on the data to model" (Sec. 2.1); "the best variance parameterization must be carefully chosen by the user looking at the data distribution and with testing" (Conclusion); flow cytometry (Bost et al. 2021): "we used GateMeClass with an equal variance parameter" (Sec. 3.2). The tutorial uses V (mass cytometry only). V crashes on the shared input (Mclust on the many values at the log10 cut-off, `legacy/gatemeclass_bug_report.md`; reproduced in the smoke test); such a failure is a result of the grid value (config out), not worked around. More sources: `DONE.md`, step 6.
  - `gmc_sampling` (annotate): share of a test sample's cells annotated via the marker table; the rest is labelled by k-NN (0.1 → 90 % of the cells get their label from k-NN). Applied before the mixtures in both E and V. The only parameter the authors varied systematically: 10 %–100 % (Sec. 3.1): "using only the 10% of the cells results in the best accuracy in many cases. Therefore, this sampling percentage was set as the default configuration of GateMeClass". Our grid stays around the default (0.05, 0.1, 0.2) instead of the authors' 10–100 % range (decision: grids are defined around the methods' defaults).
  - `gmc_k` (annotate): number of neighbours of the k-NN refinement (and of the MNN step with `reject_option=TRUE`, off here, D8). Independent of E/V. No value is reported in the paper and there is no sensitivity analysis; tutorial and function default: 20. Grid: low / mid / high around 20.
  - `gmc_sampling_imp_vars` (train): cap on the number of cells per cell type in each pairwise decision tree, `floor(sampling_imp_vars × total reference cells)`; larger cell types are subsampled to it. Independent of E/V. Not mentioned in the paper or in the tutorial's parameter list; only a function default (0.05). **Fixed, not tuned:** at full training data (~2.5M events, `TRAIN_FRAC = 1.0`) the cap is ~125,000 cells per type and pair, so nearly all Imstat cell types enter the trees completely (only the largest, granulocytes, are subsampled); it acts as a speed cap, and keeping the undocumented default is the faithful choice.
- Paper: Caligola et al. 2024, Bioinformatics 40(5):btae322, doi:10.1093/bioinformatics/btae322 (full text read at PMC11136448; supplementary not accessible). Further notes: the benchmark numbers in the paper are the best of 10 random runs ("For each tool, the best run, in terms of accuracy, among 10 random executions was reported"); all tools were run with default settings.

**LDA-Abdelaal** (1 config, 15 atoms; reviewed 2026-11-05)
- Only config: `transformation=False` (shared input, the method's own transformation off), `rejection_threshold=0`. Nothing to tune; kept in the tuning for completeness (inner scores next to the other methods; trivially selected).
- Internal transformations of the official code (`CyTOF_LDAtrain.R` L69–77 @ `19c2f8f`) are not run:
  - `'log'`: natural log without cut-offs (−Inf → 0), a cruder version of the shared log10 with cut-offs; also gives NaN on negative values (fails on Imstat).
  - `'arcsinh'`: `asinh(x/5)` with the cofactor 5 hard-coded (mass-cytometry setting; flow data would need ~150). Changing it would modify the authors' preprocessing, so it is not a native arm anymore.
- No other hyperparameters: `CyTOF_LDAtrain` only takes data, labels, `RelevantMarkers` (all channels used), `LabelIndex` and `Transformation`; `MASS::lda` runs with its defaults (prior = class proportions, method "moment"); exposing its options would be our addition (a tuned LDA is covered by LDA-sklearn).
- `rejection_threshold` (cells with max posterior below it → unknown, −1) not tuned: D8 (full assignment, optional abstention off for all methods); under the D8 scoring rule a tuned threshold could raise F1 by abstaining on uncertain wrong predictions (removes an FP, no extra FN). Abstention behaviour is a step 8 analysis.

**DeepCyTOF** (24 configs, 360 atoms; revised in the grid review 2026-11-05)
- Sources: code @ `ac924b1` (links below), paper bioRxiv v5 (§ numbers; details in `working_notes/deepcytof.md`).
- Grid: input arm {shared + `preprocess_log=False`, raw + `preprocess_log=True`} × `calibrate` {True, False} × `hidden_layer_sizes` {(12,6,3), (128,64,32)} × `lr_initial` {1e-4, 1e-3, 1e-2}.
- Default = paper / code: `preprocess_log=True`, `calibrate=True`, (12, 6, 3), lr 1e-3.
- Fixed (code values): `denoise=False` (not used by the authors on flow data, see `deepcytof.md`); softplus, l2 1e-4, RMSprop, lr halved every 50 epochs, ≤ 80 epochs with early stopping (10 % validation, patience 25; last weights kept, no restore), batch 128 ([`Util/feedforwadClassifier.py#L30-L38`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py#L30-L38), [`#L75-L117`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py#L75-L117)); `rejection_threshold=0` (D8).
- The authors tuned nothing: §3.4 only states the values ("the hidden layer sizes have been set to 12, 6, and 3"; "All networks were trained using RMSprop"; classifiers "μ(0) = 10−3, δ = .5, and T = 50"; "mini-batches of size 128"); epochs and patience are only in the code. Batch size is therefore not tuned (with the 80-epoch cap, 1024 would give far fewer updates).
- Architecture: only the published default and the FLAG-X MLP default, so the comparison shows whether DeepCyTOF's own parts (one reference sample, z-scoring on it, softplus + l2, RMSprop schedule, calibration) add anything over MLP / DGCyTOF.
- Input arm: `preprocess_log=True` = `log(1 + |x|)` on the raw data ([`Util/DataHandler.py#L25-L27`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py#L25-L27)); the internal z-scoring on the reference stays in both arms. A legitimate native arm (unlike LDA-Abdelaal's): it is the authors' flow preprocessing (§2.1.3: "For FlowCAP-I datasets, we apply a logarithmic transform, followed by rescaling, as described in Appendix A."). After z-scoring the log base does not matter; the arms differ only at low values (no cut-offs, negatives mirrored by `|x|`).
- `calibrate` (MMD-ResNet per test sample onto the reference): tuned as a defining feature of the method, although the authors' usage argues against it here. It targets run-to-run / instrument batch effects, and the authors skip it when these are absent or small:
  - Purpose (§2.2.2): "To account for machine based technical bias and variability, we include a preprocessing option in DeepCyTOF to calibrate each batch to a reference using the MMD-ResNet approach."
  - Off for FlowCAP-I (§2.3): "the MMD-ResNet option is also disabled because the training and test sets are from the same run and thus do not require calibration."
  - Off for the single-instrument CyTOF datasets of §3.2 (§2.3): "Batch effects in these two datasets were not substantial allowing classification of cells into major cell populations without employing a calibration step".
  - On for the multi-center data (§3.3): "The multi-center dataset consists of samples of the same subject, which were measured on different CyTOF machines in different locations. It is therefore reasonable that the different instrument conditions will result in calibration differences, which need to be accounted for."
  - Our samples come from one dataset (one centre, one panel), like the single-instrument case, but are separate runs (unlike FlowCAP-I), so residual batch effects are possible; the tuning decides.
  - MMD learning rate: the code's 1e-5 (`initial_lrate = 1e-5#1e-3`, [`Util/MMDNet.py#L27-L32`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/MMDNet.py#L27-L32)), not the paper's 1e-3 (§3.4: "μ(0) = 10−3, δ = .1, and T = 15"), to keep the authors' code. With it, calibration changed F1 by ≤ 0.002 in the real-data check, but dominates the runtime (timing run: 91 min per atom).
- Without calibration, DeepCyTOF is a small softplus MLP trained on one reference sample, chosen among the training samples by the original covariance criterion ([`Util/DataHandler.py#L59-L77`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py#L59-L77)); classes absent from it cannot be predicted.

**RF** (54 configs, 810 atoms; revised in the grid review 2026-11-05)
- sklearn `RandomForestClassifier` (scikit-learn 1.9.1, env `fx_sklearn`). Grid around the sklearn defaults; 10–12 channels per dataset; ~2.5M training events per inner split (`TRAIN_FRAC = 1.0`).
- Default = sklearn defaults (with `n_estimators=100`): `max_depth=None`, `min_samples_leaf=1`, `max_features='sqrt'`, `criterion='gini'`.

| Tuned | sklearn default | Values | Reason |
|---|---|---|---|
| `max_depth` | None | {10, 20, None} | Main complexity control: None (fully grown) is the default, 20 / 10 a moderate / strong limit. With ~2.5M events, unlimited trees get very large; a limit may generalise better across patients and saves memory. |
| `min_samples_leaf` | 1 | {1, 10, 100} | Second regulariser (smoother leaves); unlike depth it acts locally, in sparse regions (rare classes, class boundaries). Low / mid / high on a log scale; 100 is still small relative to ~2.5M events. |
| `max_features` | `'sqrt'` | {sqrt, 0.5, None} | Features tried per split, controls the diversity of the trees. For 10–12 channels: sqrt = 3, 0.5 = 5–6, None = all (plain bagging) → low / mid / high. `'log2'` left out: also 3 for 10–12 channels. |
| `criterion` | `'gini'` | {gini, entropy} | Categorical: all distinct values. `'log_loss'` left out: identical to `'entropy'` in sklearn. |

| Fixed | Value | Reason not tuned |
|---|---|---|
| `class_weight` | None (default) | Class weighting is a follow-up experiment (PLAN.md step 8), as for the MLP. |
| `n_estimators` | 100 (default) | More trees do not overfit; they only reduce variance with diminishing returns, at the cost of time and memory. The default halves the memory of fully grown trees compared with 200. |
| `min_samples_split` | 2 (default) | Largely redundant with `min_samples_leaf` (a split already needs ≥ 2 × `min_samples_leaf` events). |
| `max_leaf_nodes`, `min_impurity_decrease`, `min_weight_fraction_leaf` | None / 0 / 0 (defaults) | Alternative pre-pruning (stopping) rules for the tree size; they overlap with `max_depth` / `min_samples_leaf`, so only that pair is tuned. |
| `ccp_alpha` | 0 (default, no pruning) | Minimal cost-complexity post-pruning: after a tree is grown, subtrees whose impurity reduction per extra leaf is below α are pruned. Not tuned: another size regulariser (overlaps with depth / leaf size); rarely used in forests, where averaging already controls the variance of deep trees; no natural scale for α (depends on the impurity of the data); the pruning path on fully grown trees from ~2.5M events adds runtime. |
| `bootstrap` | True (default) | Part of the random-forest definition; without it every tree sees the same data and diversity comes only from `max_features`. |
| `max_samples` | None (default: bootstrap of the training-set size) | Mainly a speed knob; smaller per-tree samples would be our addition. |
| `oob_score`, `warm_start`, `verbose`, `monotonic_cst` | defaults | Do not change the fitted model, or need domain constraints we do not have (`monotonic_cst`). |
| `random_state`, `n_jobs` | task seed, job threads | Runtime settings, set by the runner. |

- Memory risk: fully grown trees (`max_depth=None`, `min_samples_leaf=1`, the default config) on ~2.5M events can have hundreds of thousands of nodes each, i.e. several GB per forest; an OOM kill would put the default config out (failure rule) → generous `--mem` for `sklearn_cpu`.

**HGB** (27 configs, 405 atoms; revised in the grid review 2026-11-05)
- sklearn `HistGradientBoostingClassifier` (scikit-learn 1.9.1, env `fx_sklearn`). Grid around the sklearn defaults; 10–12 channels, 8 Imstat classes, ~2.5M training events per inner split.
- Default = sklearn defaults: `learning_rate=0.1`, `max_leaf_nodes=31`, `l2_regularization=0`, with early stopping (on by default for > 10,000 samples) and the cap `max_iter=1000` (default 100, see below).

| Tuned | sklearn default | Values | Reason |
|---|---|---|---|
| `learning_rate` | 0.1 | {0.01, 0.1, 0.3} | Shrinkage per boosting step, the most important boosting parameter: 0.01 slow (needs many iterations), 0.3 aggressive; 1.0 left out (usually unstable). Interacts with the number of iterations, which early stopping decides. |
| `max_leaf_nodes` | 31 | {15, 31, 63} | Tree size; HGB grows trees leaf-wise, so this is the main complexity control. ~half / default / double. |
| `l2_regularization` | 0 | {0, 1, 10} | L2 penalty on the leaf values (shrinks updates of leaves with few events): default / moderate / strong. Often a minor effect, but the only regulariser that does not limit the tree structure. |

| Fixed | Value | Reason not tuned |
|---|---|---|
| `class_weight` | None (default) | Class weighting is a follow-up experiment (PLAN.md step 8), as for RF and the MLP. |
| `max_iter` | 1000 (default 100) | Only a cap for early stopping: generous (like the MLP's early-stopping ceiling of 1000), so that early stopping and not the cap ends the training, also at lr 0.01 (with the default 100, lr 0.01 would be cut short). `fit_info` stores `n_iter` / `early_stopped`, so the aggregation can check whether the cap was hit. |
| `early_stopping` | True | Identical to the default (`'auto'` = on for > 10,000 samples); explicit so tiny smoke-test data behave the same. |
| `validation_fraction`, `n_iter_no_change`, `tol`, `scoring` | 0.1, 10, 1e-7, `'loss'` (defaults) | Early-stopping settings, not model parameters. |
| `min_samples_leaf` | 20 (default) | Practically never binds: with ~2.5M events and ≤ 63 leaves, an average leaf holds tens of thousands of events; the tree size is controlled by `max_leaf_nodes`. |
| `max_depth` | None (default) | Redundant: leaf-wise growth is already limited by `max_leaf_nodes`. |
| `max_features` | 1.0 (default) | Feature subsampling per split; little room with 10–12 features, and boosting does not rely on tree diversity like RF. |
| `max_bins` | 255 (default, the maximum) | Resolution of the feature binning; already the finest, coarser bins only save time. |
| `loss` | `'log_loss'` | The only classification loss. |
| `categorical_features`, `interaction_cst`, `monotonic_cst` | defaults | No categorical features, no domain constraints. |
| `warm_start`, `verbose`, `random_state` | defaults / task seed | Runtime settings (the seed is set by the runner). |

**LogReg** (7 configs, 105 atoms; revised in the grid review 2026-11-05)
- sklearn `LogisticRegression` (scikit-learn 1.9.1, env `fx_sklearn`): with lbfgs a multiclass problem is always one multinomial model (`multi_class` removed in 1.8); `penalty` is deprecated (`l1_ratio` selects the penalty, `C=np.inf` = no penalty).
- Default = sklearn defaults: `C=1` (L2), lbfgs; `max_iter=1000` (default 100, see below).
- Scale of `C`: sklearn minimises ½‖w‖² + C · Σᵢ lossᵢ (a sum over events), so the per-event penalty weight is ~1/(C·n). With n ≈ 2.5M training events and 10–12 features: C = 1 → ~4e-7 (no effect), C = 1e-3 → ~4e-4, C = 1e-5 → ~0.04 (noticeable). Values above 1 are practically identical to C = 1; with so few parameters and so many events, regularisation hardly matters for this model.

| Tuned | sklearn default | Values | Reason |
|---|---|---|---|
| `C` | 1.0 | {1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, None} | Inverse L2 strength, the only real hyperparameter. Log scale from the default down to where the penalty starts to act (see above); 10 / 100 dropped (identical to C = 1 at this n). `None` = no penalty at all (completely unregularised model; runner: `C=inf`), kept as the reference case. |

| Fixed | Value | Reason not tuned |
|---|---|---|
| `class_weight` | None (default) | Class weighting is a follow-up experiment (PLAN.md step 8), as for RF, HGB and the MLP. |
| `l1_ratio` (penalty type) | 0 = L2 (default) | lbfgs supports only L2; L1 / elastic net need saga, which converges poorly on unscaled input; with 10–12 features the feature selection of L1 is pointless. |
| `solver` | `'lbfgs'` (default) | Optimiser, not model: the problem is convex, so every solver reaches the same solution if it converges. |
| `max_iter` | 1000 (default 100) | Convergence cap; lbfgs on unscaled input may need more than 100 iterations. A non-converged fit still counts as `ok` (the sklearn warning is not stored). |
| `tol` | 1e-4 (default) | Convergence criterion, not a model parameter. |
| `fit_intercept` | True (default) | Without intercept the model is misspecified (input not centred). |
| `intercept_scaling`, `dual` | defaults | Only used by liblinear. |
| `warm_start`, `verbose` | defaults | Runtime settings. |
| `random_state` | task seed | Only used by sag / saga / liblinear; no effect with lbfgs. |
| `n_jobs` | not set | No effect with lbfgs multinomial (threads via BLAS / `OMP_NUM_THREADS`). |

**kNN** (20 configs, 300 atoms; revised in the grid review 2026-11-05)
- sklearn `KNeighborsClassifier` (scikit-learn 1.9.1, env `fx_sklearn`); deterministic (no seed). Shared input (log10 with cut-offs, no scaling, D3), so all channels are on a similar log10 scale. Fit only stores the ~2.5M training events; the cost is in predict (extrapolated from the timing run, not measured: up to ~1–2 h per atom at k = 50, more at k = 150).
- Default = sklearn defaults: `n_neighbors=5`, `weights='uniform'`, `p=2`.

| Tuned | sklearn default | Values | Reason |
|---|---|---|---|
| `n_neighbors` | 5 | {1, 5, 15, 50, 150} | The main (and only real) complexity control: small k = local fit, large k = smoothing. Roughly log-spaced (factor ~3) around the default; 30 (old grid) dropped (close to 15 and 50); 150 added at the top, since even k = 50 is very local with ~2.5M training events. |
| `weights` | `'uniform'` | {uniform, distance} | Categorical: all values. `'distance'` weights closer neighbours more, mostly relevant at large k. Test events that coincide with training events (frequent at the log10 cut-off) give those training events weight 1 and all others 0 (sklearn rule), no division by zero. |
| `p` | 2 | {1, 2} | Minkowski exponent: 1 = Manhattan, 2 = Euclidean (default); the two standard distances. Manhattan is less dominated by a single channel with a large difference. |

| Fixed | Value | Reason not tuned |
|---|---|---|
| `metric` | `'minkowski'` (default) | Covers L1 / L2 via `p`; other metrics (Chebyshev, Mahalanobis, …) are unusual here, and Mahalanobis would effectively rescale the input, which is fixed by D3. |
| `metric_params` | None (default) | Only for metrics with extra parameters. |
| `algorithm` | `'auto'` (default; kd-tree for dense low-dimensional data) | Only the search speed: all algorithms return the exact neighbours (with exactly tied distances the kept neighbour may differ, not a model choice). |
| `leaf_size` | 30 (default) | Speed / memory of the tree only. |
| `n_jobs` | job threads | Runtime setting, set by the runner. |

- No class weighting option in `KNeighborsClassifier` (nothing to fix for the step 8 follow-up).

**LDA-sklearn** (6 configs, 90 atoms; revised in the grid review 2026-11-05)
- sklearn `LinearDiscriminantAnalysis` (scikit-learn 1.9.1, env `fx_sklearn`); deterministic (no seed). 10–12 channels, ~2.5M training events per inner split.
- Default: no shrinkage (`shrinkage=None`) with `solver='lsqr'` = the same model as the sklearn default (`solver='svd'`, which cannot shrink). Checked: `shrinkage=None` and `0.0` give identical coefficients and predictions with lsqr, so None is the shrinkage-0 extreme.
- Solvers: all three fit the same classifier (Gaussian classes with a shared covariance, same decision rule) and differ only in the computation. `'svd'` (default) avoids the covariance matrix and is recommended by the docs "for data with a large number of features" (not our case), but does not support shrinkage; `'lsqr'` solves Σ w = μ_k directly (cheapest, classification only, supports shrinkage); `'eigen'` gives the same predictions as lsqr and is only needed for `transform`. One solver (lsqr) for all configs, so the configs differ only in the shrinkage; with 10–12 features and ~2.5M events the covariance is well conditioned, so svd and lsqr give the same predictions up to rounding. The old grid (svd, or {lsqr, eigen} × shrinkage, 11 configs) contained 6 duplicate models.

| Tuned | sklearn default | Values | Reason |
|---|---|---|---|
| `shrinkage` | None (= 0) | {None, 0.1, 0.5, 0.9, 1, auto} | The only parameter that changes the model. sklearn standardises the features, shrinks their covariance toward the identity and scales back: 0 = empirical covariance, 1 = per-feature variances only (diagonal LDA, no correlations). Both extremes are included: None = 0 = default (no shrinkage), 1 = diagonal LDA; 0.1 / 0.5 / 0.9 = low / mid / high regularisation in between; `'auto'` = Ledoit-Wolf (data-driven; probably close to 0 here, since the covariance of 10–12 features from ~2.5M events is estimated very precisely). |

| Fixed | Value | Reason not tuned |
|---|---|---|
| `solver` | `'lsqr'` | See above: same model for all solvers; lsqr supports shrinkage and is the cheapest. |
| `priors` | None = class proportions of the training data (default) | Uniform priors amount to class weighting, a follow-up experiment (PLAN.md step 8). |
| `covariance_estimator` | None (default) | Alternative covariance estimator (e.g. OAS) for lsqr / eigen; overlaps with `shrinkage` (Ledoit-Wolf already covered by `'auto'`). |
| `n_components` | None (default) | Only used by `transform` (dimensionality reduction); no effect on `predict`. |
| `tol` | 1e-4 (default) | Rank threshold of the svd solver only (not used with lsqr). |
| `store_covariance` | False (default) | Only stores the covariance; no effect on the model. |

**GMM-Mocking** (20 configs, 300 atoms; reviewed 2026-11-05, grid unchanged, `max_iter` cap raised)
- Wrapper `validation/gating/gmm_mocking.py` (env `fx_sklearn`, scikit-learn 1.9.1) around the authors' `GMMClassifier` (Mocking et al. 2024, repo `AUMC-HEMA/cMRD-manuscript` @ `df1b26e`, [GMMClassifier.py](https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/GMMClassifier.py)): one `GaussianMixture` per class, prediction = argmax of the per-class log-likelihoods. The wrapper exposes all scalar `GaussianMixture` parameters with sklearn defaults (the original only sets `n_components`, `covariance_type` and `random_state`, so every other value is a sklearn default there as well).
- Authors' tuning ([02-GMMclf.ipynb](https://github.com/AUMC-HEMA/cMRD-manuscript/blob/df1b26e6035f22af6b32f8881bad1636f04acabd/02-GMMclf.ipynb), cells 7 and 14): `GridSearchCV` over `n_components_class0` 1–20 × `n_components_class1` 1–10 (10-fold `GroupKFold`, refit on F1), `covariance_type='full'` fixed, 5K events per sample (BLAST110). `random_state` is effectively None in the original (L11 drops it).
- Our generalisation: one shared `n_components` for all classes (per-class values would be 5⁸ combinations for 8 Imstat classes). The decision rule has no class priors (= uniform priors, an implicit class balancing inherent to the method).
- ~2.5M training events per inner split (500× the authors' 5K per sample): each per-class mixture is fitted on up to ~1M events (largest class).
- Default = original default: 1 component per class, full covariance (with `max_iter=1000`, see below).

| Tuned | Default (original / sklearn) | Values | Reason |
|---|---|---|---|
| `n_components` (per class, shared) | 1 | {1, 2, 5, 10, 20} | The only parameter the authors tuned; from the default to their maximum (20), roughly log-spaced. 1 component per class = QDA (Gaussian classes with own covariances); more components model multimodal classes; with ~2.5M events even 20 full components per class are well determined. |
| `covariance_type` | `'full'` | {full, tied, diag, spherical} | Categorical: all values (the authors fixed `'full'`). full = free covariance per component; tied = one covariance per class shared by its components; diag / spherical are cheaper and less flexible (can approximate correlations with more components). Cheap (timing run: atom well under 0.1 min at 10 % data). |

| Fixed | Value | Reason not tuned |
|---|---|---|
| `tol` | 1e-3 (sklearn default, as in the original) | EM convergence criterion, not a model parameter. |
| `max_iter` | 1000 (default and original: 100) | Only a cap: EM stops when the per-event mean log-likelihood improves by less than `tol`, independent of the number of events. Few components converge fast, but with 10–20 full components EM often creeps and 100 can bind; sklearn then only warns and uses the unconverged mixture. Generous cap, consistent with HGB / LogReg / MLP (the original used 100 on 5K events per sample). Cost only for fits that would stop at 100 (worst case ~1–2 s per iteration on ~1M events with 20 full components, i.e. a few hours per atom, < 24 h). `fit_info` stores per class `converged` / `n_iter`, so the aggregation can check whether the cap binds. |
| `reg_covar` | 1e-6 (default, as in the original) | Numerical safeguard against singular covariances (e.g. components collapsing on the many identical values at the log10 cut-off); tuning it would be a model change the authors did not make. A failing fit puts the config out (failure rule). |
| `n_init` | 1 (default, as in the original) | EM restarts: only reduce the dependence on the initialisation, at multiple cost. |
| `init_params` | `'kmeans'` (default, as in the original) | Initialisation method. |
| `random_state` | task seed | Seed of the mixtures (our change: the original drops it, L11). |
| `verbose`, `verbose_interval` | 0, 10 (defaults) | Logging only. |
| `weights_init`, `means_init`, `precisions_init`, `warm_start` | not exposed (sklearn defaults) | The init arrays are specific to one mixture and cannot be shared by the per-class mixtures; `warm_start` has no effect, since `fit` creates new mixtures. |
| Class priors / weighting | none (uniform, inherent to the decision rule) | Part of the original method; class weighting is a follow-up experiment (PLAN.md step 8). |

**Atoms per array**

| Array | Atoms |
|---|---|
| `sklearn_cpu` | 227 × 15 = 3405 |
| `flagx_cpu` | 64 × 15 = 960 |
| `flagx_gpu` | 126 × 15 = 1890 |
| `gatemeclass_cpu` | 270 |
| `lda_cpu` | 45 |
| `deepcytof_cpu` | 540 |

## 7. Docs (after implementation)
- `PLAN.md`:
  - step 4: methods list (+ sklearn baselines, GMM-Mocking, envs `fx_sklearn` / `fx_flagx_gpu`);
  - step 5: helpers written;
  - step 6: atoms / arrays / grids;
  - D3 wording: internal preprocessing as a grid parameter.
- `CLAUDE.md`: wrapper table (`gmm_mocking.py`), envs, `validation/core` modules, `validation/runners`, `tune_hyperparameters.py`, `slurm_scripts/`.
- No commits unless asked.

## Verification
1. Byte-compile `validation/core/*`, `validation/runners/deepcytof.py` and `tune_hyperparameters.py` in `fx_deepcytof` (Python 3.5).
2. Check the core helpers in a scratch script:
   - stratified downsampling keeps the class proportions and is seed-reproducible;
   - `join_samples` keeps `X`, `y` and `sample_ids` aligned;
   - `confusion_counts` / `class_counts` match `sklearn.metrics.confusion_matrix`, including -1;
   - `track_resources` registers a large numpy allocation.
3. `GMMMockingClassifier` on 2 classes gives the same predictions as upstream `GMMClassifier` (same `random_state` passed) on a BLAST110 subsample.
4. Create `fx_sklearn` (and `fx_flagx_gpu` when a GPU is reachable).
5. Smoke atoms: one atom per method in its env, on a tiny subsample, output to the scratchpad. Check that the JSON + counts files are complete and that the counts files concatenate.
6. Timing run: the default config, split 0 / inner 0, `TRAIN_FRAC=0.1`, per method. Report fit / predict time and memory, then propose `VAL_FRAC`, `--time` / `--mem` per array and final grid sizes. The full arrays are submitted by the user.
