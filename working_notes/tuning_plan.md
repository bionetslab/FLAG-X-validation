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

## 6. Proposed grids (for review; sizes checked against the timing run; atoms = configs × 15)

**FLAG-X SOM** (28 configs, 420 atoms; revised in the grid review 2026-11-05)
- Tuned: `som_dimensions` {15², 20², 25², 30²} × `n_epochs` {50, 100, 250, 500, 750, 1000, 2000}.
- Fixed at the values selected in the original tuning (legacy/main.py L928-943): planar, rectangular, gaussian with `gaussian_neighborhood_sigma=0.1`, `initialization='pca'`, exponential radius cooling and learning-rate decay, `learning_rate_0=0.1`, `learning_rate_n=0.001`, `radius_0=-0.5`, `radius_n=0.1`.
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

**DGCyTOF** (54 configs)
- `layer_sizes` {(64,32), (128,64,32), (256,128,64)}
- `n_epochs` {10, 20, 50}
- batch {128, 512}
- `val_size` {0.1, 0.2, 0.3}
- The learning rate is hard-coded in the method.

**GateMeClass** (18 configs)
- `gmc_gmm_parameterization` {E, V}
- `gmc_k` {10, 20, 40}
- `gmc_sampling` {0.05, 0.1, 0.2}
- `gmc_sampling_imp_vars` 0.05 (optional ×3: {0.01, 0.05, 0.1}).

**LDA-Abdelaal** (3 configs)
- `transformation` {False (shared input), `'log'` (raw), `'arcsinh'` (raw)}.
- The wrapper has no other model hyperparameters; rejection is off (D8).

**DeepCyTOF** (36 configs)
- input arm {shared + `preprocess_log=False`, raw + `preprocess_log=True`}
- `calibrate` {True, False}
- `hidden_layer_sizes` {(12,6,3), (24,12,6), (48,24,12)}
- `lr_initial` {1e-4, 1e-3, 1e-2}
- Fixed: `denoise=False`, rest original.

**RF** (108 configs)
- `max_depth` {10, 20, None}
- `min_samples_leaf` {1, 10, 100}
- `max_features` {sqrt, None}
- `class_weight` {None, balanced, balanced_subsample}
- `criterion` {gini, entropy}
- Fixed:
  - `n_estimators=200`: more trees only cost runtime;
  - `log2` is dropped: identical to `sqrt` for 10–12 channels;
  - `log_loss` is dropped: identical to `entropy`.

**HGB** (54 configs)
- `learning_rate` {0.01, 0.1, 0.3}
- `max_leaf_nodes` {15, 31, 63}
- `l2_regularization` {0, 1, 10}
- `class_weight` {None, balanced}
- Fixed: `max_iter=500` with early stopping (val 0.1, 10 iterations).

**LogReg** (14 configs)
- penalty l2 with `C` {1e-3, 1e-2, 0.1, 1, 10, 100}, or no penalty **[impl]** (`C: null` in the grid → `C=inf` in the runner; sklearn 1.9 deprecates `penalty`)
- `class_weight` {None, balanced}
- Fixed: lbfgs, multinomial, `max_iter=1000`.
- l1 / elasticnet are excluded: they need saga, which converges poorly on unscaled input.

**kNN** (20 configs)
- `n_neighbors` {1, 5, 15, 30, 50}
- `weights` {uniform, distance}
- `p` {1, 2}

**LDA-sklearn** (11 configs)
- solver svd, or solver {lsqr, eigen} × `shrinkage` {None, auto, 0.1, 0.5, 0.9}

**GMM-Mocking** (20 configs)
- `n_components` {1, 2, 5, 10, 20} (their range was 1–20)
- `covariance_type` {full, tied, diag, spherical}
- Fixed: `n_init=1`, `max_iter=100`, `reg_covar=1e-6` (sklearn defaults, as upstream).

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
