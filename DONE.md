# Done: log of completed work

What was implemented per step of `PLAN.md`, with outputs, checks, findings, and the reasoning and sources behind each decision. Append only, in chronological order; `PLAN.md` keeps the rules, the open todos and a one-line summary of each decision. Detailed notes are linked, not copied (`working_notes/`).

---

## Step 0: Repository scaffolding (2026-11-03)
- Scripts at the repo root with descriptive names, no numbering; figure scripts start with `plot_`. Each script has its parameters as constants at the top and an `OVERWRITE` flag.
- `config.py` (stdlib-only, Py3.5-compatible): `ROOT` (via `__file__`), `DATA_RAW`, `DATA_CURATED`, `RESULTS`, `FIGURES`, `DATASETS`, `CHANNELS`. No registries. **[D5]**
- Layout: `data/raw/` (gitignored), `data/curated/<dataset>/` (`.npy` gitignored, meta CSVs tracked), `data/splits/` (tracked), `results/<script>/`, `figures/<script>/`.
- Envs: `fx_flagx_cpu` (`envs/flagx_cpu.yml`, curation + FLAG-X runs), `fx_plot` (`envs/plot.yml`), one env per method. All with `channel_priority: strict`.

## Step 1: Data curation (2026-11-03)
- `curate_data.py` (env `fx_flagx_cpu`): one function per dataset. Load with flagx (one `FlowDataManager` per file) → compensate if the FCS file has a spillover matrix → rename channels (per-dataset map in the script) → keep/order as in `config.CHANNELS` → map labels (per-dataset map in the script, contiguous from 0) → save. Raises on unmapped channels/labels.
- Output in `data/curated/<dataset>/`: `sample_<NNN>_X.npy` (float32) and `sample_<NNN>_y.npy` (int32), `meta_samples.csv` (sample, original filename, extra info such as `patient_id`, `subtype`, `sample_type`), `meta_labels.csv` (label, original label, name, description). Data is stored untransformed. **[D2]**
- Datasets (10): `imstat`, `lt1`, `lt2`, `lt1b`, `lt2b`, `flowcyt`, `blast110_p1`–`p4`. **[D1]**
- Imstat sample meta (`donor_id`, `lysis`) comes from `data/raw/imstat/Imstat_samples_2026-10-29_clean.csv`, cleaned from the data provider's table (donors `a`–`l` for the `ER_` files; for the named files the `YYYYMMDD-n` token as written). 26 donors with up to 5 lysis reagents each (R4.4, R4.6).
- Compensation only for BLAST110 (only dataset with `$SPILL`). **[D7]**
- BLAST110: one dataset per tube, blast vs other, FSC-A/SSC-A only, channels named by antibody (`$PnS`).
- Overview figures: `plot_dataset_overview.py` (env `fx_plot`) → `results/dataset_overview/*.csv`, `figures/dataset_overview/` (sample sizes, pooled and per-sample class balance, optional hue by meta column, e.g. `sample_type`).

## Step 2: Splits (2026-11-03)
- `make_splits.py` (env `fx_flagx_cpu`): patient-level k-fold with scikit-learn (`StratifiedKFold` / `KFold`, `config.SEED`).
- Folds are assigned once per split group on the union of its patients, so a patient is in the same fold in every dataset of the group: `lymphoma` (lt1, lt1b, lt2, lt2b; stratified by `subtype`), `blast110` (p1–p4; stratified by `sample_type`), `imstat` (by `donor_id`), `flowcyt` (unstratified). Rare strata (BL: 2 patients) do not occur in every fold.
- Output in `data/splits/` (versioned in git): `<dataset>.csv` (sample, patient_id, [stratum], `split_<i>` = train/test), `imstat_inner.csv` (all samples, `split_<i>_inner_<j>` = train/val/test) for tuning, `summary.csv` and `summary_inner.csv` (sample counts per split).
- `K_OUTER = 5`, `K_INNER = 3` (fixed; more inner folds are computationally infeasible for tuning). One CV repetition; variability is covered by model seeds (step 7) and an optional repeated CV for the FLAG-X methods.

## Step 3: Preprocessing definition (2026-11-03)
- `validation/core/preprocessing.py`: `Preprocessor(channels, trafo, trafo_params, scaling, scaling_params)` with `fit` / `transform`. Transformations `log10_cutoff` (`{'cutoff': value or {channel: value}}`), `arcsinh` (`{'cofactor': value}`), `none`; scalings `zscore`, `minmax` (`{'feature_range': (a, b)}`, default (0, 1)), `none`.
- Numpy-only, Python 3.5-compatible reimplementation (no sklearn/flagx), so it runs in every method env. Checked against flagx `log10_w_custom_cutoffs`, pytometry `normalize_arcsinh` and the sklearn scalers.
- Fitted on the fly inside each task on the task's (possibly downsampled) training data; the transformed data is never stored. Each experiment states its preprocessing as constants. A fitted preprocessor can be saved next to the model with `save(path)` / `Preprocessor.load(path)` (JSON, loadable in every env).
- Cut-offs: `config.LOG10_CUTOFFS` (per dataset; a dict per channel or one uniform value).
- **Finding:** flagx 0.2.1 `log10_w_cutoff` (uniform cut-off, used for Flowcyt in the paper) sets values <= cutoff to ln(cutoff) instead of log10(cutoff) (4.61 instead of 2 for cut-off 100). This affects up to 5 % of the values per Flowcyt channel (HLA_PB 4.8 %, CD14_FITC 1.9 %). Our implementation uses log10(cutoff); the paper's Flowcyt numbers will change slightly. To fix in the flagx package.
- **[D3]** Default: the same input for every method, i.e. the shared preprocessing (D3a, D3b). Deviations are allowed only for two reasons, and each one is stated with the results:
  - **Tuning:** for methods with their own preprocessing (LDA-Abdelaal, DeepCyTOF), shared vs native input is a grid parameter, and the tuning may select native (step 6); with native, the method gets the curated (untransformed) data. None of the sklearn baselines has internal preprocessing (shared input only, no scaling arm).
  - **Stability:** a method that cannot handle the shared input gets the input it can handle. None so far: GateMeClass crashed on log10 with cut-offs in the legacy runs (`legacy/gatemeclass_bug_report.md`), but rechecked in step 4 this happens only with `GMM_parameterization="V"`; with `"E"` (package default) it runs on the shared input (step 6).
  - The effect of the shared preprocessing itself is checked post hoc with a small ablation (step 8).
- **[D3a]** log10 with lower cut-offs, as in the paper: per-channel cut-offs for Imstat and LT1/LT2 (incl. binary versions; values from `legacy/main.py`), uniform cut-off 100 for Flowcyt and BLAST110. Alternatives (uniform cut-off 100 everywhere, R4.m1; arcsinh with cofactor 150) only in the ablation (step 8).
- **[D3b]** No scaling (as in the paper): log10 with cut-offs only. z-score / min-max only in the ablation (step 8). Quick check: `results/d3b_quick_check/`.

## Step 4: Method integration (2026-10 to 2026-11-04, partly done)
- Wrappers, envs and findings of the newer wrappers: `working_notes/additional_methods.md`, `working_notes/deepcytof.md`.
- Methods: FLAG-X SOM (env `fx_flagx_cpu`), FLAG-X MLP and DGCyTOF (`fx_flagx_gpu`, `envs/flagx_gpu.yml`: flagx 0.2.1 + CUDA torch), GateMeClass (env `fx_gatemeclass`, `envs/gatemeclass.yml`, replaces the legacy setup), LDA-Abdelaal (`fx_lda`), DeepCyTOF (`fx_deepcytof`).
- Tabular baselines (R4.3), env `fx_sklearn` (`envs/sklearn.yml`, lightweight, no flagx/torch): scikit-learn RandomForest, HistGradientBoosting, LogisticRegression, kNN, LDA (`lda_sklearn`, in addition to LDA-Abdelaal), and the per-class GMM classifier of Mocking et al. 2024 (doi:10.1038/s43856-024-00700-x; wrapper `validation/gating/gmm_mocking.py`, generalized from binary to K classes by argmax over the per-class log-likelihoods, identical to the original rule for 2 classes; all GMM hyperparameters shared by the per-class GMMs).
  - GMM-Mocking: the input is cast to float64; with float32 input GaussianMixture fails with 'ill-defined empirical covariance' on BLAST110 (many identical values at the log10 cut-off), as does the upstream class. With float64, predictions are identical to the upstream class (same `random_state`) on BLAST110 p1 (2026-11-04).
- Runners: `validation/runners/<method>.py`, one small module per method with the same interface (`input_arm`, `build`, `fit`, `predict`, `MODULES`), imported by the experiment scripts with importlib so only the method's own env is loaded. Smoke-tested on Imstat (2026-11-04), every method in its env.
- `envs/flagx_gpu.yml`: solves (pytorch 2.5.1 CUDA 12, flagx 0.2.1, scanpy 1.12.5); not yet created (no GPU on the development server).
- **Finding (D8):** FLAG-X SOM `predict` returns -1 for events whose best-matching unit has no labelled training events (flagx `som_classifier.py` L278); the runner keeps it as built-in abstention. The FLAG-X MLP never abstains.
- **[D4]** ACDC and Bayesian trees are not included in the benchmark. Both need an expert cell-type × marker table as input, and neither the original code nor the tutorials provide a way to derive it from labelled data (checked: ACDC archived core @ `b6e9821` + `howchihlee/ACDC_tutorial` @ `b76e01e`, `disiji/fc_mondrian` @ `97420ff`; all tables are hand-written CSVs). Writing tables ourselves or deriving them from labels would not be the published method. GateMeClass stays, since it learns its marker table from labelled data (`GateMeClass_train`). The exclusion is justified in the paper's method selection criteria.

## Step 5: Shared helpers (2026-11-04)
Implemented in `validation/core/` (numpy + stdlib, Python 3.5-compatible; compiled in `fx_deepcytof`, checked in a scratch test: stratified downsampling keeps the class proportions and is seed-reproducible, `join_samples` keeps rows aligned, `confusion_counts` / `class_counts` match `sklearn.metrics.confusion_matrix` incl. -1, `derive_seed` gives identical seeds in Python 3.5 and 3.12, `track_resources` registers a 1.5 GB allocation):
- **Data loading** (`data.py`): `load_sample`, `load_meta_samples` / `load_meta_labels` / `load_labels`, `load_split(dataset, column, inner=False)` → train / val / test sample lists.
- **Downsampling** (`sampling.py`): `downsample_indices(y, n=None, frac=None, stratified=False, seed=None)`, random or class-stratified, by number or fraction of events; port of legacy `get_downsampling_bool` (`legacy/validation/utils/val_utils.py` L750) with a local RandomState.
- **Joining samples** (`sampling.py`): `join_samples(Xs, ys, sample_ids, shuffle=True, seed=None)` → concatenated, (seeded) shuffled `X`, `y` and an aligned `sample_ids` vector (needed e.g. for DeepCyTOF's reference sample).
- **Seeding** (`seeding.py`): `derive_seed(*keys)`, sha256 of `config.SEED` + any str / int / float / bool / None keys → `[0, 2**31-1)`.
- **Evaluation counts** (`evaluation.py`): `confusion_counts(y_true, y_pred, labels)` → (K, K+1) with an abstention column; `class_counts(cm)` → per-class n_true, n_pred, n_abstained, TP, FP, FN, TN (D8 rule). The class list comes from `meta_labels.csv`, so all samples have the same shape. No score helpers yet (step 9).
- **Resource tracking** (`resources.py`): `track_resources(gpu=False, interval=0.05)` context manager → wall-clock time, CPU RSS baseline / peak / increase of the process tree (pure polling of `/proc`; peaks shorter than the interval can be missed; polling costs ~3 % of one core), GPU memory via `nvidia-smi`; port of `scalability_wrapper` / `track_memory_cpu` / `track_memory_gpu` (legacy `val_utils.py`, psutil-free).
- **Output / run helpers** (`runs.py`): atomic `save_json` / `save_csv`, `run_metadata(modules)` (versions, conda env, git commit + dirty flag, host, CPU model / threads, GPUs, SLURM ids), `n_threads()`, `now()`.

## Step 6: Hyperparameter tuning, implementation (2026-11-04)
Plan, grids and atom format in detail: `working_notes/tuning_plan.md`.
- Input as decided in step 3 (D3): the shared preprocessing (log10 with cut-offs, no scaling) by default. Tuning and benchmark use the same input per method (for LDA / DeepCyTOF the selected arm, see below).
- Method-internal preprocessing (D3, deviation by tuning): for methods with their own preprocessing (LDA, DeepCyTOF), the input pipeline is a grid parameter with two arms, selected like any other parameter (D6):
  - **shared:** shared preprocessing (log10 with cut-offs, no scaling), the method's own preprocessing switched off (LDA `transformation=False`, DeepCyTOF `preprocess_log=False`);
  - **native:** curated (untransformed) data + the method's own preprocessing as in the original (LDA `transformation='log'` as in the official flow example `Examples/FCMexample.ipynb` @ `19c2f8f`, DeepCyTOF `preprocess_log=True`). LDA `'log'` gives NaN on negative values (e.g. compensated BLAST110) and `MASS::lda` fails; such a failure is a result of the native arm, not worked around.
  - Steps that are part of the model stay in both arms (DeepCyTOF internal z-scoring on the reference sample). DeepCyTOF `denoise=False` in both arms (not used by the authors on flow data, see `working_notes/deepcytof.md`).
- GateMeClass: `GMM_parameterization` ∈ {"E", "V"} is part of the grid (wrapper default "E", the package default). "V" crashes on the shared input (Mclust fails on the many values at the log10 cut-off, see `legacy/gatemeclass_bug_report.md`); like the LDA native arm, such a failure is a result of that grid value (never selected), not worked around. Sources (Caligola et al. 2024, doi:10.1093/bioinformatics/btae322; repo @ `a93e020`):
  - Paper Sec. 2.1: "The choice of the variance parameter strictly depends on the data to model." Sec. 4: it "must be carefully chosen by the user looking at the data distribution and with testing."
  - Mass cytometry (paper Sec. 3.1, E vs V with/without RSS on AML, BMMC, PANORAMA): E and V comparable on AML, E better on BMMC, V better on PANORAMA; "GateMeClass E achieved over 90% of accuracy in all the datasets". Table 1 does not state the value; its numbers match V for AML and PANORAMA and E for BMMC (inferred).
  - Flow cytometry (paper Sec. 3.2, Bost et al. 2021, compensated + Logicle): "we used GateMeClass with an equal variance parameter" (E). The repo has no flow example that runs GateMeClass (`test/flow_cytometry/load_FC_test_dataset.R` only loads and transforms the data).
  - Tutorial (`tutorial/readme.md` L73, L170, L209, L236, L267, L282): "V" in all examples, all on Levine32 (mass cytometry, arcsinh cofactor 5); L132 suggests choosing E/V from per-marker density plots.
- **[D6]** Both uses, from the same tuning runs:
  - **Per fold (Imstat):** for outer split i, use the parameter set with the best mean inner score of split i. This is clean nested CV: the outer test samples of split i play no part in choosing its parameters.
  - **Consensus (all other datasets):** one parameter set per method, with the best mean inner score over all outer splits (K_OUTER × K_INNER inner splits). It is used unchanged for Lt1/Lt2/Lt1b/Lt2b, Flowcyt and BLAST110 ("tuned once on Imstat"). There is no leakage, since tuning uses a different dataset. It is not used for the Imstat benchmark scores, which would be slightly optimistic.
- `tune_hyperparameters.py` (grids as constants; Py3.5-compatible, runs in every env):
  - Atom = (method, config, outer split, inner split) = one SLURM array task: fit on the inner train samples (each class-stratified downsampled to `TRAIN_FRAC = 0.1`, data seeds per sample independent of method and config), predict every inner val sample (class-stratified `VAL_FRAC`, 1.0 = full), store per val sample the confusion counts + per-class counts and the time / memory of fit and of each predict.
  - Output: `results/tune_hyperparameters/atoms/<method>/<config_id>/split_<i>_inner_<j>.json` (+ `_counts.csv`, long format, concatenates directly); `config_id` = hash of the params. Frozen task lists `manifests/<array>.csv`, config tables `configs/<method>.csv`.
  - One array per (env, cpu/gpu): `sklearn_cpu`, `flagx_cpu`, `gatemeclass_cpu`, `lda_cpu`, `deepcytof_cpu` (Woody), `flagx_gpu` (TinyGPU); scripts and usage in `slurm_scripts/`. `MaxArraySize` 10000 on both clusters.
  - Failure rule: an atom writes `status: started` first and overwrites it with `ok` / `failed`; an atom that is not `ok` (Python error, 24 h timeout, OOM) puts its config out (per fold for its outer split; consensus needs all 15 atoms).
- Smoke test (2026-11-04, 1 % of 3 train / 2 val samples): every method runs in its env and writes complete atoms; the counts files concatenate. Expected failures confirmed: GateMeClass "V" (`predict.Mclust ... object not of class 'Mclust'`) and LDA-Abdelaal native `log` (NaN from negative values, also on Imstat).
- **Timing run** (2026-11-04): the default config of every method on split 0 / inner 0, `TRAIN_FRAC = 0.1` (249,527 training events), `VAL_FRAC = 1.0` (35 full val samples, 1,392,592 events), 4 threads each on the development server (AMD EPYC 7402, 12 atoms in parallel), all on CPU (MLP and DGCyTOF in `fx_flagx_cpu`, no GPU available). Atoms in the session scratchpad, not kept. F1 = macro F1 per val sample (classes in true ∪ pred, D8 rule), mean over val samples; a sanity check of the default configs, not a tuning result.

  | Method | Fit (min) | Predict per sample (s) | Predict total (min) | Atom total (min) | Peak RAM (GB) | Abstained (%) | F1 |
  |---|---|---|---|---|---|---|---|
  | DeepCyTOF (native, calibrate) | 0.1 | 156.5 | 91.3 | 91.3 | 1.8 | 0 | 0.75 |
  | DGCyTOF (CPU) | 1.1 | 82.3 | 48.0 | 49.1 | 8.8¹ | 0.05 | 0.94 |
  | SOM (25², 1000 epochs) | 14.7 | 0.8 | 0.5 | 15.1 | 1.8 | 0 | 0.93 |
  | GateMeClass (E) | 1.5 | 4.9 | 2.9 | 4.3 | 2.2 | 0 | 0.81 |
  | kNN | <0.1 | 6.7 | 3.9 | 3.9 | 0.2 | 0 | 0.95 |
  | MLP (CPU) | 1.5¹ | <0.1 | <0.1 | 1.5 | 2.4¹ | 0 | 0.94 |
  | RF | 0.6 | 0.1 | 0.1 | 0.7 | 0.4 | 0 | 0.96 |
  | LogReg | 0.5 | <0.1 | <0.1 | 0.5 | 0.2 | 0 | 0.94 |
  | LDA-Abdelaal (shared) | <0.1 | 0.7 | 0.4 | 0.4 | 0.5 | 0 | 0.90 |
  | HGB | 0.1 | 0.3 | 0.1 | 0.2 | 0.3 | 0 | 0.92 |
  | GMM-Mocking | <0.1 | <0.1 | <0.1 | <0.1 | 0.2 | 0 | 0.92 |
  | LDA-sklearn | <0.1 | <0.1 | <0.1 | <0.1 | 0.2 | 0 | 0.90 |

  ¹ Caveat: in this run the PyTorch data loaders used more than 1 worker process: MLP 3 (training) + 1 (flagx validation loader); DGCyTOF 3 (training) + 6 (validation step in `fit`, hard-coded upstream default in the wrapper). The workers bring no speed-up for in-memory tensors, but each is a separate process, so the peak RSS of the process tree is inflated (copy-on-write pages are counted per process). Fixed afterwards (2026-11-04): both runners use 1 worker, and the DGCyTOF wrapper's validation step takes `num_workers` from `train_params`. Fit time and memory of MLP / DGCyTOF are to be remeasured (on the GPU).
  - Every default atom finishes in well under the 24 h limit; peak RAM ≤ 9 GB everywhere.
  - The slow parts are prediction-side calibrations that run once per val sample: DeepCyTOF's MMD-ResNet (`calibrate=True`, about 2.6 min per sample) and DGCyTOF's calibration step (about 1.4 min per sample on CPU). SOM is dominated by the fit (1000 epochs).
- **Change after the timing run** (2026-11-04): `TRAIN_FRAC` 0.1 → 0.2 (about 500k training events per atom). Fit is not the bottleneck (slowest fit: SOM, ~15 min at 10 %, expected ~30 min at 20 %; kNN predict scales with the training set, ~4 → ~8 min per atom); the per-sample calibrations of DeepCyTOF / DGCyTOF do not depend on the training size. Closer to the benchmark's training size, which matters for size-dependent parameters (e.g. SOM grid size, kNN k).
- **Change** (2026-11-04): `TRAIN_FRAC` 0.2 → 1.0 (all training events, ~2.5M per inner split). The benchmark trains on all available data (step 7) and the data-requirements experiment (step 8) shows how much is needed, so tuning uses the full inner training set; the inner train set is still ~2/3 of the outer one (inherent to nested CV). Extrapolated from the timing run (not measured): SOM fit ~2.5 h (25²), kNN predict up to ~1–2 h per atom (k=50), other fits ~5–15 min, calibrations unchanged; risks: memory of RF with unlimited depth, the 30² SOM grids near the 24 h limit. No worst-case timing check was run (user decision); `--mem` / `--time` are set without it.

## Step 6 (cont.): Hyperparameter tuning, grid review (2026-11-05)
All grids reviewed one by one with the user; justification of every grid (tuned values, fixed parameters with reasons, defaults, sources and quotes) in `working_notes/tuning_plan.md`, section 6. Principles:
- Grids are defined around each method's default / published config (the default is always in the grid, at position 0); categorical parameters get all distinct values, continuous ones low / mid / high.
- No class weighting in any grid (MLP, RF, HGB, LogReg tuned without); class weighting is a step 8 follow-up experiment.
- Iteration / epoch limits under early stopping or a convergence criterion are generous caps (MLP ES 1000 epochs, HGB `max_iter=1000`, LogReg `max_iter=1000`, GMM-Mocking `max_iter=1000`), so the criterion and not the cap ends the training; `fit_info` (MLP epochs / early stopping, HGB `n_iter`, GMM per-class convergence) shows whether a cap binds.

| Method | Configs | Atoms | Change |
|---|---|---|---|
| SOM | 28 | 420 | map size {15², 20², 25², 30²} × epochs {50 … 2000}; planar, rectangular, gaussian σ 0.1, pca, exponential decays, lr 0.1 → 0.001, radius −0.5 → 0.1 fixed (legacy analysis: map size most influential) |
| MLP | 36 | 540 | layer widths × lr × batch {128, 1024} × schedule (20 epochs, or early stopping with cap 1000); no weight decay, no class weighting |
| DGCyTOF | 54 | 810 | layer widths × epochs × lr (exposed in the wrapper) × batch; `val_size` 0.2 fixed (not tuned by the authors) |
| GateMeClass | 18 | 270 | unchanged (E/V × k × sampling) |
| LDA-Abdelaal | 1 | 15 | shared input only: the native arms (`'log'` without cut-offs, `'arcsinh'` with the hard-coded mass-cytometry cofactor 5) are dropped, no other hyperparameters, rejection off (D8); kept for completeness. Supersedes the two-arm LDA design in the implementation entry above |
| DeepCyTOF | 24 | 360 | input arm × `calibrate` × architecture {(12,6,3) published, (128,64,32) = MLP default} × lr; `denoise=False`. Calibration kept as a defining feature, although the authors skip it within one instrument / run (§2.3) and the code's MMD lr is 1e-5 vs the paper's 1e-3. Without calibration DeepCyTOF is a small MLP trained on one reference sample |
| RF | 54 | 810 | `max_depth` × `min_samples_leaf` × `max_features` {sqrt, 0.5, None} × criterion; `n_estimators=100` (default), no class weighting |
| HGB | 27 | 405 | lr × `max_leaf_nodes` × `l2_regularization`; `max_iter=1000` cap with early stopping, no class weighting |
| LogReg | 7 | 105 | `C` {1e-5 … 1} + unpenalised (C above 1 is practically unpenalised at ~2.5M events), no class weighting |
| kNN | 20 | 300 | k {1, 5, 15, 50, 150} × weights × p |
| LDA-sklearn | 6 | 90 | lsqr shrinkage sweep {None, auto, 0.1, 0.5, 0.9, 1}; the old grid had 6 duplicate solver configs |
| GMM-Mocking | 20 | 300 | grid unchanged; `max_iter=1000`; wrapper exposes `verbose` / `verbose_interval` |

- Arrays: `sklearn_cpu` 2010, `flagx_cpu` 420, `flagx_gpu` 1350, `gatemeclass_cpu` 270, `lda_cpu` 15, `deepcytof_cpu` 360 atoms (total 4425; before the review 7110). All below `MaxArraySize` 10000.
- With D3 now only DeepCyTOF has a native input arm (PLAN.md D3 updated).
- Manifests and config tables written and committed (`results/tune_hyperparameters/`, checked against the grids: rows, order, indices 0..N-1, unique atom ids, all 15 inner splits); `results/` at the repo root is now tracked in git (except `d3b_quick_check/`), the log folders via `.gitkeep`.

### Resources and SLURM scripts (2026-11-06)
- **Woody** (5 CPU arrays): `--cpus-per-task=4`, no `--mem`: Woody allocates 7.75 GB per requested core ("For each requested core 7.75 GB of main memory are allocated", doc.nhr.fau.de/clusters/woody), i.e. 31 GB per task; 4 cores run on every Woody node type. Timing run: peak RAM ≤ 9 GB at `TRAIN_FRAC = 0.1`; more memory would need more cores (> 4 only on the Ice Lake nodes, `--constraint=icx`).
- **TinyGPU** (`flagx_gpu`): `--gres=gpu:rtx3080:1 --partition=rtx3080`, the fixed host share per GPU of 8 cores and 46 GB RAM (TinyGPU partitions table); `SLURM_CPUS_PER_TASK` falls back to 8 for the thread variables (not set without `--cpus-per-task`). RTX 3080 (Ampere, 10 GB): enough for the small MLP / DGCyTOF nets, largest pool (48 GPUs); DGCyTOF's prediction and calibration run on the 8 CPU cores. A100 (32 cores per GPU) not used: scarce, longer queues.
- All arrays: `--time=24:00:00` (cluster maximum, failure rule), `#SBATCH --array=0-<N-1>` hard-coded from the frozen manifests, no throttle (Slurm schedules as available), one log file per task (`logs/tune_<array>/<jobid>_<task>.out`, stdout + stderr, `PYTHONUNBUFFERED=1`), own conda (`module purge`, `source ~/miniforge3/etc/profile.d/conda.sh`, no NHR python module), program started directly without `srun` (one process per task, as in the NHR OpenMP / Python examples), task id passed explicitly (`run <array> "$SLURM_ARRAY_TASK_ID"`), `--export=NONE` + `unset SLURM_EXPORT_ENV` (NHR recommendation).
