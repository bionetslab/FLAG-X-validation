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
- **TinyGPU update (2026-11-06):** GPU pool widened to RTX 2080 Ti / RTX 3080 / V100: `--gres=gpu:1 --partition=work,v100` (untyped request; `work` holds the 2080 Ti and 3080 nodes, `v100` the V100 nodes; Slurm starts the job in whichever partition frees a GPU first). Host share per GPU is 8 cores and 22 GB RAM in both partitions (46 GB only in the dedicated `rtx3080` partition); GPU memory 10–32 GB, enough for the small nets. Peak host RAM in the timing run was ≤ 9 GB. Shorter queue wait across 96 GPUs instead of 48.

## Step 4 (cont.): Method integration completed (2026-11-06)
- All envs created and checked on the cluster by the user (Woody: `fx_flagx_cpu`, `fx_sklearn`, `fx_gatemeclass`, `fx_lda`, `fx_deepcytof`; TinyGPU: `fx_flagx_gpu`), `data/curated/imstat/` copied; the tuning arrays are running (step 6).
- **[D8] FLAG-X SOM abstention kept as built-in:** `predict` returns -1 for cells whose best-matching unit has no labelled training events (flagx `som_classifier.py` L278). It cannot be switched off; it is inherent to the SOM (a node without training events has no label). It matters only for large SOMs with few training events. Treated like DGCyTOF's built-in abstention: run as published, abstained cells count as FN for their true class, abstention rate reported (step 9). The FLAG-X MLP never abstains.
- **`VAL_FRAC = 1.0` confirmed (2026-11-06):** the inner val samples are predicted in full (timing run: prediction cost fine at 1.0). The aggregation goes into a separate script (`tune_hyperparameters_aggregate.py`), not into `tune_hyperparameters.py`; no plot script for now.

## Step 6 (cont.) + step 5: Tuning evaluation: aggregation and best configs (2026-11-10)
- Plan: `working_notes/tuning_plan.md` section 8 (reviewed and simplified 2026-11-10). Chronological: the pairwise tests were added, then removed; the best-config selection was added (bullets at the end of this section).
- **Decisions:**
  - Score per val sample: macro F1 over the classes with `n_true + n_pred > 0` (legacy / sklearn union convention); binary datasets (lt1b, lt2b, BLAST110) F1 of label 1, NaN if undefined (does not occur: every binary sample has ≥ 103 positives).
  - Aggregation chain sample → donor → inner → outer → overall, + `donor_pooled` (donor over its 4 outer splits); mean / SD (ddof 1) / n per level, mean passed upwards.
  - Failed atoms → NaN at the sample level, propagated (no nanmean).
  - Tests are an add-on that works on any level: paired Wilcoxon (scipy), Holm over all pairs per method × scope (statsmodels), not reimplemented.
  - Medians / MAD dropped.
- **Code:**
  - `validation/core/scores.py` (`f1_from_counts`, `abstention_rate`; numpy, Py3.5);
  - `tune_hyperparameters_eval.py` (env `fx_plot`, ~40 s);
  - `envs/plot.yml` + scipy, statsmodels (statsmodels 0.15.0; scipy was already present via seaborn).
- **Outputs** (`results/tune_hyperparameters_eval/`):
  - `atoms.csv`, `failures.csv`;
  - `scores_{sample,donor,inner,outer,overall,donor_pooled}.csv` (147,500 / 30,680 / 4,425 / 1,475 / 295 / 7,670 rows);
  - `pvalues_donor_split.csv` (per outer split, 20–21 donors), `pvalues_donor_pooled_all.csv` (26 donors), `pvalues_inner_all.csv` (15 inner folds; comparison only).
- **Checks:**
  - `f1_from_counts` = sklearn macro F1 on y_true / y_pred rebuilt from the confusion rows (rf + DGCyTOF atoms incl. 1,248 abstained events, max |diff| 1e-16), binary toy case equal.
  - Every donor once per outer split (20–21 donors per split); each level's mean = mean of the level below (≤ 3e-16), NaN exactly where a lower unit is NaN; n: outer 3, overall 5, pooled 4, inner 6–7 donors, donor 1–5 samples (sd NaN for single-sample donors).
  - DGCyTOF p-value tables contain exactly the configs complete in that split.
  - Row counts = m(m−1)/2 per method × scope; rank_a < rank_b ⇔ mean_a ≥ mean_b; p_holm ≥ p.
- **Findings:**
  - Atoms: 4157 ok, 268 failed (none started / missing).
    - DGCyTOF: 124 failed in `predict` (calibration: `spearmanr` / IndexError / float64 item assignment). Complete configs per outer split 20 / 48 / 42 / 20 / 24, overall 3 of 54; the default config is not complete.
    - GateMeClass: 144 failed (the 9 V configs entirely; partial E failures); 3 configs complete.
  - `cap_hit`: HGB 110 / 405 atoms reach `max_iter=1000` (mostly lr 0.01); GMM all converged; MLP with early stopping never reaches the 1000-epoch cap. LogReg stores no `n_iter` (unchecked).
  - Overall best mean macro F1 (SD over the 5 outer splits): RF 0.9665, HGB 0.9639, MLP 0.9634, kNN 0.9608, DGCyTOF 0.9597 (3 complete configs), SOM 0.9517, LogReg 0.9500, GMM-Mocking 0.9445, DeepCyTOF 0.9373, LDA-sklearn 0.9173, LDA-Abdelaal 0.9171, GateMeClass 0.8905.
  - Configs not significantly different from the best (`donor_pooled`, Holm 0.05): RF 36 / 54, SOM 25 / 28, MLP 17 / 36, kNN 16 / 20, DeepCyTOF 8 / 24, GMM 7 / 20, HGB 3 / 27.
- **Runtimes (2026-11-10, after review):** fit and predict wall time kept separate (`fit_s`, `predict_s` per atom; predict summed over the atom's val samples), aggregated like the score to `fit_mean / fit_sd / predict_mean / predict_sd` (outer, overall); NaN for non-ok atoms. Checked: equal to the groupby mean / SD of the level below, NaN aligned with the score.
- **Best configs (2026-11-10):** `select_best` / `best_configs` in `tune_hyperparameters_eval.py` (Part 3): per method the argmax of `mean` at the outer level (per outer split; Imstat benchmark) and at the overall level (consensus; all other datasets), configs with NaN mean excluded, ties → `is_default`, then lowest `config_id` (no exact ties at the top currently). One JSON per level × method in `best_configs/` (`params_overall_<method>.json`: config_id, params, mean, sd, n, is_default, n_candidates; `params_outer_<method>.json`: the same per split under `splits`). Output layout reorganised: `scores/`, `best_configs/`, later `pvalues/`. Pairwise tests commented out until the test level is decided. Checked: argmax and `n_candidates` equal an independent recomputation, params identical to the config tables, for all 12 methods × (overall + 5 splits). DGCyTOF consensus from 3 complete configs (per split 20 / 48 / 42 / 20 / 24 candidates); GateMeClass from 3.
- **Pairwise tests removed (2026-11-10):** the commented-out add-on (`rank_and_test`, `TESTS`, Wilcoxon over all config pairs + Holm) was deleted from `tune_hyperparameters_eval.py`. Reason: whether tuning matters is an effect-size question; non-significance in an all-pairs matrix (1431 pairs for RF / DGCyTOF) does not show equivalence. Follow-up as a todo in `PLAN.md` (best vs. default on `donor_pooled`, grid flatness). `fx_plot` keeps scipy / statsmodels. Outputs unchanged (diff of a rerun against `results/` identical).
- **`donor_pooled` removed (2026-11-11):** the side table (each donor averaged over its 4 outer splits) existed only as independent units for the dropped pairwise tests. The effect-of-tuning analysis will use the levels the best configs are selected on (outer, overall), so the table and `scores_donor_pooled.csv` were removed from `tune_hyperparameters_eval.py` and `results/`. Checked: a rerun into a scratch dir is identical to `results/tune_hyperparameters_eval/` apart from the removed file.
- **Effect of tuning (2026-11-11):** Part 4 of `tune_hyperparameters_eval.py` + `plot_effect_of_tuning.py` (env `fx_plot`).
  - Method: best vs. default config, at the selection levels of D6. Δ = best − default per unit one level below: overall (consensus best) per outer split (5 units), outer (that split's best) per inner fold (3 units); mean / SD / min over the units. Descriptive only; Δ is optimistic (best selected on the same scores). Best vs. worst config was checked and dropped: it mostly measures grid width (one bad config gives LogReg 0.28, DeepCyTOF 0.21).
  - Partial results (this step only; the chain still propagates NaN): units where best or default failed are dropped (`n_valid` of `n`). Needed for DGCyTOF (default incomplete; 3 / 5 splits) and GateMeClass (default E config fails in 1 of 15 atoms, split 2 / inner 0; 4 / 5 splits).
  - Grid flatness over complete configs: `range` of config means, `n_within` (configs within `FLAT_TOL` = 0.01 of the best).
  - Outputs: `results/tune_hyperparameters_eval/effect_of_tuning/effect_{overall,outer}.csv`, `effect_units_{overall,outer}.csv`; `figures/tune_hyperparameters_eval/effect_{outer,overall}_box.pdf`.
  - Checks: Δ mean = best_mean − default_mean (≤ 2e-16); best IDs = the JSONs; mean / SD / min / flatness and the per-unit tables equal an independent recomputation; earlier outputs unchanged.
  - Findings (overall Δ macro F1, mean ± SD over splits): DeepCyTOF +0.030 ± 0.011, GateMeClass +0.027 ± 0.004, GMM-Mocking +0.013 ± 0.003, HGB +0.008 ± 0.003; all others ≤ 0.002 (SOM +0.001, MLP +0.001, RF +0.001, kNN +0.001, DGCyTOF +0.002, LogReg / LDA ≈ 0). Flat grids: RF 54 / 54, SOM 28 / 28, MLP 31 / 36 configs within 0.01 of the best. So tuning barely changes the FLAG-X methods; the earlier untuned MLP was not handicapped (R2.15, R3.m2, R4.3).

## Step 7: Main benchmark, run script (2026-11-11)
- **Plan:** `working_notes/benchmark_plan.md` (reviewed; implemented as planned).
- **Implemented:**
  - `benchmark.py` (root, Py3.5, every env): same structure and arrays as `tune_hyperparameters.py`; atom = method × dataset × outer split × model seed; commands `manifest` / `run` / `missing` / `status`.
  - `slurm_scripts/bench_<array>.sh` ×6 (copies of the tuning scripts) and a benchmark section in `slurm_scripts/README.md`.
  - Helpers moved to `validation/core/` (second user): `shared_preprocessor(dataset, arm)` and `preprocessor_state` (`preprocessing.py`), `read_manifest` and `index_ranges` (`runs.py`). `tune_hyperparameters.py` now imports them.
  - `.gitignore`: `results/benchmark/predictions/`.
- **Decisions:**
  - Parameters (D6): Imstat `params_outer_<method>.json` per outer split, other datasets `params_overall_<method>.json`; frozen into the manifests (`config_id`, `param_source`, `params`).
  - Model seeds: 3 for the stochastic methods (som, mlp, dgcytof, deepcytof, gatemeclass, rf, hgb, gmm_mocking), 1 for the deterministic ones (logreg, knn, lda_sklearn, lda_abdelaal). Model seed `derive_seed('benchmark', method, dataset, split, seed)`; data seeds (downsampling, join shuffle) `derive_seed('benchmark', dataset, split, seed, sample | 'join')`, the same for all methods.
  - No training downsampling in the first run. `TRAIN_DOWNSAMPLING` per (method, dataset), with `{'n': cap}` or `{'frac': f}` per sample (stratified), is reserved for a second run of atoms that hit OOM / timeout. A downsampled atom gets its own files (suffix `__ds_<tag>`), so the failed full-data atom stays as a record. Test samples are always predicted in full.
  - Per test sample: counts (incl. `pred_abstained`, as in the tuning) in `counts/`; predicted labels (`SAVE_LABELS`, int16 `.npz`) in `predictions/` (git-ignored); probabilities off (`SAVE_PROBA`).
  - The repeated-CV robustness check (FLAG-X only) was dropped.
- **Outputs:** `results/benchmark/manifests/<array>.csv`: sklearn_cpu 600, flagx_cpu 150, flagx_gpu 300, gatemeclass_cpu 150, lda_cpu 50, deepcytof_cpu 150 (1400 atoms).
- **Checks:**
  - All 1400 manifest params / config_ids equal the best-config JSONs.
  - The tuning manifests regenerated after the helper move are byte-identical to the committed ones; a tuning smoke atom (lda_cpu 0) is ok.
  - Smoke (`--smoke`: 3 train / 2 test samples, 1 % of the events) of every method on Imstat and BLAST110 p1 in its env (MLP / DGCyTOF in `fx_flagx_cpu`): 23 / 24 ok. GMM-Mocking on Imstat failed only because the 1 % subsample left 12 events in one class for 20 components; this config ran all 15 tuning atoms on full data.
  - A full Imstat LDA-Abdelaal atom with `{'n': 2000}`: atom ID `…__ds_n2000`, 198,000 training events (99 × 2000), manifest indices unchanged, shown by `status`. Its counts equal the saved labels for all 26 test samples.
- **Expected in the first run** (projected from the best-config tuning atoms at ~2.5M training events): SOM timeout / OOM from LT on; GateMeClass OOM from LT / Flowcyt on; kNN, DeepCyTOF and DGCyTOF prediction time / memory on BLAST110 / Flowcyt. Downsampling and memory are to be discussed after the run.

## Step 7 (cont.): Run setup: 32 cores, progress per atom, pilot and caps (2026-11-11)
- **Resources:**
  - The Woody scripts `slurm_scripts/bench_*_cpu.sh` request `--cpus-per-task=32` (248 GB). Such jobs run on the Ice Lake nodes; no `--constraint` needed. TinyGPU is unchanged (1 GPU, 22 GB host RAM).
  - Reason: the 5-fold training sets (Imstat 4.0M, LT 7.7M, Flowcyt 17.6M, BLAST110 53–65M events) exceed what 4 cores / 31 GB allow.
- **Extrapolation** (from the best-config tuning atoms, cross-checked with `legacy/results/time_table_aggregation/`):
  - Legacy (one split, all events, 32 cores / 240 GB): SOM 1000 epochs Flowcyt 2 h 50 m / 61.5 GB; GateMeClass Flowcyt 54 GB fit / 46 GB predict; FCNN Flowcyt 20.7 GB host RAM.
  - SOM ~5.3–5.9 GB and GateMeClass ~4.6 GB of RAM per million training events, independent of the number of cores. 32 vs. 4 cores ≈ 5.5× faster for the SOM.
  - Expected with 32 cores:
    - Imstat / LT / Flowcyt run for all CPU methods.
    - BLAST110: SOM (~320–345 GB, ~47–50 h) and GateMeClass (~285–305 GB) fail; MLP (~60 GB) and DGCyTOF (~46–49 GB) exceed the 22 GB host RAM of TinyGPU. DeepCyTOF (~16–19 h, per-test-sample calibration) and kNN (1–24 h) are borderline.
    - MLP × Flowcyt is borderline (18–30 GB).
- **Strategy:**
  - A pilot (split 0 / seed 0 of every method × dataset, full data) measures what fits.
  - Every dataset with a failure gets one per-sample event cap for all methods. The budget is the largest size that went through, scaled with a margin by the measured fit time / peak RAM of the binding method.
  - The pilot results are then moved to `results/benchmark_pilot/` (record) and the final run starts from scratch in `results/benchmark/` with new manifests: a clean division between pilot and final results.
  - The cap addresses the fit only; prediction failures are reported as results. All methods must be fitted.
  - Rejected: splitting fit and predict into separate steps (model storage: kNN keeps the training data, RF trees on 60M events reach GB per model; rpy2 / Keras 1.2 models do not pickle reliably; twice the jobs). If a prediction timeout occurs and the atom must be completed, a temporary checkpoint for the affected method only (deleted once the atom is ok) is the fallback.
- **Implemented in `benchmark.py`:**
  - Progress per atom: `phase` (`fit` / `predict` / `done`). After the fit the JSON is saved with the fit resources; after each test sample the counts CSV (all rows so far) and the JSON (`n_test_done`, prediction resources) are rewritten. A job killed during prediction keeps fit time / RAM and its progress. Counts are complete only if the JSON is `ok`.
  - `status`: also max training events, fit time, peak RAM of fit / predict, and the progress of unfinished atoms.
  - `pilot <array>`: index list of split 0 / seed 0 per method × dataset (120 atoms).
  - `cap <dataset> <total>`: largest per-sample cap `n` with at most `<total>` training events in every split (binary search over the `.npy` header sizes).
  - CLI with subcommands (unchanged for the existing commands).
  - `K_OUTER` replaced by the split columns of `data/splits/<dataset>.csv` (`outer_splits`).
- **Checks:**
  - Rerun of deterministic / seeded smoke atoms (logreg, lda_sklearn on Imstat + BLAST110 p1, rf on Imstat): counts CSVs byte-identical to the earlier run; JSONs differ only in `phase`, `n_test_done`, timings and `git_commit`.
  - A full Imstat kNN atom killed with SIGKILL during prediction: JSON `started` / `predict`, fit resources present, 6 / 26 test samples in JSON, counts and `.npz`; `status` shows `started:predict 6/26`.
  - `pilot` over the six arrays: 120 rows, all split 0 / seed 0, one per method × dataset.
  - `cap`: for 5 datasets / targets, `n` is maximal (n + 1 exceeds the target in some split, checked by brute force); a target above the full size gives no cap.
  - DeepCyTOF smoke atom, `pilot`, `cap` and `status` run in `fx_deepcytof` (Python 3.5).
  - Manifests unchanged (byte-identical after the `outer_splits` change).

## Step 7 (cont.): Pilot run and SOM memory fix (2026-11-11 to 2026-11-12)
- **Run:** `benchmark.py pilot <array>` (split 0 / seed 0 of every method × dataset, full data, 120 atoms), submitted 2026-11-11 15:35 (Woody, 32 cores / 248 GB; TinyGPU from tinyx, RTX 2080 Ti); last atom finished 2026-11-12 10:34. Commit `983657d4`.
- **Result:** 116 / 120 ok. The 4 failures are SOM × BLAST110 p1–p4, all out of memory in the fit (p3 / p4 killed by SLURM after 14.7 / 15.7 h, `OUT_OF_MEMORY`, atom JSON stays `started:fit`; p1 / p2 `MemoryError` after 17.4 / 17.8 h, JSON `failed`). Every other method went through on every dataset at full size, incl. BLAST110 (52–63M training events).
- **Resources** (split 0; fit / predict = peak CPU RSS; total = start to end of the atom):

  | method | Imstat (3.9M) | LT (7.4–7.6M) | Flowcyt (15.8M) | BLAST110 (51.9–63.3M) |
  |---|---|---|---|---|
  | som | 1.2 h, 20 GB | 3.7 h, 38–39 GB | 6.3 h, 80 GB | OOM after 14.7–17.8 h |
  | mlp (GPU) | 0.15 h, 4 GB | 0.2–0.4 h, 6 GB | 0.4 h, 11 GB | 1.8–4.0 h, 27–32 GB |
  | dgcytof (GPU) | 1.4 h, 3 / 9 GB | 0.7–0.8 h, 5 / 9–21 GB | 1.4 h, 9 / 17 GB | 3.5–4.2 h, 21–26 / 11–12 GB |
  | gatemeclass | 0.4 h, 17 GB | 0.6–0.8 h, 28–34 GB | 3.1 h, 71 GB | 7.1–9.1 h, 162–211 GB |
  | deepcytof | 0.5 h, 1.5 GB | 1.1–1.4 h, 3 GB | 3.8 h, 6 GB | 9.0–11.1 h (prediction 8–10 h), 16–20 GB |
  | knn | 0.2 h, 1 GB | 0.7–1.0 h, 3 GB | 6.1 h (prediction), 6 GB | 4.8–6.2 h (prediction), 16–20 GB |
  | rf | <0.1 h, 8 GB | <0.1 h, 12 GB | 0.1 h, 27 GB | 0.3–0.4 h, 82–101 GB |
  | gmm_mocking | <0.1 h, 3 GB | <0.1 h, 9–10 GB | 0.1 h, 19 GB | 0.2–0.5 h, 60–74 GB |
  | lda_abdelaal | <0.1 h, 4 GB | <0.1 h, 8 GB | <0.1 h, 16 GB | 0.1 h, 45–48 GB |
  | hgb, logreg, lda_sklearn | ≤0.2 h, ≤2 GB | ≤0.2 h, ≤3 GB | ≤0.7 h, ≤7 GB | ≤0.6 h, ≤22 GB |

  - Tightest on memory after the SOM fix: GateMeClass on BLAST110 p1 (211 GB of 248 GB). Longest atom: SOM on BLAST110 (~18 h expected with the fix, 24 h limit). Host RAM of MLP / DGCyTOF on TinyGPU up to 32 GB went through; GPU memory ≤0.4 GB.
  - The projections of the run setup were too pessimistic for most methods (e.g. SOM BLAST110 ~17.5 h of training instead of 47–50 h; MLP / DGCyTOF fit on TinyGPU).
  - Imstat atoms of the other outer splits use other configs (per-split best configs, e.g. split 1 SOM 30×30 / 100 epochs); the pilot only covers split 0.
- **SOM memory problem (cause):** not the Somoclu training (float32 data + BMUs, small), but the unit annotation in `SOMClassifier.fit` → `annotate_som` (flagx 0.2.1 `som_classifier.py` L226 / L395): `_custom_get_surface_state` → `_compute_distances` (L848–884) allocates the full float64 distance matrix training events × SOM units, of which `_custom_get_bmus` only takes the row-wise argmin. 25×25 units × 8 B = 5 GB per 1M events, matching the measured fit RAM exactly (Imstat 19.5 vs. 20.0 GB, LT 37 vs. 38 GB, Flowcyt 79 vs. 80 GB) and 260–317 GB on BLAST110. It is allocated only after the training, hence the OOM after 15–18 h. The chunked branch (L859–866) only applies to ≤10×10 SOMs with ≤100k events and stacks the chunks back into the full matrix (upstream issue for flagx). `predict` uses the same path but per test sample (≤7 GB). Fewer epochs would not help; a smaller grid only proportionally.
- **Fix** (`validation/runners/som.py`, commit `10116c13`): `_ChunkedBMUSOMClassifier(SOMClassifier)` overrides the two helpers (marked `# MOD:`): `_custom_get_surface_state` passes the data through, `_custom_get_bmus` computes the distances in chunks of `BMU_CHUNK_SIZE = 100000` events with flagx's own `_compute_distances` and keeps only the argmin. Somoclu training and flagx are untouched; `topographic_error` / `transform` are not supported by the subclass (not used). Checks (imstat, 3 samples, 25×25, uneven chunks of 7777 events): on the same trained SOM, unit class counts, unit labels, `predict` and `predict_proba` (train and a test sample) are identical to the original path. Memory: 10M events 6 GB peak with 1M chunks vs. +50 GB for the full matrix (constant in N; 100k chunks: ~0.5 GB per chunk). Time equal or faster (40k events 0.18 vs. 0.20 s, 3M events 14.0 vs. 16.1 s, 8 threads).
- **SOM is not deterministic** (commit `7f1df6f8`): two identical fits (`initialization='pca'`) give different Somoclu codebooks (max abs diff ~0.03), likely from the multithreaded OpenMP updates. SOM already is a stochastic method with 3 seeds in `N_SEEDS`; only the runner docstring was wrong. The seed does not control this variation, so a single SOM atom is not bit-reproducible (to be stated in the methods).
- **Consequence:** no caps needed: with the SOM fix every method fits on every dataset at full size, so `TRAIN_DOWNSAMPLING` stays off. Estimated final run (pilot total time × 5 splits × seeds; SOM BLAST110 at 18 h): Woody ~3000 node-h (flagx_cpu 1415, deepcytof_cpu 738, gatemeclass_cpu 575, sklearn_cpu 266, lda_cpu 3), TinyGPU ~500 GPU-h.

## Step 7 (cont.): Evaluation tables, `benchmark_eval.py` (2026-11-11)
- **Implemented:**
  - `benchmark_eval.py` (env `fx_plot`, parts skipped if their outputs exist), `--in` for the pilot (`results/benchmark_pilot`).
  - `validation/eval_helpers.py`: `write_csv`, `last_line`, `cap_hit` moved from `tune_hyperparameters_eval.py` (which imports them now), plus `failure_type`.
  - `validation/core/scores.py`: `class_scores(tp, fp, fn)` (per-class precision / recall / F1, NaN for a zero denominator; numpy, Py3.5), for reuse beyond the eval.
- **Decisions:**
  - Score tables wide, one per dataset: one row per atom × test sample, per class `n_<l>` / `prec_<l>` / `rec_<l>` / `f1_<l>`, so macro (NaN-mean), weighted (`n_<l>`) or binary (`f1_1`) aggregation can be chosen later. One file per dataset because label ids mean different classes in different datasets.
  - Class scores follow D8, with NaN for a zero denominator (F1: class neither present nor predicted, the sklearn / legacy convention of `f1_from_counts`).
  - Unpredicted test samples (failed, killed, missing atoms) are NaN rows with `predicted = False`, as in the tuning eval.
  - Failure type per atom, with the patterns in `validation/eval_helpers.py`:
    - `python_error` (traceback last line);
    - `oom` (MemoryError / CUDA OOM in the traceback, or an OOM kill in the SLURM log);
    - `timeout` (`DUE TO TIME LIMIT`);
    - `node_failure`;
    - `killed_unknown`;
    - `missing`.

    The phase (`fit` / `predict`) comes from the JSON. A killed atom's log is found via `start <atom_id>`; the latest job id counts.
  - Fit table `atoms.csv` separate from the score tables (fit time, peak / increase RSS, GPU peak, training size, status, failure).
- **Checks** (scratch input: smoke atoms, the full downsampled LDA-Abdelaal Imstat atom, the kNN atom killed after 6 / 26 test samples, fake SLURM logs):
  - Row count per dataset = Σ manifest atoms × test samples; `n_cells` = curated sample sizes; all scores / prediction resources NaN for unpredicted rows.
  - Macro F1 as NaN-mean of `f1_*` = `f1_from_counts` and `n_<l>` = `n_true` for all 44 predicted rows.
  - Precision / recall / F1 = sklearn `precision_recall_fscore_support` on the saved labels (26 full test samples, max |diff| 1e-16).
  - Killed kNN atom: `timeout` / `predict` from the newer of two logs (the older one reported an OOM kill), 6 predicted + 20 NaN rows.
  - GMM smoke error → `python_error`. `failure_type` unit cases: node failure, OOM kill, MemoryError, CUDA OOM, IndexError, no reason → `killed_unknown`.
  - `tune_hyperparameters_eval.py` after the helper move: rerun into scratch, `diff -r` against `results/tune_hyperparameters_eval/` identical.
- **Addition (2026-11-13): predicted counts per class.** `scores/<dataset>.csv` got `npred_<l>` (events predicted as class l = tp + fp, from the counts' `n_pred`) after `n_<l>`, for true vs. predicted class fractions (`/ n_cells`). It could not be recovered from prec / rec when tp = 0 and fp > 0 (7 such cells in the pilot). Checks on the pilot (`--in results/benchmark_pilot`, into scratch): all other columns and `atoms.csv` / `failures.csv` identical to the earlier output; `npred_<l>` = `n_pred` = tp + fp; Σ `npred_<l>` + `n_abstained` = Σ `n_<l>` = `n_cells` for every predicted row; NA in unpredicted rows; `npred_<l>` = rec · n / prec where prec > 0, and prec NaN ⇔ `npred_<l>` = 0.

## Step 7 (cont.): Benchmark box plot, `plot_benchmark.py` (2026-11-13)
- **Implemented:** `plot_benchmark.py` (env `fx_plot`, `--in` / `--out`, figure skipped if it exists): `scores_box.pdf`, one panel per dataset (5 × 2 grid, shared axes), one box per method (colored, alpha 0.9, no fliers; style of legacy `plot_performance_score_box_plot`), the points jittered on top (deterministic, seed 0) with one marker per outer test fold, 'k/n' above incomplete boxes and 'n/a' for methods without predictions, legend under the figure (methods in rows of 6, then the folds).
- **Global order and colors** (user request): new section Plotting in `config.py`: `PLOT_DATASETS`, `DATASET_NAMES`, `PLOT_METHODS` (FLAG-X, published methods, baselines), `METHOD_NAMES`, `METHOD_COLORS` (colorblind-friendly, muted: Paul Tol's muted scheme + 3 of his light scheme, as hex). Separate from `DATASETS`, the run order: `benchmark.py` numbers the manifest rows by looping over `DATASETS`, so reordering it would change the array indices (atom ids, seeds and results would not change); the plot order can be changed freely. `plot_effect_of_tuning.py` and `plot_dataset_overview.py` (panel order) import the names from there (same values and order as before, so no output change).
- **Decisions (with the user):** panel per dataset (not the legacy single axis with 120 boxes); score as in the tuning (macro F1 = NaN-mean of `f1_*`, binary datasets `f1_1`); one point = one test sample, averaged over the model seeds that predicted it; the box pools all outer folds.
- **Checks** (pilot, `--in results/benchmark_pilot_eval`): all 2246 plotted points equal an independent recomputation with `f1_from_counts` from the pilot counts CSVs (binary for lt1b / lt2b / BLAST110); seed averaging on a synthetic table (2 predicted seeds + 1 failed seed: mean of the two, failed one skipped); figure inspected (10 panels, SOM × BLAST110 'n/a' from the pilot OOM, split-0 marker only).
- **Y-axis labels** (user request): `F1 macro` for multi-class datasets, `F1 binary` (F1 of the positive class, label 1) for the datasets with 2 labels (lt1b, lt2b, BLAST110); binary is decided by the number of labels, as in the eval scripts.

## Step 8: Training data requirements, run script (2026-11-17)
- **Own step:** training data requirements moved out of the follow-ups into PLAN step 8 (follow-ups → 9, evaluation → 10, figures → 11, documentation → 12; references in `PLAN.md`, `CLAUDE.md`, `validation/core/evaluation.py` updated, older notes left as they are). Design: `working_notes/training_data_requirements_plan.md`.
- **Decisions (user, 2026-11-17):**
  - methods `mlp`, `som`, `hgb`, `rf` for now (final choice = top performers of step 7);
  - atom = method × dataset × outer split × trial × n_samples × n_events; 3 trials (TODO);
  - log-spaced default grid (TODO: per dataset); test samples in full; counts + resources, no predictions;
  - arrays `sklearn_cpu` / `flagx_cpu` / `flagx_gpu`, 4 cores, no partition, 1 h per atom (TODO);
  - fold first, small → large;
  - timeout / OOM = cutoff for all larger atoms of the method × dataset (any split / trial);
  - skipped atoms must not cost an allocation; node types not pinned (lower-bound claim + later calibration);
  - (`'all'`, `'all'`) left out (= main benchmark).
- **Execution design:** lane workers in pure Python, decided over one job per atom + `scancel`, a start-of-job check (an allocation per skip), `srun --time` steps (only testable on the cluster) and `aftercorr` dependency arrays (cutoff only per lane, ~70 arrays). Reasons: no allocation for skips, the cutoff is shared across lanes, it runs the same sequentially and on SLURM (local tests with short limits), and small atoms do not pay a job start each.
- **Implemented:** `training_data_requirements.py` (`manifest` / `run` / `atom` / `status` / `missing`), `slurm_scripts/tdr_{sklearn_cpu,flagx_cpu,flagx_gpu}.sh`, README section.
  - Shared helpers moved because there is now a second user: `outer_splits` → `validation/core/data.py`, `best_config(method, dataset, split, best_dir, tuning_dataset)` → `validation/core/runs.py`.
  - `load_sample(..., mmap=False)` added: training samples are memory-mapped and only the downsampled rows are read.
  - The benchmark manifests written before and after the move are byte-identical for all 6 arrays. The committed `results/benchmark/manifests/` already differ from what the current code writes, independently of this change (deepcytof_cpu, flagx_cpu, flagx_gpu, sklearn_cpu; probably the best configs changed after the manifests were written) → check before the final benchmark run.
- **Grid (default):** duplicates of `'all'` are dropped per lane; 600 lanes, 36,604 atoms (rf / hgb / som / mlp 9,151 each).
- **Seeds:** order `derive_seed('train_req', dataset, split, trial)`, event order per sample `(…, trial, sample)`, join `(…, trial, n_samples, n_events, 'join')`, model `('train_req', method, dataset, split, trial)`.
- **Smoke tests** (local, `taskset -c 0-59`, 4 threads, scratch output):
  - `--smoke` lanes ok in `fx_sklearn` (rf Imstat, ~2 s per atom), `fx_flagx_cpu` (som Imstat, 18–49 s) and mlp lt1b in `fx_flagx_cpu`.
  - mlp lt1b with 1 sample × 100 events → `failed`: the positive class has 1 training event, and flagx's stratified validation split raises a ValueError. This is a Python error, so not a cutoff; recorded as a result.
  - Forced timeout (`--atom-limit 6`, RF Imstat):
    - timeouts → markers → larger atoms skipped, also in the second lane (other trial);
    - `--lane-limit` stops the lane cleanly and the rerun resumes it; `status` / `missing` count the dominated atoms as finished;
    - no orphaned processes.
  - An atom killed while its interpreter was shutting down, after writing `ok`, was first classified as `timeout` → fixed: a child JSON `ok` wins.
  - Python 3.5 syntax check (`fx_deepcytof`) ok.
- **Nested event downsampling (2026-11-18, user decision):**
  - Found: `validation.core.sampling.downsample_indices` with a fixed seed is not nested over n. On 200 random class distributions, 31 % of the steps 100 → 300 → … → 30000 lost events of the smaller subset (on average 19 %, up to 99 %). Cause: a class whose quota rounds to 0 draws no random numbers, which shifts the stream for all later classes, plus the final correction to exactly n.
  - The legacy experiment was not nested either: one global `np.random.seed` for the whole grid, so a sample's events changed even between n_samples cells.
  - Change in `training_data_requirements.py` only (tuning / benchmark keep `downsample_indices`): `event_order(y, seed)` gives one fixed stratified order per sample (random permutation; within each class the j-th of m_c events gets rank (j + 0.5) / m_c; stable sort by rank). n_events = the first n.
  - Checked: 0 non-nested steps on the same 200 distributions; class counts within 1.22 events of n · m_c / m; deterministic per seed; ~0.7 s per 3M-event sample (vs 0.16 s for `downsample_indices`), outside the fit timing.
  - Smoke tests repeated (rf / som Imstat, mlp lt1b): ok, mlp lt1b 1 × 100 again `failed` (same cause as above); Py3.5 syntax ok.
- **Training class counts per sample (2026-11-18, for the minority count analysis):** the atom JSON also stores `n_train_events_sample_per_class` ({sample: {class: count}}, after downsampling; sums to `n_train_events_per_class`). ~2 KB per atom, ~72 MB over all 36,604 atoms. As before, the counts are written only together with the fit results, so they are missing for timeout / OOM / skipped atoms. Accepted: those cells are expected to score like the full data, and the counts can be rebuilt from the seeds if needed.
- **Grid per dataset (2026-11-18, user decision):** `N_SAMPLES` is now a dict per dataset: default 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, all; FlowCyt 1–16 + all (24 training samples). `N_EVENTS` is unchanged and the same for all datasets (100 … 300k, all).
  - Basis: the legacy SOM / MLP grids (`legacy/results/num_samples_num_events/*/random/res_df_f1_*.csv`). The knees lie at 2–16 samples and 100 → 5k events (Imstat, LT); FlowCyt needs ≥20k–all events and ≥4–10 samples; MLP on LT1 / LT2 rises up to 30–58 samples; no BLAST110 reference.
  - Dataset sizes: Imstat 125 samples (99–101 training per fold), median 33k events; LT 98 (78–79), 100k; FlowCyt 30 (24), 722k; BLAST110 95–107 (75–86), ~850k (p1 up to 5.3M).
  - Atoms 36,604 → 59,524, still 600 lanes; the added atoms are mostly small. Storage estimate: ~0.73 GB of content (~0.95 GB on disk, ~110k files) for the old grid, ~1.2 GB for the new one.
- **Method selection (2026-11-24, user decision):** `METHODS` stays `['mlp', 'som', 'hgb', 'rf']` (all datasets, all folds / trials).
  - RF / HGB / MLP: the top performers of step 7 (mean rank over the 10 datasets of the mean per-test-sample score: RF 1.2, HGB 2.1, MLP 3.1, DGCyTOF 3.6). HGB is equivalent to RF at δ = 0.02 on 9/10 datasets and superior on FlowCyt; MLP is non-inferior to RF on 8/10 (not on lt2, FlowCyt). DGCyTOF is not included: no better than MLP, prediction ~100× slower (200–1400 vs ~1.5 µs/cell).
  - HGB is also the low-memory alternative to RF: peak fit memory with full training data 16–20 vs 83–103 GB (BLAST110), 7 vs 30 GB (FlowCyt), ~3 vs ~13 GB (LT). If RF needs too much memory at its minimal training set for local training, HGB is used instead; the step-8 OOM cutoffs (4-core lanes) show this directly.
  - SOM is kept although it is not a top performer (mean rank 6.6, below RF by 0.02–0.24 on 9/10 datasets, only non-inferior on Imstat): Workflow 1 (SOM representation as a visualization / clustering aid for manual gating) is central to the paper, so the data needed for a good SOM representation must be shown; it was also in the original version of this analysis. The step-8 criterion compares each method to its own full-data model, so the lower SOM ceiling does not affect it (state this, cf. R4.5 "plateau at own ceiling"). SOM is the most expensive lane (only method in `flagx_cpu`, 150 of 600 lanes).
- **Cutoff by total training events, no calibration (2026-11-24, user decision):**
  - Rule (online and post hoc): if an atom with x total training events (`n_train_events` from the manifest) ends with timeout / OOM, every atom of the same method × dataset (any split / trial) with `n_train_events` ≥ x is skipped (first `>`, switched to `≥` the same day: without sample-size limits `n_train_events` = n_samples × n_events in every lane, so with `>` the same cell would still run, and possibly time out, in the other 14 lanes; a cell with any timeout is not eligible in the eval anyway). Before: ≥ n_samples and ≥ n_events. Reason: fit time and memory depend mainly on the total number of events; the new rule is stricter and the remaining region is still monotone in both axes.
  - Code: the cutoff marker stores `n_train_events`; `dominated_by` compares it with the atom's (ties dominated); `status` shows the cutoff (`cutoff_ev`) and the number of `ok` atoms above it (`ok>cut`).
  - Post hoc: the eval applies the rule again over all atoms. Atoms that finished `ok` before a smaller atom of another lane timed out are not rewritten; they are few, because every larger atom is skipped once the marker exists.
  - No calibration run and no slowdown factors: each cell runs in 15 lanes on mixed nodes (all on Gold ≈ 0.3 %), so the cutoff settles on the slower node type (v6) by itself. Claim: 1 h on 4 cores of the nodes used.
  - Test (local, `fx_plot`, children killed before the method import; no method env on this machine): forced timeout → marker with `n_train_events`, larger atoms skipped; fake cutoff at 500 events → atoms with 100 / 200 / 300 events ran, 600 / 1000 / 2000 skipped; `status` reports `ok>cut`. Python 3.5 check not rerun (no `fx_deepcytof` here; the change uses no new syntax).
- **Job length and node type (2026-11-24, user decisions):**
  - `--time=24:00:00 --time-min=02:00:00` in all three `tdr_*.sh`: the backfill scheduler may start a task with any limit between 2 and 24 h (gaps). Assumed to work: Slurm accepts the option on Woody, a start with a reduced limit was not observed yet. Backfill considers at most 20 jobs per user per cycle (`bf_max_job_user=20`).
  - The worker reads its job's time left at start (`slurm_time_left`: `squeue -j $SLURM_JOB_ID -h -o %L`, formats `[D-]HH:MM:SS` / `MM:SS`) and stops before `now + ATOM_TIME_LIMIT_S + WRAPUP_BUFFER_S` passes the job end; `LANE_TIME_LIMIT_S` is only the fallback outside SLURM (or if squeue fails, with a warning); `--lane-limit` (tests) overrides both.
  - Woody: `--constraint=icx` (Gold 6326 only) instead of `"kl|icx"`: one CPU type, so no calibration and no node-dependent cutoffs; budget statement "1 h on 4 cores of a Xeon Gold 6326"; SOM ~2.3× faster per core than on v6, so state the CPU model for the local-training claim. TinyGPU stays mixed.
  - Resubmission stays manual (`missing` after the array has finished; never while tasks are pending / running, no lock).
  - Tests (local, fake `squeue` on PATH): parsing of `1-00:00:00` / `23:59:30` / `5:00` / `1:10:05`, `INVALID` → fallback with warning; 1:10:05 left → atoms start, 1:09:00 left → clean stop; without SLURM the `--lane-limit` path is unchanged.

## Step 8 (cont.): Evaluation, `training_data_requirements_eval.py` (2026-11-25)
- **Decisions (user, 2026-11-25; details in `working_notes/training_data_requirements_eval_plan.md`, "Decisions"):**
  - order: greedy (fewest samples, then fewest events per sample) primary, joint (fewest total training events) secondary. Reason: greedy keeps the number of annotated samples (labelling cost) small and matches legacy Table 1;
  - stability rule (all larger n_samples at the same n_events must also pass; ineligible ones ignored): computed with and without, primary chosen after the real run (no extra cost: same pass table);
  - bootstrap: 10,000 resamples, percentile 95 % CI (one-sided α = 0.025), unit = patient (`patient_id` of the split CSV: donors for Imstat, 125 samples / 26 donors; one sample per patient elsewhere); CI conditional on the trial-averaged models, trial variance as the spread of the per-trial minima (no two-level bootstrap);
  - criteria: non-inferiority over a δ grid of 21 log-spaced values 0.001–0.1 (R10 preferred numbers, contains 0.01 / 0.02 / 0.05 exactly; `np.logspace` would miss 0.02 and 0.05), primary δ = `config.DELTA` = 0.02; legacy criterion (mean score ≥ lower bound of the baseline CI) for comparison only;
  - cut-off cells (timeout / oom / skipped / out of budget / failed / missing) never eligible;
  - step-9 config (pooled vs. cross-fitted per fold): open, both written to `minimal_configs.json`;
  - compute budget: the cutoff rule is applied as stated; the planned check of its assumption (fit time vs. `n_train_events` by n_samples, `budget_check.csv`) and automated outlier detection were dropped; outlier cutoffs are checked and resubmitted manually.
- **Implemented:** `training_data_requirements_eval.py` (parts: atom table with post-hoc budget → score tables → per-sample trial means paired with step 7 → curves per scope → minimal cells → cross-fitting → `minimal_configs.json`; each part skipped if its output exists). Working note rewritten (decisions, method, outputs).
- **Implementation choices (documented in the working note, "Method"):**
  - cells missing in some lanes (the manifest drops duplicates of `all` per lane, e.g. Imstat 1 × 30000 when the selected sample has ≤ 30k events) are resolved to the atom with the same training data (n_samples ≥ training samples → `all`; missing cap → `all` events; (`all`, `all`) → the step-7 model, d = 0); counted in `n_alias` / `n_baseline_alias`. Cells absent in every lane have no rows;
  - per test sample: trial mean over the trials with a defined score (NaN unless every trial predicted the sample), as the step-7 seed mean;
  - scopes as one long table (`scope` column): `pooled` (main result), `trial_<t>` (per-trial minima), `fold_<k>` (per-fold minima, cross-fitted evaluation), `excl_<k>` (cross-fitted selection);
  - common random numbers: one resample matrix per dataset and fold subset, shared by all methods and cells; ratio estimator (sum of d / number of samples with a defined d) so donors weigh by their sample count;
  - `scores/<dataset>.csv` only holds predicted rows (size); ~56 MB for 2 methods × 3 datasets, so a few hundred MB for the full run (kept tracked, `.gitignore` unchanged, user decision).
- **Tests** (synthetic fixture, `fx_plot`; real step-8 results not available locally, no scikit-learn in `fx_plot`): real `sklearn_cpu` manifest restricted to rf / hgb × Imstat / LT1b / FlowCyt (7,798 atoms), simulated counts from the step-7 class counts, one timeout, `ok` atoms above the cutoff, a lane without JSONs above it, one failed atom.
  - budget: the 24 atoms ≥ the cutoff are out of budget, the missing ones become `skipped`; the failed atom makes its cell ineligible exactly in `pooled`, `trial_2`, `fold_3` and `excl_<k≠3>`;
  - one cell recomputed by hand (mean d, n) matches `curves.csv` and `sample_scores.csv`; alias: Imstat 1 × 30000 resolved in 1 of 15 lanes;
  - the joint minimal total training events are monotone non-increasing in δ; cross-fitting shows the expected optimism of the pooled selection on noisy cells (hgb FlowCyt: in-sample mean d +0.011 vs. cross-fitted −0.001);
  - rerun with the later parts deleted (read-back of `curves.csv` / `minimal.csv`): byte-identical outputs; runtime ~1 min.
- **Training class counts for the minority count analysis (2026-11-25, user decision):** new eval part 1b → `train_class_counts/<dataset>.csv`, one row per dataset × fold × trial × cell × training sample with `n_<l>` per class (from `n_train_events_sample_per_class` of the atom JSON, after downsampling).
  - Only atoms that completed (`ok`); a cell is included if any method completed it. No recomputation from the seeds for cut-off cells (not needed: the analysis uses completed atoms only).
  - No method column: the data seeds contain no method, so all methods train on the same data; the eval stops with a `ValueError` if two methods report different counts for a cell (or an `ok` atom has no counts). Per method: join with its `ok` atoms in `atoms.csv`. Size at most one method's atoms × training samples (320,073 rows from the real manifest), instead of up to 4× with a method column.
  - Only raw counts are stored; everything else is derived in the figure script: legacy minority count (`legacy/main_plotting.py::fig6s_minority_count`: per-sample minimum over the classes present after downsampling, then total / mean / median / min per cell), class counts of the whole training set. Classes absent from a downsampled training sample are explicit zeros; to reproduce legacy, drop them before the minimum (alternatively count 0 as "class missing").
  - Test (fixture with synthetic per-sample counts): Imstat 32,099 / LT1b 30,156 / FlowCyt 9,900 rows (= one method's atoms × training samples; RF cells above its cutoff are covered by HGB), no duplicates; one count changed in one HGB atom → `ValueError` naming both atoms; restored → table rebuilt.
