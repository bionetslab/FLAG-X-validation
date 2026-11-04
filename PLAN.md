# Benchmark plan

Status: steps 0–3 done (2026-11-03); steps 4–6 in progress: helpers, runners and the tuning script written, tuning runs pending (2026-11-04). Built from `NOTES.md`. Open decisions are marked **[D#]** and collected at the end. Small decisions (k, seeds, grids, ...) are made when we reach the step.

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
- Prediction and scoring are separate. Each task stores, per test sample, the raw evaluation counts (confusion matrix incl. abstentions, per-class TP/FP/FN/TN, class counts) and, where feasible, the predicted labels and probabilities (step 5). Metrics and aggregation are computed afterwards from these (step 9).

**Environments**
- One isolated conda env per method (or method group) and per analysis step (data curation, evaluation, plotting). Env files live in `envs/`, each with a header comment stating its purpose and tested versions.
- R-based methods install `r-base` and all R packages in their own env. The system R is never used.
- The shared helpers (`validation/core/`, step 5: small modular functions for data and split loading, downsampling, joining samples, preprocessing, seeding, evaluation counts, resource tracking; no orchestration) depend only on numpy (+ the standard library) and must run in every method env, including the Python 3.5 envs (`fx_deepcytof`, `fx_acdc`): no f-strings, no `typing.Self`, no walrus operator, etc.

**Methods**
- Every method is used through its wrapper in `validation/gating/` with the common sklearn-like interface (see `CLAUDE.md`). Methods are not reimplemented; copied code is cited with URL, commit and line numbers.
- Method-specific details stay in the wrapper and in a small per-method runner. The shared helpers know nothing about individual methods.

**Extensibility**
- Adding a dataset = adding one `curate_<dataset>` function (raw path, channel and label maps as constants above it) plus its entries in `config.DATASETS` and `config.CHANNELS`. Everything downstream picks it up from the curated files.
- Adding a method = adding a wrapper, an env file, and a branch in the run script that builds the wrapper with its hyperparameters. Nothing else changes.

---

## 2. Steps (chronological)

### Step 0: Repository scaffolding -- DONE
- Scripts at the repo root with descriptive names, no numbering; figure scripts start with `plot_`. Each script has its parameters as constants at the top and an `OVERWRITE` flag.
- `config.py` (stdlib-only, Py3.5-compatible): `ROOT` (via `__file__`), `DATA_RAW`, `DATA_CURATED`, `RESULTS`, `FIGURES`, `DATASETS`, `CHANNELS`. No registries. **[D5]**
- Layout: `data/raw/` (gitignored), `data/curated/<dataset>/` (`.npy` gitignored, meta CSVs tracked), `data/splits/` (tracked), `results/<script>/`, `figures/<script>/`.
- Envs: `fx_flagx_cpu` (`envs/flagx_cpu.yml`, curation + FLAG-X runs), `fx_plot` (`envs/plot.yml`), one env per method. All with `channel_priority: strict`.
- The shared helpers in `validation/core/` are added in step 5 (so far only the preprocessing, step 3).

### Step 1: Data curation -- DONE
- `curate_data.py` (env `fx_flagx_cpu`): one function per dataset. Load with flagx (one `FlowDataManager` per file) → compensate if the FCS file has a spillover matrix → rename channels (per-dataset map in the script) → keep/order as in `config.CHANNELS` → map labels (per-dataset map in the script, contiguous from 0) → save. Raises on unmapped channels/labels.
- Output in `data/curated/<dataset>/`: `sample_<NNN>_X.npy` (float32) and `sample_<NNN>_y.npy` (int32), `meta_samples.csv` (sample, original filename, extra info such as `patient_id`, `subtype`, `sample_type`), `meta_labels.csv` (label, original label, name, description). Data is stored untransformed. **[D2]**
- Datasets (10): `imstat`, `lt1`, `lt2`, `lt1b`, `lt2b`, `flowcyt`, `blast110_p1`–`p4`. **[D1]**
- Imstat sample meta (`donor_id`, `lysis`) comes from `data/raw/imstat/Imstat_samples_2026-10-29_clean.csv`, cleaned from the data provider's table (donors `a`–`l` for the `ER_` files; for the named files the `YYYYMMDD-n` token as written). 26 donors with up to 5 lysis reagents each (R4.4, R4.6).
- Compensation only for BLAST110 (only dataset with `$SPILL`). **[D7]**
- BLAST110: one dataset per tube, blast vs other, FSC-A/SSC-A only, channels named by antibody (`$PnS`).
- Overview figures: `plot_dataset_overview.py` (env `fx_plot`) → `results/dataset_overview/*.csv`, `figures/dataset_overview/` (sample sizes, pooled and per-sample class balance, optional hue by meta column, e.g. `sample_type`).

### Step 2: Splits -- DONE
- `make_splits.py` (env `fx_flagx_cpu`): patient-level k-fold with scikit-learn (`StratifiedKFold` / `KFold`, `config.SEED`).
- Folds are assigned once per split group on the union of its patients, so a patient is in the same fold in every dataset of the group: `lymphoma` (lt1, lt1b, lt2, lt2b; stratified by `subtype`), `blast110` (p1–p4; stratified by `sample_type`), `imstat` (by `donor_id`), `flowcyt` (unstratified). Rare strata (BL: 2 patients) do not occur in every fold.
- Output in `data/splits/` (versioned in git): `<dataset>.csv` (sample, patient_id, [stratum], `split_<i>` = train/test), `imstat_inner.csv` (all samples, `split_<i>_inner_<j>` = train/val/test) for tuning, `summary.csv` and `summary_inner.csv` (sample counts per split).
- `K_OUTER = 5`, `K_INNER = 3` (fixed; more inner folds are computationally infeasible for tuning). One CV repetition; variability is covered by model seeds (step 7) and an optional repeated CV for the FLAG-X methods.

### Step 3: Preprocessing definition -- DONE
- `validation/core/preprocessing.py`: `Preprocessor(channels, trafo, trafo_params, scaling, scaling_params)` with `fit` / `transform`. Transformations `log10_cutoff` (`{'cutoff': value or {channel: value}}`), `arcsinh` (`{'cofactor': value}`), `none`; scalings `zscore`, `minmax` (`{'feature_range': (a, b)}`, default (0, 1)), `none`.
- Numpy-only, Python 3.5-compatible reimplementation (no sklearn/flagx), so it runs in every method env. Checked against flagx `log10_w_custom_cutoffs`, pytometry `normalize_arcsinh` and the sklearn scalers.
- Fitted on the fly inside each task on the task's (possibly downsampled) training data; the transformed data is never stored. Each experiment states its preprocessing as constants. A fitted preprocessor can be saved next to the model with `save(path)` / `Preprocessor.load(path)` (JSON, loadable in every env).
- Cut-offs: `config.LOG10_CUTOFFS` (per dataset; a dict per channel or one uniform value).
- flagx 0.2.1 `log10_w_cutoff` (uniform cut-off, used for Flowcyt in the paper) sets values <= cutoff to ln(cutoff) instead of log10(cutoff) (4.61 instead of 2 for cut-off 100). This affects up to 5 % of the values per Flowcyt channel (HLA_PB 4.8 %, CD14_FITC 1.9 %). Our implementation uses log10(cutoff); the paper's Flowcyt numbers will change slightly. To fix in the flagx package.
- **[D3, decided]** Default: the same input for every method, i.e. the shared preprocessing (D3a, D3b). Deviations are allowed only for two reasons, and each one is stated with the results:
  - **Tuning:** for methods with their own preprocessing (LDA-Abdelaal, DeepCyTOF), shared vs native input is a grid parameter, and the tuning may select native (step 6); with native, the method gets the curated (untransformed) data. None of the sklearn baselines has internal preprocessing (shared input only, no scaling arm).
  - **Stability:** a method that cannot handle the shared input gets the input it can handle. None so far: GateMeClass crashed on log10 with cut-offs in the legacy runs (`legacy/gatemeclass_bug_report.md`), but rechecked in step 4 this happens only with `GMM_parameterization="V"`; with `"E"` (package default) it runs on the shared input (step 6).
  - The effect of the shared preprocessing itself is checked post hoc with a small ablation (step 8).
- **[D3a, decided]** log10 with lower cut-offs, as in the paper: per-channel cut-offs for Imstat and LT1/LT2 (incl. binary versions; values from `legacy/main.py`), uniform cut-off 100 for Flowcyt and BLAST110. Alternatives (uniform cut-off 100 everywhere, R4.m1; arcsinh with cofactor 150) only in the ablation (step 8).
- **[D3b, decided]** No scaling (as in the paper): log10 with cut-offs only. z-score / min-max only in the ablation (step 8).

### Step 4: Method integration
- Per method: env file, wrapper, runner. The runner composes the shared helpers (step 5); the usual order in the main benchmark is: load the task's samples → downsample the training samples → join + shuffle → fit the preprocessing on the training data → `fit` → `predict` / `predict_proba` per test sample → evaluation counts (+ optional labels / probabilities) → save model, outputs, runtime and memory. Other experiments may compose them differently.
- Methods: FLAG-X SOM (env `fx_flagx_cpu`), FLAG-X MLP and DGCyTOF (`fx_flagx_gpu`, `envs/flagx_gpu.yml`: flagx 0.2.1 + CUDA torch), GateMeClass (env `fx_gatemeclass`, `envs/gatemeclass.yml`, replaces the legacy setup), LDA-Abdelaal (`fx_lda`), DeepCyTOF (`fx_deepcytof`).
- Tabular baselines (R4.3), env `fx_sklearn` (`envs/sklearn.yml`, lightweight, no flagx/torch): scikit-learn RandomForest, HistGradientBoosting, LogisticRegression, kNN, LDA (`lda_sklearn`, in addition to LDA-Abdelaal), and the per-class GMM classifier of Mocking et al. 2024 (doi:10.1038/s43856-024-00700-x; wrapper `validation/gating/gmm_mocking.py`, generalized from binary to K classes by argmax over the per-class log-likelihoods, identical to the original rule for 2 classes; all GMM hyperparameters shared by the per-class GMMs).
- Runners: `validation/runners/<method>.py`, one small module per method with the same interface (`input_arm`, `build`, `fit`, `predict`, `MODULES`), imported by the experiment scripts with importlib so only the method's own env is loaded. Smoke-tested on Imstat (2026-11-04).
- **[D4, decided]** ACDC and Bayesian trees are not included in the benchmark. Both need an expert cell-type × marker table as input, and neither the original code nor the tutorials provide a way to derive it from labelled data (checked: ACDC archived core @ `b6e9821` + `howchihlee/ACDC_tutorial` @ `b76e01e`, `disiji/fc_mondrian` @ `97420ff`; all tables are hand-written CSVs). Writing tables ourselves or deriving them from labels would not be the published method. GateMeClass stays, since it learns its marker table from labelled data (`GateMeClass_train`). The exclusion is justified in the paper's method selection criteria.
- Smoke test of every runner on a tiny subsample of one dataset before any real run.

### Step 5: Shared helpers (`validation/core/`) -- IN PROGRESS
The helpers are implemented alongside the next steps (runners of step 4, tuning of step 6), as they are needed there; the list below is the plan, not a spec.
Small, modular helpers for recurring tasks, plugged into the experiment scripts. **No orchestration:** the core does not run a benchmark or experiment and has no pipeline, task framework or registry; each script (tuning, main benchmark, follow-up experiments) composes the helpers it needs and writes its own code where none fits (e.g. the training sets of the training-data requirements experiment).
- Each helper does one thing, is stateless (seeds are passed in), numpy + stdlib only and Python 3.5-compatible (runs in every method env), and knows nothing about individual methods. Like `Preprocessor` (step 3).
- Helpers are added when a second script needs them; names, signatures and file formats are decided when they are written.

**Implemented (2026-11-04):** `data.py` (load_sample, load_meta_samples / load_meta_labels / load_labels, load_split), `sampling.py` (downsample_indices: random or class-stratified, by n or frac, port of legacy get_downsampling_bool; join_samples), `seeding.py` (derive_seed(*keys): sha256 of config.SEED + any str/int/float keys), `evaluation.py` (confusion_counts with abstention column, class_counts), `resources.py` (track_resources: wall time, peak RSS of the process tree via /proc + VmHWM, GPU via nvidia-smi), `runs.py` (atomic save_json / save_csv, run_metadata, n_threads, now). No score helpers yet (come with the aggregation, step 9).

**Helpers (planned):**
- **Data loading:** load one curated sample (`sample_<NNN>_X.npy` / `_y.npy`), the meta tables (`meta_samples.csv`, `meta_labels.csv`) and a split (`<dataset>.csv`, `imstat_inner.csv` → train / val / test sample lists).
- **Downsampling:** one sample → indices of the kept cells; reproducible (seed passed in), random or class-stratified, by number or fraction of cells.
- **Joining samples:** list of `(X, y)` → concatenated and (seeded) shuffled `X`, `y` and an aligned `sample_ids` vector (needed by methods that use sample structure, e.g. DeepCyTOF's reference sample).
- **Seeding:** derive a seed from `config.SEED` + task identity (e.g. experiment, dataset, split, method, parameter set, seed index).
- **Evaluation counts:** `y_true`, `y_pred`, class list → confusion matrix (true class × predicted class, plus a column for abstentions, -1) and the derived per-class TP, FP, FN, TN, true and predicted class counts, number of cells and abstentions. The class list comes from `meta_labels.csv` (all classes of the dataset, also those absent from a sample), so all samples have the same shape.
- **Resource tracking:** a context manager / wrapper around one block → wall-clock time and peak memory:
  - CPU: peak RSS of the process and its children (e.g. data loader workers; R runs in-process via rpy2), by a polling thread; the increase over the baseline before the block is reported as well;
  - GPU (where used): peak device memory during the block, by polling `nvidia-smi` (framework-independent);
  - stdlib only (`time.perf_counter`, `/proc`, `subprocess`); port of `scalability_wrapper` / `track_memory_cpu` / `track_memory_gpu` from `legacy/validation/utils/val_utils.py` (which use psutil).
- **Output / run helpers (only if shared):** atomic save, skip finished outputs, run metadata (env versions, git commit, hardware: CPU model / threads, GPU model), selecting the task(s) to run (all, or the one given by `SLURM_ARRAY_TASK_ID`).

**What every experiment stores** (convention, using the helpers above):
- per test sample: the evaluation counts. Any metric (abstention-aware or not, per class or averaged) is computed later from them, without the per-cell predictions (step 9);
- optional per-cell outputs (flag per experiment): predicted labels and, where the method has `predict_proba`, predicted probabilities. On where feasible (main benchmark); off where storage would blow up (e.g. training-data requirements) and in the tuning;
- wall-clock time and peak CPU / GPU memory for training (`fit`, incl. the preprocessing fit) and for the inference on each test sample separately (`predict` / `predict_proba` + preprocessing transform); the scalability experiment (step 8) reuses these;
- run metadata: parameters, seed, env versions, git commit, hardware.
- Downsampling is applied to the training samples; test samples are predicted in full unless an experiment states otherwise.

### Step 6: Hyperparameter tuning (Imstat)
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
- For each method, parameter set and outer Imstat split i: fit on the inner train and score on the inner val samples of every inner split of i (`imstat_inner.csv`; scores computed from the stored counts with the scoring rule of D8, step 9) → mean inner score per (method, parameter set, outer split).
- **[D6, decided] Both uses, from the same tuning runs:**
  - **Per fold (Imstat):** for outer split i, use the parameter set with the best mean inner score of split i. This is clean nested CV: the outer test samples of split i play no part in choosing its parameters.
  - **Consensus (all other datasets):** one parameter set per method, with the best mean inner score over all outer splits (K_OUTER × K_INNER inner splits). It is used unchanged for Lt1/Lt2/Lt1b/Lt2b, Flowcyt and BLAST110 ("tuned once on Imstat"). There is no leakage, since tuning uses a different dataset. It is not used for the Imstat benchmark scores, which would be slightly optimistic.
- Output: the inner scores of all parameter sets, plus the selected per-fold and consensus parameter sets per method.
- Grids per method: decided at this step, kept small for the slow methods (e.g. GateMeClass). Grid sizes are chosen from the runtimes of the step 4 smoke tests (cost: K_OUTER × K_INNER = 15 fits per parameter set).
- **Implementation (2026-11-04, details in `working_notes/tuning_plan.md`):** `tune_hyperparameters.py` (grids as constants; Py3.5-compatible, runs in every env).
  - Atom = (method, config, outer split, inner split) = one SLURM array task: fit on the inner train samples (each class-stratified downsampled to `TRAIN_FRAC = 0.1`, data seeds per sample independent of method and config), predict every inner val sample (class-stratified `VAL_FRAC`, 1.0 = full), store per val sample the confusion counts + per-class counts and the time / memory of fit and of each predict.
  - Output: `results/tune_hyperparameters/atoms/<method>/<config_id>/split_<i>_inner_<j>.json` (+ `_counts.csv`, long format, concatenates directly); `config_id` = hash of the params. Frozen task lists `manifests/<array>.csv`, config tables `configs/<method>.csv`.
  - One array per (env, cpu/gpu): `sklearn_cpu`, `flagx_cpu`, `gatemeclass_cpu`, `lda_cpu`, `deepcytof_cpu` (Woody), `flagx_gpu` (TinyGPU); scripts in `slurm_scripts/`. `MaxArraySize` 10000 on both clusters.
  - Failure rule: an atom writes `status: started` first and overwrites it with `ok` / `failed`; an atom that is not `ok` (Python error, 24 h timeout, OOM) puts its config out (per fold for its outer split; consensus needs all 15 atoms). Expected failures: GateMeClass "V", LDA-Abdelaal native `log` (NaN from negative values, also on Imstat).
  - Not yet implemented: aggregation (scores per val sample with the D8 rule, inner means, per-fold / consensus selection, failure report, plots).

### Step 7: Main benchmark
- Tasks: method × dataset × outer fold × model seed. Each task trains on the training samples and predicts every test sample.
- Model seeds (R4.4, R4.m3 seed variability): 3 seeds per fold for stochastic methods (FLAG-X MLP/SOM, DGCyTOF, DeepCyTOF; initialization and on-the-fly downsampling), 1 for deterministic ones (e.g. LDA). Seeds are derived from `config.SEED` + task identity. Report the mean and SD over seeds. If BLAST110 is too expensive, reduce its seeds or downsample harder and state this. The splits stay unchanged (1 CV repetition); tuning uses a single seed (stated as a limitation).
- Optional robustness check: a repeated CV with a different partition seed, for the FLAG-X methods only.
- **[D8, decided] Abstention:** the task is full assignment of every cell.
  - Where abstention is only an option of a method, it is switched off: LDA `rejection_threshold=0` (default), GateMeClass `reject_option=False` (default), DeepCyTOF `rejection_threshold=0` (the 0.4 rejection exists only in the code, not in the paper; it also filters the cells used for the MMD calibration). FLAG-X: to be checked when integrated.
  - Where it is built into the method (DGCyTOF's calibration step), the method is run as published and its abstentions are scored as below.
- Hyperparameters (D6): Imstat tasks use the per-fold parameters of their outer split; all other datasets use the consensus parameters.
- Run sequentially or as a SLURM array job; GPU only where it helps (FLAG-X MLP, DGCyTOF).

### Step 8: Follow-up experiments
Reuse the same splits, curated data and runners; only the task list changes. Candidates from the legacy results (`legacy/results/`), to be confirmed:
- number of training samples / number of events per sample (training data requirements; determines the reduced/minimal training sets used below),
- local training: FLAG-X SOM and MLP trained on the reduced/minimal training sets in the local (CPU, limited resources) setting of Table 1,
- scalability (runtime and memory vs data size),
- robustness to sample order (random sample order trials),
- probabilistic predictions / abstention behaviour,
- channel leave-out,
- preprocessing ablation (decided, see D3): rerun a subset (e.g. a few methods × datasets) with the alternative transformations (uniform cut-off 100, R4.m1; arcsinh) and normalizations, on the reduced/minimal training sets from the training data requirements experiment.

### Step 9: Evaluation
- Separate env. Reads the stored per-sample counts (step 5; per-cell labels / probabilities where stored, e.g. for calibration and abstention analyses), computes sample-wise scores (precision/recall/F1 etc.; port from `legacy/validation/utils/val_utils.py`), writes score tables. The scores implemented in the wrappers (`score`) are not used.
- Scoring rule (D8): an abstained cell (-1) counts as a false negative for its true class and as a false positive for no class. The abstention rate is reported per sample alongside the scores.
- Main comparison on the hard labels only. Predicted probabilities (where stored) are used only for secondary analyses on the methods that provide them (e.g. calibration, abstention behaviour), since not all methods have them (GateMeClass) and they differ in meaning (e.g. DGCyTOF's softmax is from before its reassignment step).
- Aggregation across folds, datasets and methods; statistical comparison (to be decided at this step).

### Step 10: Figures and tables
- Separate env. Reads only score tables, writes to `figures/`.

### Step 11: Documentation
- README "Scripts" section: order of scripts, env per script, how to run sequentially and on SLURM.

---

## 3. Open decisions

| ID | Decision | Options | Recommendation | Status |
|---|---|---|---|---|
| D1 | Datasets in the benchmark | Imstat, Lt1, Lt2, Lt1b, Lt2b, Flowcyt, BLAST110, FlowCAP-I | | **decided:** Imstat (also used for tuning), Lt1, Lt2, Lt1b, Lt2b, Flowcyt, BLAST110 |
| D2 | Storage format of curated data | per-sample `.npy` / CSV | | **decided:** per-sample `.npy` (X, y) + `meta_samples.csv`, `meta_labels.csv` |
| D3 | Preprocessing ownership | shared / method-native / both | | **decided:** default the same shared input for all; deviations only by tuning (shared vs native for LDA-Abdelaal, DeepCyTOF; step 6; no sklearn baseline has internal preprocessing) or for stability (if needed; none so far, GateMeClass runs on the shared input with "E"); post hoc ablation |
| D3a | Shared transformation | arcsinh / log10 with cutoffs | | **decided:** log10 with cut-offs as in the paper (per-channel for Imstat/LT, 100 for Flowcyt/BLAST110); alternatives in the ablation |
| D3b | Shared normalization | z-score / min-max / none | | **decided:** none; z-score / min-max in the ablation |
| D4 | Prior-knowledge methods (ACDC, Bayesian trees) | expert tables / label-derived tables / separate category / drop | | **decided:** dropped; both need an expert marker table and cannot derive it from labelled data |
| D5 | Config format | Python constants / YAML/JSON registry | | **decided:** small shared `config.py` at root, per-script constants otherwise, no registries |
| D6 | Use of tuned parameters | per-fold / consensus / both | | **decided:** both. Per-fold best for Imstat (nested CV); one consensus set (best mean inner score over all outer splits) for all other datasets |
| D7 | Compensation | during curation / as is | | **decided:** with flagx if the FCS file has a spillover matrix (only BLAST110) |
| D8 | Abstention | allow / switch off / score as error | | **decided:** switched off where it is an option (LDA, GateMeClass, DeepCyTOF); built-in abstention (DGCyTOF) kept; abstained cells count as FN for their true class, ab stention rate reported (step 9) |
