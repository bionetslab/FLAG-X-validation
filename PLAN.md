# Benchmark plan

Status: steps 0–1 done (2026-10-29). Built from `NOTES.md`. Open decisions are marked **[D#]** and collected at the end. Small decisions (k, seeds, grids, ...) are made when we reach the step.

---

## 1. General rules

**Reproducibility**
- All randomness is seeded. Seeds are derived deterministically from one global seed plus the task identity (dataset, fold, method, repetition), so any single task can be rerun in isolation with identical results.
- Splits are generated once, saved, and are the single source of truth. Every later step (tuning, benchmark, follow-up experiments) reads them; no step creates its own splits.
- Downsampling happens on the fly inside a task, with the task's seed. Downsampled data is never stored.
- Every task records the software versions of its env, the git commit, the runtime and the peak memory next to its outputs.

**No leakage**
- Test samples are never used for tuning, fitting, or for fitting any data-dependent preprocessing (scalers, cutoffs, channel selection). Fixed transformations with constant parameters (e.g. arcsinh with a fixed cofactor) are allowed.
- Tuning uses only the inner (train/val) splits of the outer training folds.

**Structure**
- One script per step. Parameters are constants at the top of the script, and the docstring names the env to run it in.
- Steps communicate only via files on disk. Each step reads the outputs of the previous one and writes into its own directory. Intermediate results are always saved.
- Every step that runs many tasks enumerates them as a list (e.g. method × dataset × fold). A task is addressed by its index, so the same script runs sequentially (loop over all indices) or as a SLURM array job (`SLURM_ARRAY_TASK_ID`).
- Tasks are idempotent and resumable: a task skips if its finished outputs exist, and writes atomically (temp file → rename) so a killed job leaves no half-written results.
- Fitting/prediction and scoring are separate steps. Methods save predictions (labels + probabilities where available); scores are computed afterwards from the saved predictions.

**Environments**
- One isolated conda env per method (or method group) and per analysis step (data curation, evaluation, plotting). Env files live in `envs/`, each with a header comment stating its purpose and tested versions.
- R-based methods install `r-base` and all R packages in their own env. The system R is never used.
- The shared benchmark code (data loading, split loading, task enumeration, writing outputs) depends only on numpy (+ the standard library) and must run in every method env, including the Python 3.5 envs (`fx_deepcytof`, `fx_acdc`): no f-strings, no `typing.Self`, no walrus operator, etc.

**Methods**
- Every method is used through its wrapper in `validation/gating/` with the common sklearn-like interface (see `CLAUDE.md`). Methods are not reimplemented; copied code is cited with URL, commit and line numbers.
- Method-specific details stay in the wrapper and in a small per-method runner. The benchmark core knows nothing about individual methods.

**Extensibility**
- Adding a dataset = adding one `curate_<dataset>` function (raw path, channel and label maps as constants above it) plus its entries in `config.DATASETS` and `config.CHANNELS`. Everything downstream picks it up from the curated files.
- Adding a method = adding a wrapper, an env file, and a branch in the run script that builds the wrapper with its hyperparameters. Nothing else changes.

---

## 2. Steps (chronological)

### Step 0: Repository scaffolding -- DONE
- Scripts at the repo root with descriptive names, no numbering; figure scripts start with `plot_`. Each script has its parameters as constants at the top and an `OVERWRITE` flag.
- `config.py` (stdlib-only, Py3.5-compatible): `ROOT` (via `__file__`), `DATA_RAW`, `DATA_CURATED`, `RESULTS`, `FIGURES`, `DATASETS`, `CHANNELS`. No registries. **[D5]**
- Layout: `data/raw/` (gitignored), `data/curated/<dataset>/` (gitignored), `results/<script>/`, `figures/<script>/`.
- Envs: `fx_flagx_cpu` (`envs/flagx_cpu.yml`, curation + FLAG-X runs), `fx_plot` (`envs/plot.yml`), one env per method. All with `channel_priority: strict`.
- Still to add when first needed: the shared core `bench/` (numpy-only, Py3.5-compatible: data/split loading, task enumeration, seeding, output writing), `data/splits/` (step 2), the global seed in `config.py`.

### Step 1: Data curation -- DONE
- `curate_data.py` (env `fx_flagx_cpu`): one function per dataset. Load with flagx (one `FlowDataManager` per file) → compensate if the FCS file has a spillover matrix → rename channels (per-dataset map in the script) → keep/order as in `config.CHANNELS` → map labels (per-dataset map in the script, contiguous from 0) → save. Raises on unmapped channels/labels.
- Output in `data/curated/<dataset>/`: `sample_<NNN>_X.npy` (float32) and `sample_<NNN>_y.npy` (int32), `meta_samples.csv` (sample, original filename, extra info such as `patient_id`, `subtype`, `sample_type`), `meta_labels.csv` (label, original label, name, description). Data is stored untransformed. **[D2]**
- Datasets (10): `imstat`, `lt1`, `lt2`, `lt1b`, `lt2b`, `flowcyt`, `blast110_p1`–`p4`. **[D1]**
- Compensation only for BLAST110 (only dataset with `$SPILL`). **[D7]**
- BLAST110: one dataset per tube, blast vs other, FSC-A/SSC-A only, channels named by antibody (`$PnS`).
- Overview figures: `plot_dataset_overview.py` (env `fx_plot`) → `results/dataset_overview/*.csv`, `figures/dataset_overview/` (sample sizes, pooled and per-sample class balance, optional hue by meta column, e.g. `sample_type`).

### Step 2: Splits
- Per dataset: outer k-fold CV on the **group** level (patient, not file), e.g. BLAST110 patients with up to 4 tubes, so no patient appears in both train and test. Optionally stratified on sample-level class composition (e.g. BLAST110 `sample_type`: Dx/FU/NBM differ strongly in size and blast fraction).
- Imstat (tuning dataset): additionally nested inner train/val splits for every outer training fold.
- Saved as `data/splits/<dataset>.json` (fold → train/test sample ids; inner splits for Imstat).
- Small decisions later: k, number of repetitions, stratification.

### Step 3: Preprocessing definition
- Shared preprocessing per dataset: channel selection, transformation (arcsinh with cofactor / log10 with cutoffs), optional scaling (fit on training data only).
- Implemented once in `bench/` (numpy-only) and applied inside each task, so the transformed data is never stored.
- **[D3, decided]** Same treatment for all methods: one shared transformation + normalization. GateMeClass is the only allowed exception, and only if it cannot handle the shared transformation (in the legacy runs it crashed on log10 with cutoffs, see `legacy/gatemeclass_bug_report.md`). The effect of preprocessing is checked post hoc with a small ablation (step 7).
- **[D3a]** Which transformation: arcsinh (cofactor) vs log10 with per-channel cutoffs.
- **[D3b]** Which normalization: e.g. per-channel z-score or min-max, fitted on the training samples only.
- Method-internal preprocessing (DeepCyTOF `preprocess_log`/`denoise`/internal scaler, LDA `transformation`): to be decided per method in step 4.

### Step 4: Method integration
- Per method: env file, wrapper, runner. The runner does: load fold data → preprocess → (downsample) → `fit` → `predict`/`predict_proba` on each test sample → save model, predictions, probabilities, runtime.
- Methods: FLAG-X SOM, FLAG-X MLP (`flagx` 0.2.1 in its own env), DGCyTOF, GateMeClass (needs a new isolated env with its own R, replacing the legacy setup), LDA, DeepCyTOF, ACDC, Bayesian trees. **[D4]**
- Smoke test of every runner on a tiny subsample of one dataset before any real run.

### Step 5: Hyperparameter tuning (Imstat)
- For each method and each outer Imstat fold: grid/random search on the inner splits → best parameters per outer fold.
- **[D6]** How tuned parameters are used: per-fold best (only for Imstat's own outer folds) and/or one consensus parameter set (e.g. best mean inner score across all outer folds) that is then used for all other datasets.
- Grids per method: decided at this step, documented next to the method registry.

### Step 6: Main benchmark
- Tasks: method × dataset × outer fold (× repetition). Each task trains on the training samples and predicts every test sample.
- Run sequentially or as a SLURM array job; GPU only where it helps (FLAG-X MLP, DGCyTOF).

### Step 7: Follow-up experiments
Reuse the same splits, curated data and runners; only the task list changes. Candidates from the legacy results (`legacy/results/`), to be confirmed:
- number of training samples / number of events per sample,
- scalability (runtime and memory vs data size),
- robustness to sample order (random sample order trials),
- probabilistic predictions / abstention behaviour,
- channel leave-out,
- preprocessing ablation (decided, see D3): rerun a subset (e.g. a few methods × datasets) with the alternative transformation and without normalization.

### Step 8: Evaluation
- Separate env. Reads saved predictions, computes sample-wise scores (abstention-aware precision/recall/F1, confusion matrices; port from `legacy/validation/utils/val_utils.py`), writes score tables.
- Aggregation across folds, datasets and methods; statistical comparison (to be decided at this step).

### Step 9: Figures and tables
- Separate env. Reads only score tables, writes to `figures/`.

### Step 10: Documentation
- README "Scripts" section: order of scripts, env per script, how to run sequentially and on SLURM.

---

## 3. Open decisions

| ID | Decision | Options | Recommendation | Status |
|---|---|---|---|---|
| D1 | Datasets in the benchmark | Imstat, Lt1, Lt2, Lt1b, Lt2b, Flowcyt, BLAST110, FlowCAP-I | | **decided:** Imstat (also used for tuning), Lt1, Lt2, Lt1b, Lt2b, Flowcyt, BLAST110 |
| D2 | Storage format of curated data | per-sample `.npy` / CSV | | **decided:** per-sample `.npy` (X, y) + `meta_samples.csv`, `meta_labels.csv` |
| D3 | Preprocessing ownership | shared / method-native / both | | **decided:** shared transformation + normalization for all; GateMeClass exception only if needed; post hoc ablation |
| D3a | Shared transformation | arcsinh / log10 with cutoffs | | open |
| D3b | Shared normalization | z-score / min-max / none | | open |
| D4 | Prior-knowledge methods (ACDC, Bayesian trees) | expert tables / label-derived tables / separate category / drop | | open |
| D5 | Config format | Python constants / YAML/JSON registry | | **decided:** small shared `config.py` at root, per-script constants otherwise, no registries |
| D6 | Use of tuned parameters | per-fold / consensus / both | | open |
| D7 | Compensation | during curation / as is | | **decided:** with flagx if the FCS file has a spillover matrix (only BLAST110) |
