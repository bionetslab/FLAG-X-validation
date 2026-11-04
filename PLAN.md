# Benchmark plan

Status: steps 0–3 done (2026-11-03); steps 4–6 in progress: helpers, runners and the tuning script written, tuning runs pending (2026-11-04). Built from `NOTES.md`.

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
- Adding a method = adding a wrapper, an env file, and a runner (`validation/runners/<method>.py`) that builds the wrapper with its hyperparameters, plus its grid / array entry in the experiment scripts. Nothing else changes.

---

## 2. Steps (chronological)

### Steps 0–3 -- DONE
Scaffolding, data curation (10 datasets), splits (`data/splits/`, K_OUTER = 5, K_INNER = 3), shared preprocessing (`Preprocessor`). Details: `DONE.md`.

### Step 4: Method integration -- IN PROGRESS
Done: wrappers, envs and runners (`validation/runners/`) for FLAG-X SOM / MLP, DGCyTOF, GateMeClass, LDA-Abdelaal, DeepCyTOF and the tabular baselines (RF, HGB, LogReg, kNN, LDA-sklearn, GMM-Mocking); smoke-tested on Imstat. Details: `DONE.md`.
- [ ] Create `fx_flagx_gpu` on TinyGPU (`envs/flagx_gpu.yml`, check `cuda-version` against the node driver) and run one MLP / DGCyTOF atom on a GPU.
- [ ] Create all envs on Woody / TinyGPU and copy `data/curated/imstat/` there.
- [ ] D8 for FLAG-X: SOM returns -1 for BMUs without labelled training events (currently kept as built-in abstention); the MLP never abstains. Confirm.

### Step 5: Shared helpers (`validation/core/`) -- IN PROGRESS
Done: data loading, downsampling, joining, seeding, evaluation counts, resource tracking, output / run helpers (`DONE.md`). Rules: each helper does one thing, is stateless (seeds are passed in), numpy + stdlib only, Python 3.5-compatible, and knows nothing about individual methods. **No orchestration:** experiment scripts compose the helpers and write their own code where none fits. Helpers are added when a second script needs them.
- [ ] Score helpers (D8 rule) together with the tuning aggregation / step 9.

**What every experiment stores** (convention, using the helpers above):
- per test sample: the evaluation counts. Any metric (abstention-aware or not, per class or averaged) is computed later from them, without the per-cell predictions (step 9);
- optional per-cell outputs (flag per experiment): predicted labels and, where the method has `predict_proba`, predicted probabilities. On where feasible (main benchmark); off where storage would blow up (e.g. training-data requirements) and in the tuning;
- wall-clock time and peak CPU / GPU memory for training (`fit`, incl. the preprocessing fit) and for the inference on each test sample separately (`predict` / `predict_proba` + preprocessing transform); the scalability experiment (step 8) reuses these;
- run metadata: parameters, seed, env versions, git commit, hardware.
- Downsampling is applied to the training samples; test samples are predicted in full unless an experiment states otherwise.

### Step 6: Hyperparameter tuning (Imstat) -- IN PROGRESS
Done: `tune_hyperparameters.py`, atom format, SLURM scripts, smoke test (`DONE.md`, `working_notes/tuning_plan.md`).
- [ ] Timing run (default config per method, split 0 / inner 0): add the results to `working_notes/tuning_plan.md`; set `--cpus-per-task` / `--mem` per array and `VAL_FRAC`.
- [ ] Final review of the grids (`working_notes/tuning_plan.md` section 6) by the user.
- [ ] Write the manifests and submit the six arrays (`slurm_scripts/README.md`).
- [ ] Aggregation script: macro F1 per val sample from the counts (D8 rule) → mean over val samples → mean inner score per (method, config, outer split); per-fold and consensus selection (D6); configs with a non-`ok` atom are out; failure / missing-atom report; plots of the inner scores.

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

## 3. Decisions

One line per decision; reasoning and sources in `DONE.md`.

| ID | Decision | Status |
|---|---|---|
| D1 | Datasets | **decided:** Imstat (also used for tuning), Lt1, Lt2, Lt1b, Lt2b, Flowcyt, BLAST110 (p1–p4) |
| D2 | Storage of curated data | **decided:** per-sample `.npy` (X, y) + `meta_samples.csv`, `meta_labels.csv` |
| D3 | Preprocessing ownership | **decided:** the same shared input for all; native input only as a tuning arm for LDA-Abdelaal / DeepCyTOF (no sklearn baseline has internal preprocessing), or for stability (none so far); post hoc ablation |
| D3a | Shared transformation | **decided:** log10 with cut-offs as in the paper (per-channel for Imstat/LT, 100 for Flowcyt/BLAST110); alternatives in the ablation |
| D3b | Shared normalization | **decided:** none; z-score / min-max in the ablation |
| D4 | Prior-knowledge methods (ACDC, Bayesian trees) | **decided:** dropped; both need an expert marker table and cannot derive it from labelled data |
| D5 | Config format | **decided:** small shared `config.py` at root, per-script constants otherwise, no registries |
| D6 | Use of tuned parameters | **decided:** both. Per-fold best for Imstat (nested CV); one consensus set (best mean inner score over all outer splits) for all other datasets |
| D7 | Compensation | **decided:** with flagx if the FCS file has a spillover matrix (only BLAST110) |
| D8 | Abstention | **decided:** switched off where it is an option (LDA, GateMeClass, DeepCyTOF); built-in abstention (DGCyTOF) kept; abstained cells count as FN for their true class, abstention rate reported (step 9) |
