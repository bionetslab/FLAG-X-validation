# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reproduction code for the FLAG-X paper (*FLAG-X: Hybrid machine learning workflows for automated gating of clinical flow cytometry data*). FLAG-X itself is **not** in this repo; it is the separate `flagx` package (github.com/bionetslab/FLAG-X, v0.2.1 used for the paper). This repo benchmarks FLAG-X against competing cell-type classification methods.

The repo is mid-restructure (Oct 2026). The new benchmark pipeline is being built step by step following `PLAN.md`; steps 0 (scaffolding) and 1 (data curation) are done. The pre-revision code is frozen in `legacy/`.

- `PLAN.md`: **the authoritative plan**: general rules, chronological steps (completed ones marked `-- DONE`), and the decision table (D1–D7). Read it before starting a new step. Open decisions (D3a, D3b, D4, D6) are settled only when the step needs them; don't decide them unasked.
- `config.py`: constants shared by several scripts (`ROOT`, `DATA_RAW`, `DATA_CURATED`, `RESULTS`, `FIGURES`, `DATASETS`, `CHANNELS`). Stdlib-only and Python 3.5-compatible, since every env imports it.
- `curate_data.py` (env `fx_flagx_cpu`): raw FCS/CSV → `data/curated/<dataset>/` (`sample_<NNN>_X.npy` float32, `sample_<NNN>_y.npy` int32, `meta_samples.csv`, `meta_labels.csv`). Untransformed; compensated with flagx only if the file has a spillover matrix (BLAST110).
- `plot_dataset_overview.py` (env `fx_plot`): sample size and class balance figures → `results/dataset_overview/`, `figures/dataset_overview/`.
- `validation/gating/`: sklearn-compatible wrappers around competing methods.
- `envs/`: one conda env file per step or method: `flagx_cpu.yml` (`fx_flagx_cpu`, flagx 0.2.1 from bioconda + CPU torch; curation and FLAG-X runs), `plot.yml` (`fx_plot`), and the method envs below. Create with strict channel priority (`CONDA_CHANNEL_PRIORITY=strict conda env create -f envs/<name>.yml`); otherwise flagx resolves to ancient scanpy/anndata.
- `legacy/`: the old `main.py` / `main_plotting.py` / `run_experiments.py` drivers, slurm scripts, old envs (`flowcy`), the old `validation` package (incl. `utils/val_utils.py` with abstention-aware metrics and `plt/`), and old results. It is reference material; don't modify it.
- `NOTES.md`: the original design notes that `PLAN.md` was built from.
- `working_notes/additional_methods.md`: the task spec and detailed findings for the newer wrappers (supervision status, sources, env versions, smoke-test results, open decisions). Read it before touching those wrappers.
- `data/raw/` and `data/curated/` are gitignored, as are `results/`, `*.csv` and `*.pdf`. Datasets (`config.DATASETS`): `imstat` (also the tuning dataset), `lt1`, `lt2`, `lt1b`, `lt2b`, `flowcyt`, `blast110_p1`–`p4` (one dataset per BLAST110 tube; group CV splits by `patient_id`).

There is no test suite, linter, or build step.

## Conventions for new scripts

- One script per step at the repo root, descriptive name, no numbering. Figure scripts start with `plot_`.
- Parameters are constants at the top; the docstring names the env. Paths come from `config.py`, never hard-coded.
- Each part has an `OVERWRITE` flag and skips finished outputs.
- Keep code simple and linear: one function per dataset/figure, small helpers only, no registries or frameworks.
- Don't rerun the full curation or other long jobs unasked; the user runs them manually. Smoke-test on a subset and write test output to a scratch location.

## Wrapper interface (`validation/gating/`)

Every wrapper follows the same contract. Keep new ones consistent with `dgcytof.py` / `gatemeclass.py`:
- The class subclasses `sklearn.base.BaseEstimator, ClassifierMixin`. All method hyperparameters are exposed in `__init__`.
- `predict` returns the original labels. `-1` means abstention/unassigned. Labels are mapped internally via `_process_class_labels`.
- `score` is macro F1 (abstentions get weight 0).
- `save`/`load` use pickle.
- `predict_proba` exists only where the method natively yields probabilities or votes.
- Code copied from upstream repos must be marked in comments with the URL, commit and line numbers. Changes to copied code are marked `# MOD:`. The methods are not reimplemented.

| Wrapper | Method | Env | Notes |
|---|---|---|---|
| `dgcytof.py` (`DgcytofClassifier`) | DGCyTOF, torch | legacy `flowcy` | uses `typing.Self` → Python ≥3.11 |
| `gatemeclass.py` (`GateMeClassClassifier`) | GateMeClass, R via rpy2 | legacy (`legacy/gmc_install.txt`) | known issues: `legacy/gatemeclass_bug_report.md` |
| `lda.py` (`LDAClassifier`) | Abdelaal et al. LDA, official R code via rpy2 | `fx_lda` (`envs/lda.yml`) | **The env must be activated** (`conda activate fx_lda` / `conda run -n fx_lda`). Otherwise rpy2 picks up the system R and crashes. |
| `deepcytof.py` (`DeepCyTOFClassifier`) | DeepCyTOF, Keras 1.2.2 / TF 1.0.1 | `fx_deepcytof` (Python 3.5) | must stay Python 3.5 syntax (no f-strings); CPU only |
| `acdc.py` (`ACDCClassifier`) | ACDC + PhenoGraph | `fx_acdc` (Python 3.5) | must stay Python 3.5 syntax; needs a cell-type × marker table, so it is not truly supervised |
| `bayesian_trees.py` (`BayesianTreesClassifier`) | Mondrian-process Bayesian trees | `fx_bayesian_trees` | needs `prior_table`, so it is not truly supervised; expensive at paper defaults |

The envs are mutually incompatible (Python 3.5 vs 3.12, different R/TF stacks). A benchmark must run each method in its own process/env rather than importing all wrappers together. Create an env with `conda env create -f envs/<method>.yml`. Every env includes numpy.

R-based methods get `r-base` and all R packages inside their own conda env. Never use the system R.

## Benchmark design (summary of `PLAN.md`)

- Pipeline: curated data → splits → preprocessing → tuning (Imstat) → fit/predict → scores → figures. Each step saves its outputs and the next reads them.
- Splits are generated once (group-level k-fold CV, nested inner splits for Imstat) and are the single source of truth.
- One shared transformation + normalization for all methods, applied on the fly and fitted on training data only. Downsampling is also on the fly, with fixed seeds.
- The planned shared core `bench/` (data/split loading, task enumeration, seeding) must be numpy-only and Python 3.5-compatible, so it runs in every method env.
- Every experiment must be runnable sequentially or as a SLURM array job (task index → method × dataset × fold).

## Legacy drivers (reference)

`legacy/main.py` defines `main_<experiment>()` functions with docstrings (e.g. `main_data_processing`, `main_som_classifier`, `main_dgcytof`, `main_gatemeclass`). `legacy/run_experiments.py -m {som,dgcytof,fcnn,gatemeclass,som_parameter_tuning}` dispatches to them for the slurm jobs in `legacy/slurm_scripts/`. These scripts import `validation.*` / `flagx` from the old layout, so they will not run unchanged against the current tree.
