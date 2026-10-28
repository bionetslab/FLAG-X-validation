# Additional methods to be added to the benchmark

- flowDensity (https://doi.org/10.1093/bioinformatics/btu677)
- OpenCyto (https://doi.org/10.1371/journal.pcbi.1003806)
- DeepCyTOF (https://doi.org/10.1093/bioinformatics/btx448)
- LDA (https://doi.org/10.1002/cyto.a.23738)
- ACDC (https://doi.org/10.1093/bioinformatics/btx054)
- Bayesian trees (http://proceedings.mlr.press/v85/ji18a/ji18a.pdf)

## Task
Investigate and report all findings in ``additional_methods.md``
- Investigate which of those methods are truly supervised
- Investigate which of those methods have their source code publicly available
- Check if it is possible to wrap this in a sklearn-like API, see ``../validation/gating`` for examples

Implement:
- For those methods for which it is possible 
    - Add file ``method_name.py`` to ``../validation/gating``
    - Implement the wrapper
        - The wrapper should expose all hyperparameters of the underlying method
    - Add file ``method_name.yml`` to ``../envs`` 
            - should contain an env with which the wrapped method can be used
            - numpy should be installed in each env as well

## Rules
- Use one agent per method
- Never use more than 60 CPUs at once, ensure via taskset
- Yo can set up a conda env per method and one additional one if your work requires, document this as well in this md
- Always create an isolated conda env and install all deps in it; never use system R (R methods get ``r-base`` + all R packages inside their env)
- flowDensity and OpenCyto (gating-template based): only investigate and report in this md, no wrapper
- Try to install with the original deps; if impossible, use any env/versions that work
- Do not reimplement anything from papers; if no usable code exists, just report. Copying source code from repos is fine, but must be declared clearly (url + line numbers in code comments and in this md)
- CPU-only machine for now (will be deployed with GPU): if a CPU-only env does not work, report it and keep the env file anyways
- Smoke tests on a very small subsample of the data only, keep compute short
- Wrapper interface as in ``dgcytof.py`` / ``gatemeclass.py``: sklearn ``BaseEstimator, ClassifierMixin``, ``predict`` returns original labels (``-1`` = abstention/unassigned), ``score`` = macro F1, ``save``/``load``; ``predict_proba`` only where the method natively provides probabilities
- No git branches, no commits



---

# Findings (2026-10-27)

## Overview

| Method | Truly supervised? | Code public? | Wrapped? | Env | Smoke test (macro F1, tiny imstat subsample) |
|---|---|---|---|---|---|
| flowDensity | **No.** Expert-defined sequential density gating; the user tunes parameters by hand. | Yes (Bioconductor, Artistic-2.0) | No (report only, per rules) | none | – |
| OpenCyto | **No.** Expert gating template; the paper says it "doesn't rely on a training data set". | Yes (Bioconductor / RGLab GitHub, AGPL-3.0) | No (report only, per rules) | none | – |
| DeepCyTOF | **Yes** (classifier trained on labelled cells), with unsupervised parts (DAE, MMD-ResNet calibration per test sample) | Yes (KlugerLab/deepcytof, Apache-2.0) | Yes: `validation/gating/deepcytof.py` | `fx_deepcytof` (`envs/deepcytof.yml`), original 2017 stack, works on CPU | 0.456 with original defaults; 0.753 with `denoise=False`; 0.708 with `preprocess_log=False` |
| LDA (Abdelaal et al.) | **Yes**, fully | Yes (tabdelaal/CyTOF-Linear-Classifier, R + MATLAB, MIT) | Yes: `validation/gating/lda.py` (official R code via rpy2) | `fx_lda` (`envs/lda.yml`) | 0.959 |
| ACDC | **No.** Uses a cell-type × marker table (+1/−1/0) and unlabelled cells; labels are used only for evaluation. | Partly. The official Bitbucket repo is deactivated; there is an archived copy on Software Heritage. **No license.** | Yes: `validation/gating/acdc.py` (+ optional, clearly marked, non-original derivation of the table from labels) | `fx_acdc` (`envs/acdc.yml`), original-era stack | 0.850 (with the label-derived table) |
| Bayesian trees (Ji et al.) | **No.** Expert prior table + unlabelled cells; the paper lists it under "Methods without Cell-Level Labels". | Yes (disiji/fc_mondrian, MIT) | Yes: `validation/gating/bayesian_trees.py` (`prior_table` is a required hyperparameter) | `fx_bayesian_trees` (`envs/bayesian_trees.yml`) | 0.543 (4 chains × 20 iterations instead of 50 × 3000; heuristic table built for the test only) |

**Smoke-test setup:**
- Data: 5000 training and 2000 test cells, randomly subsampled (seed 0) from `data/np_files/imstat/arcsinh_cofactor150/{x,y}_{train,test}.npy`. 10 channels, 8 classes (labels 1–8); chance macro F1 ≈ 0.10–0.13.
- Scope: this only checks that fit/predict/predict_proba/score/save/load run. It is **not** a benchmark.
- Reproduced: all four smoke tests were re-run independently after the agents finished and gave the same numbers.

**Common interface:**
- All four wrappers are sklearn `BaseEstimator, ClassifierMixin`.
- `predict` returns the original labels, with `-1` for abstention.
- `score` is macro F1 with abstentions given weight 0, the same convention as `gatemeclass.py`. Note that sklearn still counts `-1` as a class with F1 = 0 in the macro average, so `score` is slightly pessimistic when cells are rejected. For the benchmark, the abstention-aware metrics in `val_utils` are the better choice.
- `save`/`load` use pickle, and `_process_class_labels` matches the existing wrappers.
- `predict_proba` is provided by all four, because all four methods natively produce probabilities or votes.

## Conda envs (all created and tested on this machine, CPU only; numpy in each)

| Env | File | Key versions | Notes |
|---|---|---|---|
| `fx_deepcytof` | `envs/deepcytof.yml` | Python 3.5.6, Keras 1.2.2, TensorFlow 1.0.1 (CPU), numpy 1.12.1, scipy 0.19.1, scikit-learn 0.18.2, h5py 2.7.1 (+ protobuf 3.19.6, theano 1.0.5 via pip) | Original-era stack; no porting needed. Python 3.5, so the wrapper cannot be imported from `flowcy`: the benchmark must run it in its own process. |
| `fx_lda` | `envs/lda.yml` | Python 3.12, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, rpy2 3.6.6, r-base 4.4.3, r-mass 7.3-66 | **Must be used activated** (`conda activate fx_lda` / `conda run -n fx_lda`). Calling the env's python directly makes rpy2 find the system R, which crashes with `undefined symbol: Rf_asBool`. |
| `fx_acdc` | `envs/acdc.yml` | Python 3.5.6, numpy 1.13.3, scipy 1.0.1, scikit-learn 0.19.2, pandas 0.22.0, PhenoGraph 1.5.2 (git pin `7f4bc18`), psutil, `libgfortran=3.0` | Original-era stack. The wrapper is written to also run on Python ≥3.12, but was only tested in this env. |
| `fx_bayesian_trees` | `envs/bayesian_trees.yml` | Python 3.12, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1, joblib 1.6.0 | The original pins no versions; the code runs unchanged on these. |

No extra tooling env was created. Create each env with `conda env create -f envs/<method>.yml`.

**GPU:**
- Only DeepCyTOF could use a GPU, and the GPU build of TF 1.0.1 needs CUDA 8 + cuDNN 5.1, which do not support modern GPUs.
- The networks are tiny (12-6-3 MLP, 25-unit DAE/ResNet), so running on CPU with `n_threads` is recommended.
- A GPU would require porting the copied Keras 1 code to Keras 2/TF 2. This was not done because the original stack works.

## Decisions needed before benchmarking

1. **Supervised vs knowledge-based categories.**
   - Only DeepCyTOF and LDA are supervised in the sense used by FLAG-X/DGCyTOF.
   - ACDC and Bayesian trees need an expert marker/prior table. We have no expert tables for our datasets.
   - Options:
     - (a) Write expert tables per dataset from the marker panels and gating strategies.
     - (b) Derive tables from the training labels. This is not part of either original method; ACDC has an optional, clearly marked derivation in the wrapper, while for Bayesian trees the only derivation is a heuristic in the smoke-test script, not in the wrapper.
     - (c) Report them in a separate "prior-knowledge" category or drop them.
2. **DeepCyTOF preprocessing defaults for arcsinh input.**
   - The original `log(1+|x|)` step and the DAE hurt badly on the smoke data: 0.456 with the defaults vs 0.753 with `denoise=False`.
   - Decide whether to keep the original defaults or tune/report `preprocess_log`/`denoise`.
3. **LDA defaults.** The original functions have no defaults for these, so the wrapper chose:
   - `transformation=False`, because our data is already transformed;
   - `rejection_threshold=0.0`, as in the official notebooks.
4. **Bayesian trees compute.** At the paper defaults (50 chains × 3000 iterations), fitting takes about 2 CPU-hours per 5k cells, growing roughly linearly with N. Use subsampling and/or `n_jobs`.
5. **Inductive vs transductive.**
   - The papers apply ACDC, Bayesian trees and DeepCyTOF's calibration to the data being classified.
   - The wrappers fit on training data and predict on test data by default.
   - `BayesianTreesClassifier(transductive=True)` reproduces the paper setting.
   - DeepCyTOF trains one MMD calibration network per test sample at predict time; pass `sample_ids`.
   - ACDC's random-walk graph is built over the test batch.
6. **ACDC license.** There is no license file in the original repo or the tutorial repo. The code is copied with attribution, but redistribution rights are unclear.
7. **flowLearn.** If a density-gating competitor that learns from labelled data is wanted, flowLearn (Lux et al. 2018, doi:10.1093/bioinformatics/bty082) is the closest one. It is semi-supervised: thresholds from a few manually gated prototype samples are transferred to the rest.

---

## flowDensity (Malek et al. 2015, doi:10.1093/bioinformatics/btu677): report only

Sources:
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4325545/
- Source: https://git.bioconductor.org/packages/flowDensity, devel @ `68dcd62af76ad46f0a2068de7a62b5c225ffde6b`
- Package page: https://bioconductor.org/packages/flowDensity/

**Supervised? No.**
- The vignette calls it a "supervised clustering algorithm". There, "supervised" means the gating strategy is known in advance; it does not mean training on labels.
- The user chains `flowDensity(obj, channels, position, ...)` calls into a gating hierarchy.
- `deGate()` finds each 1-D threshold per file from the density shape: the valley between peaks, slope tracking, sd from a peak, or a percentile.
- The paper describes tuning only as the user setting parameters by hand, e.g. "Parameters can be adjusted on a population-specific basis when extra information is given by a user (e.g. desired percentile cut-off, number of standard deviations from the peak)".
- No function in `R/` takes labels or manual gates as input.

**Availability:**
- Bioconductor 3.23 release 1.46.0; devel is 1.47.0.
- License Artistic-2.0; written in R with C++11.
- Bioconda: newest is `bioconductor-flowdensity` 1.44.0.
- The GitHub mirror github.com/Bioconductor/flowDensity returns 404.

**Wrapper feasibility (not implemented):**
- Needs a hand-built gating hierarchy for each dataset: per node the parent, channel pair, position and `deGate` arguments, and a mapping from each leaf to a label.
- `fit` would be a no-op, unless we add our own tuning of percentiles against y. That would be our extension, not the method.
- `predict` would go through rpy2: flowFrame → sequential gates → `CellPopulation@index`. Cells in no leaf become `-1`.

**Use elsewhere:**
- Finak et al. 2016 (Sci Rep, doi:10.1038/srep20686) found flowDensity and OpenCyto to be the two best algorithms. They were combined into one pipeline and compared with central manual gating.
- flowLearn does **not** compare against flowDensity. It uses flowDensity gates as its ground truth.
- No benchmark was found that uses flowDensity as a fit/predict classifier baseline; the search was not exhaustive.

## OpenCyto (Finak et al. 2014, doi:10.1371/journal.pcbi.1003806): report only

Sources:
- Paper: PMC4148203
- Source: https://github.com/RGLab/openCyto @ `c59b599caa508c0db221481970e7981bc063cbbc`
- Package page: https://bioconductor.org/packages/openCyto

**Supervised? No.**
- The paper says the gating templates "don't rely on a training data set", in contrast to Xcyt, which "fits a model to training data".
- Gate parameters were tuned so the gates are "not obviously wrong, rather than being tuned to provide a good fit to manual gating". Manual gates were imported only for comparison.
- Template columns: `alias, pop, parent, dims, gating_method, gating_args, collapseDataForGating, groupBy, preprocessing_method, preprocessing_args`.
- Built-in gating methods: `gate_quantile`, `rangeGate`, `gate_flowclust_1d/2d`, `gate_mindensity(2)`, `boundary`, `singletGate`, `gate_quad_tmix`, `gate_quad_sequential`, `gate_template`, `gate_custom`. `gate_tail` no longer exists.
- `prior_flowClust` estimates flowClust priors from pooled samples without labels (unsupervised, cross-sample).

**Availability:**
- Bioconductor 3.23 release 2.24.0. License AGPL-3.0-only.
- Imports flowWorkspace, flowCore, flowClust, ncdfFlow, RBGL, graph, data.table; uses cpp11 for C++.
- The GitHub `devel` branch lags behind (2.21.1); the canonical source is git.bioconductor.org.
- Bioconda: newest is `bioconductor-opencyto` 2.22.0 (R 4.5).

**Wrapper feasibility (not implemented):**
- `fit` has nothing to learn from y.
- Needs an expert gating template for each dataset and a mapping from populations to our labels.
- Data must be in flowFrame/GatingSet form with real channel names.
- Output is boolean membership per population, possibly overlapping, which must be resolved to one label per cell.
- Sketch: rpy2 → `GatingSet` → `gt_gating(gatingTemplate(csv), gs)` → `gh_pop_get_indices`.

**Use elsewhere:**
- Finak et al. 2016 used OpenCyto as the framework in an evaluation against manual gating.
- Abdelaal et al. 2019 does not include OpenCyto at all.

## DeepCyTOF (Li et al. 2017, doi:10.1093/bioinformatics/btx448): wrapped

**Supervised? Yes, at its core.**
- A softmax MLP (12-6-3, softplus, L2) is trained on the labelled cells of **one reference sample**. Cells labelled 0 are left out of training.
- Unsupervised parts:
  - The reference sample is chosen by minimal mean covariance distance to the other samples.
  - A denoising autoencoder is trained on all cells.
  - An MMD-ResNet calibration network is trained per test sample at predict time, mapping the sample onto the reference distribution.
- Rejection: cells with max probability < 0.4 get `-1`.

**Code:**
- https://github.com/KlugerLab/deepcytof @ `ac924b10a77cf2d5c7ea43ea39a172898d71abb1` (2017-03-15, unmaintained). License Apache-2.0.
- Python 3 scripts for Keras 1.x; no requirements file.

**Wrapper:** `DeepCyTOFClassifier` in `validation/gating/deepcytof.py`, written to run on Python 3.5 (no f-strings, no `typing.Self`).
- `fit(X, y, sample_ids=None)`: with `sample_ids`, the original reference-selection rule is used and the classifier is trained only on the reference sample.
- `predict(X, sample_ids=None)` runs the per-sample calibration; `predict_proba` returns the native softmax output.
- `save`/`load`: the Keras weights are pickled and the models rebuilt on predict.
- `predict` calls `K.clear_session()`, which also resets any other Keras model in the same process.

**Exposed hyperparameters (original defaults):**
- Preprocessing: `preprocess_log=True`, `unlabeled_label`.
- DAE: `denoise=True`, `dae_keep_prob=0.8`, `dae_num_zeros_ok=1`, `dae_encoding_dim=25`, `dae_l2_penalty=1e-2`, `dae_lr=1e-3`, `dae_n_epochs=80`, `dae_batch_size=128`, `dae_validation_split=0.1`, `dae_patience=25`.
- Classifier: `hidden_layer_sizes=(12,6,3)`, `activation='softplus'`, `l2_penalty=1e-4`, `n_epochs=80`, `batch_size=128`, `validation_split=0.1`, `patience=25`, `lr_initial=1e-3`, `lr_drop=0.5`, `lr_epochs_drop=50`, `rejection_threshold=0.4`.
- MMD-ResNet: `calibrate=True`, `mmd_layer_sizes=(25,25)`, `mmd_l2_penalty=1e-2`, `mmd_init_scale=1e-5`, `mmd_subsample_fraction=0.2`, `mmd_n_epochs=500`, `mmd_batch_size=1000`, `mmd_validation_split=0.1`, `mmd_patience=20`, `mmd_lr_initial=1e-5`, `mmd_lr_drop=0.1`, `mmd_lr_epochs_drop=15`, `mmd_target_validation_split=0.1`, `mmd_target_sample_size=1000`, `mmd_n_neighbors=20`, `mmd_scales=None`, `mmd_weights=None`.
- Wrapper-only, not in the method: `seed`, `n_threads`, `verbosity`.

**Copied code** (base URL `https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/`; each block is marked in the wrapper with URL, file, lines and a change list):

| Original | Lines | Wrapper |
|---|---|---|
| `Util/DataHandler.py` | 25-27, 29-34, 59-77 | `_pre_process_samples_cytof_data`, `_standard_scale`, `_choose_reference_sample` |
| `Util/denoisingAutoEncoder.py` | 20-86 (arch. 57-65) | `_build_dae`, `_train_dae`, `_predict_dae` |
| `Util/feedforwadClassifier.py` | 30-38, 75-125 (arch. 95-106), 139-141 | `_make_step_decay`, `_build_classifier`, `_train_classifier`, `_prediction` |
| `Util/MMDNet.py` | 27-32, 34-74, 76-156 | `_make_mmd_step_decay`, `_construct_mmd`, `_calibrate` |
| `Util/CostFunctions.py` | 11-19, 21-109 | module helpers, `MMD` |

**Deviations from the original:**
- Hard-coded constants became arguments with the same defaults.
- numpy arrays replace the original's sample objects and file I/O; model saving to disk was removed.
- The matplotlib monitoring callbacks were removed.
- The DAE trains on all cells passed to `fit`. This is identical to the original when `sample_ids` is given.
- A GvHD-specific label hack was removed.
- `== None` became `is None`.
- Original quirks are kept: the MMD scale weights default to 3, and Keras `validation_split` takes the last 10% of cells unshuffled.

**Open issues:**
- Must run in its own process (Python 3.5).
- Not bit-reproducible: the MMD cost uses unseeded random sampling.
- With its default learning rate (1e-5), calibration had no effect on the smoke data.
- TF 1.0 prints a harmless `UnboundLocalError` at interpreter exit.

## LDA (Abdelaal et al. 2019, doi:10.1002/cyto.a.23738): wrapped

**Supervised? Yes, fully.**
- `MASS::lda` (all default arguments: priors = class proportions, `tol=1e-4`, `method="moment"`) is trained on labelled cells.
- Optional rejection: if the max posterior is below `RejectionThreshold`, the cell is labelled unknown.
- Optional transformation: `asinh(x/5)` or `log`, plus marker selection.
- Nothing else; the rest of the code is FCS/CSV I/O.

**Code:**
- https://github.com/tabdelaal/CyTOF-Linear-Classifier @ `19c2f8f524ab01d99a46e32989be385f5de20a8c`, MIT.
- Written in R and MATLAB; there is no Python LDA in the repo.
- The MATLAB `CyTOF_LDApredict.m` applies arcsinh twice (L40-48 / L58-66), which looks like a bug. The R version doesn't do this, so the wrapper uses the R version.

**Wrapper:** `LDAClassifier` in `validation/gating/lda.py`.
- Calls the official R code through rpy2.
- The R model is stored as serialized bytes so the object can be pickled; save/load was tested in a fresh process.
- `predict_proba` returns the native posteriors.
- Predictions agree 100% with sklearn's `LinearDiscriminantAnalysis`.

**Hyperparameters** (the originals have no defaults, so these are the wrapper's choices):

| Wrapper | Original | Wrapper default |
|---|---|---|
| `relevant_markers` (0-based) | `RelevantMarkers` | `None`, meaning all columns |
| `transformation` (`'arcsinh'`/`'log'`/`False`) | `Transformation` | `False` |
| `rejection_threshold` | `RejectionThreshold` | `0.0`, as in the official notebooks; the paper sweeps 0.5–0.9 |

**Copied code:**
- `CyTOF_LDAtrain.R` L69-77 (transformation) and L92-95 (`MASS::lda`) → `_R_LDA_FIT`.
- `CyTOF_LDApredict.R` L68-81 (transformation, predict, rejection) → `_R_LDA_PREDICT`.

**Deviations from the original:**
- In-memory matrices replace reading FCS/CSV folders.
- `loadNamespace("MASS")` was added so a loaded model can predict in a fresh R session. It doesn't change results.
- Labels are re-indexed to 0..K-1 internally and the posterior columns reordered back.

**Open issues:**
- `'log'` gives NaN on negative values, and `MASS::lda` then fails, exactly as the original would. This matters for compensated flow data.
- The cost of copying data from Python to R on large datasets was not measured.

## ACDC (Lee et al. 2017, doi:10.1093/bioinformatics/btx054): wrapped

**Supervised? No.**
- Inputs are a cell-type × marker table (+1/−1/0) and unlabelled cells. The paper's tables were written by hand from Cytobank gating hierarchies.
- How it works:
  1. A 2-component GMM is fit per marker.
  2. Each cell gets a score per cell type, plus an "unknown" score.
  3. High-scoring cells per type are clustered with PhenoGraph; the cluster centres become landmarks.
  4. A random walk on a 10-NN graph of cells plus landmarks labels every cell.
- Labels are used only for evaluation. The original has no mechanism for deriving the table from labels; GateMeClass has `GateMeClass_train`, ACDC does not.

**Code:**
- The official repo, https://bitbucket.org/dudleylab/acdc, is **deactivated**.
- Software Heritage has an archived copy: master @ `b6e98212cab060261f13760daf581ac299fc2fa3` (2017-02-08), `swh:1:rev:b6e98212cab060261f13760daf581ac299fc2fa3`.
- The pipeline that connects the functions exists only in the tutorial notebooks: https://github.com/howchihlee/ACDC_tutorial @ `b76e01e6dac838456f1f51f0bff89295200d4d37`.
- **Neither repo has a license.**
- Python 3.5, scikit-learn > 0.18, scipy, pandas, PhenoGraph (MIT).

**Wrapper:** `ACDCClassifier` in `validation/gating/acdc.py`.
- `fit(X, y=None)` builds the landmarks from the training data. This follows the paper's Sec. 2.2.2: shared landmarks from pooled data, then classify each sample.
- `predict` runs the original random walk on new data; unknown cells get `-1`.
- `predict_proba` returns the random-walk probabilities, with **an extra last column for unknown (−1)**.
- **Non-original glue:** if `marker_table=None`, the table is derived from the labels. For each class and marker, the mean of ACDC's own per-marker positivity score is computed: ≥0.75 → +1, ≤0.25 → −1, otherwise 0.

**Exposed hyperparameters (original defaults):**
- `marker_table`, `marker_names`, `n_neighbor=10`, `thres=0.5`, `marker_model_thres=0.0`, `gmm_n_init=10`, `min_events_landmark_clustering=60`, `rw_tol=1e-6`.
- PhenoGraph settings ACDC hard-codes: `pg_k=30`, `pg_directed=False`, `pg_prune=False`, `pg_min_cluster_size=10`, `pg_jaccard=True`, `pg_primary_metric='euclidean'`, `pg_n_jobs=-1`, `pg_q_tol=1e-3`.
- PhenoGraph defaults ACDC doesn't set: `pg_louvain_time_limit=2000`, `pg_nn_method='kdtree'`.
- Wrapper-only, not in the method: `table_pos_thres=0.75`, `table_neg_thres=0.25`, `random_state`, `time_fit_pred`, `verbosity`.

**Copied code** (each block is marked in the wrapper with URL, commit, file and lines):
- `ACDC/cell_type_annotation.py` L26-37, 39-43, 45-64, 66-71, 75-85, 87-107, 111-127, 129-145, 154-159, 183-205.
- `ACDC/random_walk_classifier.py` L7-10, 12-21, 23-27, 29-53, 55-84.
- The pipeline order is adapted from `notebooks/AML_event_classification.ipynb` (JSON lines 40, 55-56, 287-313).

**Deviations from the original** (all marked `# MOD:` or "glue" in the code):
- Hard-coded constants became arguments with the same defaults.
- Changes that don't affect behaviour: `.as_matrix()`→`.values`, `np.float`→`np.float64`, bicg `tol`→`rtol` on scipy ≥1.12, an explicit `scipy.sparse.linalg` import.
- "unknown" is stored internally as the integer −1 instead of the string.
- Landmarks are built on the training data and the random walk runs on the test data.
- The label-derived table and `random_state` seeding are additions.

**Open issues:**
- Only the label-derived table was tested, because the smoke data has no column-to-marker mapping. The fair input would be an expert table.
- PhenoGraph's Louvain step is a bundled binary that can't be seeded, so results are not deterministic.
- Set `pg_n_jobs` explicitly on shared machines; the default −1 uses every CPU.
- PhenoGraph writes temporary files into the working directory.
- The random walk uses pure-Python loops over all test cells and was not tested at million-cell scale.

## Bayesian trees (Ji et al., MLHC 2018, PMLR 85): wrapped

**Supervised? No.**
- The paper requires "(1) a prior knowledge table relating markers and cell-types, and (2) unlabeled ... data" and states the model "requires no cell-level labels". Table 3 lists it under "Methods without Cell-Level Labels", next to ACDC.

| Variant | Inputs | Labels used? |
|---|---|---|
| **MP** (single-sample cell classification; wrapped) | Table + unlabelled cells. Tree structures are drawn from a Mondrian prior shaped by the table, cut positions refined by MCMC on a Gaussian-leaf likelihood, and the final call is a majority vote. | none |
| MP-Prior (baseline) | Table + data range only | none |
| MP-RE (random effects, multi-sample) | Table + several unlabelled samples. The diagnosis use adds per-subject healthy-vs-AML labels and a logistic regression on cell-type proportions. | subject-level only |
| Global MP (diagnosis baseline) | Table + pooled cells per group | group-level only |

**Code:**
- https://github.com/disiji/fc_mondrian @ `97420ff311242afe103c45130ada509e1e60a0ac` (2019-11-11, last commit), by the first author. MIT.
- Python; `src/` was ported from Python 2 to Python 3 in commit `0220daf`.
- No requirements file. Imports numpy, scipy, pandas, matplotlib, scikit-learn, joblib.

**Wrapper:** `BayesianTreesClassifier` in `validation/gating/bayesian_trees.py`.
- `prior_table` is **required**. `y` is only used to set up and check the label space.
- `fit` runs the MCMC chains in parallel with joblib; `predict` returns the majority vote.
- `predict_proba` returns the normalised vote matrix.
- It never abstains, and labels without a table row can never be predicted.
- `transductive=True` reproduces the paper's setting (MCMC on the data being classified); the default is `False`.

**Exposed hyperparameters (original defaults):**
- `prior_table` (required), `n_mcmc_chains=50`, `n_mcmc_samples=3000`, `mcmc_gaussian_std=0.1` (halved every n/10 iterations, as in the original), `n_burnt_samples=20`.
- `cut_beta_params`: the original Beta table (paper φ0=5, φ1=2).
- `dim_priority`: 0/1/100/1000 (paper γ).
- `cov_reg=0.001`, `random_state=0` (chain i uses seed `random_state+i`).
- Wrapper-only: `transductive=False`, `n_jobs=1`, `time_fit_pred`, `verbosity`.
- The original `budget` argument is not exposed because it has no effect for this variant.

**Copied code** (base URL `https://github.com/disiji/fc_mondrian/blob/97420ff311242afe103c45130ada509e1e60a0ac/`):

| Original | Lines | Changes |
|---|---|---|
| `src/flowMP_sample.py` `draw_informed_Mondrian` | 50-73 | prior dicts passed as arguments |
| `src/flowMP_sample.py` `draw_informed_Mondrian_at_t` | 76-154 | priority dict passed as argument |
| `src/flowMP_sample.py` `Mondrian_Gaussian_perturbation` | 157-189 | verbatim |
| `src/flowMP_sample.py` `MP_mcmc` | 192-227 | passes the new parameters through |
| `src/flowMP_sample.py` default prior values | 61-69, 98-102 | copied as defaults |
| `src/flowMP_compute.py` `comp_log_p_sample` | 6-36 | `cov_reg` made a parameter |
| `src/flowMP_classify.py` `classify_cells` | 55-102 | verbatim |
| `src/flowMP_classify.py` `classify_cells_majority` | 105-125 | also returns the vote matrix; vote loop vectorised |
| `run/flowMP-AML-run-v1.0.ipynb` cells 3 and 5 | – | data bounds and chain loop, adapted |

**Fidelity:** one chain and the majority vote give exactly the same result as the original repo code at the same seed.

**Deviations from the original:**
- The default mode is inductive rather than the paper's transductive setting.
- `predict_proba` is an addition.
- `predict` seeds the global numpy RNG so random tie-breaks are reproducible.
- Original quirks are kept: the likelihood can underflow to −inf, the annealing schedule is odd when `n_mcmc_samples < 10`, and the table splits in classification differ from those in sampling.

**Open issues:**
- Needs an expert prior table per dataset (see the decisions above).
- Expensive at the default settings.
- MP-RE and the diagnosis variants are not wrapped.

---

# Supervised gating methods: literature search (2026-10-27)

Dates are online/first-published dates. "(note: …)" marks borderline cases.

| Method | Publication DOI | Date | Code availability |
|---|---|---|---|
| SVM leukemic-cell detection (Toedling et al.) | 10.1186/1471-2105-7-282 | 2006-06 | on request (R package "cytomics", per paper; uses e1071 SVM) |
| Statistical pattern recognition / SVM (Quinn et al.) (note: link to the FlowCAP-I "RadialSVM" entry unverified) | 10.1002/cyto.a.20416 | 2007-05 | not found (paper not checked for a statement) |
| Transfer learning for auto-gating (Lee, Stoolman, Scott) (note: no DOI; PMLR v27, proceedings.mlr.press/v27/lee12a) | none (PMLR) | 2012-06 | not found |
| PhenoGraph classify (Levine et al.) (note: mainly unsupervised clustering paper; `classify` labels cells by transfer from manually gated reference populations) | 10.1016/j.cell.2015.05.047 | 2015-06 | https://github.com/jacoblevine/PhenoGraph (Python; `phenograph/classify.py`) |
| Scaffold maps (Spitzer et al.) (note: assigns unsupervised clusters to manually gated "landmark" populations; no trained classifier) | 10.1126/science.1259425 | 2015-07 | https://github.com/nolanlab/scaffold (R; README says it moved to ParkerICI/grappolo, vite, panorama) |
| DeepCyTOF | 10.1093/bioinformatics/btx448 | 2017-07 | https://github.com/KlugerLab/deepcytof (Python) |
| flowLearn (note: semi-supervised; transfers gates from a few gated samples) | 10.1093/bioinformatics/bty082 | 2018-02 | https://github.com/mlux86/flowLearn, which redirects to github.com/mlx404/flowLearn (R); removed from Bioconductor |
| GateFinder (note: learns a 2D polygon gating strategy for a given target population) | 10.1093/bioinformatics/bty430 | 2018-05 | https://bioconductor.org/packages/GateFinder (R) |
| hypergate (note: learns a hyperrectangle gate for a given labelled population) | 10.1093/bioinformatics/bty491 | 2018-06 | https://cran.r-project.org/package=hypergate ; https://github.com/ebecht/hypergate (R) |
| CyTOF Linear Classifier (LDA; Abdelaal et al.) | 10.1002/cyto.a.23738 | 2019-03 | https://github.com/tabdelaal/CyTOF-Linear-Classifier (MATLAB + R) |
| Supervised MRD assessment in B-ALL (Reiter et al.) (note: clinical MRD, blast vs non-blast cells) | 10.1002/cyto.a.23852 | 2019-07 | not found (paywalled; no statement checked) |
| optimalFlow (optimalFlowTemplates + optimalFlowClassification) | 10.1186/s12859-020-03795-w | 2020-10 | https://bioconductor.org/packages/optimalFlow (R) |
| CyAnno | 10.1093/bioinformatics/btab409 | 2021-05 | https://github.com/abbioinfo/CyAnno (Python) |
| CytoPy (note: analysis framework; its CellClassifier module trains supervised classifiers on gated data) | 10.1371/journal.pcbi.1009071 | 2021-06 | https://github.com/burtonrj/CytoPy (Python; repo archived) |
| Flowformer, set transformer for blast identification (Wödlinger et al.) (note: MRD in B-ALL) | 10.1016/j.compbiomed.2022.105314 | 2022-02 | https://github.com/mwoedlinger/flowformer (Python) |
| DGCyTOF | 10.1371/journal.pcbi.1008885 | 2022-04 | https://github.com/lijcheng12/DGCyTOF (Python/Jupyter) |
| DNN for CLL MRD (Salama et al., Mayo) (note: clinical event-level DNN) | 10.3390/cancers14102537 | 2022-05 | not available (no code statement; Python/TensorFlow/Keras used) |
| AI-assisted CLL MRD (Bazinet et al.) (note: clinical; AI model trained on 41 manually gated files; software not identified, possibly commercial) | 10.1002/cyto.b.22116 | 2023-02 | not verified (paywalled) |
| CytOpT (note: optimal-transport label/proportion transfer from a gated source sample; outputs class proportions rather than per-cell gates) | 10.1214/22-aoas1660 | 2023-06 (arXiv 2006.09003, 2020) | https://cran.r-project.org/package=CytOpT (R); https://pypi.org/project/CytOpT/ (Python) |
| CyGate | 10.1021/acs.analchem.3c03006 | 2023-11 | https://github.com/seungjinna/cygate, which redirects to HanyangBISLab/cygate (Java .jar only) |
| FATE (feature-agnostic transformer; Weijler et al.) (note: MRD blast detection across panels) | 10.1109/wacv57701.2024.00777 | 2024-01 | https://github.com/lisaweijler/FATE (Python) |
| MAGIC-DR (note: clinical AML MRD; single-cell XGBoost) | 10.1002/cyto.b.22168 | 2024-02 | not found |
| EverFlow DL cell classifier, ALOT tube (Cheng et al.) (note: clinical acute leukemia) | 10.1038/s41598-024-58580-z | 2024-04 | not available (no code statement; data on request) |
| GateMeClass | 10.1093/bioinformatics/btae322 | 2024-05 | https://github.com/simo1c/GateMeClass (R) |
| GateNet | 10.1016/j.compbiomed.2024.108820 | 2024-07 (arXiv 2312.07316, 2023-12) | https://github.com/wwu-mmll/gatenet (Python/Jupyter) |
| AutoGater (note: weakly supervised; live/dead gating from scatter only) | 10.1038/s41598-024-66936-8 | 2024-10 (preprint 10.1101/2022.12.07.519491, 2022-12) | https://github.com/netrias/autogater (Python/Jupyter) |
| flowNetworks, local+global feature learning for MRD (Weijler et al., ICPR 2024) | 10.1007/978-3-031-78198-8_21 | 2024-12 | https://github.com/lisaweijler/flowNetworks (Python) |
| UNITO | 10.1038/s41467-025-56622-2 | 2025-02 (preprint 10.1101/2024.05.06.592739, 2024-05) | https://github.com/KyleeCJ/UNITO (Python) |
| ElastiGate | 10.1038/s41598-025-99118-1 | 2025-05 | FlowJo plugin https://www.flowjo.com/exchange/plugin/bd-elastigate-plugin (proprietary, BD/FlowJo); modified bUnwarpJ at https://github.com/airvine2/bUnwarpJ (Java) |
| flowMagic (preprint) | 10.1101/2025.10.07.679685 | 2025-10 | https://github.com/semontante/flowMagic (R) |
| CCADDAS (Seheult et al., Mayo) (note: clinical pipeline; DNN event classifier trained on manually annotated controls; follow-ups 10.3390/cancers17101688 and 10.1038/s41375-026-02986-3) | 10.1182/bloodadvances.2025016126 | 2025-12 | on request (stated in the 2026 Leukemia follow-up; Python/TensorFlow on Google Vertex AI); Blood Adv paper itself not checked |
