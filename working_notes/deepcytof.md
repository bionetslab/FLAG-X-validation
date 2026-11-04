# DeepCyTOF in the benchmark (2026-11-03)

Wrapper: `validation/gating/deepcytof.py` (`DeepCyTOFClassifier`), env `fx_deepcytof`. General wrapper notes (copied code, deviations, env): `additional_methods.md`, section DeepCyTOF.

**Sources**
- Code: https://github.com/KlugerLab/deepcytof @ `ac924b10a77cf2d5c7ea43ea39a172898d71abb1`. Links below use the base `https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/` (abbreviated as `src/`).
- Paper: Li et al. 2017, "Gating mass cytometry data by deep learning", Bioinformatics 33(21):3423-3430, doi:10.1093/bioinformatics/btx448.
  - Quotes and section numbers are from the bioRxiv preprint v5 (doi:10.1101/054411, posted 2017-06-20): https://www.biorxiv.org/content/10.1101/054411v5.full.pdf.
  - The journal version renumbers the sections (bioRxiv §2.3 = journal §3.2). It also moves Appendix A to the Supplementary Material.
  - Not read: the PMC full text (captcha), the journal supplement, and any arXiv version (none exists).

## Summary

- At its core, DeepCyTOF is a small MLP: 12-6-3 softplus layers with a softmax output.
- Everything else addresses mass cytometry across batches with one hand-gated sample:
  - reference-sample selection,
  - the denoising autoencoder (DAE),
  - MMD-ResNet calibration per sample.
- Our setting is different: flow data, all training samples labelled, prediction on held-out samples. The authors never consider it (see 4.).
- We use the method as the authors intended. The caveats are listed below.

## Pipeline in the original code

Mass cytometry, `src/DeepCyTOF.py`:
1. Pick the reference sample. It is the training-pool sample with the smallest mean Frobenius norm of covariance differences, computed on log-preprocessed data.
   - [`src/DeepCyTOF.py#L101-L111`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L101-L111)
   - [`src/Util/DataHandler.py#L59-L78`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py#L59-L78)
2. Apply `log(1 + |x|)`: [`src/Util/DataHandler.py#L25-L27`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py#L25-L27).
3. Train the DAE, unsupervised, on all pooled training-pool samples: [`src/Util/denoisingAutoEncoder.py#L22-L71`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/denoisingAutoEncoder.py#L22-L71).
4. Z-score with a scaler fitted on the denoised reference: [`src/DeepCyTOF.py#L125-L126`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L125-L126).
5. Train the MLP on the labelled cells of the reference sample only. Label-0 ("unlabelled") cells are dropped.
   - [`src/DeepCyTOF.py#L133-L139`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L133-L139)
   - [`src/Util/feedforwadClassifier.py#L79-L117`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py#L79-L117)
6. For each test sample: denoise, scale, predict, train one MMD-ResNet onto the reference, then predict again. The MMD-ResNet uses a random 20 % of the cells and drops cells predicted as rejected.
   - [`src/DeepCyTOF.py#L151-L194`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L151-L194)
   - [`src/Util/MMDNet.py#L115-L156`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/MMDNet.py#L115-L156)
7. Reject a cell (label 0) if its max softmax probability is < 0.4. This exists only in the code; the paper doesn't mention it.
   - [`src/Util/feedforwadClassifier.py#L139-L141`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py#L139-L141)

Flow cytometry, FlowCAP-I, `src/FlowCAP_CellClassifier.py`:
- Per sample: `log(1 + |x|)`, z-scoring, a random cell-level split (25 % train / 75 % test), then the MLP. There is no DAE, no calibration and no reference selection.
  - [`src/FlowCAP_CellClassifier.py#L75-L101`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/FlowCAP_CellClassifier.py#L75-L101)
- The script imports only `DataHandler`, `FileIO` and `feedforwadClassifier`: [`#L20-L25`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/FlowCAP_CellClassifier.py#L20-L25).

## Decisions

### 1. Input pipeline: tuned in step 6, shared vs native (see `PLAN.md`, step 6)
- **shared:** shared preprocessing (log10 with cut-offs, no scaling) + `preprocess_log=False`.
- **native:** curated (untransformed) data + `preprocess_log=True` (original default), as the authors did on flow data (bioRxiv §2.1.3: "For FlowCAP-I datasets, we apply a logarithmic transform, followed by rescaling, as described in Appendix A."; Appendix A: "A_ij ← log(1 + A_ij) … we standardize each column of A to have zero mean and unit variance").
- The internal z-scoring (fitted on the reference sample) is part of the model and stays in both arms.
- The combination shared preprocessing + `preprocess_log=True` is not used: it log-transforms twice. Together with the DAE it broke the method completely in the real-data check (DAE output constant in at least one channel, classifier predicting a single class even on its own reference sample; F1 0.005–0.012 on test, 0.10 on the reference).
- The native arm (raw data + `preprocess_log=True`, `denoise=False`) was not tested yet.
- Paper vs code: the paper writes `log(1 + A)`, the code `log(1 + |x|)` ([`src/Util/DataHandler.py#L26`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/DataHandler.py#L26)). They differ only for negative values, which raw flow data can contain (e.g. compensated BLAST110); the wrapper uses the code version.

### 2. Denoising (DAE): specific to mass cytometry, not used by the authors on flow data
- **Paper:**
  - The DAE is motivated by CyTOF zeros: "All samples in our Mass Cytometry dataset contain large proportions of zero values across different markers. This usually does not reflect biological phenomenon, but rather, occurs due to instabilities of the CyTOF instrument" (bioRxiv §2.2.1).
  - It was disabled for FlowCAP-I: "Here the DAE option is disabled because there are no missing values and the MMD-ResNet option is also disabled because the training and test sets are from the same run and thus do not require calibration" (bioRxiv §2.3, p. 9; journal §3.2).
  - §2.2, p. 7: "DeepCyTOF was employed to classify cells without denoising and without calibration (Section 3.1) [= FlowCAP-I], without calibration but with the denoising preprocessing step (Section 3.2) and with the denoising and with the calibration preprocessing steps (Section 3.3)."
  - Table 1, footnote a: "DeepCyTOF without denoising (DAE) and without calibration (MMD-ResNets)."
- **Code:** the flow script never imports the DAE ([`src/FlowCAP_CellClassifier.py#L20-L25`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/FlowCAP_CellClassifier.py#L20-L25)).
- **Consequence:** `denoise=False` for the benchmark, as the authors did on flow data. In the real-data check the DAE also clearly hurt: F1 0.41–0.55 with it vs 0.68–0.74 without, both with `preprocess_log=False`.
- Paper vs code, CyTOF DAE details (not relevant if it is disabled):
  - The paper trains "a DAE for each batch … selecting the cells with no zeros … dropout probability = proportion of zeros".
  - The code trains one DAE on the pooled training pool, keeps cells with at most 1 zero (`numZerosOK=1`), and uses a fixed `keepProb` of 0.8 or 0.9 ([`src/Util/denoisingAutoEncoder.py#L27-L55`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/denoisingAutoEncoder.py#L27-L55), [`src/DeepCyTOF.py#L76`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L76), [`#L86`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L86)).

### 3. Calibration (MMD-ResNet): a hyperparameter, tuned in step 6 (`calibrate` ∈ {True, False})
- Its purpose is batch effects between samples. The authors disabled it for FlowCAP-I only because training and test cells come from the same sample (quote in 2.). Our test samples are different samples from different patients and runs, so calibration could help or not.
- With the default learning rate (1e-5, [`src/Util/MMDNet.py#L27-L32`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/MMDNet.py#L27-L32)) it changed almost nothing in the real-data check (≤ 0.002 F1). It still accounts for most of the predict time (about 20–25 s per test sample of 5000 cells).

### 4. One reference sample: the authors never consider fully labelled training data, so we use the method as intended
- **Paper:**
  - Single labelled sample as the design premise:
    - Abstract: "DeepCyTOF requires labeled cells from only a single sample."
    - §2.2.3: "we select the sample with the smallest average distance to all other samples to be the reference sample. Once the reference sample is chosen, we use manual gating to label its cells … This is the only label information DeepCyTOF requires, regardless of the total number of samples we want to gate."
  - The motivation is the cost of manual gating (Intro): "In order to avoid gating each dataset separately (which therefore requires labeled samples from each dataset), a domain adaptation procedure is used."
  - All CyTOF experiments use one reference. §3.2: near-perfect performance "despite the fact that it was only given labels from the reference sample." §3.3, multi-center: reference subject 2, applied to the remaining 15.
  - FlowCAP-I is a within-sample setting (25 %/75 % of the cells of each subject, one classifier per subject), not training across samples.
  - Discussion: no variant, ablation or recommendation for several or all labelled samples. Only: unsupervised pre-training helps "especially, when labeled training data is scarce".
- **Code:**
  - The options are only `isCalibrate`, `denoise`, `loadModel` ([`src/DeepCyTOF.py#L49-L51`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L49-L51)). None controls the number of labelled samples.
  - `trainIndex` is only the candidate pool for reference selection and the DAE input, not a labelled training set ([`src/DeepCyTOF.py#L69-L96`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L69-L96)).
- **Consequence:**
  - Reference selection among the training samples (pass `sample_ids` to `fit`), with the MLP trained on the reference sample only. Training on all labelled samples pooled would be our own adaptation, and we don't do it.
  - **Caveat:** classes missing from the reference sample can never be predicted. In the check, one of 6 Imstat training samples had 7 of 8 classes. This should be stated in the paper.
  - The DAE input (all training cells) is irrelevant with `denoise=False`.

## Benchmark settings (summary)

| Parameter | Value | Reason |
|---|---|---|
| input pipeline | tuned: shared (+ `preprocess_log=False`) vs native (raw + `preprocess_log=True`) | 1. |
| `denoise` | `False` | 2. not used by the authors on flow data |
| `calibrate` | tuned, {True, False} | 3. |
| reference sample | original covariance criterion over the training samples (`sample_ids` in `fit`) | 4. |
| all others | original defaults | |

## Real-data check (2026-11-03)

- Setup:
  - Imstat `split_0`: the first 6 training samples and the first 2 test samples, 5000 random cells each.
  - Shared log10 with cut-offs, no scaling.
  - `seed=0`, `n_threads=4`, CPU.
- Script: `check_dcy.py` in the session scratchpad, not kept.

| Configuration | F1 test 1 | F1 test 2 | Fit | Predict per sample |
|---|---|---|---|---|
| Original defaults (`preprocess_log=True`, `denoise=True`) | 0.012 | 0.005 | 18 s | 19 s |
| `preprocess_log=False` | 0.414 | 0.553 | 18 s | 22–25 s |
| `preprocess_log=False, denoise=False` | 0.742 | 0.681 | 3 s | 27 s |

- Calibrated vs uncalibrated F1 differed by ≤ 0.002 in all configurations.
- Wrapper bug ruled out: the DAE rebuilt from the stored weights gives exactly the same output as the live model.
- The F1 values are only a sanity check, not benchmark numbers: 5000-cell reference sample, few samples.

## Notes for the runner

- Call `predict_proba` once per test sample and apply the threshold (0.4) yourself. `predict` and `predict_proba` each retrain the MMD-ResNet, so calling both doubles the cost and gives slightly different results.
- Pass `sample_ids` to `fit`. Without it, the whole training set is treated as one reference sample.
- Predict one test sample at a time, or pass `sample_ids` to `predict`. Otherwise one MMD-ResNet is trained for the pooled test set.
- Python 3.5 env and its own process. `K.clear_session()` in `fit`/`predict` resets every Keras model in the process.
- Not bit-reproducible: the MMD cost samples randomly inside TF.

## Other differences between paper and code (for reference)

- The 0.4 rejection threshold and the randomised F1 handling of rejected cells ([`src/Util/feedforwadClassifier.py#L40-L73`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/Util/feedforwadClassifier.py#L40-L73)) exist only in the code.
- `DeepCyTOF.py` evaluates every sample in `testIndex`, including the reference sample ([`src/DeepCyTOF.py#L151-L154`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/DeepCyTOF.py#L151-L154)). The paper reports on the remaining samples.
- `FlowCAP_CellClassifier.py`:
  - Fits the z-score scaler on the whole sample before the 25/75 split ([`#L84-L89`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/FlowCAP_CellClassifier.py#L84-L89)).
  - Stores its results one index off: `acc[i-1], F1[i-1]` ([`#L100`](https://github.com/KlugerLab/deepcytof/blob/ac924b10a77cf2d5c7ea43ea39a172898d71abb1/src/FlowCAP_CellClassifier.py#L100)).
