# Training data requirements: evaluation plan (PLAN.md step 8)

Status: draft for review (2026-11-18). Script: `training_data_requirements_eval.py` (skeleton only, not implemented).
Experiment design: `working_notes/training_data_requirements_plan.md`. Reviewer comment: R4.5
(`../flagx_revision/REVISION_PLAN.md`), also R2.19 / R2.21 and the test set reuse note (REVISION_PLAN.md:687).

## Goal

From the step-8 atoms (`results/training_data_requirements/`) and the full-data baseline of step 7
(`results/benchmark_eval/scores/`): the score curves over n_samples × n_events, the minimal training configuration
per method × dataset under a fixed criterion, its variability over trials / folds, a sensitivity analysis over
criteria, and a selection-bias check (cross-fitting). The minimal configurations are read by step 9.

## Planned outputs (`results/training_data_requirements_eval/`)

- `atoms.csv`: one row per manifest atom: status (ok / failed / timeout / oom / skipped / missing), phase, fit time /
  peak memory, n_train_events, skipped_by.
- `scores/<dataset>.csv`: one row per atom × test sample (per-class counts → f1 / prec / rec, macro F1 or F1 of
  label 1 for binary datasets), as `benchmark_eval.py`.
- `sample_scores.csv`: per method × dataset × cell × test sample: score averaged over the trials (NaN if any trial
  has no score), plus the step-7 baseline (seed mean) and the paired difference d = cell − baseline.
- `curves.csv`: per method × dataset × cell: mean score and mean d over all test samples (pooled over folds),
  bootstrap CI of mean d (donor-level resampling for Imstat), n, and the per-fold / per-trial means.
- `minimal.csv`: per method × dataset × criterion: minimal cell, selected on all folds (pooled); also per trial and
  per fold (spread), and joint minimum (fewest total training events) vs. greedy (samples first, then events).
- `crossfit.csv`: per method × dataset × criterion × fold k: cell selected on the folds ≠ k, its score on fold k,
  and the pooled selection for comparison (in-sample vs. cross-fitted score at the minimal config).
- `minimal_configs.json`: the minimal cell per method × dataset under the primary criterion (pooled; and the
  per-fold cross-fitted cells), read by step 9.

## Open decisions (to settle before implementation)

- Margin δ (decided 2026-11-18): default δ = 0.02 macro F1, kept as a stated convention; no data-driven δ (the seed
  variability of step 7 rests on 3 seeds only, too few). Instead the dependence on δ is shown explicitly: a δ
  sensitivity figure (minimal samples, events per sample and total training events of the minimal cell over δ = 0.001–0.1, per
  method × dataset; prototype `scratchpad/plot_delta_sensitivity.py`). Testing "Δ = 0" instead was rejected: it
  accepts noisy cells early and rejects precise ones (the margin is then the CI width, i.e. depends on the test size).
- Order of the grid for "minimal" (greedy samples-first as in legacy, or joint by total events; both reported, one
  primary).
- Step 9 uses the pooled or the cross-fitted per-fold config (decide after comparing them).
- Treatment of cells cut off (timeout / oom / skipped): not eligible as minimal; a cell counts only if all folds
  and trials have scores.
- Bootstrap: N_BOOT, percentile CI, resampling unit (test sample; donor for Imstat).
- Stability rule: the minimal cell must also hold for all larger n_samples at the same n_events (see the back-check).

## Back-check on the legacy results (2026-11-18)

Data: `legacy/results/num_samples_num_events/{fcnn,som}/<dataset>/.../random/detailed_res/nevents_<e>_nsamples_<s>/res_df_sw_avg_f1.csv`
(macro F1 per test sample; one split, one sample order). Baseline: `legacy/results/gating_performance/` (full data, as
in `legacy/main_plotting.py::fig5s_num_samples_num_events`). Same test samples in both (50 / 40 / 12 rows); paired by
row. Legacy rule: greedy (samples first, then events), n_samples ≥ 2. On the legacy figure grid (n_samples multiples
of 5 + full, events 5k–all), `legacy_ci` (mean ≥ lower bound of the baseline bootstrap CI) reproduces Table 1.

| grid | dataset | method | base | CI hw | legacy_ci | Δ ≥ −0.01 | Δ ≥ −0.02 | Δ ≥ −0.05 | noninf 0.01 | noninf 0.02 | noninf 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | Imstat | MLP | 0.956 | 0.011 | 10×20k | 10×20k | 5×10k | 5×5k | 10×50k | 10×20k | 5×5k |
| legacy | Imstat | SOM | 0.945 | 0.013 | 5×5k | 5×5k | 5×5k | 5×5k | 5×10k | 5×5k | 5×5k |
| legacy | LT1 | MLP | 0.857 | 0.020 | 15×all | 15×all | 15×all | 15×50k | 15×all | 15×all | 15×all |
| legacy | LT1 | SOM | 0.702 | 0.012 | 5×5k | 5×20k | 5×5k | 5×5k | 10×all | 5×20k | 5×5k |
| legacy | LT2 | MLP | 0.848 | 0.034 | 15×all | 30×all | 25×50k | 10×all | 58×all | 30×all | 15×all |
| legacy | LT2 | SOM | 0.632 | 0.018 | 5×5k | 5×5k | 5×5k | 5×5k | 5×10k | 5×5k | 5×5k |
| legacy | LT1b | MLP | 0.971 | 0.020 | 5×5k | 5×5k | 5×5k | 5×5k | 5×10k | 5×5k | 5×5k |
| legacy | LT1b | SOM | 0.957 | 0.024 | 5×5k | 5×5k | 5×5k | 5×5k | 5×50k | 5×5k | 5×5k |
| legacy | LT2b | MLP | 0.956 | 0.026 | 5×5k | 5×5k | 5×5k | 5×5k | 5×50k | 5×5k | 5×5k |
| legacy | LT2b | SOM | 0.944 | 0.028 | 5×5k | 5×5k | 5×5k | 5×5k | 5×5k | 5×5k | 5×5k |
| legacy | Flowcyt | MLP | 0.865 | 0.066 | 5×50k | 5×all | 5×all | 5×50k | 10×all | 10×50k | 5×all |
| legacy | Flowcyt | SOM | 0.798 | 0.043 | 5×20k | 5×20k | 5×20k | 5×20k | 5×all | 5×20k | 5×20k |
| full | Imstat | MLP | | | 3×50k | 4×50k | 3×10k | 2×10k | 7×all | 3×all | 2×20k |
| full | Imstat | SOM | | | 2×10k | 2×10k | 2×5k | 2×5k | 2×10k | 2×10k | 2×5k |
| full | LT1 | MLP | | | 7×all | 8×all | 7×all | 6×all | 13×50k | 8×all | 6×all |
| full | LT1 | SOM | | | 3×20k | 3×all | 3×20k | 2×5k | 7×50k | 3×all | 2×all |
| full | LT2 | MLP | | | 6×all | 30×all | 13×all | 6×all | 58×all | 30×all | 6×all |
| full | LT2 | SOM | | | 2×5k | 3×5k | 2×5k | 2×5k | 3×5k | 3×5k | 2×5k |
| full | LT1b | MLP | | | 2×10k | 3×50k | 2×20k | 2×5k | 3×50k | 3×50k | 2×10k |
| full | LT1b | SOM | | | 3×5k | 3×5k | 3×5k | 2×all | 5×50k | 4×10k | 3×5k |
| full | LT2b | MLP | | | 2×20k | 3×all | 3×all | 2×1k | 5×50k | 3×all | 3×1k |
| full | LT2b | SOM | | | 2×10k | 2×50k | 2×10k | 2×5k | 3×50k | 3×20k | 2×10k |
| full | Flowcyt | MLP | | | 2×all | 5×all | 4×all | 2×all | 10×all | 10×50k | 3×all |
| full | Flowcyt | SOM | | | 3×50k | 5×20k | 4×50k | 3×50k | 5×all | 5×20k | 3×50k |

`Δ ≥ −x`: mean paired difference to the baseline (point estimate, no CI); `noninf x`: lower bound of the 95 % paired
bootstrap CI of the mean difference > −x. `full` = all legacy cells (n_samples 2–20 in steps of 1, then steps of 5;
events 100 – all).

Findings:
- **Leniency is real where the baseline CI is wide.** LT2 MLP (CI half-width 0.034): `legacy_ci` accepts 15×all at
  Δ = −0.028; Flowcyt MLP (0.066): 5×50k at Δ = −0.030. `noninf 0.02` gives 30×all and 10×50k there and agrees with
  `legacy_ci` on the datasets with narrow CIs.
- **Pairing is much more precise:** median half-width of the paired CI (cells with Δ > −0.03) is 0.005–0.017 vs.
  0.011–0.066 for the baseline CI (Flowcyt MLP 0.017 vs. 0.066). A non-inferiority test on the paired difference is
  therefore not more lenient for small test sets, and more powerful than the legacy rule.
- **The legacy "5 samples" is mostly the grid floor** (first cell ≥ 2 of the figure grid is 5). On the full grid most
  minima are 2–4 samples (LT1 / LT2 MLP 6–30).
- **Single order → noisy, non-monotone curves.** The first passing cell is often followed by failing larger cells
  (same n_events): e.g. `noninf 0.02` LT2 MLP 30×all fails again at 40, 45, 50, 55; LT1b SOM 4×10k fails again at
  11 of 24 larger n; also for `legacy_ci` (LT2 MLP 6×all: 10 of 22). The trial × fold mean of step 8 smooths this;
  in addition a stability rule ("passes and all larger n_samples at the same n_events pass") is worth considering.
- δ = 0.01 is too strict for a single-order curve (LT2 MLP needs all 58 samples); δ = 0.05 adds little over the point
  estimate. δ = 0.02 as primary looks reasonable.

Scripts, tables and figure (git-ignored `scratchpad/`, env `fx_plot`): `scratchpad/legacy_criterion_check.py` →
`scratchpad/results/legacy_criterion_check/legacy_deltas.csv` (Δ, SD, paired CI per method × dataset × cell) and
`legacy_minimal.csv`; `scratchpad/plot_legacy_criterion.py` →
`scratchpad/figures/legacy_criterion_check/legacy_criterion.png` (Δ over n_samples per events cap with the
paired CI band, −δ, the legacy threshold, and both selected cells; the CI bar at the circle is the tested quantity).
`scratchpad/plot_delta_sensitivity.py` → `scratchpad/figures/legacy_criterion_check/delta_sensitivity.png` and
`scratchpad/results/legacy_criterion_check/delta_sensitivity.csv` (minimal cell over δ). On the legacy data the
greedy minimal sample number falls steeply up to δ ≈ 0.02 and is flat afterwards for most datasets (LT2 / LT1 MLP:
slow decline up to δ ≈ 0.05–0.09); the events per sample of the minimal cell jump around (greedy selection, one
order), so a joint rule (fewest total events) may give a smoother second axis.
Joint rule added (2026-11-18; total events per cell = sum over the first n samples of min(events, cap), counts from
the legacy np_files via `scratchpad/legacy_train_event_counts.py` →
`scratchpad/results/legacy_criterion_check/legacy_train_event_counts.csv`): the minimal total events fall smoothly with
δ (monotone by construction), but the joint rule trades samples for events: it picks many samples with few events
(SOM: 20–60 samples × 100–1k events at δ ≥ 0.02), and the sample number of the minimal cell jumps. Greedy keeps the
number of annotated samples (the clinical labelling cost) small; joint minimizes compute / data volume. They answer
different questions, so the rule should follow the claim (open decision above).

## Theory

Note on the answer below: the step-8 grid does **not** contain the ('all', 'all') cell (it is the main benchmark and
is left out, see `training_data_requirements.py`). The baseline is therefore always the step-7 score of the same test
sample (seed mean; 32 cores, other model seeds), not a step-8 cell; the fallback in point 3 of section 2 becomes the
only case.

### Prompt (verbatim)

Some reviewers critisized the following:

```
Direct criticism
- R4.5 (REVISION_PLAN.md:1253) is the most specific:
  - With small test sets the 95% CI is wide, so the criterion is lenient.
  - The minimum is selected on the test set, which introduces selection bias.
```

I think we can adress the first point since now we do multiple folds instead of siple train-test this way the CIs hopefully get smaller. However i would average the results across seeds per fold. Is this sensible? How should we aggregate the data requirement results across folds?
The second point we cannot resolve easily. I would use the minimal data requirements as determined with the test data (first config within 95% empirical CI of full dataset baseline) for all followup experiments (local training, preprocessing ablation, probabiliatic predictions, class weighting, ...). Strictly speaking this introduces selection bias. What strategy do you recommend?

### Answer (verbatim)

Short version: yes, average over trials within a fold, then pool across folds at the level of test samples. I'd also change the criterion, because "within the CI" gets the logic slightly backwards. For the selection bias there's a fix that needs no new runs.

#### 1. Seeds/trials within a fold

Averaging is fine for the point estimate. In step 8, though, a trial is more than a model seed: it is the training sample order plus the event order plus the model seed. R4.5 explicitly asks for "multiple random orderings with variance", so the trials are the evidence for that request and shouldn't only be averaged away. I'd use them twice:
- **Point estimate:** per test sample, take the mean score over the trials. This is the expected performance for a random draw of n samples.
- **Variability:** apply the criterion once per trial and report how the minimal (n_samples, n_events) varies across trials. Step 9 adds more trials around the elbow, which tightens this.

#### 2. Aggregation across folds

The test sample is the unit, and every sample is tested exactly once across the 5 folds. Per (method, dataset, grid cell):
1. Score per test sample, averaged over trials (within its fold).
2. Pool across folds, giving one value per sample. That is N values, not 40% of N as in the old single split.
3. Pair with the full-data baseline on the same test sample: d_i = score(cell) − score(all × all). Use the step-8 `all × all` cell of the same fold and trial where it exists. Where the 1 h / 4-core budget cut it off, use the step-7 score.
4. Take the mean of d_i with a bootstrap CI over test samples. For Imstat, resample donors rather than samples, because some donors have several samples.

Don't decide per fold and then average the per-fold minima. With about 8 test patients per fold on LT1 you'd be back to the original small-test-set problem. Report the per-fold minima only as a spread.

One caveat to state in the Methods: a CI over test samples doesn't capture the training-set variance between folds (cf. Bates, Hastie & Tibshirani 2023). For deciding where the curve plateaus that's acceptable.

#### 3. The criterion itself

Pooling the folds does narrow the CI, by roughly √2.5 compared with the 60:40 split. But "inside the baseline's 95% CI" makes how lenient the criterion is depend on test-set size: a small dataset gets a lenient threshold and a large one a strict threshold. Not being able to reject equality is also not evidence of equivalence. My recommendation:
- **Primary:** a paired non-inferiority criterion with a fixed margin δ. The minimal config is the smallest cell whose lower 95% bound of mean(d_i) is above −δ, e.g. δ = 0.02 macro F1, fixed before you look at the results.
- **Sensitivity table:** δ = 0.01 / 0.05 and the legacy CI criterion. R4.5's to-do already asks for stricter criteria, so this covers it.

This answers the leniency point directly, not just through the larger test set.

#### 4. Selection bias

The bias isn't in the curve: each grid point is an honest test estimate. It enters only where you pick the minimum and then claim "n samples are enough". Two cheap fixes:

**a) Cross-fitted selection for the step-8 claim, at no compute cost.** For each fold k, choose the minimal config with the criterion applied to folds ≠ k only, and evaluate that config on fold k. That gives a selection-free estimate of the score at the "minimal" config. Compare it with the in-sample estimate: if the two agree, you can say the bias is negligible, quantified rather than only acknowledged. One honest caveat: this is only approximately nested. The models of the other folds had fold k's patients in their training data, so fold k leaks into the selection indirectly. The leak is weak, but say so. Strict nesting would mean rerunning the grid on inner splits, which only Imstat has.

**b) Follow-up experiments (local training, preprocessing ablation, class weighting, …).** Your plan is defensible, with two points:
- These experiments compare arms at the same training size on the same test samples. The selection bias mostly cancels in those comparisons, so your conclusions there are about differences between arms, not absolute claims.
- One subtle asymmetry: the size was chosen because the *default* arm reaches baseline there, so the default is slightly favoured at that size. This matters most for the preprocessing ablation and class weighting. Either state it, or use the cross-fitted per-fold config from (a) in the follow-ups as well. That costs nothing extra because the follow-ups run per fold anyway; the only drawback is that the config can differ between folds.

What I'd do in practice: compute both the pooled and the cross-fitted configs. If they mostly coincide, use the pooled config for all follow-ups, report the comparison in the supplement, and add a sentence on test-set reuse to the limitations (already planned at REVISION_PLAN.md:687). Fix the criterion and δ in `PLAN.md` before running step 9; the plan already says the criterion must be fixed before the follow-up.

R4.5 also notes that the old selection was greedy (samples first, then events). Because the new grid is nested, a joint minimum (e.g. fewest total events among the qualifying cells) costs nothing to add to the sensitivity table.

Should I write this into `working_notes/training_data_requirements_plan.md` as an evaluation section for review?

### Prompt 2 (verbatim)

is the ci of the delta actually idiomatic. what do we measure by this.

### Answer 2 (verbatim)

A confidence interval is a statement about the procedure: if you drew new test sets from the same population over and over and recomputed the interval each time, 95% of those intervals would contain the true mean difference. For one realized interval the reading is "the plausible range for the true mean delta, given this test set".

**What the delta CI measures here.** The parameter is the expected paired difference in per-sample macro F1, between the model trained on the reduced data and the model trained on full data, on a new patient from the same population. The bootstrap resamples only the test samples, so it captures only one source of uncertainty: which patients happened to end up in the test set. The two trained models are held fixed. Training order, the model seed and which patients were in the training set are not part of it. So in the legacy check it reads: "for *these two fitted models*, how much worse is the small one on new patients?"

**Is it idiomatic?** Yes, in two respects:
- **Paired bootstrap CI of a mean difference:** the standard way to compare two models on the same test set (e.g. Koehn 2004 in NLP; generally in ML model comparison). Pairing removes the variance between test samples. That is why it is 2–4× narrower than the baseline CI.
- **Lower CI bound > −δ:** this is the standard non-inferiority design from clinical trials (CONSORT extension for non-inferiority trials, Piaggio et al. 2012). It is equivalent to a one-sided test at α = 0.025 with the null hypothesis "the small model is worse by at least δ". The legacy rule ("mean inside the baseline CI") is not a recognized test. It compares a point estimate with an interval for a different quantity and ignores the pairing.

**Caveats to state (or handle in step 8):**
1. **Training variance is not in the CI.** In step 8 the per-sample score is averaged over 3 trials first. The parameter then becomes the expected performance over random training orders, which is better. The CI still contains only test-sample variance, though. To include the trial variance, use a two-level bootstrap: resample trials, then test samples. Alternatively, keep it separate as the spread of the per-trial minima. The spread of the per-trial minima is what R4.5 asks for anyway.
2. **Multiple testing.** We test many cells and take the first that passes. Each test has a 2.5% one-sided false-pass rate, but the chance that *some* small cell passes by luck grows with the number of cells scanned. The legacy check shows exactly this: cells pass and larger cells fail again. The stability rule (all larger n_samples must also pass) limits this; so do the smoother trial × fold means.
3. **Small n.** With 12 test samples (Flowcyt in the legacy split), the percentile bootstrap is coarse. Pooling over 5 folds in step 8 helps there (all ~18 samples, plus trial averaging), but Flowcyt stays the weakest case.
4. **Imstat:** some donors contribute several samples, so resample by donor (already in the plan).

In short: the paired CI against a fixed margin is the standard tool and measures the right thing for the criterion. It conditions on the training runs, though, so the variability across orders has to be reported separately, as planned.

### Prompt 3 (verbatim)

how is the lower ci bound equivalent to a test? and why alpha=0.25

### Answer 3 (verbatim)

First, a correction: α = 0.025, not 0.25.

**Why the lower bound is a test (test–CI duality)**

The non-inferiority question as a one-sided test:
- H₀: μ ≤ −δ (the reduced model is worse by at least δ)
- H₁: μ > −δ (it is worse by less than δ, i.e. non-inferior)

Here μ is the true mean paired difference.

A one-sided (1 − α) confidence interval [L, ∞) is, by construction, the set of all values μ₀ that a level-α test would *not* reject for H₀: μ ≤ μ₀. So:

- If L > −δ, then −δ lies outside the interval, and the test rejects H₀: μ ≤ −δ at level α.
- If L ≤ −δ, then −δ lies inside the interval, and H₀ cannot be rejected.

"Lower bound above −δ" and "the one-sided test rejects at level α" are the same decision. If the true μ were exactly −δ, the procedure would wrongly declare non-inferiority with probability α. The interval misses the true value with probability α in total, and for a one-sided interval all of that miss sits on the side that matters here.

**Why 0.025**

The lower bound of a **two-sided 95% CI** leaves 2.5% in each tail. Its lower bound is therefore the lower bound of a **one-sided 97.5% CI**, which corresponds to a one-sided test at α = 0.025. We used the two-sided 95% CI (percentiles 2.5 / 97.5), so the implied level is 0.025.

This is a convention, not a law:
- **α = 0.025 one-sided** (= a two-sided 95% CI) is the usual choice for non-inferiority trials. It holds the non-inferiority claim to the same false-positive rate in the direction that matters as a classical two-sided 5% superiority test. It is what CONSORT and regulatory guidance (ICH E9 / E10) expect.
- **α = 0.05 one-sided** (= the lower bound of a two-sided **90%** CI) is also common, e.g. in the "two one-sided tests" (TOST) procedure for bioequivalence (Schuirmann 1987). It is more lenient and would pick smaller minimal cells.

For us, the 95% CI / α = 0.025 version is the conservative and more widely accepted choice. The level only needs to be stated in the Methods.

**One caveat for the bootstrap:** with a percentile bootstrap CI the duality holds only approximately, because the interval's coverage is approximate. With ≥ 40 test samples this is fine. With 12 (legacy Flowcyt) it is rough.

Should I add this to the Theory section as Prompt 3 / Answer 3?

### References

- **Koehn P. (2004).** Statistical Significance Tests for Machine Translation Evaluation. *Proceedings of EMNLP 2004*,
  Barcelona, pp. 388–395. No DOI; ACL Anthology W04-3250, https://aclanthology.org/W04-3250/.
  Where (checked in the PDF, 2026-11-18): **Section 5 "Bootstrap Resampling"** (p. 392): bootstrap confidence
  interval of one system's score by resampling test sentences with replacement; **Section 6 "Paired Bootstrap
  Resampling"** (p. 393, Figs. 7–8): both systems are scored on the *same* resampled test sets and the share of
  resamples in which one system wins gives the significance of the difference (our per-test-sample pairing is the
  same idea); Section 7 "Validation Experiments" checks the methods empirically.
- **Piaggio G., Elbourne D.R., Pocock S.J., Evans S.J.W., Altman D.G.; CONSORT Group (2012).** Reporting of
  noninferiority and equivalence randomized trials: extension of the CONSORT 2010 statement. *JAMA* 308(24):2594–2604.
  doi:10.1001/jama.2012.87802, PMID 23268518. Predecessor: Piaggio et al. (2006), *JAMA* 295(10):1152–1160,
  doi:10.1001/jama.295.10.1152.
  Where (**not verified**, full text not accessible here, JAMA returned 403; from memory, check before citing):
  Figure 1 (the possible positions of the two-sided CI of the treatment difference relative to the noninferiority
  margin −Δ and to 0; noninferiority is concluded if the whole CI lies above −Δ); checklist items 7a (sample size,
  incl. the margin and its rationale) and 17a (result: effect estimate with its CI relative to the margin); the
  text on the correspondence of a one-sided α = 0.025 test to the lower bound of a two-sided 95 % CI.
- **Bates S., Hastie T., Tibshirani R. (2024).** Cross-validation: what does it estimate and how well does it do it?
  *Journal of the American Statistical Association* 119(546):1434–1445. doi:10.1080/01621459.2023.2197686,
  arXiv:2104.00673. Cited in Answer 1 (section 2): CV estimates the average error over training sets from the
  population, not the error of the model at hand, and naive CV intervals undercover; nested CV as a remedy.
  Section pointers not checked.
- **Schuirmann D.J. (1987).** A comparison of the two one-sided tests procedure and the power approach for assessing
  the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics* 15(6):657–680.
  doi:10.1007/BF01068419 (DOI from memory, **not verified**). Cited in Answer 3: TOST at α = 0.05 per side, i.e. the
  two-sided 90 % CI. Section pointers not checked.
- **ICH E9 (1998)** Statistical Principles for Clinical Trials, and **ICH E10 (2000)** Choice of Control Group and
  Related Issues in Clinical Trials (ICH guidelines, no DOI; https://www.ich.org/page/efficacy-guidelines). Cited in
  Answer 3 for the one-sided α = 0.025 / two-sided 95 % CI convention of non-inferiority trials. Pointers (from
  memory, **not verified**): E9 section 3.3.2 (trials to show equivalence or non-inferiority) and 5.5 (estimation,
  confidence intervals and hypothesis testing); E10 section 1.5 (non-inferiority margin).
