# Statistics: tests, confidence intervals and margins (working note)

Background for the statistical criteria of PLAN.md step 7 (`benchmark_eval.py`: pairwise method comparisons,
`pairwise_deltas.csv` columns `superior` / `noninferior` / `equivalent`) and step 8 (minimal training data,
`working_notes/training_data_requirements_eval_plan.md`). Prompts and answers are kept verbatim; the decisions
themselves live in the plans and in `PLAN.md`. Margin: `config.DELTA` (= 0.02). References with DOIs and the passages
the statements are taken from: section References at the end.

## Step 8: criterion for the minimal training data (2026-11-18, moved from the evaluation plan)

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

## Equivalence vs. non-inferiority; criteria table; type of bootstrap test (2026-11-23)

Context: the pairwise method comparison of step 7 (`pairwise_f1.pdf`) uses equivalence (95 % CI of the mean paired
difference within ±δ); step 8 uses non-inferiority (lower bound > −δ). Same δ (`config.DELTA`).

### Prompt 4 (verbatim)

And why not use equivalence for the minimal data requirements as well? I mean sometimes the performance with fewer data is better than with all data (this can also be due to chance of course). If this is the case, what happens with the non inferiority test? Can you show me the criteria for a comparison of A (method1, few data) to B (method 2, all data) given ci_lower, ci_upper and +/- delta. Also what kinfd of bootstrap test is this is this common practice? Do not implement just answer.

### Answer 4 (verbatim)

Non-inferiority is the right test for the minimal training data; equivalence would penalize cells where less data does better.

#### What happens when fewer data do better

For d = A − B, with A = the reduced data and B = all data:

- **Non-inferiority** (lower bound > −δ) asks only "is A not relevantly worse?". If A is better, by chance or not, d > 0, the lower bound moves up, and the cell passes more easily. That is the intended behaviour: the claim is "enough data", and being better doesn't contradict it.
- **Equivalence** also needs the upper bound < +δ. A cell where the reduced model is clearly better (upper bound ≥ δ) would fail and not count as minimal. It would be rejected for being too good. Equivalence is also stricter on noisy cells, because a wide CI breaks through +δ, so it would pick larger minimal cells for no substantive reason.

So for step 8 the question is one-sided and non-inferiority fits. If a reduced cell is significantly better than full data, that is worth reporting separately (e.g. as an overfitting or tuning effect: the parameters were tuned on full data). It is not a reason to reject the cell.

For the method comparison the question is symmetric ("are A and B practically the same?"), so equivalence fits there. Different questions call for different tests, with the same δ.

#### Criteria for d = A − B, 95 % CI [L, U], margin δ

| Conclusion | Criterion |
|---|---|
| A better than B | L > 0 |
| A relevantly better (by more than δ) | L > δ |
| A non-inferior to B | L > −δ |
| A non-superior to B | U < δ |
| A and B equivalent | −δ < L and U < δ |
| A worse than B | U < 0 |
| A relevantly worse (inferior by more than δ) | U < −δ |
| Inconclusive for non-inferiority | L ≤ −δ < U (the CI straddles −δ) |

Typical CI positions:

```
            -δ         0         +δ
             |         |          |
1.               [---]                  equivalent, non-inferior, not significant
2.                       [---]          equivalent, non-inferior, A significantly better
3.                         [-------]    non-inferior, A better; not equivalent (U > δ)
4.                                 [--] A relevantly better (L > δ); non-inferior
5.        [------------]                inconclusive (L < -δ): not non-inferior, not equivalent
6.       [-----]                        A significantly worse (U < 0); NI inconclusive
7. [---]                                A relevantly worse (U < -δ)
```

Case 3 is the "fewer data do better" case: non-inferiority passes and equivalence fails. Case 4 is clearly better; equivalence fails, but non-inferiority still passes.

#### What kind of bootstrap test this is

It is a **paired nonparametric (case-resampling) bootstrap** with a **percentile confidence interval** for the mean per-sample difference. The test is obtained by **inverting the CI**: you reject if the CI excludes the null value. There is no separate bootstrap p-value.

- **Paired bootstrap:** comparing two models on the same test set by resampling test cases is standard in ML and NLP (Koehn 2004; Berg-Kirkpatrick et al. 2012; Dror et al. 2018 review significance testing in NLP). The percentile interval itself is textbook (Efron & Tibshirani 1993, ch. 13).
- **CI-based non-inferiority and equivalence:** comparing the CI with ±δ is how the clinical guidelines define these designs (CONSORT extension, Piaggio et al. 2012; ICH E9 / E10). Equivalence as TOST is from Schuirmann 1987. Clinical trials usually use t- or Wald intervals; a bootstrap CI is an accepted nonparametric substitute.
- **The combination** (a paired bootstrap CI used against a margin) is less common in ML benchmarks than plain significance testing, but each part is standard and easy to cite.

**Points to state in the Methods:**
- **α:** a 95 % CI means α = 0.025 one-sided for non-inferiority and α = 0.025 per side for equivalence. That is conservative compared with the classic TOST, which uses a 90 % CI (α = 0.05 per side).
- **The percentile interval is only approximate.** With skewed differences, the BCa interval (Efron 1987) has better coverage. A paired t-interval gives nearly the same answer at n ≈ 100 and is the simplest to defend; it's a reasonable robustness check.
- **Only test-sample variance is in the CI.** It conditions on the trained models, so fold and seed variance is not included, as discussed before.

## References

DOIs checked against Crossref (api.crossref.org) on 2026-11-23. "Where" names the passage a statement is taken from;
**verified** = read in the full text on the date given, **not verified** = full text not accessible here (paywall /
bot check), from memory: check before citing.

- **Koehn P. (2004).** Statistical Significance Tests for Machine Translation Evaluation. *Proceedings of EMNLP 2004*,
  Barcelona, pp. 388–395. No DOI (EMNLP 2004 predates ACL DOIs); ACL Anthology W04-3250,
  https://aclanthology.org/W04-3250/. Cited in Answers 2 and 4 (paired bootstrap).
  Where (verified in the PDF, 2026-11-18): **Section 5 "Bootstrap Resampling"** (p. 392): bootstrap confidence
  interval of one system's score by resampling test sentences with replacement; **Section 6 "Paired Bootstrap
  Resampling"** (p. 393, Figs. 7–8): both systems are scored on the *same* resampled test sets and the share of
  resamples in which one system wins gives the significance of the difference (our per-test-sample pairing is the
  same idea); Section 7 "Validation Experiments" checks the methods empirically.
- **Berg-Kirkpatrick T., Burkett D., Klein D. (2012).** An Empirical Investigation of Statistical Significance in NLP.
  *Proceedings of EMNLP-CoNLL 2012*, Jeju Island, pp. 995–1005. No DOI (ACL DOIs start in 2017); ACL Anthology
  D12-1091, https://aclanthology.org/D12-1091/. Cited in Answer 4 (paired bootstrap is standard).
  Where (verified in the PDF, 2026-11-23): **Section 2.1 "Hypothesis Tests"** (p. 996): "We use the paired
  bootstrap (Efron and Tibshirani, 1993) because it is one of the most widely used [...] and because it can be easily
  applied to any performance metric, even complex metrics like F1-measure"; the same paragraph: with the paired
  Student's t-test, "to the extent that the assumptions of the t-test are met, it is likely that the results would be
  very similar". **Section 2.2 "The Bootstrap"** (p. 996) and **Figure 1** (p. 996, pseudocode): test sets resampled
  from the observed one with replacement; p-value with the re-centred statistic (count of δ(x⁽ⁱ⁾) > 2δ(x)); footnote 2:
  the variant δ(x⁽ⁱ⁾) < 0 is equivalent if the bootstrap distribution is symmetric around δ(x).
- **Dror R., Baumer G., Shlomov S., Reichart R. (2018).** The Hitchhiker's Guide to Testing Statistical Significance
  in Natural Language Processing. *Proceedings of the 56th Annual Meeting of the ACL (Volume 1: Long Papers)*,
  pp. 1383–1392. doi:10.18653/v1/P18-1128. Cited in Answer 4 (review of significance testing).
  Where (verified in the PDF, 2026-11-23): **Section 3.2 "Prominent Significance Tests"**, paragraph **"Paired
  bootstrap test"** (p. 1387): sampling with replacement, samples used "as surrogate populations"; used in MT,
  summarization, semantic parsing; "less effective for small test sets, as it assumes that the test set distribution
  does not deviate too much from the population distribution". **Figure 1** (p. 1387): decision tree for choosing a
  test (bootstrap / randomization for small data with unknown statistic distribution). **Section 5 "Open
  Questions"** (p. 1389): testing over several cross-validation splits is an open issue.
- **Efron B., Tibshirani R.J. (1993).** *An Introduction to the Bootstrap.* Chapman & Hall, New York (Monographs on
  Statistics and Applied Probability 57). doi:10.1007/978-1-4899-4541-9 (Springer record of the book). Cited in
  Answer 4 (percentile interval) and by Berg-Kirkpatrick et al. for the paired bootstrap.
  Where (chapter titles and pages from the Crossref chapter records, content **not verified**): **Ch. 13
  "Confidence intervals based on bootstrap percentiles"**, pp. 168–177 (the percentile interval we use); Ch. 14
  "Better bootstrap confidence intervals", pp. 178–201 (BCa); Ch. 16 "Hypothesis testing with the bootstrap",
  pp. 220–236.
- **Efron B. (1987).** Better Bootstrap Confidence Intervals. *Journal of the American Statistical Association*
  82(397):171–185. doi:10.1080/01621459.1987.10478410. Cited in Answer 4 (BCa has better coverage than the
  percentile interval for skewed statistics).
  Where (**not verified**, publisher page returned 403): the paper introduces the BCa (bias-corrected and
  accelerated) interval as a second-order accurate correction of the percentile interval.
- **Piaggio G., Elbourne D.R., Pocock S.J., Evans S.J.W., Altman D.G.; CONSORT Group (2012).** Reporting of
  noninferiority and equivalence randomized trials: extension of the CONSORT 2010 statement. *JAMA* 308(24):2594–2604.
  doi:10.1001/jama.2012.87802 (verified), PMID 23268518. Predecessor: Piaggio et al. (2006), *JAMA* 295(10):1152–1160,
  doi:10.1001/jama.295.10.1152. Cited in Answers 2 and 4.
  Where (**not verified**, full text not accessible here, JAMA returned 403 / a bot check; from memory, check before
  citing): Figure 1 (the possible positions of the two-sided CI of the treatment difference relative to the
  noninferiority margin −Δ and to 0; noninferiority is concluded if the whole CI lies above −Δ); checklist items 7a
  (sample size, incl. the margin and its rationale) and 17a (result: effect estimate with its CI relative to the
  margin); the text on the correspondence of a one-sided α = 0.025 test to the lower bound of a two-sided 95 % CI.
  The verified ICH E9 passages below cover the same points and can be cited instead.
- **Schuirmann D.J. (1987).** A comparison of the Two One-Sided Tests Procedure and the Power Approach for assessing
  the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics* 15(6):657–680.
  doi:10.1007/BF01068419 (verified). Cited in Answers 3 and 4: equivalence as two one-sided tests (TOST) at α = 0.05
  per side, i.e. the two-sided 90 % CI inside the equivalence limits.
  Where (**not verified**, paywalled): the TOST procedure is introduced and compared with the power approach. ICH E9
  section 3.3.2 (below, verified) states the CI / TOST equivalence and can be cited instead.
- **Bates S., Hastie T., Tibshirani R. (2024).** Cross-validation: what does it estimate and how well does it do it?
  *Journal of the American Statistical Association* 119(546):1434–1445 (online 2023-05-15).
  doi:10.1080/01621459.2023.2197686 (verified), arXiv:2104.00673. Cited in Answer 1 (section 2) and in the step-7
  discussion: a CI over test samples does not contain the training-set variance.
  Where (verified in arXiv v4, 18 Jul 2022, 2026-11-23): **Section 1 "Introduction"**, 2nd paragraph: the estimand
  of CV "is not the accuracy of the model fit on the data at hand, but is instead the average accuracy over many
  hypothetical data sets"; "naïve intervals based on CV can fail badly, giving coverage far below the nominal level",
  because the variance estimate "does not account for the correlation between the error estimates in different
  folds"; nested CV (NCV) as the remedy. **Section 3.1** (Theorem 1, Corollary 1, Figure 2): targets Err, Err_X,
  Err_XY; CV has lower MSE for Err_X than for Err_XY. **Section 4 "Confidence intervals with nested
  cross-validation"** (4.1 "Dependence structure of CV errors", 4.3 "A nested CV estimate of MSE").
- **ICH E9 (1998).** Statistical Principles for Clinical Trials. ICH Harmonised Tripartite Guideline, Step 4 version
  of 5 February 1998. No DOI; https://database.ich.org/sites/default/files/E9_Guideline.pdf. Cited in Answers 3
  and 4.
  Where (verified in the PDF, 2026-11-23): **Section 3.3.2 "Trials to Show Equivalence or Non-inferiority"**
  (pp. 15–16): an equivalence margin "is the largest difference that can be judged as being clinically acceptable";
  "both the upper and the lower equivalence margins are needed" for equivalence, "only the lower margin" for
  non-inferiority; "Equivalence is inferred when the entire confidence interval falls within the equivalence
  margins. Operationally, this is equivalent to the method of using two simultaneous one-sided tests"; "For
  non-inferiority trials a one-sided interval should be used", with its one-sided test counterpart against the lower
  margin; concluding equivalence or non-inferiority from a non-significant test of no difference is rejected (the
  sentence starting "Concluding equivalence or non-inferiority based on observing a non-significant test", p. 16).
  **Section 5.5 "Estimation, Confidence Intervals and Hypothesis Testing"** (p. 25): "The approach of setting type I
  errors for one-sided tests at half the conventional type I error used in two-sided tests is preferable in
  regulatory settings. This promotes consistency with the two-sided confidence intervals" (our α = 0.025 one-sided
  = lower bound of the two-sided 95 % CI).
- **ICH E10 (2000).** Choice of Control Group and Related Issues in Clinical Trials. ICH Harmonised Tripartite
  Guideline, Step 4 version of 20 July 2000. No DOI; https://database.ich.org/sites/default/files/E10_Guideline.pdf.
  Cited in Answer 3.
  Where (verified in the PDF, 2026-11-23): **Section 1.5.1.1 "Historical Evidence of Sensitivity to Drug Effects
  and Choosing the Non-inferiority Margin"** (p. 9): "an equivalence or non-inferiority margin, sometimes called
  delta, is selected. This margin is the degree of inferiority of the test treatments to the control that the trial
  will attempt to exclude statistically. If the confidence interval for the difference [...] excludes a degree of
  inferiority [...] as large as, or larger than, the margin, the test treatment can be declared non-inferior".
