# Sample order robustness (PLAN.md step 9)

Script: `sample_order_robustness.py` (implemented 2026-11-25). SLURM: `slurm_scripts/sor_<array>.sh <scheme>`.

## Question
Step 8 gives one minimal cell (n_min training samples × ne_min events per sample) per method × dataset, selected on the mean over 3 trials (sample orders). How much does that result depend on the sample order? The same runs answer two questions:
1. **Share of passing trials:** the share of single trials that meet the step-8 primary criterion at the minimal cell (paired non-inferiority to the full-data model, δ = `config.DELTA` = 0.02).
2. **Spread of the per-trial minimum:** for each trial, the smallest n_samples (at ne_min) that passes.

Caveat for the text: a single trial's d is noisier than the 3-trial mean that selected n_min, so the share is expected to be below 1 even without any order effect.

## Design (decided 2026-11-25, user)
- Mirrors `training_data_requirements.py` (step 8) on a smaller grid and imports its machinery: one child process per atom, per-atom time limit, cutoff / dominance, lanes, resubmission via `missing`, `status`.
- **Atom** = method × dataset × outer split × trial × n_samples, with n_samples = every integer 1 … n_min (values ≥ the split's training samples are dropped) and n_events = ne_min.
- **Lane** = method × dataset × split × trial, ordered split → dataset → method → trial; atoms small → large.
- **Minimal cell:** the pooled cell of the step-8 eval's `minimal_configs.json` (primary criterion: noninf, δ = 0.02, greedy); `STABLE` selects the variant with or without the stability rule (TODO: decide after the real step-8 eval). Method × dataset without a minimal cell or with n_min = 'all' are skipped.
- **Trials:** `N_TRIALS = 30` (TODO: final number; legacy Fig S6 used 100 random orders on LT only). Numbered from 0 with their own seeds. The results stand on their own: the step-8 trials are **not** reused (this replaces the earlier plan to continue the trial numbering).
- **Two arms**, one output root each (`results/sample_order_robustness/<scheme>/`):
  - `random`: a random permutation of the outer training samples per trial (as in step 8, new seeds).
  - `stratified`: round-robin over strata. The samples are shuffled within each stratum. Each round takes the next sample of every stratum that still has samples left, in a stratum order that is drawn anew each round (with n not a multiple of the number of strata, different strata are left out). Which sample comes from a stratum is random. Strata (`STRATA`, a `meta_samples.csv` column): LT1 / LT2 / LT1b / LT2b `subtype` (11 subtypes; BL has 2 samples, NB 9), Imstat `lysis` (5 protocols × 25; the donor is the CV group), BLAST110 `sample_type` (AML_Dx / AML_FU / NBM). Flowcyt has no strata (healthy bone marrow), so it runs only in the random arm.
- **Methods / arrays / params / time limits:** as in step 8 (RF, HGB, MLP, SOM; D6 params frozen into the manifest; 1 h per atom). Test samples are predicted in full; counts + resources, no predictions.
- **Seeds** (key `('order_robust', scheme)`): order `derive_seed(*key, dataset, split, trial)`; event order per sample `derive_seed(*key, dataset, split, trial, sample)` (`event_order`, nested over n_samples); join `derive_seed(*key, dataset, split, trial, n_samples, n_events, 'join')`; model `derive_seed(*key, method, dataset, split, trial)`. The data seeds do not contain the method.

## Size (dummy minimal cells, 2026-11-25)
Lanes: random 6,000 (sklearn_cpu 3,000, flagx_cpu 1,500, flagx_gpu 1,500), stratified 5,400 (2,700 / 1,350 / 1,350); all below `MaxArraySize` = 10,000. Atoms (dummy cells): random 56,250, stratified 54,750. The real counts depend on the step-8 minimal cells.

TODO before the run: estimate the lane time from `pred_s` in the step-8 `atoms.csv` (the full test-set prediction dominates, e.g. BLAST110) and set the SLURM `--time` / the lane split accordingly.

## Changes to `training_data_requirements.py` (step 8 unchanged)
- `run_atom(..., order=None, seed_key=('train_req',), order_scheme=None)`: training sample order and seed prefix as arguments; `order_scheme` written to the JSON if given.
- `run_child` / `run_lane(..., atom_cmd=None)`: the child command (incl. `--out`) of the calling script; `run_lane(..., overwrite=None)`.
- Checked: manifest, counts and atom JSONs (except times / metadata) of a step-8 smoke lane are identical to the version before the change.

## Eval (to do)
`sample_order_robustness_eval.py`, reusing the step-8 eval helpers (scores, bootstrap over patients, step-7 baseline): per scheme × method × dataset × trial × n_samples mean d, CI, pass; share of passing trials at n_min with a Clopper–Pearson CI; distribution of the per-trial minimal n_samples; share of trials passing by n (Fig S6-style). Then a plot script.
