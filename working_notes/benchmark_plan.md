# Benchmark plan: step 7, part 1 (`benchmark.py`, main benchmark run)

> Status: implemented (2026-11-11); smoke-tested in every env (`DONE.md`, step 7).
> Run setup changed afterwards (2026-11-11): Woody arrays with 32 cores, per-atom progress, pilot → caps → final run (`results/benchmark_pilot/`, `results/benchmark/`); see `DONE.md` step 7 (cont.) and `slurm_scripts/README.md`.

## Context
Steps 0–6 are done; tuning produced `results/tune_hyperparameters_eval/best_configs/params_{outer,overall}_<method>.json`. Step 7 runs every method on every dataset with its best parameters. It follows the tuning code layout. `benchmark.py` only generates atoms (fit + predict + counts). A later `benchmark_eval.py` (not in this plan) aggregates them. Atom = method × dataset × outer split × model seed. There is one SLURM array per (env, cpu/gpu), with the same envs, clusters and resources as the tuning. The first run uses no downsampling. A seeded on-the-fly training downsampling option exists for a second run of the atoms that hit OOM or timeout.

## Decisions (from the request + answers)
- **Hyperparameters (D6):** Imstat atoms use `params_outer_<method>.json['splits'][str(split)]['params']`. All other datasets use `params_overall_<method>.json['params']`. The params are read at manifest time and frozen into the manifest, together with `config_id` and `param_source` (`outer` / `overall`). A missing file or split raises an error.
- **Model seeds:** 3 (`seed` 0–2) for som, mlp, dgcytof, deepcytof, rf, hgb, gmm_mocking and gatemeclass. 1 seed for logreg, knn, lda_sklearn and lda_abdelaal. Splits are unchanged; there is one CV repetition.
- **Seeds:** model `derive_seed('benchmark', method, dataset, split, seed)`. Data (downsampling per sample, join shuffle): `derive_seed('benchmark', dataset, split, seed, sample|'join')`. These are method-independent, so all methods with the same seed see the same training order/subset.
- **Abstention (D8):** already handled by the params and runners. The counts keep the `pred_abstained` column, as in the tuning.
- **Downsampling:** off by default. Test samples are always predicted in full (PLAN rule).
- **Per-cell outputs:** predicted labels are saved (`SAVE_LABELS = True`); probabilities are not (`SAVE_PROBA = False`, flag only). Predictions and counts go into separate folders. Predictions are git-ignored; the counts are tracked later by the user.
- The PLAN line "Optional robustness check: repeated CV …" is removed.

## `benchmark.py` (repo root, Py3.5-compatible, runs in every env)
It has the same structure and CLI as `tune_hyperparameters.py` (copy its `run_atom` skeleton).

**Constants:** `DATASETS = config.DATASETS`, `OUT = results/benchmark`, `OVERWRITE = False`, `ARRAYS` (same 6 arrays / envs / methods as the tuning), `N_SEEDS = {method: 3|1}`, `SAVE_LABELS = True`, `SAVE_PROBA = False`, `STRATIFIED = True`, `SMOKE`, and
```python
# Training downsampling per (method, dataset), per training sample: {'n': 20000} (cap) or {'frac': 0.1}.
# Empty = all events. Second run only, for atoms that hit OOM / timeout (decided after the first run).
TRAIN_DOWNSAMPLING = {}
```

**Commands:**
- `manifest <array> [--out]`: writes rows (index, atom_id, method, dataset, split, seed, config_id, param_source, params, train_downsampling) for the loop method → dataset → split → seed. It also creates `logs/bench_<array>/` and prints N.
- `run <array> [index|all] [--out] [--smoke]`: runs one atom (default `$SLURM_ARRAY_TASK_ID`). It skips the atom if its JSON exists.
- `missing <array>`: lists atoms without a JSON as a SLURM range, as in the tuning.
- `status <array>`: prints a table method × dataset of ok / failed / started / missing, plus the SLURM range of all non-ok atoms. This is the input for deciding the second run.

**Atom ID / paths:** `atom_id = <method>__<dataset>__split_<i>_seed_<r>[__ds_<tag>]`. The tag (e.g. `n20000`, `frac0.1`) is set only when downsampled. A downsampled rerun therefore writes new files, and the failed full-data atom stays on disk as evidence. The manifest defines which atoms are current, and the eval reads atoms via the manifest.
```
results/benchmark/
  manifests/<array>.csv
  atoms/<method>/<dataset>/split_<i>_seed_<r>[__ds_<tag>].json        status + everything else
  counts/<method>/<dataset>/split_<i>_seed_<r>[__ds_<tag>].csv         per test sample × true class
  predictions/<method>/<dataset>/split_<i>_seed_<r>[__ds_<tag>]/<sample>.npz   y_pred (int16); proba if SAVE_PROBA
  logs/bench_<array>/
```

**One atom:** the JSON is written with `started`, then:
1. Load the runner and `run_metadata`, then `load_split(dataset, 'split_<i>')` → train/test.
2. Load every training sample, downsample if a spec applies, and `join_samples(shuffle=True)`.
3. `runner.input_arm(params)` → preprocessor. Under `track_resources`, run the preprocessor fit + `build` + `fit`; store `fit_info` if the runner has it.
4. For each test sample (full): `track_resources` around `predict(pre.transform(X), proba=SAVE_PROBA)` → `confusion_counts` / `class_counts` rows (same columns as the tuning plus `dataset`, `seed`) and an optional npz.
5. Save the counts CSV and then the JSON with `ok` (or `failed` + traceback).

The JSON content is the same as in the tuning: identity, params, param_source, seeds, train/test samples, n_train_events(+ per class), downsampling spec, preprocessor state, resources fit/predict per sample, metadata.

## Shared helpers (rule: move to `validation/core/` once a second script needs them)
Move these from `tune_hyperparameters.py` and import them in both scripts. Tuning behaviour stays identical.
- `make_preprocessor(arm)` → `validation/core/preprocessing.py: shared_preprocessor(dataset, arm)`, which uses `config.CHANNELS` / `config.LOG10_CUTOFFS` for any dataset; plus `preprocessor_state(p)`.
- `index_ranges`, `read_manifest(path)` → `validation/core/runs.py`.

Reused unchanged: `validation/core/{data,sampling,seeding,evaluation,resources,runs}.py`, all runners in `validation/runners/`.

## SLURM
- `slurm_scripts/bench_<array>.sh` ×6: copies of `tune_<array>.sh` with job name `bench_<array>`, `--output=results/benchmark/logs/%x/%A_%a.out`, `python benchmark.py run <array>`, and `--array=0-<N-1>`.
- Expected N: sklearn_cpu 600, flagx_cpu 150, flagx_gpu 300, gatemeclass_cpu 150, lda_cpu 50, deepcytof_cpu 150, about 1400 in total.
- `slurm_scripts/README.md`: add a benchmark section (table, resume, status / second run).
- The cluster needs all `data/curated/<dataset>/` directories (so far only Imstat is there).

## Expected trouble in the first run (projected from the best-config tuning atoms, which trained on ~2.5M events; outer train ≈ Imstat 3.9M, LT 7.5M, Flowcyt ~17M, BLAST110 ~50–63M events)
- SOM (2000 epochs, 39k s / 15 GB at 2.5M): likely to time out from LT onward, and to OOM on Flowcyt/BLAST110.
- GateMeClass (13 GB at 2.5M): likely OOM from LT/Flowcyt onward.
- kNN (prediction 40 s/sample), DeepCyTOF (MMD retraining per test sample, 300 s/sample), DGCyTOF (calibration, 10 GB at prediction): prediction time/memory is likely to fail on BLAST110/Flowcyt.

These are estimates only. As requested, the plan is to run everything at full size, then set `TRAIN_DOWNSAMPLING` from the `status` output. Test-side failures (kNN, DeepCyTOF prediction) are not fixed by training downsampling; this would then be a separate decision.

## Files
- New: `benchmark.py`, `slurm_scripts/bench_*.sh` (6), `working_notes/benchmark_plan.md`
- Edit: `tune_hyperparameters.py` (imports moved helpers), `validation/core/preprocessing.py`, `validation/core/runs.py`, `.gitignore` (`results/benchmark/predictions/`), `PLAN.md` (step 7 status, drop the robustness line), `DONE.md`, `CLAUDE.md`, `slurm_scripts/README.md`

## Verification
- `python benchmark.py manifest <array> --out <scratch>` for all 6 arrays: check N and the params against the best-config JSONs (Imstat per split, others overall).
- Smoke test: `run <array> 0 --smoke --out <scratch>` in each env (`fx_sklearn`, `fx_flagx_cpu`, `fx_lda`, `fx_gatemeclass`, `fx_deepcytof` for Py3.5, plus `fx_flagx_cpu` for the mlp/dgcytof rows) on Imstat and one BLAST110 atom. Check the JSON status ok, the counts sum to n_events per sample, and the npz labels match the counts.
- Set one `TRAIN_DOWNSAMPLING` entry in the scratch copy: check the new atom_id suffix, `n_train_events`, and that `status` shows both atoms.
- Tuning regression after the helper move: `tune_hyperparameters.py manifest lda_cpu --out <scratch>` + `run lda_cpu 0 --smoke`. The manifest must be identical to the committed one.
