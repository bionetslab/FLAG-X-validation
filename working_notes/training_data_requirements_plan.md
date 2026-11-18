# Training data requirements plan: step 8 (`training_data_requirements.py`)

> Status: implemented and smoke-tested (2026-11-17; `DONE.md`, step 8). Not run yet. Open TODOs: grid per dataset, number of trials, time limit, method selection, hardware calibration, eval harness.

## Context
Step 7 shows how well every method performs with all training data. Step 8 asks how much training data the best methods need: the score, fit / predict time and memory as a function of the **number of training samples** × **events per sample**. It runs under a fixed small budget (4 CPUs or 1 GPU, 1 h per atom). The result also sets the reduced / minimal training sets used in step 9 (local training, preprocessing ablation). The legacy version of this experiment is `legacy/main.py::main_num_samples_num_events_experiment` (SOM / FCNN, one split, fixed sample order, grid 1–75 samples × 100–50k events + all).

## Decisions (from the request and the discussion of 2026-11-17)
- **Methods:** a constant at the top of the script. For now `mlp`, `som`, `hgb`, `rf`; the final choice = the top performers of step 7 (probably none of the published methods).
- **Atom** = (method, dataset, outer split, trial, n_samples, n_events).
  - **Trial** = training sample order + downsampling seed; the model seed is tied to the trial.
  - 3 trials (`N_TRIALS = 3  # TODO: revisit before the final run`).
- **Grid:** log-spaced defaults, dataset-specific later (`# TODO`), see below.
- **Data:** downsampling on the fly. Test samples are predicted in full (PLAN rule).
- **Outputs per atom:** as in the benchmark: resources + counts per test sample, **no predictions** (storage). The eval harness comes later in its own script.
- **Arrays:** `sklearn_cpu` (rf, hgb), `flagx_cpu` (som), `flagx_gpu` (mlp).
- **Resources:** 4 CPUs (GPU array: 1 GPU, 4 threads), no partition, **1 h per atom** (`ATOM_TIME_LIMIT_S = 3600  # TODO`).
- **Order:** one outer fold completely before the next. Within a fold, small → large: all event sizes of n samples, then n + 1 samples.
- **Cutoff:** a timeout or OOM marks where a method stops under this budget. Every larger atom (≥ samples **and** ≥ events) of the same method × dataset, **in any fold and trial**, is then skipped.
- **Skipping must not cost a SLURM allocation.** Hence lane workers instead of one job per atom (decided over: one job per atom + `scancel`, `srun` steps, dependency arrays; see below).
- **Hardware:** not pinned to one CPU / GPU type (queue time). The claim is a lower bound for the weakest node type, see "Hardware".
- **Hyperparameters:** as in step 7 (D6): Imstat per outer split, all other datasets the consensus config. Frozen into the manifest.

## Execution: lane workers (pure Python)
**Lane** = (method, dataset, outer split, trial) = one SLURM array task. The worker (`run <array> <lane>`) walks the lane's grid cells in order (n_samples ascending, and within each, n_events ascending) and handles each atom as follows:
1. **Done already?** If the atom JSON has status `ok` / `timeout` / `oom` / `failed` / `skipped` → skip. A `started` atom (lane killed earlier) is rerun.
2. **Dominated?** It looks for an atom of the same method × dataset (any split, any trial) with n_samples' ≤ n and n_events' ≤ e (`all` = ∞) and status `timeout` / `oom`. If there is one, it writes a small JSON `{status: skipped, skipped_by: <atom_id>}` and moves on. This takes milliseconds and needs no allocation. Other lanes that run at the same time see these failures via the shared FS.
3. **Enough time left in the lane?** It starts the atom only if `elapsed + ATOM_TIME_LIMIT_S + WRAPUP_BUFFER_S <= LANE_TIME_LIMIT_S` (`LANE_TIME_LIMIT_S` = SLURM `--time`, buffer ~10 min). Otherwise it stops cleanly (exit 0). This way the job limit never cuts an atom, and every `timeout` is a real cutoff.
4. **Run** the atom as a child process: `python training_data_requirements.py atom <array> <index>` via `subprocess.Popen(start_new_session=True)`, with stdout/stderr going to `logs/<method>/<dataset>/<stem>.log`. When the child exits, or `ATOM_TIME_LIMIT_S` is reached, the worker classifies the outcome:
   - child JSON `ok` → `ok` (counts complete, even if the kill came during interpreter shutdown);
   - over the time limit → `os.killpg(SIGKILL)` on the child's session (this also kills loky / DataLoader workers) → `timeout`;
   - exit −9 / SIGKILL without our timeout → `oom` (cgroup OOM killer);
   - child status `failed` with `MemoryError` / `TerminatedWorkerError` in the traceback → `oom`;
   - other Python errors → `failed` (e.g. a missing class at n_samples = 1). These do **not** dominate;
   - child status `ok` → `ok`.

   timeout / oom also write a marker `cutoffs/<method>/<dataset>/split_<i>__trial_<t>__ns_<s>__ne_<e>.json`, and the dominance check only lists this directory. The worker rewrites the child's JSON with that status and keeps the child's `phase` (`fit` / `predict`) and resources so far.

**Lane order in the array:** split → dataset → method → trial, so fold 0 is scheduled first. Within a fold, all lanes start small, so the whole array runs roughly from small to large.

**Restart:** a lane that stopped because of the time budget (step 3) or was lost (node failure) has unfinished atoms. `missing <array>` lists these lanes as a SLURM range, and resubmitting them continues where they stopped.

**Sequential = SLURM:** `run <array> all` runs all lanes one after another locally. The time logic is the same, so a local smoke test with a 10 s limit tests timeout and skipping.

**Size:** 4 methods × 10 datasets × 5 folds × 3 trials = **600 lanes** (sklearn_cpu 300, flagx_cpu 150, flagx_gpu 150), all below MaxArraySize. Atoms: about 40k, but most of them are small or get skipped.

### Rejected alternatives
- **One job per atom** (incl. `scancel` of dominated pending atoms): about 36k jobs in chunks of ≤ 10000 with offsets. Every tiny atom pays job startup + conda + imports. Needs `squeue` / `scancel` from compute nodes. If the OOM kill hits the batch shell, the cancel doesn't run.
- **A start-of-job check only:** one allocation per skipped atom.
- **`srun --time` steps inside the lane:** no extra allocation, and per-atom records in `sacct`. But it can only be tested on the cluster and depends on step behaviour there (environment export, GPU binding on TinyGPU).
- **Dependency arrays (`aftercorr` + `--kill-on-invalid-dep`):** about 70 arrays per env, cutoffs only within a lane, and Python errors must exit 0.

## Atom
Same as `benchmark.py::run_atom`, with these changes:
- **Training samples:** the first n of a per-trial permutation of the outer training samples of the split, `derive_seed('train_req', dataset, split, trial)`. The sets are nested over n and the same for all methods. `'all'` = all training samples.
- **Events (nested, 2026-11-18):** per training sample a fixed stratified event order `event_order(y, seed=derive_seed('train_req', dataset, split, trial, sample))`; n_events = the first n of it (`'all'` = no downsampling). This makes the subsets nested over n_events, and a sample brings the same events to every n_samples. Within each class the events are in random order; the j-th of m_c events gets the rank (j + 0.5) / m_c, and the events are stably sorted by rank. Class counts are n · m_c / m within ~1 event, and a class appears once its quota reaches ~0.5 (as with `downsample_indices`). `STRATIFIED = False` would give a plain random permutation. `validation.core.sampling.downsample_indices` is not nested (tested: 31 % of the steps 100 → … → 30000 lost events of the smaller subset). A class with quota 0 draws no random numbers, which shifts the stream for all later classes, and the final correction to exactly n adds more noise. It stays unchanged for tuning / benchmark. The legacy experiment was not nested either: one global seed for the whole grid. Cost: ~0.7 s per 3M-event sample (vs 0.16 s), outside the fit timing. Then `join_samples(shuffle=True, seed=derive_seed('train_req', dataset, split, trial, n_samples, n_events, 'join'))`.
- **Model seed:** `derive_seed('train_req', method, dataset, split, trial)`.
- **Loading:** `load_sample(..., mmap=True)` (new optional argument, default off), so a 100-event subset doesn't read a 3M-event file. Only the selected rows are copied.
- **Test samples:** predicted in full. Counts CSV per test sample × class as in the benchmark. **No** predictions.
- **JSON:** identity incl. trial / n_samples / n_events and the selected training samples, params + source, seeds, `n_train_events` (+ per class and per training sample × class, for the minority count analysis), preprocessor state, resources of the fit and of each test sample's predict, `phase` / `n_test_done` progress, metadata (`cpu_model`, `gpu_names`, host), and the status set by the worker.

## Grid (constants; `# TODO: dataset-specific grid`)
```python
N_SAMPLES = [1, 2, 4, 8, 16, 32, 64, 'all']        # values >= n_train of the split are dropped; 'all' always
N_EVENTS = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 'all']
```
Outer training samples per split: Imstat ~99, LT ~78, Flowcyt ~24, BLAST110 ~76–85. Events per sample: ~40k (Imstat) to ~3M (BLAST110). A cap above a sample's size = all its events, so for small samples the high caps duplicate `'all'`.

**Decided (2026-11-17):** duplicates are dropped per lane at manifest time: n_samples ≥ the number of training samples (= `'all'`), and event caps ≥ the largest selected sample (= `'all'` events). The cell (`'all'`, `'all'`) is left out, since the main benchmark (step 7) covers it. Default grid: 600 lanes, 36,604 atoms (sklearn_cpu 18,302; flagx_cpu 9,151; flagx_gpu 9,151; about 55 atoms per Imstat lane and 70 per BLAST110 lane).

## Outputs (`results/training_data_requirements/`)
```
manifests/<array>.csv          index, lane, atom_id, method, dataset, split, trial, n_samples, n_events,
                               n_train_samples_split, n_train_events (expected), config_id, param_source, params
cutoffs/<method>/<dataset>/split_<i>__trial_<t>__ns_<s>__ne_<e>.json   one marker per timeout / oom atom
atoms/<method>/<dataset>/split_<i>/trial_<t>/ns_<s>__ne_<e>.json
counts/<method>/<dataset>/split_<i>/trial_<t>/ns_<s>__ne_<e>.csv
logs/<method>/<dataset>/split_<i>/trial_<t>/ns_<s>__ne_<e>.log   child stdout/stderr
logs/tdr_<array>/<jobid>_<task>.out                              worker log (SLURM)
```
`atom_id = <method>__<dataset>__split_<i>__trial_<t>__ns_<s>__ne_<e>`.

## Commands
- `manifest <array> [--out]`: writes the frozen atom list (lane order as above, atom order inside a lane small → large), creates the log dirs, and prints #lanes / #atoms / `--array=0-<L-1>`.
- `run <array> [<lane> | all] [--out] [--smoke]`: the lane worker (default `$SLURM_ARRAY_TASK_ID`).
- `atom <array> <index> [--out] [--smoke]`: a single atom (the child process; also useful for debugging).
- `status <array>`: per method × dataset, the counts ok / timeout / oom / failed / skipped / started / missing, the frontier (largest ok cell per n_samples), max fit time / RAM, and the lanes that are not finished.
- `missing <array>`: lanes with unfinished atoms, as a SLURM range.
- `--smoke`: 3 test samples, 1 % of the events, a tiny grid, short time limits (only with `--out`).

## SLURM (`slurm_scripts/tdr_<array>.sh`)
- Woody (`sklearn_cpu`, `flagx_cpu`): `--cpus-per-task=4`, no partition, `--time=24:00:00` (= `LANE_TIME_LIMIT_S`), job name `tdr_<array>`, thread variables from `SLURM_CPUS_PER_TASK`, `python training_data_requirements.py run <array> $SLURM_ARRAY_TASK_ID`.
- TinyGPU (`flagx_gpu`): `--gres=gpu:1`, no partition, 4 threads, the rest as above.
- `--time` and `ATOM_TIME_LIMIT_S` are TODOs (1 h per atom may go up or down).
- `slurm_scripts/README.md`: new section.

## Hardware (lower bound)
Earlier atoms ran on Woody E3-1240 v5 / v6 (4-core nodes) and Gold 6326 (Ice Lake). The GPU nodes are mixed (rtx2080ti / rtx3080). With 4 cores, both Woody types give ~31 GB of RAM.

The online cutoff uses the raw 1 h. A success on a fast node does not certify the weakest node. So for a claim like "with 4 cores of type X one gets at least this far":
- **[TODO] Calibration:** rerun a few atoms (several methods / sizes) per node type with `--constraint` / `--gres=gpu:<type>:1` → slowdown factor per CPU / GPU model relative to the slowest.
- **Eval:** an atom counts as within budget only if `wall × slowdown(cpu_model) ≤ budget`. This only removes cells, so the frontier stays monotone and conservative. `cpu_model` / `gpu_names` are already recorded per atom (`run_metadata`).

## Shared helpers (the second user now exists)
- `outer_splits(dataset)` → `validation/core/data.py`.
- `best_config(method, dataset, split, best_dir, tuning_dataset)` → `validation/core/runs.py`.
- `load_sample(dataset, sample, mmap=False)`.
- `benchmark.py` imports them; its manifests must stay byte-identical.

## Risks
- **n_samples = 1 / 100 events:** classes can be missing from the training data (rare classes; binary datasets without positive events). The counts use the labels from `meta_labels.csv` and are not affected. A method that crashes ends as `failed` and doesn't dominate. Smoke test for this.
- **BLAST110 test folds** (~20 × ~3M events) take time with 4 cores, even with tiny models. A predict-phase timeout is a cutoff like any other (phase recorded).
- **Leftover child processes on timeout** (loky, DataLoader): process-group kill, plus a smoke test with `ps`.
- **OOM of the worker itself:** unlikely, because it never loads data.
- **OOM kill of the whole job (untested, accepted 2026-11-18):** normally the kernel OOM killer kills only the child (largest memory user in the job's cgroup) → return code < 0 → `oom` + marker, and the lane continues. If the cluster's cgroup setup kills the whole group instead (`memory.oom.group`), the worker dies as well. The atom then stays `started` without a cutoff marker, and every resubmit via `missing` would rerun it and crash the lane again. No real OOM was triggered in the smoke tests. Since OOM is unlikely with these methods and grid sizes, the current setup is kept. Visible in `status` (atom stuck at `started`, lane always `missing`). Fix if it happens: the worker records each attempt, and an atom found `started` again on a rerun gets `oom` (perhaps only after the 2nd crash, so node failures don't count). Optionally raise the child's `oom_score_adj` so it is killed first.

## Files
- New: `training_data_requirements.py`, `slurm_scripts/tdr_{sklearn_cpu,flagx_cpu,flagx_gpu}.sh`
- Edit: `validation/core/{data,runs}.py`, `benchmark.py` (imports), `slurm_scripts/README.md`, `PLAN.md`, `DONE.md`, `CLAUDE.md`, `.gitignore` if needed (child logs)

## Verification
- `manifest` for the 3 arrays into a scratch `--out`: lane / atom counts, ordering, params against the best-config JSONs.
- `run <array> <lane> --smoke` in `fx_sklearn` and `fx_flagx_cpu`, incl. n_samples = 1, n_events = 100 on Imstat.
- Forced timeout (tiny `ATOM_TIME_LIMIT_S`): `timeout` in the JSON, then `skipped` for the larger atoms of this lane and of a second lane. RF with `n_jobs=4`: no leftover processes.
- Tiny `LANE_TIME_LIMIT_S`: the lane stops cleanly; a rerun resumes it.
- `benchmark.py manifest` after the helper move: identical to the committed manifests.
