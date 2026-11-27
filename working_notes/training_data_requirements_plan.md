# Training data requirements plan: step 8 (`training_data_requirements.py`)

> Status: implemented and smoke-tested (2026-11-17; `DONE.md`, step 8). Not run yet. Open TODO: eval harness (incl. the post-hoc budget filter).

## Context
Step 7 shows how well every method performs with all training data. Step 8 asks how much training data the best methods need: the score, fit / predict time and memory as a function of the **number of training samples** × **events per sample**. It runs under a fixed small budget (4 CPUs or 1 GPU, 1 h per atom). The result also sets the reduced / minimal training sets used in step 9 (local training, preprocessing ablation). The legacy version of this experiment is `legacy/main.py::main_num_samples_num_events_experiment` (SOM / FCNN, one split, fixed sample order, grid 1–75 samples × 100–50k events + all).

## Decisions (from the request and the discussion of 2026-11-17)
- **Methods:** a constant at the top of the script: `rf`, `hgb`, `mlp`, `som` (decided 2026-11-24, reasons in `DONE.md`, step 8).
- **Atom** = (method, dataset, outer split, trial, n_samples, n_events).
  - **Trial** = training sample order + downsampling seed; the model seed is tied to the trial.
  - 3 trials (`N_TRIALS = 3`; fixed 2026-11-18, more trials only around the elbow in the follow-up, PLAN.md step 9).
- **Grid:** log-spaced defaults, dataset-specific later (`# TODO`), see below.
- **Data:** downsampling on the fly. Test samples are predicted in full (PLAN rule).
- **Outputs per atom:** as in the benchmark: resources + counts per test sample, **no predictions** (storage). The eval harness comes later in its own script.
- **Arrays:** `sklearn_cpu` (rf, hgb), `flagx_cpu` (som), `flagx_gpu` (mlp).
- **Resources:** 4 CPUs (GPU array: 1 GPU, 4 threads), no partition, **1 h per atom** (`ATOM_TIME_LIMIT_S = 3600`, fixed 2026-11-18, the same for all lanes; see Hardware).
- **Order:** one outer fold completely before the next. Within a fold, small → large: all event sizes of n samples, then n + 1 samples.
- **Cutoff:** a timeout or OOM marks where a method stops under this budget. Every atom of the same method × dataset with **at least as many total training events** (`n_train_events` ≥ that of the timeout / OOM atom), **in any fold and trial**, is then skipped (changed 2026-11-24 from "≥ samples and ≥ events": fit time and memory depend mainly on the total number of events).
- **Skipping must not cost a SLURM allocation.** Hence lane workers instead of one job per atom (decided over: one job per atom + `scancel`, `srun` steps, dependency arrays; see below).
- **Hardware:** not pinned to one CPU / GPU type (queue time), except that the slow Woody E3-1240 v5 nodes are excluded (2026-11-18). The claim is a lower bound for the weakest remaining node type (E3-1240 v6), see "Hardware".
- **Hyperparameters:** as in step 7 (D6): Imstat per outer split, all other datasets the consensus config. Frozen into the manifest.

## Execution: lane workers (pure Python)
**Lane** = (method, dataset, outer split, trial) = one SLURM array task. The worker (`run <array> <lane>`) walks the lane's grid cells in order (n_samples ascending, and within each, n_events ascending) and handles each atom as follows:
1. **Done already?** If the atom JSON has status `ok` / `timeout` / `oom` / `failed` / `skipped` → skip. A `started` atom (lane killed earlier) is rerun.
2. **Dominated?** It looks for an atom of the same method × dataset (any split, any trial) with at most as many total training events (`n_train_events` from the manifest; the marker stores it) and status `timeout` / `oom`. Ties are dominated (2026-11-24): without sample-size limits `n_train_events` = n_samples × n_events in every lane, so the same cell in the other 14 lanes is skipped instead of timing out again (a cell with any timeout is not eligible in the eval anyway). If there is one, it writes a small JSON `{status: skipped, skipped_by: <atom_id>}` and moves on. This takes milliseconds and needs no allocation. Other lanes that run at the same time see these failures via the shared FS.
3. **Enough time left in the lane?** It starts the atom only if `now + ATOM_TIME_LIMIT_S + WRAPUP_BUFFER_S <= deadline` (buffer ~10 min). Deadline (2026-11-24): the end of the SLURM job, from its time left (`squeue -j $SLURM_JOB_ID -h -o %L`) read at the worker's start, since with `--time-min` the limit varies per task; outside SLURM lane start + `LANE_TIME_LIMIT_S`. Otherwise it stops cleanly (exit 0). This way the job limit never cuts an atom, and every `timeout` is a real cutoff.
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

**Size:** 4 methods × 10 datasets × 5 folds × 3 trials = **600 lanes** (sklearn_cpu 300, flagx_cpu 150, flagx_gpu 150), all below MaxArraySize. Atoms: 59,524 (grid of 2026-11-18), but most of them are small or get skipped.

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

## Grid (constants; decided 2026-11-18)
```python
N_SAMPLES = {'default': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 'all'],   # values >= n_train are dropped
             'flowcyt': [1, 2, 3, 4, 6, 8, 12, 16, 'all']}                   # 24 training samples
N_EVENTS = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 'all']       # same for all datasets
```
**Reasoning:** based on the legacy SOM / MLP results (`legacy/results/num_samples_num_events/`; one trial, older curation, no BLAST110).
- Macro F1 comes within ~0.02 of the plateau mostly at 2–16 samples (Imstat SOM ≥1k events: ~3; LT1b SOM: ~3 at ≥5k, ~8 at 1k). MLP on LT1 / LT2 keeps improving up to 30–58 samples.
- Over the events, the transitions run 100 → 1k → 5k (Imstat, LT) and 10k → 50k → all (FlowCyt).
- Hence ~×1.5 steps over the samples (dense at the knee, up to 64) and ×3 steps over the events (the effect is multiplicative; 2–4 points per transition).
- One event grid for all datasets: a sample with fewer events than the cap goes in completely. 100 events stay in as the lower end (minority count analysis).
- BLAST110 has no reference and gets the default grid. Its large cells are expected to hit the time limit (= cutoff).

**Justification of the spacing (for the paper; 2026-11-18):**
- **Log-uniform grid.** Learning curves are roughly linear in log(training size), so the ratio between grid points matters, not the difference.
  - The event grid has a constant ratio of ~×3 (half-decade steps of ~0.5 in log₁₀), so 3k → 10k is no larger a step than 1k → 3k.
  - Equal steps over 3.5 decades resolve every part of the curve equally, without knowing in advance where each method's knee is. The knee differs by method and dataset, and BLAST110 has no prior data.
- **Resolution backed by legacy data.** Mean change in macro F1 per step, by n_samples 1 / 2–4 / >4:
  - The steep part is 1k → 5k (e.g. MLP Imstat +0.43 / +0.16 / +0.02); the grid has two points there (1k, 3k).
  - 5k → 10k: +0.00–0.03 for 2–4 samples, ~0 for >4, within the single-trial noise (±0.1 at 1 sample).
  - Exception: FlowCyt rises up to 'all' (MLP 5k → 10k → 20k: +0.05 / +0.10 at 2–4 samples). This range is covered by 10k, 30k, 100k, 300k, all.
- **Samples finer (~×1.5).** The sample axis spans only ~2 decades (1–100) and is limited to integers, and the knees lie at 2–16 samples. With ×2 steps that range would get only 4–5 points.
- **Cost.** Each extra event value adds ~5,000 atoms. Refining 3k–10k (e.g. adding 5k) would add resolution where the legacy curves are flat.
- **Methods sentence (draft):** "Training set size was varied on log-spaced grids: the number of training samples in steps of ≈1.5× (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, all) and the number of events per sample in half-decade steps (100, 300, 1,000, …, 300,000, all), giving uniform resolution on the logarithmic scale on which learning curves are typically approximately linear."
Outer training samples per split: Imstat ~99, LT ~78, Flowcyt ~24, BLAST110 ~76–85. Events per sample: ~40k (Imstat) to ~3M (BLAST110). A cap above a sample's size = all its events, so for small samples the high caps duplicate `'all'`.

**Decided (2026-11-17):** duplicates are dropped per lane at manifest time: n_samples ≥ the number of training samples (= `'all'`), and event caps ≥ the largest selected sample (= `'all'` events). The cell (`'all'`, `'all'`) is left out, since the main benchmark (step 7) covers it. First default grid (log-spaced samples 1, 2, 4, …, 64): 36,604 atoms. Grid of 2026-11-18: 600 lanes, **59,524 atoms** (sklearn_cpu 29,762; flagx_cpu 14,881; flagx_gpu 14,881; about 90 atoms per Imstat / LT lane, 80 per FlowCyt lane and 115 per BLAST110 lane).

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
- Woody (`sklearn_cpu`, `flagx_cpu`): `--cpus-per-task=4`, `--constraint=icx` (Gold 6326 only, 2026-11-24), no partition, `--time=24:00:00 --time-min=02:00:00`, job name `tdr_<array>`, thread variables from `SLURM_CPUS_PER_TASK`, `python training_data_requirements.py run <array> $SLURM_ARRAY_TASK_ID`.
- TinyGPU (`flagx_gpu`): `--gres=gpu:rtx2080ti:1` (RTX 2080 Ti only, default partition `work`, decided 2026-11-27, see Hardware), 4 threads, the rest as above.
- `--time` = 24 h (cluster maximum) per lane; `ATOM_TIME_LIMIT_S` = 1 h (decided 2026-11-18).
- `--time-min=02:00:00` (decided 2026-11-24, all three scripts): the backfill scheduler may start a task with any limit between 2 h and 24 h, so tasks also fit into scheduling gaps and long slots are still used fully. Woody uses `sched/backfill`; Slurm accepted the option (TimeMin shown next to TimeLimit), but a start with a reduced limit was not yet observed (assumed to work). Backfill considers at most 20 jobs per user per cycle (`bf_max_job_user=20`), so the gain is limited. The worker reads its job's time left, so no constant has to match `--time`. Resubmission stays manual: `missing` after the array has finished (`squeue` empty), never while tasks of the previous submission are pending or running (a lane would run twice; there is no lock).
- `slurm_scripts/README.md`: new section.

## Hardware (lower bound)
Earlier atoms ran on Woody E3-1240 v5 / v6 (4-core nodes) and Gold 6326 (Ice Lake). The GPU nodes are mixed (rtx2080ti / rtx3080). With 4 cores, both Woody types give ~31 GB of RAM.

**Time limit 1 h per atom (decided 2026-11-18).** Basis: fit time per training event on 4 cores from the tuning atoms (best configs, ~2.5M training events, all three Woody node types; MLP on TinyGPU):

| Method | Gold 6326 (icx) | E3-1240 v6 (kl) | E3-1240 v5 (sl) | Max. training events in 1 h |
|---|---|---|---|---|
| RF | 134 µs | 129 µs | 189 µs | ~27M → ~19M (v5) |
| HGB | 177 µs | 174 µs | 530 µs | ~20M → ~7M (v5) |
| SOM | 3.0 ms | 6.9 ms | 16.1 ms | ~1.2M → ~0.2M (v5) |
| MLP (GPU) | 303 µs (range 120–1060) | | | ~12M |

- Predicting the validation set (~1.4M events) took 1–60 s (HGB on v5: ~5 min). The limit is mostly fit time.
- Legacy elbows in pooled training events (SOM / MLP): SOM ≤ ~0.1M (FlowCyt ~4 × 20k; Imstat, LT far below); MLP ~1.5–7M (LT1 / LT2, FlowCyt). So 1 h covers every elbow even on the slowest node, with the least margin for SOM on v5 and MLP (~2×).
- 30 min would put SOM on v5 at the FlowCyt elbow and make prediction a noticeable part of the budget for HGB. 2 h would move the frontier by only about one grid step, and SOM lanes are already long. A single limit for all lanes and methods keeps the budget statement simple: "1 h on 4 cores".
- The tuning-based SOM rates are ~3× (Gold) to ~15× (v5) slower than the extrapolation from the 32-core pilot. So the compute estimate for the SOM array is too optimistic: more SOM atoms near the frontier time out.

**Node types (Woody docs, doc.nhr.fau.de/clusters/woody, read 2026-11-18):** v5 (`sl`, w12xx/w13xx) 62 nodes × 4 cores; v6 (`kl`, w14xx/w15xx) 108 × 4; Gold 6326 (`icx`, w22xx–w25xx) 110 × 32 (80 of them financed by ECAP / physics). Only partition `work`, 24 h max. Selection by `--constraint=sl|kl|icx`. The docs do not mention excluding a type, but Slurm's OR syntax `--constraint="kl|icx"` would exclude only v5. **Decided (2026-11-18): exclude v5**, `#SBATCH --constraint="kl|icx"` in both Woody scripts. Reasons:
  - v5 is only ~6 % of the 4-core slots (62 of ~1,050), so the scheduling loss is small.
  - A cutoff applies to all folds / trials, so one lane landing on v5 could pull in the frontier for a whole method × dataset (SOM v5 ≈ 2.3× v6, 5× Gold; HGB ~3×). Every needless timeout also costs a full hour.
  - The lower bound is then the E3-1240 v6 (4 cores, 3.7 GHz, 32 GB, 2017 desktop class), still a modest machine for the local-training claim. Superseded 2026-11-24: icx only, see below.
  - The OR syntax is standard Slurm but not shown in the NHR docs: test once with `sbatch --constraint="kl|icx" --wrap hostname`.

**Woody: Gold 6326 (`icx`) only (decided 2026-11-24, replaces "kl|icx").** `#SBATCH --constraint=icx` in both Woody scripts. All CPU atoms then run on one CPU type, so the cutoff and the timings need no calibration or slowdown factors and do not depend on node assignment. The budget statement becomes "1 h on 4 cores of an Intel Xeon Gold 6326 (Ice Lake, 2021)". Per core, RF / HGB are as fast as on v6, SOM ~2.3× faster (table above), so for SOM the frontier is less conservative for older desktop / laptop CPUs than with v6; state the CPU model. 110 × 32 cores = ~880 4-core slots (v6: 432), part of them financed by ECAP / physics. TinyGPU (MLP) was mixed (rtx2080ti / rtx3080) until 2026-11-27, see below.

**TinyGPU: RTX 2080 Ti only (decided 2026-11-27, user, replaces the mixed work partition).** `#SBATCH --gres=gpu:rtx2080ti:1` in `tdr_flagx_gpu.sh` and `sor_flagx_gpu.sh`. One GPU and CPU type (RTX 2080 Ti 11 GB, Gold 6134, 8 cores / 22 GB host share per GPU, 32 GPUs) for all MLP atoms, as `icx` on Woody; budget statement "1 h on 1 RTX 2080 Ti + 4 threads". Chosen over the RTX 3080 (48 GPUs, 1.1–1.9× faster MLP fits in step 7, 46 GB in partition `rtx3080`) for the shorter queue: in step 7 (untyped request) 126 of 150 MLP atoms landed on a 2080 Ti, 16 on a 3080. V100 no faster than the 2080 Ti in step 7; A100 owned by groups with priority access, 32 cores per GPU. Expected limits (step-7 MLP fits on full data, ~3 min and ~0.6 GB per million training events): the 1 h timeout comes at ~15–20M training events, before the 22 GB host RAM (~28M). Imstat (3.9M), LT (7.4–7.6M) and Flowcyt (17M, 46 min) fit with full data; the legacy MLP convergence points (`scratchpad/results/legacy_criterion_check/legacy_minimal.csv`, noninf δ = 0.02: ≤ ~3M events) lie well below the cutoff. BLAST110 (50–63M) is cut off at ~25–35 % of the full data; no legacy reference, so whether MLP converges below the cutoff is open.

**Post-hoc check (eval):** lanes run in parallel, so an atom above the final cutoff can finish `ok` before a smaller atom of another lane times out (faster node, or split / trial differences). Finished atoms are not rewritten, but once the marker exists every larger atom is skipped, so only a few atoms per method × dataset are affected. The eval applies the rule again over all atoms (`training_data_requirements_eval_plan.md`, "Compute budget"); `status` shows the cutoff and the number of `ok` atoms above it (`ok>cut`).

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
- **OOM kill of the whole job (accepted 2026-11-18; not expected on Woody, evidence 2026-11-24 below):** normally the kernel OOM killer kills only the child (largest memory user in the job's cgroup) → return code < 0 → `oom` + marker, and the lane continues. If the cluster's cgroup setup kills the whole group instead (`memory.oom.group`), the worker dies as well. The atom then stays `started` without a cutoff marker, and every resubmit via `missing` would rerun it and crash the lane again. No real OOM was triggered in the smoke tests. Since OOM is unlikely with these methods and grid sizes, the current setup is kept. Visible in `status` (atom stuck at `started`, lane always `missing`). Fix if it happens: the worker records each attempt, and an atom found `started` again on a rerun gets `oom` (perhaps only after the 2nd crash, so node failures don't count). Optionally raise the child's `oom_score_adj` so it is killed first.
  - Evidence from the step-7 pilot (two SOM BLAST110 atoms OOM on Woody): the SLURM log shows `slurm_script: line 28: <pid> Killed python benchmark.py run …` followed by `error: Detected 1 oom_kill event in StepId=<job>.batch`. The batch shell, a second process in the same cgroup, survived and printed the message, so the OOM killer killed only the largest process, not the whole cgroup. Same on TinyGPU (DGCyTOF OOM, task 250). The login node mounts cgroup v1, which has no group kill (`memory.oom.group` exists only in v2); compute nodes not checked directly. "Detected oom_kill event" is only a log entry: it doesn't end the step, at most it sets the job state to `OUT_OF_MEMORY` at the end.
  - For the lanes (shell → worker → child): the child is by far the largest process, so it is killed (return code −9), the worker survives and writes `oom` + the cutoff marker, and the lane continues. A lane with a handled child OOM may still end with job state `OUT_OF_MEMORY`; this doesn't mean the lane failed. The planned `sacct` check of `started` atoms is therefore not implemented.
  - Keep starting the worker directly in the batch script, **not with `srun`**: Woody has `KillOnBadExit=1`, so under `srun` a task exiting non-zero would end the whole step.
  - joblib / loky worker processes (if a method uses them; RF uses threads, HGB OpenMP): the OOM killer may kill such a worker instead of the child → `TerminatedWorkerError` (in `OOM_MARKERS` → `oom`) or a hang → `timeout`. Both are cutoffs; the lane continues.
  - `start_new_session=True` changes only the process group, not the cgroup: the ~31 GB limit applies to the whole lane.
  - Still to do once: forced-OOM atom in the cluster smoke test (confirms the parent / child behavior on a compute node).

## Files
- New: `training_data_requirements.py`, `slurm_scripts/tdr_{sklearn_cpu,flagx_cpu,flagx_gpu}.sh`
- Edit: `validation/core/{data,runs}.py`, `benchmark.py` (imports), `slurm_scripts/README.md`, `PLAN.md`, `DONE.md`, `CLAUDE.md`, `.gitignore` if needed (child logs)

## Verification
- `manifest` for the 3 arrays into a scratch `--out`: lane / atom counts, ordering, params against the best-config JSONs.
- `run <array> <lane> --smoke` in `fx_sklearn` and `fx_flagx_cpu`, incl. n_samples = 1, n_events = 100 on Imstat.
- Forced timeout (tiny `ATOM_TIME_LIMIT_S`): `timeout` in the JSON, then `skipped` for the larger atoms of this lane and of a second lane. Rerun 2026-11-24 for the total-events rule (local, `fx_plot`, children killed before the method import): the marker stores `n_train_events`; with a cutoff at 500 events (rule `>`), atoms with 100 / 200 / 300 events ran and 600 / 1000 / 2000 were skipped; the switch to `≥` was tested with a cutoff at exactly 300 events; `status` reports the cutoff and `ok>cut`. RF with `n_jobs=4`: no leftover processes.
- Tiny `LANE_TIME_LIMIT_S`: the lane stops cleanly; a rerun resumes it.
- `benchmark.py manifest` after the helper move: identical to the committed manifests.
