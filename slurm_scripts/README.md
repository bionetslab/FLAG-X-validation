# SLURM scripts

One script per (env, cluster) array. Every array task runs one atom (training data requirements: one lane of atoms). CPU arrays run on Woody (`sbatch`), the GPU array on TinyGPU (`sbatch.tinygpu`, frontend `tinyx`). Docs: https://doc.nhr.fau.de/

`MaxArraySize` is 10000 on Woody and on TinyGPU, so every array fits into one submission.

## Hyperparameter tuning (`tune_hyperparameters.py`, PLAN.md step 6, `working_notes/tuning_plan.md`)

| Array | Script | Env | Cluster | Methods | Atoms (N) |
|---|---|---|---|---|---|
| `sklearn_cpu` | `tune_sklearn_cpu.sh` | `fx_sklearn` | Woody | rf, hgb, logreg, knn, lda_sklearn, gmm_mocking | 2010 |
| `flagx_cpu` | `tune_flagx_cpu.sh` | `fx_flagx_cpu` | Woody | som | 420 |
| `flagx_gpu` | `tune_flagx_gpu.sh` | `fx_flagx_gpu` | TinyGPU | mlp, dgcytof | 1350 |
| `gatemeclass_cpu` | `tune_gatemeclass_cpu.sh` | `fx_gatemeclass` | Woody | gatemeclass | 270 |
| `lda_cpu` | `tune_lda_cpu.sh` | `fx_lda` | Woody | lda_abdelaal | 15 |
| `deepcytof_cpu` | `tune_deepcytof_cpu.sh` | `fx_deepcytof` | Woody | deepcytof | 360 |

N as of the grid review (2026-11-05), from the committed manifests (`results/tune_hyperparameters/manifests/<array>.csv`). The scripts hard-code the array range `#SBATCH --array=0-<N-1>`, without throttle (Slurm schedules as many tasks as the cluster and the fair-share allow); after a grid change, write a new manifest and update the range in the script. `--array` on the `sbatch` command line overrides the script (e.g. to resubmit selected indices).

**Prerequisites on the cluster**
- The repo is cloned and `data/curated/imstat/` is copied into it, for example under `$WORK`.
- Own conda installation in `~/miniforge3` (the scripts run `module purge` and `source ~/miniforge3/etc/profile.d/conda.sh`, no NHR python module); the envs are created there from `envs/*.yml`. Mind the `$HOME` quotas (also inodes); NHR recommends `$WORK` for conda.

**What a Woody script does:** one array task = one atom. Each task gets 4 cores (memory: 7.75 GB per core on Woody, so 31 GB; no `--mem`), 24 h (the cluster maximum, on purpose: an atom that does not finish in 24 h puts its config out), activates the array's env, sets the thread variables from `SLURM_CPUS_PER_TASK` and runs `python tune_hyperparameters.py run <array> "$SLURM_ARRAY_TASK_ID"` directly (no `srun`: one process per task, as in the NHR OpenMP example) from the submit directory (`SLURM_SUBMIT_DIR`, hence: submit from the repo root).

**The TinyGPU script** (`tune_flagx_gpu.sh`) is the same, except: 1 GPU of any type among RTX 2080 Ti, RTX 3080 (partition `work`) and V100 (partition `v100`): `--gres=gpu:1 --partition=work,v100`, the job starts in whichever partition frees a GPU first. Fixed host share per GPU: 8 cores and 22 GB RAM (no `--cpus-per-task`; `SLURM_CPUS_PER_TASK` falls back to 8 for the thread variables), submitted with `sbatch.tinygpu` on `tinyx`. Create `fx_flagx_gpu` in an interactive TinyGPU job, so conda picks the CUDA build of torch (NHR: "For GPU support of your packages you must use an interactive job"). DGCyTOF uses the GPU only for training; its prediction (incl. the calibration) runs on the 8 cores.

**Run, from the repo root:**

```bash
# 1. Freeze the task list (already done and committed for the current grids). This prints N and creates the log
#    dir. Any env with numpy works; the manifest imports no method code.
python tune_hyperparameters.py manifest sklearn_cpu

# 2. Submit (the array range is in the scripts)
sbatch slurm_scripts/tune_sklearn_cpu.sh
sbatch.tinygpu slurm_scripts/tune_flagx_gpu.sh     # on the TinyGPU frontend tinyx

# 3. Resume: resubmit the same command. Atoms that already have a JSON are skipped.
#    Or only the missing atoms (no JSON yet): python tune_hyperparameters.py missing sklearn_cpu prints a range for
#    sbatch --array=<range> slurm_scripts/tune_sklearn_cpu.sh (the command line overrides the script's range).
```

**Atom status**

Each atom writes `results/tune_hyperparameters/atoms/<method>/<config_id>/split_<i>_inner_<j>.json`. Its `status` is:
- `ok`: finished.
- `failed`: Python error; the traceback is in the JSON.
- `started`: killed by a timeout, OOM or node failure; see the log `results/tune_hyperparameters/logs/tune_<array>/<jobid>_<task>.out` (stdout and stderr together, unbuffered; Slurm's timeout / OOM message lands there too) or `sacct -j <jobid>`. The log folders are committed (`.gitkeep`), so they exist on a fresh clone.

Any status other than `ok` puts the config out. To rerun an atom deliberately, delete its JSON.

**Do not run `manifest` again between submission and the end of the array.** The array indices refer to the frozen manifest. After a grid change, write a new manifest and submit again; finished atoms are kept, because `config_id` is a hash of the params.

## Main benchmark (`benchmark.py`, PLAN.md step 7, `working_notes/benchmark_plan.md`)

Same setup as the tuning (clusters, GPU, 24 h, env per array), except that the Woody scripts request 32 cores per task (248 GB, for the memory and runtime of the larger datasets; such jobs run on the Ice Lake nodes `w22xx`–`w25xx`). One array task = one atom = (method, dataset, outer split, model seed).

| Array | Script | Env | Cluster | Methods (seeds) | Atoms (N) |
|---|---|---|---|---|---|
| `sklearn_cpu` | `bench_sklearn_cpu.sh` | `fx_sklearn` | Woody | rf, hgb, gmm_mocking (3); logreg, knn, lda_sklearn (1) | 600 |
| `flagx_cpu` | `bench_flagx_cpu.sh` | `fx_flagx_cpu` | Woody | som (3) | 150 |
| `flagx_gpu` | `bench_flagx_gpu.sh` | `fx_flagx_gpu` | TinyGPU | mlp, dgcytof (3) | 300 |
| `gatemeclass_cpu` | `bench_gatemeclass_cpu.sh` | `fx_gatemeclass` | Woody | gatemeclass (3) | 150 |
| `lda_cpu` | `bench_lda_cpu.sh` | `fx_lda` | Woody | lda_abdelaal (1) | 50 |
| `deepcytof_cpu` | `bench_deepcytof_cpu.sh` | `fx_deepcytof` | Woody | deepcytof (3) | 150 |

N = 10 datasets × 5 outer splits × seeds per method, from the manifests `results/benchmark/manifests/<array>.csv` (2026-11-11). The parameters are frozen into the manifests from `results/tune_hyperparameters_eval/best_configs/` (Imstat: per outer split; other datasets: consensus). The cluster needs all `data/curated/<dataset>/` directories.

```bash
# 0. Task lists: committed (results/benchmark/manifests/); rewrite only after a change (prints N, creates the log dirs)
python benchmark.py manifest sklearn_cpu
# 1. Pilot: split 0 / seed 0 of every method x dataset, full data (120 atoms); the command prints the index list
python benchmark.py pilot sklearn_cpu
sbatch --array=<printed list> slurm_scripts/bench_sklearn_cpu.sh
sbatch.tinygpu --array=<printed list> slurm_scripts/bench_flagx_gpu.sh     # on tinyx
# 2. Status per method x dataset: ok / failed / started / missing, max training events, fit time, peak RAM of fit /
#    predict (also of killed atoms), progress of unfinished atoms (e.g. started:predict 6/22), non-ok SLURM range
python benchmark.py status sklearn_cpu
# 3. Caps for the datasets with failures (see below), then the final run of all atoms
python benchmark.py cap blast110_p1 20000000          # per-sample cap n for <= 20M training events in every split
mv results/benchmark results/benchmark_pilot          # after editing TRAIN_DOWNSAMPLING and committing
python benchmark.py manifest sklearn_cpu              # new manifests in results/benchmark (also the log dirs)
sbatch slurm_scripts/bench_sklearn_cpu.sh             # full array range (as in the script)
```

Outputs per atom (`<stem>` = `split_<i>_seed_<r>[__ds_<tag>]`): `atoms/<method>/<dataset>/<stem>.json` (status as in the tuning: `ok` / `failed` / `started` = killed by timeout, OOM or node failure; plus `phase` `fit` / `predict` / `done` and `n_test_done`: the JSON is rewritten after the fit and after each test sample, so a job killed during prediction keeps its fit resources and progress), `counts/<method>/<dataset>/<stem>.csv` (rewritten after each test sample; complete only if the JSON is `ok`), `predictions/<method>/<dataset>/<stem>/<sample>.npz` (predicted labels; git-ignored). Logs: `results/benchmark/logs/bench_<array>/`.

**Caps (after the pilot).** Finding the exact feasible size per dataset is too expensive, so:
- Every dataset on which a pilot atom failed gets one per-sample event cap `{'n': n}` in `TRAIN_DOWNSAMPLING`, for **all** methods of that dataset (the data seeds do not depend on the method, so all methods train on the same subsets).
- The training budget is the largest size that went through, scaled with a margin by the measured fit time / peak RAM of the binding method (`status`); `cap` turns it into `n`.
- The pilot is moved to `results/benchmark_pilot/` as a record; the final run writes new manifests into `results/benchmark/` and runs all atoms (also those of uncapped datasets). The row order is unchanged (same indices); capped atoms get the suffix `__ds_n<n>`.
- A failure during prediction (`started:predict k/n`, or `failed` in phase `predict`) is a result and is reported; the cap only addresses the fit.

## Training data requirements (`training_data_requirements.py`, PLAN.md step 8, `working_notes/training_data_requirements_plan.md`)

One array task = one **lane** = (method, dataset, outer split, trial), not one atom. The lane worker runs the lane's atoms (n_samples × n_events cells) from small to large, each as a child process with a per-atom time limit (`ATOM_TIME_LIMIT_S`, 1 h). A timeout or OOM is a cutoff: all larger atoms of the same method × dataset (any split / trial) are then skipped without running, so skipped atoms never cost an allocation. The worker starts an atom only if it can finish before the end of its SLURM job minus a buffer (time left read from `squeue` at start, since `--time-min` makes the limit vary; `WRAPUP_BUFFER_S`); otherwise it stops and the lane is resubmitted later.

| Array | Script | Env | Cluster | Methods | Lanes | Atoms |
|---|---|---|---|---|---|---|
| `sklearn_cpu` | `tdr_sklearn_cpu.sh` | `fx_sklearn` | Woody | rf, hgb | 300 | 29762 |
| `flagx_cpu` | `tdr_flagx_cpu.sh` | `fx_flagx_cpu` | Woody | som | 150 | 14881 |
| `flagx_gpu` | `tdr_flagx_gpu.sh` | `fx_flagx_gpu` | TinyGPU | mlp | 150 | 14881 |

Lanes = 10 datasets × 5 outer splits × 3 trials per method (counts of 2026-11-18, grid with ~×1.5 sample steps). Resources: Woody 4 cores (~31 GB), `--constraint=icx` (Gold 6326 only: one CPU type for all atoms, see the working note), no partition, `--time=24:00:00 --time-min=02:00:00` per lane (the scheduler may start a task with 2–24 h to fill gaps; the worker reads its job's time left from `squeue` and starts an atom only if it fits); TinyGPU 1 RTX 2080 Ti (`--gres=gpu:rtx2080ti:1`, default partition `work`, host share 8 cores / 22 GB), 4 threads; one GPU type for all atoms (the step-9 `sor_flagx_gpu.sh` as well). `cpu_model` / `gpu_names` are recorded per atom.

```bash
python training_data_requirements.py manifest sklearn_cpu    # prints the number of lanes L (--array=0-<L-1>)
sbatch slurm_scripts/tdr_sklearn_cpu.sh
sbatch.tinygpu slurm_scripts/tdr_flagx_gpu.sh                # on tinyx
python training_data_requirements.py status sklearn_cpu      # atoms per status, cutoffs, unfinished lanes
python training_data_requirements.py missing sklearn_cpu     # unfinished lanes as --array range -> resubmit them
sbatch --array=<printed range> slurm_scripts/tdr_sklearn_cpu.sh
```

Resubmit only after the previous submission has finished (`squeue -u $USER -n tdr_sklearn_cpu` empty): `missing` also lists lanes that are still pending or running, and a lane submitted twice runs twice (no lock).

Statuses: `ok`, `failed` (Python error, e.g. a class with a single training event; not a cutoff), `timeout`, `oom` (killed by a signal, or a memory error in the traceback), `skipped` (dominated by a cutoff, `skipped_by`), `started` (lane killed during the atom; rerun on resubmission). Logs: one per atom under `results/training_data_requirements/logs/<method>/<dataset>/`, the worker logs under `logs/tdr_<array>/`.

## Sample order robustness (`sample_order_robustness.py`, PLAN.md step 9, `working_notes/sample_order_robustness_plan.md`)

Same lane setup as the training data requirements (one array task = one lane = method × dataset × outer split × trial, same resources), with the scripts `sor_<array>.sh`. The order scheme (`random` / `stratified`) is the script's argument, and every scheme has its own manifests and outputs under `results/sample_order_robustness/<scheme>/`. The lane count differs per scheme (Flowcyt only in `random`) and depends on the step-8 minimal cells, so `--array` is given at submission. Worker logs: `results/sample_order_robustness/logs/sor_<array>/`. TODO before the run: set `--time` from the lane time estimate (TODO in the script).

```bash
python sample_order_robustness.py manifest random sklearn_cpu      # prints the number of lanes L
sbatch --array=0-<L-1> slurm_scripts/sor_sklearn_cpu.sh random
sbatch.tinygpu --array=0-<L-1> slurm_scripts/sor_flagx_gpu.sh random   # on tinyx
python sample_order_robustness.py status random sklearn_cpu
python sample_order_robustness.py missing random sklearn_cpu       # -> sbatch --array=<printed range> ...
```
