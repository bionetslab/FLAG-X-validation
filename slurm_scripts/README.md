# SLURM scripts

One script per (env, cluster) array. Every array task runs one atom. CPU arrays run on Woody (`sbatch`), the GPU array on TinyGPU (`sbatch.tinygpu`, frontend `tinyx`). Docs: https://doc.nhr.fau.de/

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

N as of the grid review (2026-11-05), from the committed manifests (`results/tune_hyperparameters/manifests/<array>.csv`). The scripts hard-code `#SBATCH --array=0-<N-1>%<throttle>` (at most 50 tasks running at once on Woody, 20 on TinyGPU); after a grid change, write a new manifest and update the range in the script. `--array` on the `sbatch` command line overrides the script (e.g. to resubmit selected indices).

**Prerequisites on the cluster**
- The repo is cloned and `data/curated/imstat/` is copied into it, for example under `$WORK`.
- Own conda installation in `~/miniforge3` (the scripts run `module purge` and `source ~/miniforge3/etc/profile.d/conda.sh`, no NHR python module); the envs are created there from `envs/*.yml`. Mind the `$HOME` quotas (also inodes); NHR recommends `$WORK` for conda.

**What a Woody script does:** one array task = one atom. Each task gets 4 cores (memory: 7.75 GB per core on Woody, so 31 GB; no `--mem`), 24 h (the cluster maximum, on purpose: an atom that does not finish in 24 h puts its config out), activates the array's env, sets the thread variables from `SLURM_CPUS_PER_TASK` and runs `python tune_hyperparameters.py run <array> "$SLURM_ARRAY_TASK_ID"` directly (no `srun`: one process per task, as in the NHR OpenMP example) from the submit directory (`SLURM_SUBMIT_DIR`, hence: submit from the repo root).

**The TinyGPU script** (`tune_flagx_gpu.sh`) is the same, except: 1 RTX 3080 (`--gres=gpu:rtx3080:1 --partition=rtx3080`) with its fixed host share of 8 cores and 46 GB RAM (no `--cpus-per-task`; `SLURM_CPUS_PER_TASK` falls back to 8 for the thread variables), throttle 20, submitted with `sbatch.tinygpu` on `tinyx`. Create `fx_flagx_gpu` in an interactive TinyGPU job, so conda picks the CUDA build of torch (NHR: "For GPU support of your packages you must use an interactive job"). DGCyTOF uses the GPU only for training; its prediction (incl. the calibration) runs on the 8 cores.

**Run, from the repo root:**

```bash
# 1. Freeze the task list (already done and committed for the current grids). This prints N and creates the log
#    dir. Any env with numpy works; the manifest imports no method code.
python tune_hyperparameters.py manifest sklearn_cpu

# 2. Submit (the array range and the throttle are in the scripts)
sbatch slurm_scripts/tune_sklearn_cpu.sh
sbatch.tinygpu slurm_scripts/tune_flagx_gpu.sh     # on the TinyGPU frontend tinyx

# 3. Resume: resubmit the same command. Atoms that already have a JSON are skipped.
```

**Atom status**

Each atom writes `results/tune_hyperparameters/atoms/<method>/<config_id>/split_<i>_inner_<j>.json`. Its `status` is:
- `ok`: finished.
- `failed`: Python error; the traceback is in the JSON.
- `started`: killed by a timeout, OOM or node failure; see the log `results/tune_hyperparameters/logs/tune_<array>/<jobid>_<task>.out` (stdout and stderr together, unbuffered; Slurm's timeout / OOM message lands there too) or `sacct -j <jobid>`. The log folders are committed (`.gitkeep`), so they exist on a fresh clone.

Any status other than `ok` puts the config out. To rerun an atom deliberately, delete its JSON.

**Do not run `manifest` again between submission and the end of the array.** The array indices refer to the frozen manifest. After a grid change, write a new manifest and submit again; finished atoms are kept, because `config_id` is a hash of the params.
