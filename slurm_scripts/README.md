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

N as of the grid review (2026-11-05); `manifest` prints the current value.

**Prerequisites on the cluster**
- The repo is cloned and `data/curated/imstat/` is copied into it, for example under `$WORK`.
- The envs are created there from `envs/*.yml`.
- `module load python` gives `conda`.

**Run, from the repo root:**

```bash
# 1. Freeze the task list. This prints N and creates the log dir.
#    Any env with numpy works; the manifest imports no method code.
python tune_hyperparameters.py manifest sklearn_cpu

# 2. Submit (the throttle %50 limits the number of concurrently running tasks)
sbatch --array=0-2009%50 slurm_scripts/tune_sklearn_cpu.sh
sbatch.tinygpu --array=0-1349%20 slurm_scripts/tune_flagx_gpu.sh

# 3. Resume: resubmit the same command. Atoms that already have a JSON are skipped.
```

**Atom status**

Each atom writes `results/tune_hyperparameters/atoms/<method>/<config_id>/split_<i>_inner_<j>.json`. Its `status` is:
- `ok`: finished.
- `failed`: Python error; the traceback is in the JSON.
- `started`: killed by a timeout, OOM or node failure; see the log in `results/tune_hyperparameters/logs/tune_<array>/` or `sacct -j <jobid>`.

Any status other than `ok` puts the config out. To rerun an atom deliberately, delete its JSON.

**Do not run `manifest` again between submission and the end of the array.** The array indices refer to the frozen manifest. After a grid change, write a new manifest and submit again; finished atoms are kept, because `config_id` is a hash of the params.
