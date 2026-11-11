#!/bin/bash -l
# First, run:
#   python benchmark.py manifest sklearn_cpu     # prints N; --array below must be 0-<N-1>
# Then, submit from the repo root (see slurm_scripts/README.md):
#   sbatch slurm_scripts/bench_sklearn_cpu.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-599
#SBATCH --job-name=bench_sklearn_cpu
#SBATCH --output=results/benchmark/logs/%x/%A_%a.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fx_sklearn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

python benchmark.py run sklearn_cpu "$SLURM_ARRAY_TASK_ID"

