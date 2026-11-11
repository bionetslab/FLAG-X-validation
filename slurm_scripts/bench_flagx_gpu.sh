#!/bin/bash -l
# First, run
#   python benchmark.py manifest flagx_gpu     # prints N; --array below must be 0-<N-1>
# Then, submit from the repo root on tinyx (see slurm_scripts/README.md):
#   sbatch.tinygpu slurm_scripts/bench_flagx_gpu.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=work,v100
#SBATCH --time=24:00:00
#SBATCH --job-name=bench_flagx_gpu
#SBATCH --array=0-299
#SBATCH --output=results/benchmark/logs/%x/%A_%a.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fx_flagx_gpu

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}  # Not set without --cpus-per-task; 8 = the per-GPU share
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

python benchmark.py run flagx_gpu "$SLURM_ARRAY_TASK_ID"
