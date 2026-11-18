#!/bin/bash -l
# Training data requirements (PLAN.md step 8, working_notes/training_data_requirements_plan.md).
# One array task = one lane (method x dataset x outer split x trial); the lane worker runs its atoms small -> large,
# each with a per-atom time limit (ATOM_TIME_LIMIT_S), and stops before --time is reached (LANE_TIME_LIMIT_S).
# First, run
#   python training_data_requirements.py manifest flagx_gpu     # prints the number of lanes L; --array must be 0-<L-1>
# Then, submit from the repo root on tinyx (see slurm_scripts/README.md):
#   sbatch.tinygpu slurm_scripts/tdr_flagx_gpu.sh
# --time must equal LANE_TIME_LIMIT_S in training_data_requirements.py (per-atom limit ATOM_TIME_LIMIT_S = 1 h).
# No partition: default partition work (RTX 2080 Ti / RTX 3080). The host share per GPU is fixed (8 cores); the
# methods use 4 threads, as on Woody.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tdr_flagx_gpu
#SBATCH --array=0-149
#SBATCH --output=results/training_data_requirements/logs/%x/%A_%a.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fx_flagx_gpu

export SLURM_CPUS_PER_TASK=4  # Threads used by the methods (n_threads() reads it), as on Woody
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

python training_data_requirements.py run flagx_gpu "$SLURM_ARRAY_TASK_ID"
