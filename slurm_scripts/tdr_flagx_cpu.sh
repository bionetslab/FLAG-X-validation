#!/bin/bash -l
# Training data requirements (PLAN.md step 8, working_notes/training_data_requirements_plan.md).
# One array task = one lane (method x dataset x outer split x trial); the lane worker runs its atoms small -> large,
# each with a per-atom time limit (ATOM_TIME_LIMIT_S), and stops before --time is reached (LANE_TIME_LIMIT_S).
# First, run:
#   python training_data_requirements.py manifest flagx_cpu     # prints the number of lanes L; --array must be 0-<L-1>
# Then, submit from the repo root (see slurm_scripts/README.md):
#   sbatch slurm_scripts/tdr_flagx_cpu.sh
# TODO: --time must equal LANE_TIME_LIMIT_S in training_data_requirements.py (per-atom limit 1 h, may change)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-149
#SBATCH --job-name=tdr_flagx_cpu
#SBATCH --output=results/training_data_requirements/logs/%x/%A_%a.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate fx_flagx_cpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

python training_data_requirements.py run flagx_cpu "$SLURM_ARRAY_TASK_ID"
