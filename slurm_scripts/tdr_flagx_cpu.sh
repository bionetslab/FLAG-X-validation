#!/bin/bash -l
# Training data requirements (PLAN.md step 8, working_notes/training_data_requirements_plan.md).
# One array task = one lane (method x dataset x outer split x trial); the lane worker runs its atoms small -> large,
# each with a per-atom time limit (ATOM_TIME_LIMIT_S), and stops before the job's time limit is reached.
# First, run:
#   python training_data_requirements.py manifest flagx_cpu     # prints the number of lanes L; --array must be 0-<L-1>
# Then, submit from the repo root (see slurm_scripts/README.md):
#   sbatch slurm_scripts/tdr_flagx_cpu.sh
# --time / --time-min: SLURM may start a task with any limit between 2 h and 24 h (backfill gaps); the worker reads
# its job's time left from squeue, so no constant has to match (per-atom limit ATOM_TIME_LIMIT_S = 1 h).
# --constraint=icx: Gold 6326 (Ice Lake) nodes only, one CPU type for all atoms (working note, 2026-11-24).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=icx
#SBATCH --time=24:00:00
#SBATCH --time-min=02:00:00
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
