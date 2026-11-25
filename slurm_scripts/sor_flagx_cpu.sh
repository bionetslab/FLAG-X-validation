#!/bin/bash -l
# Sample order robustness (PLAN.md step 9, working_notes/sample_order_robustness_plan.md); copy of tdr_flagx_cpu.sh.
# One array task = one lane (method x dataset x outer split x trial); the lane worker runs its atoms small -> large,
# each with a per-atom time limit (ATOM_TIME_LIMIT_S), and stops before the job's time limit is reached.
# The order scheme (random | stratified) is the first argument; the lane count differs per scheme, so pass --array.
# TODO: set --time from the lane time estimate before the run (sample_order_robustness.py, TODO below the docstring).
# First, run:
#   python sample_order_robustness.py manifest <scheme> flagx_cpu     # prints the number of lanes L; --array must be 0-<L-1>
# Then, submit from the repo root (see slurm_scripts/README.md):
#   sbatch --array=0-<L-1> slurm_scripts/sor_flagx_cpu.sh <scheme>
# --time / --time-min: SLURM may start a task with any limit between 2 h and 24 h (backfill gaps); the worker reads
# its job's time left from squeue, so no constant has to match (per-atom limit ATOM_TIME_LIMIT_S = 1 h).
# --constraint=icx: Gold 6326 (Ice Lake) nodes only, one CPU type for all atoms (working note, 2026-11-24).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=icx
#SBATCH --time=24:00:00
#SBATCH --time-min=02:00:00
#SBATCH --job-name=sor_flagx_cpu
#SBATCH --output=results/sample_order_robustness/logs/%x/%A_%a.out
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

SCHEME=${1:?give the order scheme: random | stratified}
python sample_order_robustness.py run "$SCHEME" flagx_cpu "$SLURM_ARRAY_TASK_ID"
