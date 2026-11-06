#!/bin/bash -l
# Hyperparameter tuning, array 'deepcytof_cpu' (env fx_deepcytof, Woody): one array task = one atom (tune_hyperparameters.py).
# Submit from the repo root (see slurm_scripts/README.md):
#   python tune_hyperparameters.py manifest deepcytof_cpu     # prints N
#   sbatch --array=0-<N-1>%<throttle> slurm_scripts/tune_deepcytof_cpu.sh
# --time is the cluster maximum on purpose: an atom that does not finish within 24 h puts its config out.
# --cpus-per-task / --mem: from the timing run (working_notes/tuning_plan.md).
#SBATCH --job-name=tune_deepcytof_cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --export=NONE
#SBATCH --output=results/tune_hyperparameters/logs/%x/%A_%a.out

unset SLURM_EXPORT_ENV

module load python
conda activate fx_deepcytof

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}  # TinyGPU allocates a fixed number of cores per GPU
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
python tune_hyperparameters.py run deepcytof_cpu
