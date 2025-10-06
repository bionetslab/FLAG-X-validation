#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --error=%j.error
#SBATCH --output=%j.log
#SBATCH --job-name=som_classifier
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate flowcy

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

cd $HOME/FLAG-X-validation

echo "[$(date)] Starting SOM-classifier training and inference ..."
srun python run_experiments.py -m som
