#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --error=%x_%j.error
#SBATCH --output=%x_%j.log
#SBATCH --job-name=fcnn
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate flowcy

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=true

cd $HOME/FLAG-X-validation

echo "[$(date)] Starting FCNN training and inference ..."
srun python scalability.py -m fcnn

