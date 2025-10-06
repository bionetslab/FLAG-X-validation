
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=240G
#SBATCH --error=%j.error
#SBATCH --output=%j.log
#SBATCH --job-name=gatemeclass
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gmc

cd $HOME/FLAG-X-validation

echo "[$(date)] Starting GateMeClass training and inference ..."
srun python run_experiments.py -m gatemeclass
