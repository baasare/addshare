#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a job on Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --account=def-pbranco
#SBATCH --nodes=5
#SBATCH --cpus-per-task=30
#SBATCH --mem=35G
#SBATCH --time=10:00:00

# Emails me when job starts, ends or fails
#SBATCH --mail-user=basar092@uottawa.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=addshare_plus_elliptical
#SBATCH --output=outputs/addshare_plus_elliptical_output_%j.out
#SBATCH --error=errors/addshare_plus_elliptical_error_%j.out
# ---------------------------------------------------------------------
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
module load python/3.9.6
source ~/projects/def-pbranco/baasare/thesis/venv/bin/activate

python ~/projects/def-pbranco/baasare/thesis/addshare_plus_elliptical.py cifar-10 random
python ~/projects/def-pbranco/baasare/thesis/addshare_plus_elliptical.py f-mnist random
python ~/projects/def-pbranco/baasare/thesis/addshare_plus_elliptical.py mnist random
python ~/projects/def-pbranco/baasare/thesis/addshare_plus_elliptical.py svhn random
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: $(date)"
