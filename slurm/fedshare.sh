#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a job on Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --account=def-pbranco
#SBATCH --nodes=5
#SBATCH --cpus-per-task=30
#SBATCH --mem=35G
#SBATCH --time=3-00:00:00

# Emails me when job starts, ends or fails
#SBATCH --mail-user=basar092@uottawa.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=fedshare
#SBATCH --output=outputs/fedshare_output_%j.out
#SBATCH --error=errors/fedshare_error_%j.out
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

# Run the Python script
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py cifar-10 2
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py f-mnist 2
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py mnist 2
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py svhn 2


python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py cifar-10 3
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py f-mnist 3
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py mnist 3
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py svhn 3


python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py cifar-10 5
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py f-mnist 5
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py mnist 5
python ~/projects/def-pbranco/baasare/thesis/fedshare_starter.py svhn 5

# Print job finish time and exit code
echo "Job finished with exit code $? at: $(date)"