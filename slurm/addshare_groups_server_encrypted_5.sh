#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a job on Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --account=def-pbranco
#SBATCH --nodes=3
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --time=5:00:00

# Emails me when job starts, ends or fails
#SBATCH --mail-user=basar092@uottawa.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=addshare-server-group_encrypted-5-batch
#SBATCH --output=outputs/output_addshare_server_group_encrypted_5_%j.out
#SBATCH --error=errors/error_addshare_server_group_encrypted_5_%j.out
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

python ~/projects/def-pbranco/baasare/thesis/addshare_groups_server_encrypted.py svhn 5
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: $(date)"
