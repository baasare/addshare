#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a job on Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --account=def-pbranco
#SBATCH --nodes=5
#SBATCH --cpus-per-task=30
#SBATCH --mem=35G
#SBATCH --time=2-00:00:00

# Emails me when job starts, ends or fails
#SBATCH --mail-user=basar092@uottawa.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=addshare-node-group_encrypted-3-batch
#SBATCH --output=outputs/output_addshare_node_encrypted_group_3_%j.out
#SBATCH --error=errors/error_addshare_node_encrypted_group_3_%j.out
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

python ~/projects/def-pbranco/baasare/thesis/addshare_groups_node_encrypted.py cifar-10 3
python ~/projects/def-pbranco/baasare/thesis/addshare_groups_node_encrypted.py f-mnist 3
python ~/projects/def-pbranco/baasare/thesis/addshare_groups_node_encrypted.py mnist 3
python ~/projects/def-pbranco/baasare/thesis/addshare_groups_node_encrypted.py svhn 3
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: $(date)"
